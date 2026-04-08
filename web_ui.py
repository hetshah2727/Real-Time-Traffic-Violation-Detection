import cv2
import easyocr
import inspect
import os
import re
import threading
import time
from datetime import datetime

import torch
from flask import Flask, Response, jsonify, render_template, request, send_from_directory
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
HELMET_MODEL_PATH = os.path.join(APP_ROOT, "model", "best.pt")
if not os.path.exists(HELMET_MODEL_PATH):
    HELMET_MODEL_PATH = os.path.join(APP_ROOT, "model", "helmet.pt")
SEATBELT_MODEL_PATH = os.path.join(APP_ROOT, "model", "seatbelt.pt")
VEHICLE_CONTEXT_MODEL_PATH = os.path.join(APP_ROOT, "model", "yolov8n.pt")
OUTPUT_DIR = os.path.join(APP_ROOT, "evidence")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def enable_torch_safe_globals():
    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is not None:
        add_safe_globals([DetectionModel])

    if "weights_only" in inspect.signature(torch.load).parameters:
        original_torch_load = torch.load

        def torch_load_compat(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_torch_load(*args, **kwargs)

        torch.load = torch_load_compat


enable_torch_safe_globals()


class HelmetDetector:
    VIOLATION_FRAMES_THRESHOLD = 10
    CHALLAN_COOLDOWN_FRAMES = 120
    TRACK_MAX_MISSED_FRAMES = 20
    MATCH_IOU_THRESHOLD = 0.3
    GLOBAL_VIOLATION_FRAMES_THRESHOLD = 12
    OCR_READ_EVERY_N_FRAMES = 30
    PLATE_OCR_MIN_CONF = 0.12
    PLATE_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    INDIAN_PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{3,4}$")
    DETECTION_EVERY_N_FRAMES = 3
    INFERENCE_IMGSZ = 512
    SEATBELT_CONF_THRESHOLD = 0.35
    VEHICLE_CONTEXT_CONF_THRESHOLD = 0.25
    CONTEXT_OVERLAP_THRESHOLD = 0.05

    def __init__(self, helmet_model_path, seatbelt_model_path, vehicle_context_model_path):
        self.model = YOLO(helmet_model_path)
        self.seatbelt_model = None
        self.vehicle_context_model = None

        try:
            self.seatbelt_model = YOLO(seatbelt_model_path)
        except Exception:
            self.seatbelt_model = None

        try:
            self.vehicle_context_model = YOLO(vehicle_context_model_path)
        except Exception:
            self.vehicle_context_model = None

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar cascade for face detection.")

        _, self.helmet_ids, self.no_helmet_ids, self.rider_ids, self.plate_ids = self.build_class_map(self.model)
        if not self.helmet_ids and not self.no_helmet_ids:
            raise RuntimeError("Could not determine helmet classes from model labels.")

        self.seatbelt_with_ids = set()
        self.seatbelt_without_ids = set()
        if self.seatbelt_model is not None:
            _, self.seatbelt_with_ids, self.seatbelt_without_ids = self.build_seatbelt_class_map(self.seatbelt_model)

        self.vehicle_four_wheeler_ids = set()
        self.vehicle_two_wheeler_ids = set()
        if self.vehicle_context_model is not None:
            _, self.vehicle_four_wheeler_ids, self.vehicle_two_wheeler_ids = self.build_vehicle_context_class_map(
                self.vehicle_context_model
            )

        self.reader = None
        try:
            self.reader = easyocr.Reader(["en"], gpu=False)
        except Exception:
            self.reader = None

        self.reset_session()

    def reset_session(self):
        self.tracks = {}
        self.next_track_id = 1
        self.frame_index = 0
        self.global_violation_frames = 0
        self.global_last_challan_frame = -self.CHALLAN_COOLDOWN_FRAMES
        self.last_detections = []
        self.last_seatbelt_detections = []
        self.last_vehicle_context_detections = []
        self.last_status = {
            "fps": 0.0,
            "active_tracks": 0,
            "last_plate": "UNKNOWN",
            "last_challan": "",
            "frame_index": 0,
        }

    @staticmethod
    def normalize_label(label):
        return "".join(ch.lower() for ch in str(label) if ch.isalnum())

    def build_class_map(self, model):
        raw_names = model.names
        if isinstance(raw_names, dict):
            names_by_id = {int(k): v for k, v in raw_names.items()}
        else:
            names_by_id = {i: v for i, v in enumerate(raw_names)}

        helmet_ids = set()
        no_helmet_ids = set()
        rider_ids = set()
        plate_ids = set()

        for class_id, class_name in names_by_id.items():
            normalized = self.normalize_label(class_name)

            if "withouthelmet" in normalized or "nohelmet" in normalized or "nohelmel" in normalized:
                no_helmet_ids.add(class_id)
                continue

            if "helmet" in normalized:
                helmet_ids.add(class_id)

            if any(token in normalized for token in ("rider", "motorcyclist", "person")):
                rider_ids.add(class_id)

            if any(token in normalized for token in ("numberplate", "licenseplate", "licenceplate", "plate")):
                plate_ids.add(class_id)

        if not helmet_ids and 0 in names_by_id:
            helmet_ids.add(0)

        return names_by_id, helmet_ids, no_helmet_ids, rider_ids, plate_ids

    def build_seatbelt_class_map(self, model):
        raw_names = model.names
        if isinstance(raw_names, dict):
            names_by_id = {int(k): v for k, v in raw_names.items()}
        else:
            names_by_id = {i: v for i, v in enumerate(raw_names)}

        with_seatbelt_ids = set()
        without_seatbelt_ids = set()

        for class_id, class_name in names_by_id.items():
            normalized = self.normalize_label(class_name)

            if any(token in normalized for token in ("withoutseatbelt", "noseatbelt", "withoutbelt", "nobelt")):
                without_seatbelt_ids.add(class_id)
                continue

            if any(token in normalized for token in ("withseatbelt", "seatbelt", "belted")):
                with_seatbelt_ids.add(class_id)

        if not with_seatbelt_ids and 0 in names_by_id:
            with_seatbelt_ids.add(0)
        if not without_seatbelt_ids and 1 in names_by_id:
            without_seatbelt_ids.add(1)

        return names_by_id, with_seatbelt_ids, without_seatbelt_ids

    def build_vehicle_context_class_map(self, model):
        raw_names = model.names
        if isinstance(raw_names, dict):
            names_by_id = {int(k): v for k, v in raw_names.items()}
        else:
            names_by_id = {i: v for i, v in enumerate(raw_names)}

        four_wheeler_ids = set()
        two_wheeler_ids = set()

        for class_id, class_name in names_by_id.items():
            normalized = self.normalize_label(class_name)

            if any(token in normalized for token in ("car", "truck", "bus", "van", "pickup", "suv")):
                four_wheeler_ids.add(class_id)

            if any(token in normalized for token in ("motorcycle", "motorbike", "bike", "bicycle", "scooter", "moped")):
                two_wheeler_ids.add(class_id)

        for fallback_id in (2, 5, 7):
            if fallback_id in names_by_id:
                four_wheeler_ids.add(fallback_id)
        for fallback_id in (1, 3):
            if fallback_id in names_by_id:
                two_wheeler_ids.add(fallback_id)

        return names_by_id, four_wheeler_ids, two_wheeler_ids

    def best_overlap_ratio(self, subject_box, candidate_boxes):
        best_ratio = 0.0
        subject_area = max(1, self.area(subject_box))
        for candidate_box in candidate_boxes:
            inter = self.intersection_area(subject_box, candidate_box)
            if inter <= 0:
                continue
            ratio = inter / subject_area
            if ratio > best_ratio:
                best_ratio = ratio
        return best_ratio

    @staticmethod
    def to_int_bbox(xyxy):
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        return x1, y1, x2, y2

    @staticmethod
    def area(box):
        x1, y1, x2, y2 = box
        return max(0, x2 - x1) * max(0, y2 - y1)

    @staticmethod
    def intersection_area(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        x1 = max(ax1, bx1)
        y1 = max(ay1, by1)
        x2 = min(ax2, bx2)
        y2 = min(ay2, by2)
        if x2 <= x1 or y2 <= y1:
            return 0
        return (x2 - x1) * (y2 - y1)

    def iou(self, box_a, box_b):
        inter = self.intersection_area(box_a, box_b)
        union = self.area(box_a) + self.area(box_b) - inter
        if union <= 0:
            return 0.0
        return inter / union

    def box_matches_rider(self, object_box, rider_box):
        inter = self.intersection_area(object_box, rider_box)
        if inter == 0:
            return False
        return (inter / max(1, self.area(object_box)) >= 0.3) or (inter / max(1, self.area(rider_box)) >= 0.02)

    @staticmethod
    def expand_face_to_rider(face, frame_shape):
        x, y, w, h = face
        frame_h, frame_w = frame_shape[:2]
        x1 = max(0, x - int(0.6 * w))
        y1 = max(0, y - int(0.4 * h))
        x2 = min(frame_w - 1, x + w + int(0.6 * w))
        y2 = min(frame_h - 1, y + h + int(2.2 * h))
        return x1, y1, x2, y2

    @staticmethod
    def expand_box(box, frame_shape, x_pad_ratio=0.08, y_pad_ratio=0.18):
        x1, y1, x2, y2 = box
        frame_h, frame_w = frame_shape[:2]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        pad_x = int(bw * x_pad_ratio)
        pad_y = int(bh * y_pad_ratio)
        ex1 = max(0, x1 - pad_x)
        ey1 = max(0, y1 - pad_y)
        ex2 = min(frame_w - 1, x2 + pad_x)
        ey2 = min(frame_h - 1, y2 + pad_y)
        return ex1, ey1, ex2, ey2

    def normalize_plate_text(self, text):
        cleaned = re.sub(r"[^A-Za-z0-9]", "", text).upper()
        if len(cleaned) < 5:
            return ""
        return cleaned

    def plate_text_score(self, text):
        if not text:
            return 0
        if self.INDIAN_PLATE_REGEX.fullmatch(text):
            return 3
        has_alpha = any(ch.isalpha() for ch in text)
        has_digit = any(ch.isdigit() for ch in text)
        if has_alpha and has_digit and 5 <= len(text) <= 12:
            return 2
        return 0

    def find_best_plate_box_for_rider(self, rider_box, plate_boxes):
        best_box = None
        best_score = 0.0
        for plate_box in plate_boxes:
            inter = self.intersection_area(plate_box, rider_box)
            if inter <= 0:
                continue
            plate_cover = inter / max(1, self.area(plate_box))
            rider_cover = inter / max(1, self.area(rider_box))
            score = plate_cover + rider_cover
            if score > best_score:
                best_score = score
                best_box = plate_box
        return best_box

    def fallback_plate_search_boxes(self, rider_box, frame_shape):
        x1, y1, x2, y2 = rider_box
        frame_h, frame_w = frame_shape[:2]
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)

        lower_x1 = max(0, int(x1 + 0.1 * width))
        lower_x2 = min(frame_w - 1, int(x2 - 0.1 * width))
        lower_y1 = max(0, int(y1 + 0.45 * height))
        lower_y2 = min(frame_h - 1, y2)

        full_box = self.expand_box(rider_box, frame_shape, x_pad_ratio=0.05, y_pad_ratio=0.05)
        boxes = []
        if lower_x2 > lower_x1 and lower_y2 > lower_y1:
            boxes.append((lower_x1, lower_y1, lower_x2, lower_y2))
        boxes.append(full_box)
        return boxes

    def read_number_plate_from_crop(self, crop):
        if self.reader is None or crop is None or crop.size == 0:
            return ""

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        boosted = cv2.convertScaleAbs(denoised, alpha=1.4, beta=8)

        candidates = []
        try:
            candidates.extend(
                self.reader.readtext(
                    boosted,
                    detail=1,
                    paragraph=False,
                    allowlist=self.PLATE_ALLOWLIST,
                    rotation_info=[0],
                )
            )
        except Exception:
            pass

        best_text = ""
        best_score = 0
        best_conf = 0.0
        token_entries = []

        for bbox, text, conf in candidates:
            plate_text = self.normalize_plate_text(text)
            if not plate_text:
                continue

            if conf >= (self.PLATE_OCR_MIN_CONF * 0.75):
                min_x = min(point[0] for point in bbox)
                min_y = min(point[1] for point in bbox)
                token_entries.append((min_y, min_x, plate_text))

            score = self.plate_text_score(plate_text)
            if score == 0 or conf < self.PLATE_OCR_MIN_CONF:
                continue

            if score > best_score or (score == best_score and conf > best_conf):
                best_score = score
                best_conf = conf
                best_text = plate_text

        if token_entries:
            token_entries.sort(key=lambda item: (item[0], item[1]))
            merged_tokens = []
            for _, _, token in token_entries:
                if token not in merged_tokens:
                    merged_tokens.append(token)
                if len(merged_tokens) >= 3:
                    break
            merged_text = self.normalize_plate_text("".join(merged_tokens))
            merged_score = self.plate_text_score(merged_text)
            if merged_score > best_score or (merged_score == best_score and len(merged_text) > len(best_text)):
                best_text = merged_text

        return best_text

    def read_number_plate(self, frame, plate_box):
        if plate_box is None:
            return ""

        x1, y1, x2, y2 = self.expand_box(plate_box, frame.shape)
        if x2 <= x1 or y2 <= y1:
            return ""

        crop = frame[y1:y2, x1:x2]
        return self.read_number_plate_from_crop(crop)

    def plate_text_for_pdf(self, plate_text):
        cleaned = self.normalize_plate_text(str(plate_text or ""))
        return cleaned if cleaned else "UNKNOWN"

    def generate_challan(self, frame, rider_id=None, plate_text="UNKNOWN", violation_reason="Helmet Not Worn"):
        fname = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = os.path.join(OUTPUT_DIR, f"frame_{fname}.jpg")
        pdf_path = os.path.join(OUTPUT_DIR, f"challan_{fname}.pdf")

        cv2.imwrite(img_path, frame)
        c = canvas.Canvas(pdf_path, pagesize=A4)
        _, h = A4

        c.setFont("Helvetica-Bold", 18)
        c.drawString(50, h - 50, "AI-TraQ Traffic Violation Challan")

        c.setFont("Helvetica", 12)
        c.drawString(50, h - 100, f"Violation: {violation_reason}")
        c.drawString(50, h - 120, f"Date & Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        c.drawString(50, h - 140, "Fine Amount: INR 500")
        c.drawString(50, h - 160, f"Rider ID: {rider_id if rider_id is not None else 'N/A'}")

        plate_value = self.plate_text_for_pdf(plate_text)
        c.setFont("Helvetica-Bold", 13)
        c.drawString(50, h - 182, f"Number Plate: {plate_value}")
        c.setFont("Helvetica", 11)
        if plate_value == "UNKNOWN":
            c.drawString(50, h - 200, "Plate OCR Status: Not readable")

        try:
            c.drawImage(img_path, 50, h - 500, width=400, height=300)
        except Exception:
            pass

        c.save()
        return os.path.basename(pdf_path), plate_value

    def process_frame(self, frame):
        self.frame_index += 1
        run_inference = (self.frame_index % self.DETECTION_EVERY_N_FRAMES == 1) or (not self.last_detections)

        if run_inference:
            results = self.model.predict(
                frame,
                conf=0.40,
                verbose=False,
                imgsz=((int(self.INFERENCE_IMGSZ) + 31) // 32) * 32,
                max_det=20,
            )[0]

            detections = []
            if results.boxes is not None and len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy()
                confidences = results.boxes.conf.cpu().numpy()
                for xyxy, cls_id, conf in zip(boxes, classes, confidences):
                    detections.append({
                        "bbox": self.to_int_bbox(xyxy),
                        "cls": int(cls_id),
                        "conf": float(conf),
                    })
            self.last_detections = detections

            if self.seatbelt_model is not None and (self.seatbelt_with_ids or self.seatbelt_without_ids):
                seatbelt_results = self.seatbelt_model.predict(
                    frame,
                    conf=self.SEATBELT_CONF_THRESHOLD,
                    verbose=False,
                    imgsz=((int(self.INFERENCE_IMGSZ) + 31) // 32) * 32,
                    max_det=30,
                )[0]

                seatbelt_detections = []
                if seatbelt_results.boxes is not None and len(seatbelt_results.boxes) > 0:
                    sb_boxes = seatbelt_results.boxes.xyxy.cpu().numpy()
                    sb_classes = seatbelt_results.boxes.cls.cpu().numpy()
                    sb_confidences = seatbelt_results.boxes.conf.cpu().numpy()

                    for xyxy, cls_id, conf in zip(sb_boxes, sb_classes, sb_confidences):
                        seatbelt_detections.append(
                            {
                                "bbox": self.to_int_bbox(xyxy),
                                "cls": int(cls_id),
                                "conf": float(conf),
                            }
                        )
                self.last_seatbelt_detections = seatbelt_detections
            else:
                self.last_seatbelt_detections = []

            if self.vehicle_context_model is not None and (self.vehicle_four_wheeler_ids or self.vehicle_two_wheeler_ids):
                vehicle_results = self.vehicle_context_model.predict(
                    frame,
                    conf=self.VEHICLE_CONTEXT_CONF_THRESHOLD,
                    verbose=False,
                    imgsz=((int(self.INFERENCE_IMGSZ) + 31) // 32) * 32,
                    max_det=40,
                )[0]

                vehicle_detections = []
                if vehicle_results.boxes is not None and len(vehicle_results.boxes) > 0:
                    v_boxes = vehicle_results.boxes.xyxy.cpu().numpy()
                    v_classes = vehicle_results.boxes.cls.cpu().numpy()
                    v_confidences = vehicle_results.boxes.conf.cpu().numpy()

                    for xyxy, cls_id, conf in zip(v_boxes, v_classes, v_confidences):
                        vehicle_detections.append(
                            {
                                "bbox": self.to_int_bbox(xyxy),
                                "cls": int(cls_id),
                                "conf": float(conf),
                            }
                        )
                self.last_vehicle_context_detections = vehicle_detections
            else:
                self.last_vehicle_context_detections = []
        else:
            detections = self.last_detections

        helmet_boxes = [d["bbox"] for d in detections if d["cls"] in self.helmet_ids]
        no_helmet_boxes = [d["bbox"] for d in detections if d["cls"] in self.no_helmet_ids]
        rider_boxes = [d["bbox"] for d in detections if d["cls"] in self.rider_ids]
        plate_boxes = [d["bbox"] for d in detections if d["cls"] in self.plate_ids]
        seatbelt_with_boxes = [
            d["bbox"] for d in self.last_seatbelt_detections if d["cls"] in self.seatbelt_with_ids
        ]
        seatbelt_without_boxes = [
            d["bbox"] for d in self.last_seatbelt_detections if d["cls"] in self.seatbelt_without_ids
        ]
        four_wheeler_boxes = [
            d["bbox"] for d in self.last_vehicle_context_detections if d["cls"] in self.vehicle_four_wheeler_ids
        ]
        two_wheeler_boxes = [
            d["bbox"] for d in self.last_vehicle_context_detections if d["cls"] in self.vehicle_two_wheeler_ids
        ]

        used_face_fallback = False
        if not rider_boxes:
            used_face_fallback = True
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            rider_boxes = [self.expand_face_to_rider(face, frame.shape) for face in faces]

        assigned_track_ids = set()
        frame_has_violation = False
        frame_violation_rider_id = None
        frame_violation_plate = "UNKNOWN"
        frame_violation_reason = "Helmet Not Worn"
        challan_generated = ""

        for rider_box in rider_boxes:
            best_track_id = None
            best_iou = 0.0
            for track_id, track in self.tracks.items():
                if self.frame_index - track["last_seen"] > self.TRACK_MAX_MISSED_FRAMES:
                    continue
                score = self.iou(rider_box, track["bbox"])
                if score > best_iou:
                    best_iou = score
                    best_track_id = track_id

            if best_track_id is None or best_iou < self.MATCH_IOU_THRESHOLD:
                best_track_id = self.next_track_id
                self.next_track_id += 1
                self.tracks[best_track_id] = {
                    "bbox": rider_box,
                    "last_seen": self.frame_index,
                    "violation_frames": 0,
                    "last_challan_frame": -self.CHALLAN_COOLDOWN_FRAMES,
                    "plate_text": "UNKNOWN",
                    "last_plate_read_frame": -self.OCR_READ_EVERY_N_FRAMES,
                }

            track = self.tracks[best_track_id]
            track["bbox"] = rider_box
            track["last_seen"] = self.frame_index
            assigned_track_ids.add(best_track_id)

            has_helmet = any(self.box_matches_rider(h_box, rider_box) for h_box in helmet_boxes)
            has_no_helmet = any(self.box_matches_rider(nh_box, rider_box) for nh_box in no_helmet_boxes)
            has_seatbelt = any(self.box_matches_rider(sb_box, rider_box) for sb_box in seatbelt_with_boxes)
            has_no_seatbelt = any(self.box_matches_rider(nsb_box, rider_box) for nsb_box in seatbelt_without_boxes)

            two_wheeler_overlap = self.best_overlap_ratio(rider_box, two_wheeler_boxes)
            four_wheeler_overlap = self.best_overlap_ratio(rider_box, four_wheeler_boxes)
            in_two_wheeler_context = (
                two_wheeler_overlap >= self.CONTEXT_OVERLAP_THRESHOLD and two_wheeler_overlap >= four_wheeler_overlap
            )
            in_four_wheeler_context = (
                four_wheeler_overlap >= self.CONTEXT_OVERLAP_THRESHOLD and four_wheeler_overlap > two_wheeler_overlap
            )
            if used_face_fallback and in_two_wheeler_context:
                in_two_wheeler_context = False
            if used_face_fallback and not in_four_wheeler_context:
                in_two_wheeler_context = False

            if in_four_wheeler_context:
                status = "N/A"
            elif has_no_helmet:
                status = "NO"
            elif has_helmet:
                status = "YES"
            else:
                status = "NO"

            seatbelt_applicable = (
                in_four_wheeler_context
                and self.seatbelt_model is not None
                and (self.seatbelt_with_ids or self.seatbelt_without_ids)
            )
            if not seatbelt_applicable:
                seatbelt_status = "N/A"
            elif has_no_seatbelt:
                seatbelt_status = "NO"
            elif has_seatbelt:
                seatbelt_status = "YES"
            else:
                seatbelt_status = "UNK"

            helmet_violation = status == "NO" and not in_four_wheeler_context
            seatbelt_violation = seatbelt_status == "NO" and in_four_wheeler_context
            is_violation = helmet_violation or seatbelt_violation

            if helmet_violation and seatbelt_violation:
                violation_reason = "Helmet Not Worn + Seatbelt Not Worn"
            elif helmet_violation:
                violation_reason = "Helmet Not Worn"
            elif seatbelt_violation:
                violation_reason = "Seatbelt Not Worn"
            else:
                violation_reason = "No Violation"

            best_plate_box = self.find_best_plate_box_for_rider(rider_box, plate_boxes)
            should_read_plate = (
                is_violation
                and run_inference
                and track["violation_frames"] >= max(1, self.VIOLATION_FRAMES_THRESHOLD - 3)
                and (
                    track["plate_text"] == "UNKNOWN"
                    or self.frame_index - track["last_plate_read_frame"] >= self.OCR_READ_EVERY_N_FRAMES
                )
            )
            if should_read_plate:
                detected_plate = ""
                if best_plate_box is not None:
                    detected_plate = self.read_number_plate(frame, best_plate_box)

                if not detected_plate:
                    fallback_boxes = self.fallback_plate_search_boxes(rider_box, frame.shape)
                    if fallback_boxes:
                        detected_plate = self.read_number_plate(frame, fallback_boxes[0])

                track["last_plate_read_frame"] = self.frame_index
                if detected_plate:
                    track["plate_text"] = detected_plate

            if is_violation:
                track["violation_frames"] += 1
                frame_has_violation = True
                if frame_violation_rider_id is None:
                    frame_violation_rider_id = best_track_id
                    frame_violation_plate = track["plate_text"]
                    frame_violation_reason = violation_reason
            else:
                track["violation_frames"] = 0

            if (
                is_violation
                and track["violation_frames"] >= self.VIOLATION_FRAMES_THRESHOLD
                and self.frame_index - track["last_challan_frame"] >= self.CHALLAN_COOLDOWN_FRAMES
            ):
                challan_name, plate_value = self.generate_challan(
                    frame,
                    rider_id=best_track_id,
                    plate_text=track["plate_text"],
                    violation_reason=violation_reason,
                )
                challan_generated = challan_name
                track["last_challan_frame"] = self.frame_index
                track["violation_frames"] = 0
                self.last_status["last_plate"] = plate_value

            color = (0, 255, 0) if not is_violation else (0, 0, 255)
            x1, y1, x2, y2 = rider_box
            label = f"ID {best_track_id} | Helmet: {status} | Seatbelt: {seatbelt_status} | Plate: {track['plate_text']}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if frame_has_violation:
            self.global_violation_frames += 1
        else:
            self.global_violation_frames = 0

        if (
            not challan_generated
            and frame_has_violation
            and self.global_violation_frames >= self.GLOBAL_VIOLATION_FRAMES_THRESHOLD
            and self.frame_index - self.global_last_challan_frame >= self.CHALLAN_COOLDOWN_FRAMES
        ):
            challan_name, plate_value = self.generate_challan(
                frame,
                rider_id=frame_violation_rider_id,
                plate_text=frame_violation_plate,
                violation_reason=frame_violation_reason,
            )
            challan_generated = challan_name
            self.global_last_challan_frame = self.frame_index
            self.global_violation_frames = 0
            self.last_status["last_plate"] = plate_value

        stale_tracks = [
            track_id
            for track_id, track in self.tracks.items()
            if track_id not in assigned_track_ids and self.frame_index - track["last_seen"] > self.TRACK_MAX_MISSED_FRAMES
        ]
        for track_id in stale_tracks:
            del self.tracks[track_id]

        self.last_status.update(
            {
                "active_tracks": len(self.tracks),
                "frame_index": self.frame_index,
                "last_challan": challan_generated,
            }
        )

        return frame


class RuntimeState:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.stop_event = threading.Event()
        self.latest_jpeg = None
        self.error = ""
        self.fps = 0.0
        self.camera_index = 0
        self.detector = HelmetDetector(
            HELMET_MODEL_PATH,
            SEATBELT_MODEL_PATH,
            VEHICLE_CONTEXT_MODEL_PATH,
        )


state = RuntimeState()
app = Flask(__name__, template_folder="templates", static_folder="static")


def build_challan_list(limit=40):
    entries = []
    for name in os.listdir(OUTPUT_DIR):
        if not name.lower().endswith(".pdf"):
            continue
        full_path = os.path.join(OUTPUT_DIR, name)
        entries.append(
            {
                "name": name,
                "modified": datetime.fromtimestamp(os.path.getmtime(full_path)).strftime("%Y-%m-%d %H:%M:%S"),
                "url": f"/evidence/{name}",
            }
        )
    entries.sort(key=lambda item: item["modified"], reverse=True)
    return entries[:limit]


def camera_loop():
    if os.name == "nt":
        cap = cv2.VideoCapture(state.camera_index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(state.camera_index)

    if not cap.isOpened():
        with state.lock:
            state.error = "Could not open camera."
            state.running = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    cap.set(cv2.CAP_PROP_FPS, 30)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    state.detector.reset_session()
    last_tick = time.time()
    frames_in_window = 0

    while not state.stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            with state.lock:
                state.error = "Camera frame read failed."
            time.sleep(0.03)
            continue

        annotated = state.detector.process_frame(frame)
        ok, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            continue

        frames_in_window += 1
        now = time.time()
        if now - last_tick >= 1.0:
            fps = frames_in_window / (now - last_tick)
            frames_in_window = 0
            last_tick = now
            with state.lock:
                state.fps = round(fps, 1)

        with state.lock:
            state.latest_jpeg = encoded.tobytes()
            state.error = ""

    cap.release()
    with state.lock:
        state.running = False


def mjpeg_stream():
    while True:
        with state.lock:
            frame = state.latest_jpeg
        if frame is None:
            time.sleep(0.05)
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/video_feed")
def video_feed():
    return Response(mjpeg_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.post("/api/start")
def api_start():
    payload = request.get_json(silent=True) or {}
    camera_index = int(payload.get("camera_index", 0))

    with state.lock:
        if state.running:
            return jsonify({"ok": True, "message": "Already running."})
        state.running = True
        state.camera_index = camera_index
        state.stop_event.clear()

    state.thread = threading.Thread(target=camera_loop, daemon=True)
    state.thread.start()
    return jsonify({"ok": True, "message": "Detection started."})


@app.post("/api/stop")
def api_stop():
    with state.lock:
        if not state.running:
            return jsonify({"ok": True, "message": "Already stopped."})
        state.stop_event.set()

    if state.thread is not None:
        state.thread.join(timeout=2.0)

    with state.lock:
        state.running = False
    return jsonify({"ok": True, "message": "Detection stopped."})


@app.get("/api/status")
def api_status():
    with state.lock:
        running = state.running
        fps = state.fps
        error = state.error

    detector_status = state.detector.last_status
    return jsonify(
        {
            "running": running,
            "fps": fps,
            "error": error,
            "active_tracks": detector_status.get("active_tracks", 0),
            "frame_index": detector_status.get("frame_index", 0),
            "last_plate": detector_status.get("last_plate", "UNKNOWN"),
            "last_challan": detector_status.get("last_challan", ""),
        }
    )


@app.get("/api/challans")
def api_challans():
    return jsonify({"items": build_challan_list()})


@app.get("/evidence/<path:filename>")
def evidence_file(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8502, debug=False, threaded=True)
