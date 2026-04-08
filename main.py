import cv2
import torch
import inspect
import re
import easyocr
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import os
from datetime import datetime


def enable_torch_safe_globals():
    """Configure trusted checkpoint loading for local Ultralytics weights."""
    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is None:
        pass
    else:
        add_safe_globals([DetectionModel])

    if "weights_only" in inspect.signature(torch.load).parameters:
        original_torch_load = torch.load

        def torch_load_compat(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_torch_load(*args, **kwargs)

        torch.load = torch_load_compat


enable_torch_safe_globals()

HELMET_MODEL_PATH = "model/best.pt" if os.path.exists("model/best.pt") else "model/helmet.pt"
SEATBELT_MODEL_PATH = "model/seatbelt.pt"
VEHICLE_CONTEXT_MODEL_PATH = "model/yolov8n.pt"

helmet_model = YOLO(HELMET_MODEL_PATH)
seatbelt_model = None
vehicle_context_model = None
try:
    seatbelt_model = YOLO(SEATBELT_MODEL_PATH)
except Exception as exc:
    print(f"Warning: seatbelt model initialization failed: {exc}")

try:
    vehicle_context_model = YOLO(VEHICLE_CONTEXT_MODEL_PATH)
except Exception as exc:
    print(f"Warning: vehicle context model initialization failed: {exc}")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

OUTPUT_DIR = "evidence"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VIOLATION_FRAMES_THRESHOLD = 10
CHALLAN_COOLDOWN_FRAMES = 120
TRACK_MAX_MISSED_FRAMES = 20
MATCH_IOU_THRESHOLD = 0.3
GLOBAL_VIOLATION_FRAMES_THRESHOLD = 12
OCR_READ_EVERY_N_FRAMES = 30
FULL_FRAME_OCR_EVERY_N_FRAMES = 90
PLATE_OCR_MIN_CONF = 0.12
PLATE_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
INDIAN_PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{3,4}$")
ENABLE_FULL_FRAME_OCR_FALLBACK = False
CAMERA_WIDTH = 960
CAMERA_HEIGHT = 540
CAMERA_FPS = 30
DETECTION_EVERY_N_FRAMES = 3
INFERENCE_IMGSZ = 512
INFERENCE_IMGSZ = ((int(INFERENCE_IMGSZ) + 31) // 32) * 32
SEATBELT_CONF_THRESHOLD = 0.35
VEHICLE_CONTEXT_CONF_THRESHOLD = 0.25
CONTEXT_OVERLAP_THRESHOLD = 0.05


def normalize_label(label):
    return "".join(ch.lower() for ch in str(label) if ch.isalnum())


def build_class_map(model):
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
        normalized = normalize_label(class_name)

        if "withouthelmet" in normalized or "nohelmet" in normalized or "nohelmel" in normalized:
            no_helmet_ids.add(class_id)
            continue

        if "helmet" in normalized:
            helmet_ids.add(class_id)

        if any(token in normalized for token in ("rider", "motorcyclist", "person")):
            rider_ids.add(class_id)

        if any(token in normalized for token in ("numberplate", "licenseplate", "licenceplate", "plate")):
            plate_ids.add(class_id)

    # Fallback for generic 2-class models where class names are missing or ambiguous.
    if not helmet_ids and 0 in names_by_id:
        helmet_ids.add(0)

    return names_by_id, helmet_ids, no_helmet_ids, rider_ids, plate_ids


def build_seatbelt_class_map(model):
    raw_names = model.names
    if isinstance(raw_names, dict):
        names_by_id = {int(k): v for k, v in raw_names.items()}
    else:
        names_by_id = {i: v for i, v in enumerate(raw_names)}

    with_seatbelt_ids = set()
    without_seatbelt_ids = set()

    for class_id, class_name in names_by_id.items():
        normalized = normalize_label(class_name)

        if any(token in normalized for token in ("withoutseatbelt", "noseatbelt", "withoutbelt", "nobelt")):
            without_seatbelt_ids.add(class_id)
            continue

        if any(token in normalized for token in ("withseatbelt", "seatbelt", "belted")):
            with_seatbelt_ids.add(class_id)

    # Fallback for the common 2-class setup: [with_seatbelt, without_seatbelt].
    if not with_seatbelt_ids and 0 in names_by_id:
        with_seatbelt_ids.add(0)
    if not without_seatbelt_ids and 1 in names_by_id:
        without_seatbelt_ids.add(1)

    return names_by_id, with_seatbelt_ids, without_seatbelt_ids


def build_vehicle_context_class_map(model):
    raw_names = model.names
    if isinstance(raw_names, dict):
        names_by_id = {int(k): v for k, v in raw_names.items()}
    else:
        names_by_id = {i: v for i, v in enumerate(raw_names)}

    four_wheeler_ids = set()
    two_wheeler_ids = set()

    for class_id, class_name in names_by_id.items():
        normalized = normalize_label(class_name)

        if any(token in normalized for token in ("car", "truck", "bus", "van", "pickup", "suv")):
            four_wheeler_ids.add(class_id)

        if any(token in normalized for token in ("motorcycle", "motorbike", "bike", "bicycle", "scooter", "moped")):
            two_wheeler_ids.add(class_id)

    # COCO fallback IDs for yolov8n: car=2, motorcycle=3, bus=5, truck=7, bicycle=1.
    for fallback_id in (2, 5, 7):
        if fallback_id in names_by_id:
            four_wheeler_ids.add(fallback_id)
    for fallback_id in (1, 3):
        if fallback_id in names_by_id:
            two_wheeler_ids.add(fallback_id)

    return names_by_id, four_wheeler_ids, two_wheeler_ids


def best_overlap_ratio(subject_box, candidate_boxes):
    best_ratio = 0.0
    subject_area = max(1, area(subject_box))
    for candidate_box in candidate_boxes:
        inter = intersection_area(subject_box, candidate_box)
        if inter <= 0:
            continue
        ratio = inter / subject_area
        if ratio > best_ratio:
            best_ratio = ratio
    return best_ratio


def to_int_bbox(xyxy):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    return x1, y1, x2, y2


def area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


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


def iou(box_a, box_b):
    inter = intersection_area(box_a, box_b)
    union = area(box_a) + area(box_b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def box_matches_rider(object_box, rider_box):
    inter = intersection_area(object_box, rider_box)
    if inter == 0:
        return False
    return (inter / max(1, area(object_box)) >= 0.3) or (inter / max(1, area(rider_box)) >= 0.02)


def expand_face_to_rider(face, frame_shape):
    x, y, w, h = face
    frame_h, frame_w = frame_shape[:2]

    x1 = max(0, x - int(0.6 * w))
    y1 = max(0, y - int(0.4 * h))
    x2 = min(frame_w - 1, x + w + int(0.6 * w))
    y2 = min(frame_h - 1, y + h + int(2.2 * h))
    return x1, y1, x2, y2


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


def normalize_plate_text(text):
    cleaned = re.sub(r"[^A-Za-z0-9]", "", text).upper()
    if len(cleaned) < 5:
        return ""
    return cleaned


def plate_text_score(text):
    if not text:
        return 0
    if INDIAN_PLATE_REGEX.fullmatch(text):
        return 3

    has_alpha = any(ch.isalpha() for ch in text)
    has_digit = any(ch.isdigit() for ch in text)
    if has_alpha and has_digit and 5 <= len(text) <= 12:
        return 2

    return 0


def find_best_plate_box_for_rider(rider_box, plate_boxes):
    best_box = None
    best_score = 0.0
    for plate_box in plate_boxes:
        inter = intersection_area(plate_box, rider_box)
        if inter <= 0:
            continue
        plate_cover = inter / max(1, area(plate_box))
        rider_cover = inter / max(1, area(rider_box))
        score = plate_cover + rider_cover
        if score > best_score:
            best_score = score
            best_box = plate_box
    return best_box


def fallback_plate_search_boxes(rider_box, frame_shape):
    x1, y1, x2, y2 = rider_box
    frame_h, frame_w = frame_shape[:2]
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)

    # Lower-central rider area catches bike plate region in most views.
    lower_x1 = max(0, int(x1 + 0.1 * width))
    lower_x2 = min(frame_w - 1, int(x2 - 0.1 * width))
    lower_y1 = max(0, int(y1 + 0.45 * height))
    lower_y2 = min(frame_h - 1, y2)

    full_box = expand_box(rider_box, frame_shape, x_pad_ratio=0.05, y_pad_ratio=0.05)
    boxes = []
    if lower_x2 > lower_x1 and lower_y2 > lower_y1:
        boxes.append((lower_x1, lower_y1, lower_x2, lower_y2))
    boxes.append(full_box)
    return boxes


def read_number_plate(frame, plate_box, reader):
    if reader is None:
        return ""

    if plate_box is None:
        return ""

    x1, y1, x2, y2 = expand_box(plate_box, frame.shape)
    if x2 <= x1 or y2 <= y1:
        return ""

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return ""

    return read_number_plate_from_crop(crop, reader)


def read_number_plate_from_crop(crop, reader):
    if crop is None or crop.size == 0:
        return ""

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    boosted = cv2.convertScaleAbs(denoised, alpha=1.4, beta=8)
    candidates = []
    for image in (boosted,):
        try:
            candidates.extend(
                reader.readtext(
                    image,
                    detail=1,
                    paragraph=False,
                    allowlist=PLATE_ALLOWLIST,
                    rotation_info=[0],
                )
            )
        except Exception:
            continue

    if not candidates:
        return ""

    best_text = ""
    best_score = 0
    best_conf = 0.0
    token_entries = []
    for bbox, text, conf in candidates:
        plate_text = normalize_plate_text(text)
        if not plate_text:
            continue

        if conf >= (PLATE_OCR_MIN_CONF * 0.75):
            min_x = min(point[0] for point in bbox)
            min_y = min(point[1] for point in bbox)
            token_entries.append((min_y, min_x, plate_text))

        score = plate_text_score(plate_text)
        if score == 0 or conf < PLATE_OCR_MIN_CONF:
            continue

        if score > best_score or (score == best_score and conf > best_conf):
            best_score = score
            best_conf = conf
            best_text = plate_text

    # Merge nearby OCR tokens (e.g., "GJ06AB" + "1003") for handwritten multi-line plates.
    if token_entries:
        token_entries.sort(key=lambda item: (item[0], item[1]))
        merged_tokens = []
        for _, _, token in token_entries:
            if token not in merged_tokens:
                merged_tokens.append(token)
            if len(merged_tokens) >= 3:
                break

        merged_text = normalize_plate_text("".join(merged_tokens))
        merged_score = plate_text_score(merged_text)
        if merged_score > best_score or (merged_score == best_score and len(merged_text) > len(best_text)):
            best_text = merged_text

    return best_text


def read_number_plate_full_frame(frame, reader):
    if reader is None:
        return ""

    frame_h, frame_w = frame.shape[:2]
    scaled = frame
    if frame_w > 960:
        new_h = max(1, int(frame_h * (960 / frame_w)))
        scaled = cv2.resize(frame, (960, new_h), interpolation=cv2.INTER_AREA)

    return read_number_plate_from_crop(scaled, reader)


def plate_text_for_pdf(plate_text):
    cleaned = normalize_plate_text(str(plate_text or ""))
    return cleaned if cleaned else "UNKNOWN"


def generate_challan(frame, rider_id=None, plate_text="UNKNOWN", violation_reason="Helmet Not Worn"):
    fname = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = f"{OUTPUT_DIR}/frame_{fname}.jpg"
    pdf_path = f"{OUTPUT_DIR}/challan_{fname}.pdf"

    cv2.imwrite(img_path, frame)

    c = canvas.Canvas(pdf_path, pagesize=A4)
    w, h = A4

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, h - 50, "AI-TraQ Traffic Violation Challan")

    c.setFont("Helvetica", 12)
    c.drawString(50, h - 100, f"Violation: {violation_reason}")
    c.drawString(50, h - 120, f"Date & Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(50, h - 140, "Fine Amount: INR 500")
    c.drawString(50, h - 160, f"Rider ID: {rider_id if rider_id is not None else 'N/A'}")
    plate_value = plate_text_for_pdf(plate_text)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, h - 182, f"Number Plate: {plate_value}")
    c.setFont("Helvetica", 11)
    if plate_value == "UNKNOWN":
        c.drawString(50, h - 200, "Plate OCR Status: Not readable")

    try:
        c.drawImage(img_path, 50, h - 500, width=400, height=300)
    except Exception as exc:
        print(f"Warning: could not embed evidence image in PDF: {exc}")

    c.save()
    print(f"Challan generated: {pdf_path} | Plate: {plate_value}")


if os.name == "nt":
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
else:
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Unable to open camera. Check if your webcam is connected and not in use.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
try:
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
except Exception:
    pass

if face_cascade.empty():
    raise RuntimeError("Failed to load Haar cascade for face detection.")

_, helmet_ids, no_helmet_ids, rider_ids, plate_ids = build_class_map(helmet_model)

seatbelt_with_ids = set()
seatbelt_without_ids = set()
if seatbelt_model is not None:
    _, seatbelt_with_ids, seatbelt_without_ids = build_seatbelt_class_map(seatbelt_model)
    if not seatbelt_with_ids and not seatbelt_without_ids:
        print("Warning: could not determine seatbelt classes from seatbelt model labels.")

vehicle_four_wheeler_ids = set()
vehicle_two_wheeler_ids = set()
if vehicle_context_model is not None:
    _, vehicle_four_wheeler_ids, vehicle_two_wheeler_ids = build_vehicle_context_class_map(vehicle_context_model)
    if not vehicle_four_wheeler_ids:
        print("Warning: could not determine four-wheeler classes from vehicle context model labels.")

if not helmet_ids and not no_helmet_ids:
    raise RuntimeError("Could not determine helmet classes from model labels.")

tracks = {}
next_track_id = 1
frame_index = 0
global_violation_frames = 0
global_last_challan_frame = -CHALLAN_COOLDOWN_FRAMES
last_detections = []
last_seatbelt_detections = []
last_vehicle_context_detections = []
cached_full_frame_plate_text = ""
last_full_frame_ocr_frame = -FULL_FRAME_OCR_EVERY_N_FRAMES

plate_reader = None
try:
    plate_reader = easyocr.Reader(["en"], gpu=False)
except Exception as exc:
    print(f"Warning: number-plate OCR initialization failed: {exc}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_index += 1

    gray = None
    run_inference = (frame_index % DETECTION_EVERY_N_FRAMES == 1) or (not last_detections)
    if run_inference:
        results = helmet_model.predict(frame, conf=0.40, verbose=False, imgsz=INFERENCE_IMGSZ, max_det=20)[0]

        detections = []
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()

            for xyxy, cls_id, conf in zip(boxes, classes, confidences):
                detections.append(
                    {
                        "bbox": to_int_bbox(xyxy),
                        "cls": int(cls_id),
                        "conf": float(conf),
                    }
                )
        last_detections = detections

        if seatbelt_model is not None and (seatbelt_with_ids or seatbelt_without_ids):
            seatbelt_results = seatbelt_model.predict(
                frame,
                conf=SEATBELT_CONF_THRESHOLD,
                verbose=False,
                imgsz=INFERENCE_IMGSZ,
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
                            "bbox": to_int_bbox(xyxy),
                            "cls": int(cls_id),
                            "conf": float(conf),
                        }
                    )
            last_seatbelt_detections = seatbelt_detections
        else:
            last_seatbelt_detections = []

        if vehicle_context_model is not None and (vehicle_four_wheeler_ids or vehicle_two_wheeler_ids):
            vehicle_results = vehicle_context_model.predict(
                frame,
                conf=VEHICLE_CONTEXT_CONF_THRESHOLD,
                verbose=False,
                imgsz=INFERENCE_IMGSZ,
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
                            "bbox": to_int_bbox(xyxy),
                            "cls": int(cls_id),
                            "conf": float(conf),
                        }
                    )
            last_vehicle_context_detections = vehicle_detections
        else:
            last_vehicle_context_detections = []
    else:
        detections = last_detections

    helmet_boxes = [d["bbox"] for d in detections if d["cls"] in helmet_ids]
    no_helmet_boxes = [d["bbox"] for d in detections if d["cls"] in no_helmet_ids]
    rider_boxes = [d["bbox"] for d in detections if d["cls"] in rider_ids]
    plate_boxes = [d["bbox"] for d in detections if d["cls"] in plate_ids]
    seatbelt_with_boxes = [
        d["bbox"] for d in last_seatbelt_detections if d["cls"] in seatbelt_with_ids
    ]
    seatbelt_without_boxes = [
        d["bbox"] for d in last_seatbelt_detections if d["cls"] in seatbelt_without_ids
    ]
    four_wheeler_boxes = [
        d["bbox"] for d in last_vehicle_context_detections if d["cls"] in vehicle_four_wheeler_ids
    ]
    two_wheeler_boxes = [
        d["bbox"] for d in last_vehicle_context_detections if d["cls"] in vehicle_two_wheeler_ids
    ]

    # Fallback to face detections when rider class is unavailable in the model.
    used_face_fallback = False
    if not rider_boxes:
        used_face_fallback = True
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        rider_boxes = [expand_face_to_rider(face, frame.shape) for face in faces]

    assigned_track_ids = set()
    full_frame_plate_text = cached_full_frame_plate_text
    frame_has_violation = False
    frame_violation_rider_id = None
    frame_violation_plate = "UNKNOWN"
    frame_violation_reason = "Helmet Not Worn"
    challan_generated_this_frame = False

    for rider_box in rider_boxes:
        best_track_id = None
        best_iou = 0.0
        for track_id, track in tracks.items():
            if frame_index - track["last_seen"] > TRACK_MAX_MISSED_FRAMES:
                continue
            score = iou(rider_box, track["bbox"])
            if score > best_iou:
                best_iou = score
                best_track_id = track_id

        if best_track_id is None or best_iou < MATCH_IOU_THRESHOLD:
            best_track_id = next_track_id
            next_track_id += 1
            tracks[best_track_id] = {
                "bbox": rider_box,
                "last_seen": frame_index,
                "violation_frames": 0,
                "last_challan_frame": -CHALLAN_COOLDOWN_FRAMES,
                "plate_text": "UNKNOWN",
                "last_plate_read_frame": -OCR_READ_EVERY_N_FRAMES,
            }

        track = tracks[best_track_id]
        track["bbox"] = rider_box
        track["last_seen"] = frame_index
        assigned_track_ids.add(best_track_id)

        has_helmet = any(box_matches_rider(h_box, rider_box) for h_box in helmet_boxes)
        has_no_helmet = any(box_matches_rider(nh_box, rider_box) for nh_box in no_helmet_boxes)
        has_seatbelt = any(box_matches_rider(sb_box, rider_box) for sb_box in seatbelt_with_boxes)
        has_no_seatbelt = any(box_matches_rider(nsb_box, rider_box) for nsb_box in seatbelt_without_boxes)

        two_wheeler_overlap = best_overlap_ratio(rider_box, two_wheeler_boxes)
        four_wheeler_overlap = best_overlap_ratio(rider_box, four_wheeler_boxes)
        in_two_wheeler_context = two_wheeler_overlap >= CONTEXT_OVERLAP_THRESHOLD and two_wheeler_overlap >= four_wheeler_overlap
        in_four_wheeler_context = (
            four_wheeler_overlap >= CONTEXT_OVERLAP_THRESHOLD and four_wheeler_overlap > two_wheeler_overlap
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
            in_four_wheeler_context and seatbelt_model is not None and (seatbelt_with_ids or seatbelt_without_ids)
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

        best_plate_box = find_best_plate_box_for_rider(rider_box, plate_boxes)
        should_read_plate = (
            is_violation
            and plate_reader is not None
            and run_inference
            and track["violation_frames"] >= max(1, VIOLATION_FRAMES_THRESHOLD - 3)
            and (
                track["plate_text"] == "UNKNOWN"
                or frame_index - track["last_plate_read_frame"] >= OCR_READ_EVERY_N_FRAMES
            )
        )
        if should_read_plate:
            detected_plate = ""
            if best_plate_box is not None:
                detected_plate = read_number_plate(frame, best_plate_box, plate_reader)

            if not detected_plate:
                fallback_boxes = fallback_plate_search_boxes(rider_box, frame.shape)
                if fallback_boxes:
                    detected_plate = read_number_plate(frame, fallback_boxes[0], plate_reader)

            if (
                not detected_plate
                and ENABLE_FULL_FRAME_OCR_FALLBACK
                and frame_index - last_full_frame_ocr_frame >= FULL_FRAME_OCR_EVERY_N_FRAMES
            ):
                full_frame_plate_text = read_number_plate_full_frame(frame, plate_reader)
                last_full_frame_ocr_frame = frame_index
                if full_frame_plate_text:
                    cached_full_frame_plate_text = full_frame_plate_text
                detected_plate = full_frame_plate_text

            if not detected_plate and cached_full_frame_plate_text:
                detected_plate = cached_full_frame_plate_text

            track["last_plate_read_frame"] = frame_index
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
            and track["violation_frames"] >= VIOLATION_FRAMES_THRESHOLD
            and frame_index - track["last_challan_frame"] >= CHALLAN_COOLDOWN_FRAMES
        ):
            challan_plate = track["plate_text"]
            if challan_plate == "UNKNOWN" and full_frame_plate_text:
                challan_plate = full_frame_plate_text
            generate_challan(
                frame,
                rider_id=best_track_id,
                plate_text=challan_plate,
                violation_reason=violation_reason,
            )
            track["last_challan_frame"] = frame_index
            track["violation_frames"] = 0
            challan_generated_this_frame = True

        if not is_violation:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        x1, y1, x2, y2 = rider_box
        label = (
            f"ID {best_track_id} | Helmet: {status} | Seatbelt: {seatbelt_status} "
            f"| Plate: {track['plate_text']}"
        )
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if frame_has_violation:
        global_violation_frames += 1
    else:
        global_violation_frames = 0

    # Fallback trigger when tracker IDs fluctuate and per-track streaks keep resetting.
    if (
        not challan_generated_this_frame
        and frame_has_violation
        and global_violation_frames >= GLOBAL_VIOLATION_FRAMES_THRESHOLD
        and frame_index - global_last_challan_frame >= CHALLAN_COOLDOWN_FRAMES
    ):
        challan_plate = frame_violation_plate
        if challan_plate == "UNKNOWN" and full_frame_plate_text:
            challan_plate = full_frame_plate_text
        generate_challan(
            frame,
            rider_id=frame_violation_rider_id,
            plate_text=challan_plate,
            violation_reason=frame_violation_reason,
        )
        global_last_challan_frame = frame_index
        global_violation_frames = 0

    stale_tracks = [
        track_id
        for track_id, track in tracks.items()
        if track_id not in assigned_track_ids and frame_index - track["last_seen"] > TRACK_MAX_MISSED_FRAMES
    ]
    for track_id in stale_tracks:
        del tracks[track_id]

    cv2.imshow("Helmet + Seatbelt Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
