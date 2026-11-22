from pathlib import Path
import cv2
import numpy as np


def draw_detections(
    image: np.ndarray,
    detections: list[dict],
    bbox_format: str = "xywh",
) -> np.ndarray:
    """Draw detection bounding boxes and class_id labels onto an image."""
    canvas = image.copy()
    h_img, w_img = canvas.shape[:2]
    thickness = max(round(max(h_img, w_img) / 600), 1)

    for det in detections:
        bbox = det.get("bbox") or []
        if len(bbox) < 4:
            continue

        if bbox_format == "xywh":
            cx, cy, w, h = [float(v) for v in bbox[:4]]
            x1, y1 = cx - w / 2.0, cy - h / 2.0
            x2, y2 = cx + w / 2.0, cy + h / 2.0
        else:
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]

        pt1 = (int(round(x1)), int(round(y1)))
        pt2 = (int(round(x2)), int(round(y2)))

        cv2.rectangle(canvas, pt1, pt2, (0, 200, 0), thickness)

        class_id = det.get("class_id", det.get("class", ""))
        label = str(class_id)
        text_y = max(pt1[1] - 6, 16)
        cv2.putText(
            canvas,
            label,
            (pt1[0], text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 0),
            thickness,
            cv2.LINE_AA,
        )

    return canvas


def save_drawn_detections(
    image_path: Path,
    output_path: Path,
    detections: list[dict],
    bbox_format: str = "xywh",
) -> Path:
    """Load an original image, draw detections, and save to output_path."""
    img_org = cv2.imread(str(image_path))
    if img_org is None:
        raise FileNotFoundError(f"Could not read image for drawing: {image_path}")

    rendered = draw_detections(img_org, detections, bbox_format=bbox_format)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), rendered)
    return output_path


def calculate_score(
    detections: list[dict],
    image_shape: tuple[int, int] | tuple[int, int, int] | None = None,
    face_area: float | None = None,
    max_count: int = 100,
    bbox_format: str = "xyxy",
) -> float:
    """
    Calculate Normalized Acne Severity Index (NASI) in [0, 1] from a list of detections.
    """
    if not detections:
        return 0.0

    if face_area is None:
        if image_shape is None or len(image_shape) < 2:
            raise ValueError("Provide either image_shape or face_area to calculate_score.")
        h_img, w_img = image_shape[:2]
        face_area = float(h_img * w_img)

    if face_area <= 0:
        return 0.0

    class_weights = {
        0: 1.0,   # ogólny trądzik (v3.2)
        1: 0.7,   # closed_comedo
        2: 1.5,   # nodule
        3: 0.8,   # open_comedo
        4: 1.0,   # papule
        5: 1.2,   # pustule
    }

    confidences = []
    areas = []
    weighted_counts = 0.0

    for det in detections:
        cid = int(det.get("class_id", -1))
        weight = class_weights.get(cid, 0.0)
        if weight == 0.0:
            continue

        bbox = det.get("bbox") or []
        if len(bbox) < 4:
            continue

        if bbox_format == "xywh":
            _, _, w, h = [float(v) for v in bbox[:4]]
            area = max(0.0, w) * max(0.0, h)
        else:
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

        confidences.append(float(det.get("confidence", 0.0)))
        areas.append(area / face_area)
        weighted_counts += weight

    if not confidences:
        return 0.0

    conf_arr = np.asarray(confidences, dtype=np.float64)
    area_arr = np.asarray(areas, dtype=np.float64)

    count_score = np.log1p(weighted_counts) / np.log1p(max_count)
    size_score = np.mean(area_arr ** 0.6)
    density_score = np.clip(area_arr.sum() / 0.12, 0.0, 1.0)
    confidence_score = np.mean(np.sqrt(conf_arr))

    nasi = (
        0.45 * count_score
        + 0.20 * size_score
        + 0.25 * density_score
        + 0.10 * confidence_score
    )

    return round(float(np.clip(nasi, 0.0, 1.0)), 3)