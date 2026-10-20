import base64
import html
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from IPython.display import display, clear_output
import numpy as np

DISPLAY_CLASSES = {
    0: {"name": "acne", "pl": "Trądzik", "rgb": (255, 80, 80)},
    1: {"name": "closed_comedo", "pl": "Zaskórnik zamknięty", "rgb": (70, 190, 255)},
    2: {"name": "nodule", "pl": "Guzek trądzikowy", "rgb": (190, 100, 255)},
    3: {"name": "open_comedo", "pl": "Zaskórnik otwarty", "rgb": (80, 220, 160)},
    4: {"name": "papule", "pl": "Grudka", "rgb": (255, 190, 60)},
    5: {"name": "pustule", "pl": "Krosta", "rgb": (255, 110, 190)},
}

V4_CLASS_MAP = {1: 1, 5: 2, 6: 3, 8: 4, 9: 5}
V4_CLASS_NAMES = {"closed_comedo": 1, "nodule": 2, "open_comedo": 3, "papule": 4, "pustule": 5}


def image_info(path, images_dir):
    parts = path.relative_to(images_dir).parts
    date = parts[0] if len(parts) > 1 else "Unknown"
    side = parts[1].lower() if len(parts) > 2 else "Unknown"
    return date, side

def card(label, value, card_style, border_style, text_style, muted_style):
    return f"""
    <div style="
        background:{card_style};
        border:1px solid {border_style};
        border-radius:9px;
        padding:10px 12px;
        min-width:0;
        overflow-wrap:anywhere;
    ">
        <div style="
            color:{muted_style};
            font-size:9px;
            font-weight:700;
            letter-spacing:.8px;
            margin-bottom:5px;
        ">{html.escape(str(label))}</div>
        <div style="
            color:{text_style};
            font-size:16px;
            font-weight:700;
        ">{html.escape(str(value))}</div>
    </div>
    """

def filter_detections(detections, model_version, v4_class_names, v4_class_map, display_classes):
    filtered = []

    if model_version == "v3.2":
        for det in detections:
            filtered.append({
                **det,
                "display_id": 0,
                "display_name": "acne",
                "source_model": "v3.2",
            })

    elif model_version == "v4":
        for det in detections:
            class_name = str(det.get("class", ""))
            display_id = v4_class_names.get(class_name)

            if display_id is None:
                try:
                    original_id = int(class_name)
                    display_id = v4_class_map.get(original_id)
                except (TypeError, ValueError):
                    display_id = None

            if display_id is None:
                continue

            filtered.append({
                **det,
                "display_id": display_id,
                "display_name": display_classes[display_id]["name"],
                "source_model": "v4",
            })

    return filtered

def bbox_iou(box_a, box_b):
    ax, ay, aw, ah = map(float, box_a)
    bx, by, bw, bh = map(float, box_b)

    ax1, ay1 = ax - aw / 2, ay - ah / 2
    ax2, ay2 = ax + aw / 2, ay + ah / 2

    bx1, by1 = bx - bw / 2, by - bh / 2
    bx2, by2 = bx + bw / 2, by + bh / 2

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0

def find_common_detections(v3_detections, v4_detections, iou_threshold):
    candidates = []

    for i, det3 in enumerate(v3_detections):
        for j, det4 in enumerate(v4_detections):
            overlap = bbox_iou(det3["bbox"], det4["bbox"])
            if overlap >= iou_threshold:
                candidates.append((overlap, i, j))

    candidates.sort(key=lambda item: item[0], reverse=True)

    used_v3 = set()
    used_v4 = set()
    common = []

    for overlap, i, j in candidates:
        if i in used_v3 or j in used_v4:
            continue

        used_v3.add(i)
        used_v4.add(j)

        det = dict(v4_detections[j])
        det["matched_iou"] = overlap
        det["matched_v3_confidence"] = v3_detections[i]["confidence"]
        common.append(det)

    return common

def draw_detections(
    image, 
    detections, 
    display_classes=None, 
    default_color=(0, 0, 255), 
    thickness=4, 
    font_scale=0.75, 
    text_thickness=1,
    show_class_name=True, 
    show_confidence=True, 
    use_background_box=False,
    dynamic_scale=False,
    image_is_rgb=False  # <--- NOWY PARAMETR (True dla matplotlib/RGB, False dla OpenCV/BGR)
):
    output = image.copy()
    h, w = output.shape[:2]

    if dynamic_scale:
        scale = max(h, w)
        thickness = max(1, int(scale / 600))
        font_scale = max(0.35, min(0.55, scale / 3000))
        text_thickness = max(1, int(scale / 1600))

    for det in detections:
        bbox = det.get("bbox")
        if bbox is None or len(bbox) != 4:
            continue

        cx, cy, bw, bh = map(float, bbox)
        x1 = max(0, int(round(cx - bw / 2)))
        y1 = max(0, int(round(cy - bh / 2)))
        x2 = min(w - 1, int(round(cx + bw / 2)))
        y2 = min(h - 1, int(round(cy + bh / 2)))

        if display_classes and "display_id" in det:
            class_id = det["display_id"]
            color_rgb = display_classes.get(class_id, {}).get("rgb", default_color)
            if image_is_rgb:
                color = color_rgb
            else:
                color = (color_rgb[2], color_rgb[1], color_rgb[0])  # RGB -> BGR dla OpenCV
        else:
            color = default_color

        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        label_parts = []
        if show_class_name:
            if "display_id" in det:
                label_parts.append(str(det["display_id"]))
            elif "class" in det:
                label_parts.append(str(det["class"]))
        
        if show_confidence and "confidence" in det:
            label_parts.append(f"{float(det['confidence']):.2f}")

        if not label_parts:
            continue

        label = " | ".join(label_parts)

        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
        )

        text_y = y1 - 5
        if text_y < text_height + baseline:
            text_y = min(h - 2, y1 + text_height + baseline + 15)

        if use_background_box:
            label_x = max(0, min(x1, w - text_width - 4))
            cv2.rectangle(
                output,
                (label_x, text_y - text_height - baseline),
                (label_x + text_width + 4, text_y + 2),
                color,
                -1
            )
            cv2.putText(
                output, label, (label_x + 2, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA
            )
        else:
            cv2.putText(
                output, label, (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, text_thickness, cv2.LINE_AA
            )

    return output

def build_legend(detections, display_classes):
    used_ids = sorted({det["display_id"] for det in detections})
    handles = []

    for class_id in used_ids:
        info = display_classes[class_id]
        rgb = info["rgb"]
        count = sum(det["display_id"] == class_id for det in detections)

        handles.append(
            Patch(
                facecolor=tuple(channel / 255 for channel in rgb),
                edgecolor=tuple(channel / 255 for channel in rgb),
                label=f"ID {class_id} — {info['name']}\n    {info['pl']} (×{count})",
            )
        )

    return handles


SKIN_H_MIN = 0
SKIN_H_MAX = 30      
SKIN_S_MIN = 15      
SKIN_S_MAX = 255     
SKIN_V_MIN = 40      
SKIN_CR_MIN = 130    
SKIN_CR_MAX = 190    
SKIN_CB_MIN = 70     
SKIN_CB_MAX = 140    

REF_DIAG = 1414.0    
BASE_BLUR = 5        
BASE_ADAPTIVE_BLOCK = 31 
BASE_MORPH_KERNEL = 5   

def get_adaptive_params(image):
    h, w = image.shape[:2]
    diag = np.sqrt(h**2 + w**2)
    scale = diag / REF_DIAG
    blur_k = max(3, int(BASE_BLUR * scale) | 1)
    adaptive_block = max(5, int(BASE_ADAPTIVE_BLOCK * scale) | 1)
    k_close = max(3, int(diag * 0.003) | 1)
    k_open = max(3, int(diag * 0.0015) | 1)
    return blur_k, adaptive_block, k_close, k_open

def create_skin_mask(image):
    h, w = image.shape[:2]
    blur_k, adaptive_block, k_close, k_open = get_adaptive_params(image)
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(
        hsv,
        np.array([SKIN_H_MIN, SKIN_S_MIN, SKIN_V_MIN]),
        np.array([SKIN_H_MAX, SKIN_S_MAX, 255])
    )

    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    _, cr, cb = cv2.split(ycrcb)

    cr_mask = cv2.inRange(cr, SKIN_CR_MIN, SKIN_CR_MAX)
    cb_mask = cv2.inRange(cb, SKIN_CB_MIN, SKIN_CB_MAX)
    ycrcb_mask = cv2.bitwise_and(cr_mask, cb_mask)

    skin_mask = cv2.bitwise_or(hsv_mask, ycrcb_mask)

    kernel_close = np.ones((k_close, k_close), np.uint8)
    kernel_open = np.ones((k_open, k_open), np.uint8)
    
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    return skin_mask