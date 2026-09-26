import cv2
import numpy as np
from datetime import datetime

def load_ground_truth(label_path, img_w, img_h):
    """
    Converts normalized YOLO format [class, cx, cy, w, h] 
    to absolute pixel coordinates [x1, y1, x2, y2].
        
    - cx, cy: Center of the object (e.g., 0.5 = 50% of image size).
    - w, h: Width and height of the object (e.g., 0.2 = 20% of image size).
    
    Example: 1000x1000 image, input "0 0.5 0.5 0.2 0.2"
    - cx=0.5 (500px), cy=0.5 (500px) -> Object centered at 500,500.
    - w=0.2 (200px), h=0.2 (200px) -> Object size is 200x200px.
    - Result: [x1=400, y1=400, x2=600, y2=600] (x1=cx-w/2, x2=cx+w/2).
    """
    gt_boxes = []
    if not label_path.exists():
        return gt_boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cx, cy, w, h = map(float, parts[1:5])
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            gt_boxes.append([x1, y1, x2, y2])
    return gt_boxes


def calculate_box_iou(boxA, boxB):
    """
    Calculates Intersection over Union (IoU) - the overlap ratio between two boxes.
    
    - Intersection: The overlapping area (Rectangle intersection).
    - Union: Total area covered by both boxes (AreaA + AreaB - Intersection).
    - Formula: IoU = Intersection / Union.
    
    Example: 
    - BoxA: [50, 50, 150, 150] (100x100), BoxB: [100, 100, 200, 200] (100x100)
    - Intersection: [100, 100, 150, 150] (50x50 = 2500px)
    - Union: (10000 + 10000) - 2500 = 17500px
    - Result: 2500 / 17500 ≈ 0.14
    """

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)


def evaluate_predictions(pred_boxes, gt_boxes, iou_thresh=0.3):
    """
    Classifies predictions as TP, FP, or FN based on IoU threshold.
    
    - True Positive (TP): Prediction matches Ground Truth (IoU >= threshold).
    - False Positive (FP): Prediction does not match any Ground Truth.
    - False Negative (FN): Ground Truth that was missed by any prediction.
    
    Example (thresh=0.3):
    - Prediction: [100, 100, 200, 200], Ground Truth: [105, 105, 205, 205]
    - IoU result is ~0.84. Since 0.84 >= 0.3, it counts as 1 TP.
    """
    tp, fp = 0, 0
    matched_gt = set()

    for pred in pred_boxes:
        best_iou = 0
        best_gt_idx = -1
        for idx, gt in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
            iou = calculate_box_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx
        
        if best_iou >= iou_thresh:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def compile_metrics(tp, fp, fn):
    """
    Calculates standard performance metrics.
    
    Example: 80 TP, 20 FP, 10 FN
    - Precision: 80 / (80+20) = 0.80
    - Recall: 80 / (80+10) = 0.88
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1_score": f1, "tp": tp, "fp": fp, "fn": fn}


def create_heatmap(image):
    """
    Generates a heatmap using the 'a' channel of the CIELab color space.
    
    - LAB Space: Separates Lightness (L) from color (a=Green/Red, b=Blue/Yellow).
    - 'a' channel: Highlights red/green intensity. 
    - Normalization: Scales values to 0-255 range.
    - Colormap (JET): Maps low values to blue and high values to red.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    _, a, _ = cv2.split(lab)
    heatmap = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)