"""
Ensemble voting pipeline

Automatically detects, validates, and analyzes acne lesions 
by combining traditional Computer Vision with YOLO, 
and generates a Markdown report with annotated results.

Processing Steps:
- Step 0 (Load): Imports the input image and creates a timestamped output directory.
- Step 1 (Preprocessing): Enhances contrast and details using CLAHE.
- Step 2 (Ensemble Detection): Runs YOLO and OpenCV detectors through an ensemble engine to boost and validate candidates.
- Step 3 (Filtering & NMS): Filters out low-confidence proposals and eliminates overlapping boxes using Non-Maximum Suppression (NMS).
- Step 4 (Reporting): Saves visualization layers and compiles a REPORT_VISUAL.md summary.

Outputs:
Saved inside a timestamped run folder: Models/shared/runs/ensemble_run_YYYYMMDD_HHMMSS/
- step1_cv_raw.jpg, step2_yolo_raw.jpg, step3_final.jpg
- REPORT_VISUAL.md (complete pipeline report)
"""

import io
import cv2
import torch
import sys
from pathlib import Path
from datetime import datetime
from torchvision.ops import nms

# Paths setup
SHARED_DIR = Path(__file__).resolve().parent
MODELS_DIR = SHARED_DIR.parent
YOLO_DIR = MODELS_DIR / "yolo"
CV_SRC_DIR = MODELS_DIR / "cv" / "src"
# INPUT_IMAGE = SHARED_DIR / "images" / "test-image-3.png"
INPUT_IMAGE = SHARED_DIR / "images" / "levle1_271.jpg"

for path in [CV_SRC_DIR, YOLO_DIR, YOLO_DIR / "src", SHARED_DIR]:
    if str(path) not in sys.path:
        sys.path.append(str(path))

from baseline import detect_red_regions, clean_mask, extract_boxes  # type: ignore
from image_preprocess import enhance_details  # type: ignore
from config import BEST_MODEL, CONF_THRESHOLD  # type: ignore

from detectors import CVDetector, YOLODetector
from ensemble_engine import EnsembleEngine

# Color codes (BGR format)
COLOR_CV_RAW = (110, 30, 15)
COLOR_YOLO_ONLY = (10, 45, 110)
COLOR_ENSEMBLE_BOOST = (35, 95, 20)


def generate_visual_markdown(output_path, timestamp, stats):
    """
    Generates a visual Markdown report tracking performance and statistics.
    """
    md_content = f"""# Visual Detection Analysis Report - Ensemble Pipeline

**Session Identifier:** `{timestamp}`  
**Applied Confidence Threshold (CONF_THRESHOLD):** `{CONF_THRESHOLD}`

---

## Pipeline Statistical Summary

* **Raw anomalies detected by OpenCV (CV):** {stats['raw_cv']}
* **Raw predictions registered by YOLO (conf >= 0.001):** {stats['raw_yolo']}
* **YOLO candidates validated by OpenCV (Confidence Boost +0.15 applied):** {stats['boosted_count']}
* **Low confidence candidates dropped (Failed final threshold filtering):** {stats['dropped_count']}
* **Final verified acne lesions reported:** **{stats['final_count']}**

---

## Step-by-Step Visual Analysis

### Step 1: Traditional Computer Vision Feature Extraction (Reference Only)
*This image exhibits raw red-channel anomalies and contour coordinates extracted by the OpenCV pipeline, acting as validation points for the neural network.*

![Step 1 - OpenCV Raw](step1_cv_raw.jpg)

---

### Step 2: Raw YOLO Model Proposals
*Every localized anomaly registered by the deep learning model before confidence filtering. High-density overlaps and background noise are expected at this level.*

![Step 2 - YOLO Raw](step2_yolo_raw.jpg)

---

### Step 3: Final Consolidated Output (Ensemble Voting Execution)
*The final verified result of the ensemble pipeline.*

![Step 3 - Final Consolidated](step3_final.jpg)
"""
    md_content = md_content.replace(r'\xb2', '2')
    with io.open(output_path / "REPORT_VISUAL.md", "w", encoding="utf-8") as f:
        f.write(md_content)


def main():
    image_org = cv2.imread(str(INPUT_IMAGE))
    if image_org is None:
        raise FileNotFoundError(f"Could not load image from: {INPUT_IMAGE}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = SHARED_DIR / "runs" / f"ensemble_run_{timestamp}"
    runs_dir.mkdir(parents=True, exist_ok=True)

    enhanced_image = enhance_details(image_org)

    # Initialize Object-Oriented Detectors
    yolo_model = YOLODetector(model_path=BEST_MODEL, default_conf=0.001)
    cv_model = CVDetector(detect_red_regions, clean_mask, extract_boxes, silence=False)
    
    # Initialize OOP Coordinator Engine
    engine = EnsembleEngine(yolo_detector=yolo_model, cv_detector=cv_model)

    # Execute Ensemble Pipeline with visual boosting constant
    y_boxes, boosted_confs, cv_boxes, statuses = engine.run_ensemble_boosting(
        enhanced_image, dynamic_tolerance_pct=0.10, boost_value=0.15
    )

    # Build Visualization Canvas Layers
    canvas_cv = image_org.copy()
    canvas_yolo = image_org.copy()
    canvas_final = image_org.copy()

    # Draw OpenCV Reference Boxes
    for box in cv_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(canvas_cv, (x1, y1), (x2, y2), COLOR_CV_RAW, 2)
        cv2.putText(canvas_cv, "CV", (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_CV_RAW, 1)

    # Draw YOLO Raw Bounding Boxes
    for box, conf, status in zip(y_boxes, boosted_confs, statuses):
        x1, y1, x2, y2 = map(int, box)
        color = COLOR_ENSEMBLE_BOOST if "Boost" in status else COLOR_YOLO_ONLY
        cv2.rectangle(canvas_yolo, (x1, y1), (x2, y2), color, 1)
        cv2.putText(canvas_yolo, f"{conf:.2f}", (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # Perform Final Score-Threshold Filtering & NMS
    final_boxes = []
    final_confs = []
    final_statuses = []

    for box, conf, status in zip(y_boxes, boosted_confs, statuses):
        if conf >= CONF_THRESHOLD:
            final_boxes.append(box)
            final_confs.append(conf)
            final_statuses.append(status)

    dropped_count = len(y_boxes) - len(final_boxes)

    # Execute Non-Maximum Suppression (NMS)
    if final_boxes:
        b_tensor = torch.tensor(final_boxes, dtype=torch.float32)
        s_tensor = torch.tensor(final_confs, dtype=torch.float32)
        keep = nms(b_tensor, s_tensor, iou_threshold=0.3).tolist()
        
        final_boxes = [final_boxes[idx] for idx in keep]
        final_confs = [final_confs[idx] for idx in keep]
        final_statuses = [final_statuses[idx] for idx in keep]

    # Draw Final Decided Boxes
    for box, conf, status in zip(final_boxes, final_confs, final_statuses):
        x1, y1, x2, y2 = map(int, box)
        color = COLOR_ENSEMBLE_BOOST if "Boost" in status else COLOR_YOLO_ONLY
        cv2.rectangle(canvas_final, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas_final, f"{status} {conf:.2f}", (x1, max(y1 - 7, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Save Output Assets
    cv2.imwrite(str(runs_dir / "step1_cv_raw.jpg"), canvas_cv)
    cv2.imwrite(str(runs_dir / "step2_yolo_raw.jpg"), canvas_yolo)
    cv2.imwrite(str(runs_dir / "step3_final.jpg"), canvas_final)

    stats = {
        "raw_cv": len(cv_boxes),
        "raw_yolo": len(y_boxes),
        "boosted_count": sum(1 for s in statuses if "Boost" in s),
        "dropped_count": dropped_count,
        "final_count": len(final_boxes)
    }

    generate_visual_markdown(runs_dir, timestamp, stats)

    print(f"Process complete. Output saved inside: {runs_dir}")


if __name__ == "__main__":
    main()