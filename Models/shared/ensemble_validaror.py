"""
Ensemble validation pipeline

Evaluates and compares the performance of three object detection pipelines:
Traditional OpenCV, YOLO, and their ensemble model on a validation dataset.

Processing Steps:
- Load validation images and annotations.
- Preprocess each image.
- Run OpenCV, YOLO, and ensemble detection.
- Evaluate predictions using IoU matching.
- Calculate Precision, Recall, and F1-Score.
- Save the results as JSON and Markdown reports.

Outputs in Models/shared/runs/validation/ensemble_comp_YYYYMMDD_HHMMSS/
- metrics_comparison.json
- performance_comparison.md
"""

import io
import traceback
import cv2
import torch
import json
import sys
import argparse

from pathlib import Path
from datetime import datetime
from torchvision.ops import nms
from tqdm import tqdm


# Paths configuration
SHARED_DIR = Path(__file__).resolve().parent
MODELS_DIR = SHARED_DIR.parent
YOLO_DIR = MODELS_DIR / "yolo"
CV_SRC_DIR = MODELS_DIR / "cv" / "src"

for path in [CV_SRC_DIR, YOLO_DIR, YOLO_DIR / "src", SHARED_DIR]:
    if str(path) not in sys.path:
        sys.path.append(str(path))

from baseline import detect_red_regions, clean_mask, extract_boxes  # type: ignore
from image_preprocess import enhance_details  # type: ignore
from config import BEST_MODEL, CONF_THRESHOLD, RUNS_DIR  # type: ignore

# Shared OOP and utility modules
from detectors import CVDetector, YOLODetector
from ensemble_engine import EnsembleEngine
from utils import load_ground_truth, evaluate_predictions, compile_metrics

DEFAULT_VALID_DIR = YOLO_DIR / "data" / "dataset_final_preprocessed" / "valid"
IOU_EVAL_THRESHOLD = 0.3


def run_comparative_evaluation(valid_dir_path: Path):
    images_dir = valid_dir_path / "images"
    labels_dir = valid_dir_path / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(
            f"Target verification folder must contain 'images' and 'labels' subdirectories. Checked: {valid_dir_path}"
            )

    # Build Object Instances using Adapters
    yolo_model = YOLODetector(model_path=BEST_MODEL, default_conf=0.001)
    cv_model = CVDetector(detect_red_regions, clean_mask, extract_boxes, silence=True)
    engine = EnsembleEngine(yolo_detector=yolo_model, cv_detector=cv_model)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    img_paths = [p for p in images_dir.iterdir() if p.suffix.lower() in image_extensions]

    if not img_paths:
        print(f"Error: No image assets detected inside targeting folder: {images_dir}")
        return

    print(f"Loaded {len(img_paths)} assets from target validation split directory.")
    print(f"Target Location: {valid_dir_path}")
    
    counters = {
        "yolo": {"tp": 0, "fp": 0, "fn": 0},
        "cv": {"tp": 0, "fp": 0, "fn": 0},
        "ensemble": {"tp": 0, "fp": 0, "fn": 0}
    }

    print("\n--- Executing Multi-Pipeline Evaluation ---")
    
    for img_p in tqdm(img_paths, desc="Evaluating Dataset Pipelines", unit="img", ascii=True):
        img_org = cv2.imread(str(img_p))
        if img_org is None:
            continue
        h, w, _ = img_org.shape
        
        lbl_p = labels_dir / f"{img_p.stem}.txt"
        gt_boxes = load_ground_truth(lbl_p, w, h)
        img_enhanced = enhance_details(img_org)

        # A. Evaluate Pure Traditional CV
        cv_boxes = cv_model.predict(img_enhanced)
        tp_c, fp_c, fn_c = evaluate_predictions(cv_boxes, gt_boxes, IOU_EVAL_THRESHOLD)
        counters["cv"]["tp"] += tp_c
        counters["cv"]["fp"] += fp_c
        counters["cv"]["fn"] += fn_c

        # B. Evaluate Pure YOLO
        y_boxes, y_confs = yolo_model.predict(img_enhanced)
        yolo_pure_boxes = [box for box, conf in zip(y_boxes, y_confs) if conf >= CONF_THRESHOLD]
        
        if yolo_pure_boxes:
            b_tensor = torch.tensor(yolo_pure_boxes, dtype=torch.float32)
            s_tensor = torch.tensor([c for c in y_confs if c >= CONF_THRESHOLD], dtype=torch.float32)
            keep = nms(b_tensor, s_tensor, iou_threshold=0.3).tolist()
            yolo_pure_boxes = [yolo_pure_boxes[idx] for idx in keep]

        tp_y, fp_y, fn_y = evaluate_predictions(yolo_pure_boxes, gt_boxes, IOU_EVAL_THRESHOLD)
        counters["yolo"]["tp"] += tp_y
        counters["yolo"]["fp"] += fp_y
        counters["yolo"]["fn"] += fn_y

        # C. Evaluate Ensemble Proximity Pipeline
        y_b_boxes, boosted_confs, _, _ = engine.run_ensemble_boosting(
            img_enhanced, dynamic_tolerance_pct=0.10, boost_value=0.08
        )
        ensemble_final_boxes = []
        ensemble_final_confs = []
        
        for box, conf in zip(y_b_boxes, boosted_confs):
            if conf >= CONF_THRESHOLD:
                ensemble_final_boxes.append(box)
                ensemble_final_confs.append(conf)

        if ensemble_final_boxes:
            b_tensor_e = torch.tensor(ensemble_final_boxes, dtype=torch.float32)
            s_tensor_e = torch.tensor(ensemble_final_confs, dtype=torch.float32)
            keep_e = nms(b_tensor_e, s_tensor_e, iou_threshold=0.3).tolist()
            ensemble_final_boxes = [ensemble_final_boxes[idx] for idx in keep_e]

        tp_e, fp_fp, fn_e = evaluate_predictions(ensemble_final_boxes, gt_boxes, IOU_EVAL_THRESHOLD)
        counters["ensemble"]["tp"] += tp_e
        counters["ensemble"]["fp"] += fp_fp
        counters["ensemble"]["fn"] += fn_e

    metrics_report = {
        "yolo_pure": compile_metrics(counters["yolo"]["tp"], counters["yolo"]["fp"], counters["yolo"]["fn"]),
        "cv_pure": compile_metrics(counters["cv"]["tp"], counters["cv"]["fp"], counters["cv"]["fn"]),
        "ensemble_pipeline": compile_metrics(counters["ensemble"]["tp"], counters["ensemble"]["fp"], counters["ensemble"]["fn"]),
        "dataset_summary": {
            "total_images_processed": len(img_paths),
            "applied_iou_matching_threshold": IOU_EVAL_THRESHOLD,
            "applied_yolo_conf_threshold": CONF_THRESHOLD
        }
    }

    print("\n\n=== PIPELINE PERFORMANCE SUMMARY ===")
    for name, data in [("Pure YOLO", metrics_report["yolo_pure"]), 
                       ("Pure OpenCV", metrics_report["cv_pure"]), 
                       ("Ensemble Pipeline", metrics_report["ensemble_pipeline"])]:
        print(f"\n[{name}]")
        print(f"  Precision : {data['precision']:.3f} (TP: {data['tp']}, FP: {data['fp']})")
        print(f"  Recall    : {data['recall']:.3f} (FN: {data['fn']})")
        print(f"  F1-Score  : {data['f1_score']:.3f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RUNS_DIR / "validation" / f"ensemble_comp_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    

    with open(out_dir / "metrics_comparison.json", "w", encoding="utf-8", errors="replace") as json_f:
        json.dump(metrics_report, json_f, indent=2)
    

    md_content = f"""# Comparative Validation Report - Object Detection Pipelines


**Evaluation Timestamp:** `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`  
**Target Path:** `{valid_dir_path}`  
**Operational YOLO Conf Threshold (`CONF_THRESHOLD`):** `{CONF_THRESHOLD}`  
**Evaluation Matching IoU Threshold:** `{IOU_EVAL_THRESHOLD}`  
**Total Images Scanned:** `{metrics_report['dataset_summary']['total_images_processed']}`

---

## Performance Matrix Comparison

| Pipeline Engine Model | Precision | Recall | F1-Score | True Positives (TP) | False Positives (FP) | False Negatives (FN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pure OpenCV (Traditional CV)** | {metrics_report['cv_pure']['precision']:.4f} | {metrics_report['cv_pure']['recall']:.4f} | {metrics_report['cv_pure']['f1_score']:.4f} | {metrics_report['cv_pure']['tp']} | {metrics_report['cv_pure']['fp']} | {metrics_report['cv_pure']['fn']} |
| **Pure YOLO Model (Deep Learning)** | {metrics_report['yolo_pure']['precision']:.4f} | {metrics_report['yolo_pure']['recall']:.4f} | {metrics_report['yolo_pure']['f1_score']:.4f} | {metrics_report['yolo_pure']['tp']} | {metrics_report['yolo_pure']['fp']} | {metrics_report['yolo_pure']['fn']} |
| **Ensemble Proximity Engine (Combined)** | **{metrics_report['ensemble_pipeline']['precision']:.4f}** | **{metrics_report['ensemble_pipeline']['recall']:.4f}** | **{metrics_report['ensemble_pipeline']['f1_score']:.4f}** | {metrics_report['ensemble_pipeline']['tp']} | {metrics_report['ensemble_pipeline']['fp']} | {metrics_report['ensemble_pipeline']['fn']} |

---

*Report systematically compiled by the DermaAI Automated Multi-Pipeline Cross-Validator.*
"""
    md_content = md_content.replace('\xb2', '2')
    md_content = md_content.encode('utf-8', 'ignore').decode('utf-8')
        
    with open(out_dir / "performance_comparison.md", "w", encoding="utf-8", errors="replace") as md_f:
        md_f.write(md_content)

    print(f"\n[DONE] Diagnostic reports saved successfully.")
    print(f"JSON Structure: {out_dir / 'metrics_comparison.json'}")
    print(f"Markdown Sheet: {out_dir / 'performance_comparison.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-Pipeline Object Detection Validator Engine")
    parser.add_argument(
        "--valid_dir", 
        type=str, 
        default=str(DEFAULT_VALID_DIR), 
        help="Absolute path to your validation dataset directory containing 'images' and 'labels'"
    )
    args = parser.parse_args()

    try:
        run_comparative_evaluation(Path(args.valid_dir))
    except Exception as e:
        print("\n[CRITICAL ERROR - DEBUG INFO]:")
        traceback.print_exc()