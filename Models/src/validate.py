"""

YOLOv8 VALIDATION & METRICS PIPELINE

Run this script to evaluate the performance of your trained model.
It calculates precision, recall, and mAP metrics on a specified dataset split.

What it does:

* Loads a trained YOLOv8 model (default: BEST_MODEL from config)
* Runs evaluation on a chosen split (train, val, or test)
* Generates visual plots (PR curves, Confusion Matrix, etc.)
* Saves a JSON report with core metrics
* Automatically manages timestamped session directories

Usage:

* Default (Test split): python Models/src/validate.py
* Specific Model: python Models/src/validate.py --model path/to/model.pt
* Custom Split: python Models/src/validate.py --split val
* Hardware Override: python Models/src/validate.py --device cpu

Output:

* Visuals: runs/validation/val_[model]_[timestamp]/
* Report:  runs/validation/val_[model]_[timestamp]/results.json

"""


import json
import argparse
import sys
import torch
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# Path setup to ensure local imports work
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from config import BEST_MODEL, DATA_YAML, RUNS_DIR

def load_model(model_path: Path):
    """Load trained production model."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Please train first.")
    return YOLO(str(model_path))

def safe_get(metrics, key):
    """Safely extract metric from YOLO results."""
    try:
        return float(metrics.results_dict.get(key, 0.0))
    except:
        return 0.0

def get_device():
    """Auto-select CPU/GPU."""
    return "cuda:0" if torch.cuda.is_available() else "cpu"

def run_validation_pipeline(model_arg=None, imgsz=640, split="test", device_arg=None):
    """
    Main orchestration logic for validation - analogous to run_prediction_pipeline
    """
    # 1. Path & Device setup
    model_path = Path(model_arg) if model_arg else BEST_MODEL
    device = device_arg if device_arg else get_device()
    
    if not model_path.exists():
        print(f"Error: Model path {model_path} does not exist.")
        return

    # 2. Session setup (Analogous to predict)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    session_name = f"val_{model_path.stem}_{timestamp}"
    session_dir = RUNS_DIR / "validation" / session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    # 3. Load Model
    model = load_model(model_path)
    
    print(f"--- STARTING VALIDATION SESSION ---")
    print(f"[SESSION] {session_name}")
    print(f"[MODEL]   {model_path.name}")
    print(f"[SPLIT]   {split}")
    print(f"[DEVICE]  {device}\n")

    # 4. Run Validation Execution
    metrics = model.val(
        data=str(DATA_YAML),
        imgsz=imgsz,
        split=split,
        device=device,
        project=str(session_dir.parent), 
        name=session_name,
        plots=True,
        verbose=False,
        exist_ok=True
    )

    # 5. Extract Results
    results = {
        "model": model_path.name,
        "session": session_name,
        "device": device,
        "imgsz": imgsz,
        "split": split,
        "metrics": {
            "precision": safe_get(metrics, "metrics/precision(B)"),
            "recall": safe_get(metrics, "metrics/recall(B)"),
            "mAP50": safe_get(metrics, "metrics/mAP50(B)"),
            "mAP50-95": safe_get(metrics, "metrics/mAP50-95(B)")
        },
        "timestamp": datetime.now().isoformat()
    }

    # 6. Save Report & Print Summary
    print(f"\nVALIDATION RESULTS ")
    print(f"Precision:  {results['metrics']['precision']:.3f}")
    print(f"Recall:     {results['metrics']['recall']:.3f}")
    print(f"mAP50:      {results['metrics']['mAP50']:.3f}")
    print(f"mAP50-95:   {results['metrics']['mAP50-95']:.3f}")

    report_path = session_dir / "results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[DONE] Session saved in: {session_dir}")
    return results

def main():
    parser = argparse.ArgumentParser(description="Acne Detection Validation Pipeline")

    # Arguments consistent with predict/train style
    parser.add_argument("--model", type=str, default=None, help="Path to model file")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda:0")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test", "train"])

    args = parser.parse_args()

    try:
        run_validation_pipeline(
            model_arg=args.model,
            imgsz=args.imgsz,
            split=args.split,
            device_arg=args.device
        )
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")

if __name__ == "__main__":
    main()