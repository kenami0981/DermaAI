"""
YOLOv8 PRODUCTION TRAINING PIPELINE

Run this script to train the final acne detection model.
It automates the training process, integrates optimized hyperparameters,
and handles model deployment.

What it does:

* Orchestrates training using YOLOv8 (default: yolov8s.pt)
* Auto-loads best hyperparameters from Optuna searches (if available)
* Smart Resume: automatically continues from 'last.pt' if interrupted
* Production Deployment: copies the 'best.pt' model to the BEST_MODEL path
* Hardware Management: auto-detects CUDA/GPU or optimizes for CPU
* Documentation: saves all training parameters to a YAML file for tracking

Usage:

1. Ensure dataset_final and data.yaml are ready. Use the dataset merge pipeline if needed.
2. (Optional) Place acne_hparam_search_{CURRENT_VERSION}_best.yaml in HPARAM_SEARCH_RESULT to use Optuna results.
3. Run: python Models/src/train.py

Output:

* Training Logs & Plots: runs/train/[TRAINING_RUN_NAME]/
* Deployment: Copies best weights to the production folder defined in config.py.
* Checkpoints: Saves 'last.pt' every 10 epochs for crash recovery.

Requirements:

* BASE_MODEL must be accessible in the weights folder.
* Sufficient disk space in the runs/ directory for logs and weights.
* CUDA-enabled GPU HIGHLY RECOMMENDED for production training. 
* If using CPU, ensure you have a modern processor and at least 16GB RAM.

"""


import yaml
import torch
import shutil
import sys
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# Path setup to ensure local imports work
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# Internal imports
from config import DATA_YAML, RUNS_DIR, BASE_MODEL, BEST_MODEL, TRAINING_RUN_NAME, HPARAM_SEARCH_RESULT

def merge_best_params(base_params, best_params):
    """Merge base configuration with Optuna hyperparameters."""
    merged = base_params.copy()
    merged.update(best_params or {})
    return merged

def train_production(best_params=None):
    """
    Main training orchestration logic.
    Supports resuming from checkpoints and deploying the best model.
    """
    
    # BASE TRAINING CONFIG (optimized for CPU) 
    train_params = {
        "data": str(DATA_YAML),
        "project": str(RUNS_DIR),
        "name": TRAINING_RUN_NAME,
        "exist_ok": True,
        
        "plots": True,
        "verbose": True,
        
        "device": 0 if torch.cuda.is_available() else "cpu",
        "workers": 4 if torch.cuda.is_available() else 0,
        "batch": 16 if torch.cuda.is_available() else 4,
        
        "epochs": 1, # change to 100-150 for actual training (this is just for testing the pipeline)
        "patience": 20,
        "imgsz": 640,
        "cos_lr": True,
        
        "save": True,
        "save_period": 10
    }

    # MERGE OPTUNA BEST PARAMS
    if best_params:
        print("HYPERPARAMETERS DETECTED")
        for key, value in best_params.items():
            print(f"  [OPTUNA] {key}: {value}")
        train_params = merge_best_params(train_params, best_params)
    else:
        print("NO BEST PARAMS PROVIDED\n")
        print("Status: Using default production configuration.")


    # RESUME LOGIC
    run_dir = RUNS_DIR / TRAINING_RUN_NAME
    ckpt_path = run_dir / "weights" / "last.pt"
    
    if ckpt_path.exists():
        print(f"\n[RESUME] Found checkpoint: {ckpt_path}")
        model = YOLO(str(ckpt_path))
        train_params["resume"] = True
    else:
        print(f"\n[START] No checkpoint found, starting from {BASE_MODEL}")
        model = YOLO(str(BASE_MODEL))
        train_params["resume"] = False

    # TRAINING EXECUTION
    print(f"\nSTARTING TRAINING | Run: {TRAINING_RUN_NAME}\n")
    try:
        results = model.train(**train_params)
        
        # SAVE PARAMETERS FOR DOCUMENTATION
        weights_dir = run_dir / "weights"
        config_dump_path = weights_dir / "training_params.yaml"
        with open(config_dump_path, "w") as f:
            yaml.dump(train_params, f)

        # DEPLOY TO PRODUCTION 
        best_path = weights_dir / f"best_{TRAINING_RUN_NAME}.pt" # adjust name if needed 
        if best_path.exists():
            BEST_MODEL.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(best_path, BEST_MODEL)
            print(f"\nDEPLOYMENT | Model deployed to: {BEST_MODEL}")

        return results

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Training stopped by user.")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        raise


def load_best_hparams():
    """
    Utility to load best hyperparameters from yaml if they exist.
    """
    if HPARAM_SEARCH_RESULT.exists():
        print(f"[INFO] Loading best hyperparameters from {HPARAM_SEARCH_RESULT.name} ")
        with open(HPARAM_SEARCH_RESULT, 'r') as f:
            try:
                return yaml.safe_load(f)
            except Exception as e:
                print(f"[ERROR] Could not parse yaml: {e} ")
                return None
    return None

def main():
    try:
        best_params = load_best_hparams()
        train_production(best_params=best_params)
        
        print(f"\nPROCESS COMPLETE | Results in: {RUNS_DIR / TRAINING_RUN_NAME}")
    except Exception as e:
        print(f"Critical Error in main: {e}")

if __name__ == "__main__":
    main()