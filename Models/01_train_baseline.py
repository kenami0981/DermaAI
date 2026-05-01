from ultralytics import YOLO
from config import DATA_YAML, RUNS_DIR, PRETRAINED_WEIGHTS, BEST_MODEL
from pathlib import Path
from datetime import datetime
import shutil

def train_baseline():

    run_name = f"baseline_train{datetime.now().strftime('%Y%m%d_%H%M')}"
    run_dir = RUNS_DIR / run_name

    model = YOLO(str(PRETRAINED_WEIGHTS))

    model.train(
        data=str(DATA_YAML),

        epochs=1,
        imgsz=320,
        batch=4, 
        workers=0,
        device="cpu",

        project=str(RUNS_DIR),
        name=run_name,
        exist_ok=True,

        plots=True,
        verbose=True,

        patience=5,
        save=True,
        save_period=2,

        # hsv_h=0.015,
        # hsv_s=0.7,
        # hsv_v=0.4,
        # fliplr=0.5,
        # mosaic=1.0,
        # mixup=0.1
    )

    # DEPLOY
    
    best_path = run_dir / "weights" / "best.pt"

    if best_path.exists():
        shutil.copy(best_path, BEST_MODEL)
        print(f"Model deployed to: {BEST_MODEL}")
    else:
        print("WARNING: best.pt not found, model not deployed")


def main():
    try:
        train_baseline()
        print(f"Training finished. Output in: {RUNS_DIR}")
    except Exception as e:
        print(f"Training failed: {e}")


if __name__ == "__main__":
    main()