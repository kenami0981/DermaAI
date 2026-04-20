from ultralytics import YOLO
from config import DATA_YAML, RUNS_DIR, YOLO_WEIGHTS

def train_baseline():

    model = YOLO(str(YOLO_WEIGHTS))

    model.train(
        data=str(DATA_YAML),

        epochs=10,
        imgsz=320,
        batch=4, 
        workers=0,
        device="cpu",

        project=str(RUNS_DIR),
        exist_ok=True,

        plots=True,
        verbose=True,

        patience=5,
        save=True,
        save_period=2
    )

def main():
    try:
        train_baseline()
        print(f"Training finished. Output in: {RUNS_DIR}")
    except Exception as e:
        print(f"Training failed: {e}")

if __name__ == "__main__":
    main()