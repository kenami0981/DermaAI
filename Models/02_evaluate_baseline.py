from ultralytics import YOLO
from config import BEST_MODEL, DATA_YAML, RUNS_DIR

def main():
    try:
        if not BEST_MODEL.exists():
            raise FileNotFoundError("Model not found. Train first")

        model = YOLO(str(BEST_MODEL))

        print(f"Validating: {BEST_MODEL.name}")

        metrics = model.val(
            data=str(DATA_YAML),
            imgsz=320,
            split="val",
            device="cpu",
            project=str(RUNS_DIR),
            plots=True,
            verbose=True,
            exist_ok=True
        )

        print(f"Precision: {metrics.results_dict['metrics/precision(B)']:.3f}")
        print(f"Recall: {metrics.results_dict['metrics/recall(B)']:.3f}")
        print(f"mAP50: {metrics.results_dict['metrics/mAP50(B)']:.3f}")
        print(f"mAP50-95: {metrics.results_dict['metrics/mAP50-95(B)']:.3f}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()