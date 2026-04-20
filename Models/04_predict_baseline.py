from ultralytics import YOLO
from config import BEST_MODEL, IMAGES_DIR, RUNS_DIR, CLASS_WEIGHTS
from pathlib import Path
import time
import json


def load_model():
    """
    Load trained YOLO model.
    """
    if not BEST_MODEL.exists():
        raise FileNotFoundError("Model not found. Train first")

    return YOLO(str(BEST_MODEL))


def calculate_score(results, model):
    """
    Converts detections into acne score (0-1)
    """
    total = 0.0

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            class_name = model.names[cls_id]
            weight = CLASS_WEIGHTS.get(class_name, 0.1)

            total += weight * conf

    # normalize 
    return round(min(total / 10.0, 1.0), 3)


def extract_detections(results, model):
    """
    Converts YOLO output into structured JSON.
    """
    detections = []

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            class_name = model.names[cls_id]
            xywh = box.xywh[0].tolist()

            detections.append({
                "class": class_name,
                "confidence": round(conf, 3),
                "bbox": [round(x, 2) for x in xywh]
            })

    return detections



def run_inference(image_path):
    """
    Full inference pipeline:
    image --> YOLO --> detections --> score --> output JSON
    """

    model = load_model()

    start_time = time.time()

    results = model.predict(
        source=str(image_path),
        imgsz=640,
        conf=0.25,
        save=True,
        project=str(RUNS_DIR),
        name="predict"
    )

    detections = extract_detections(results, model)
    score = calculate_score(results, model)

    end_time = time.time()

    save_dir = Path(results[0].save_dir)

    annotated_image = None
    
    for file in save_dir.glob("*"):
        if file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            annotated_image = file.name
            break

    if annotated_image is None:
        annotated_image = ""

    return {
        "score": score,
        "time": round(end_time - start_time, 3),
        "detections": detections,
        "image": {
            "input": image_path.name,
            "output": f"runs/{save_dir.name}/{annotated_image}"
        }
    }



def main():
    try:
        image_path = IMAGES_DIR / "test-image.png"

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        result = run_inference(image_path)

        output_file = RUNS_DIR / "last_result.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        print(json.dumps(result, indent=2))

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()