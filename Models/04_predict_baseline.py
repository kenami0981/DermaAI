from ultralytics import YOLO
from config import BEST_MODEL, IMAGES_DIR, RUNS_DIR
from pathlib import Path
import time
import json
import numpy as np
import argparse


def load_model():
    """
    Load trained production model.
    Stops execution if weights are missing.
    """

    if not BEST_MODEL.exists():
        raise FileNotFoundError("Model not found. Train first")

    return YOLO(str(BEST_MODEL))


def calculate_score(results):
    """
    Convert detections into a normalized acne severity score.

    Score is based on:
    - detection confidence
    - lesion size
    - number of detected lesions
    """

    if not results or results[0].boxes is None:
        return 0.0

    r = results[0]

    if len(r.boxes) == 0:
        return 0.0

    # Original image dimensions
    img_h, img_w = r.boxes.orig_shape
    image_area = img_h * img_w

    score = 0.0

    for box in r.boxes:

        # Detection confidence
        conf = float(box.conf[0])

        # Bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0]

        # Relative lesion size
        box_area = (x2 - x1) * (y2 - y1)
        area_ratio = box_area / image_area

        # Larger + more confident lesions increase severity
        score += conf * (1 + area_ratio)

    # Penalize excessive lesion counts
    num_lesions = len(r.boxes)
    score = score / (1 + num_lesions / 50)

    # Normalize to 0–1 range
    score = score / 5.0
    score = np.clip(score, 0, 1)

    return round(float(score), 3)


def extract_detections(results, model):
    """
    Extract detection metadata into JSON-friendly format.
    """

    detections = []

    for r in results:

        if r.boxes is None:
            continue

        for box in r.boxes:

            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            detections.append({
                "class": model.names[cls_id],
                "confidence": round(conf, 3),

                # YOLO xywh format:
                # center_x, center_y, width, height
                "bbox": [round(x, 2) for x in box.xywh[0].tolist()]
            })

    return detections



def run_inference(image_path):
    """
    Run full prediction pipeline on one image.
    Returns:
    - score
    - detections
    - inference time
    - output image path
    """

    model = load_model()

    start = time.time()

    # Run YOLO prediction
    results = model.predict(
        source=str(image_path),
        imgsz=640,
        conf=0.25,
        save=True,
        project=str(RUNS_DIR),
        name="predict"
    )

    detections = extract_detections(results, model)
    score = calculate_score(results)

    end = time.time()

    # Folder with annotated output image
    save_dir = Path(results[0].save_dir)

    annotated_image = next(
        (
            f.name
            for f in save_dir.glob("*")
            if f.suffix.lower() in [".jpg", ".png", ".jpeg"]
        ),
        ""
    )

    return {
        "score": score,
        "time": round(end - start, 3),
        "detections": detections,
        "image": {
            "input": image_path.name,
            "output": str(Path(save_dir.name) / annotated_image)
        }
    }



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None)

    args = parser.parse_args()

    try:
        # Use provided image or default test image
        image_path = Path(args.image) if args.image else IMAGES_DIR / "test-image.png"

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        result = run_inference(image_path)

        # Save latest result
        output_file = RUNS_DIR / "last_result.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        print(json.dumps(result, indent=2))

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()