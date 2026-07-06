"""
# ACNE DETECTION INFERENCE PIPELINE

This script automates the inference process using the best-performing YOLO model.
It handles image preprocessing, lesion detection, and diagnostic calculations.

FUNCTIONALITY:
1. Load production weights (e.g. best_v1.pt) defined in config.py.
2. Apply detail enhancement (optional) to improve detection on skin textures.
3. Calculate a normalized 'Acne Score' (0.0 - 1.0) based on:
   - Lesion quantity (count)
   - Detection confidence (model certainty)
   - Relative lesion size (box area vs. total image area)
4. Export structured results for each image.

OUTPUT STRUCTURE:
Results are saved in: Models/runs/predict/predict_[MODEL]_[TIMESTAMP]/
└── img_[IMG_NAME]/
    ├── image_clean.jpg     <-- Preprocessed input image with bounding boxes (no labels)
    ├── image0.jpg          <-- Annotated image (bounding boxes, no labels)
    └── results.json        <-- Diagnostic report (score, time, coordinates, confidence)

USAGE:
    python Models/src/predict.py                    <-- Runs inference on all images in IMAGES_DIR
    python Models/src/predict.py --image file.jpg   <-- Runs inference on a specific file
    python Models/src/predict.py --no_preprocess    <-- Disables image detail enhancement
"""

import time
import json
import numpy as np
import argparse
import cv2
import shutil
import sys
import torch

from ultralytics import YOLO
from pathlib import Path
from datetime import datetime

# Path setup to ensure local imports work
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from image_preprocess import enhance_details
from config import BEST_MODEL, IMAGES_DIR, RUNS_DIR, CONF_THRESHOLD, IMG_SIZE

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


def run_inference(image_path, model, session_dir, preprocess=True, tta=True):
    """
    Run full prediction pipeline on one image.
    Returns:
    - score
    - detections
    - inference time
    - output image path
    """

    start = time.time()

    # Load image for potential preprocessing
    img_org = cv2.imread(str(image_path))
    if img_org is None:
        return None

    img_output_folder = session_dir / f"img_{image_path.stem}"
    img_output_folder.mkdir(parents=True, exist_ok=True)

    # Apply enhancement if enabled
    img_input = enhance_details(img_org) if preprocess else img_org

    # Run YOLO prediction
    if tta:
        # Generate image variants for multi-scale prediction
        img_flipped = cv2.flip(img_input, 1)

        # Run predictions on variants (returns clean data without saving to disk)
        res_scale1 = model.predict(source=img_input, imgsz=int(IMG_SIZE * 1.25), conf=CONF_THRESHOLD, save=False, verbose=False, augment=False)
        res_scale2 = model.predict(source=img_input, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, save=False, verbose=False, augment=False)
        res_flipped = model.predict(source=img_flipped, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, save=False, verbose=False, augment=False)

        # Run the main inference pass, which will create the folder structure and save the base file
        # Dynamic line thickness calculation for the built-in save method
        h, w = img_input.shape[:2]
        dynamic_line_thickness = max(round(max(h, w) / 600), 1)

        results = model.predict(
            source=img_input,
            imgsz=IMG_SIZE,
            conf=CONF_THRESHOLD,
            save=True,
            project=str(img_output_folder),
            name="temp",
            show_labels=False, # only boxes are drawn (no class names)
            line_thickness=dynamic_line_thickness,
            exist_ok=True,
            augment=False 
        )

        # Securely merge objects using tensor operations
        if results and len(results) > 0 and results[0].boxes is not None:
            main_b = results[0].boxes
            
            # Lists for individual tensor components, starting with the main inference pass
            all_data = [main_b.data]

            def process_and_collect(source_results, is_flipped=False):
                if source_results and len(source_results) > 0 and source_results[0].boxes is not None:
                    src_b = source_results[0].boxes
                    if len(src_b) > 0:
                        tensor_data = src_b.data.clone()
                        if is_flipped:
                            # Flip the X coordinates for bounding boxes from the mirror reflection
                            img_w = main_b.orig_shape[1]
                            x1 = tensor_data[:, 0].clone()
                            x2 = tensor_data[:, 2].clone()
                            tensor_data[:, 0] = img_w - x2
                            tensor_data[:, 2] = img_w - x1
                        all_data.append(tensor_data)

            process_and_collect(res_scale1)
            process_and_collect(res_scale2)
            process_and_collect(res_flipped, is_flipped=True)

            # If we collected additional boxes, natively concatenate them into one large tensor
            if len(all_data) > 1:
                results[0].boxes.data = torch.cat(all_data, dim=0)
    else:
        # Dynamic line thickness calculation for the non-TTA block
        h, w = img_input.shape[:2]
        dynamic_line_thickness = max(round(max(h, w) / 600), 1)

        results = model.predict(
            source=img_input,
            imgsz=IMG_SIZE,
            conf=CONF_THRESHOLD,
            save=True,
            project=str(img_output_folder),
            name="temp",
            show_labels=False,
            line_thickness=dynamic_line_thickness,
            exist_ok=True,
            augment=False
        )

    if results and len(results) > 0:
        # Calculate dynamic line width: e.g. 2 pixels for smaller images, thicker for larger images
        h, w = results[0].boxes.orig_shape
        dynamic_line_width = max(round(max(h, w) / 600), 1)  # For 1280px it yields a line thickness of ~2

        annotated_img = results[0].plot(labels=False, conf=False, line_width=dynamic_line_width)
        clean_img_path = img_output_folder / "image_clean.jpg"
        cv2.imwrite(str(clean_img_path), annotated_img)

    temp_dir = img_output_folder / "temp"
    temp_file = next(temp_dir.glob("*"), None)

    if temp_file and temp_file.exists():
        final_img_path = img_output_folder / temp_file.name
        shutil.move(str(temp_file), str(final_img_path))
        shutil.rmtree(str(temp_dir))
    else:
        final_img_path = image_path

    detections = extract_detections(results, model)
    score = calculate_score(results)
    end = time.time()

    # result data
    res_data = {
        "score": score,
        "time": round(end - start, 3),
        "tta_applied": tta,
        "detections": detections,
        "image": {
            "input": image_path.name,
            "output": str(final_img_path.relative_to(RUNS_DIR))
        }
    }

    with open(img_output_folder / "results.json", "w", encoding="utf-8") as f:
        json.dump(res_data, f, indent=2)

    return res_data


def run_pipeline(input_arg, preprocess=True, tta=True):
    """Main orchestration logic for inference."""
    model = load_model()
    
    # Path logic: arg file, arg folder, or default IMAGES_DIR
    if input_arg:
        input_path = Path(input_arg)
    else:
        input_path = IMAGES_DIR

    if not input_path.exists():
        print(f"Error: Path {input_path} does not exist.")
        return

    extensions = ('.png', '.jpg', '.jpeg')
    if input_path.is_file():
        files = [input_path]
    else:
        files = [f for f in input_path.rglob("*") if f.suffix.lower() in extensions]

    if not files:
        print(f"No images found to process in {input_path}")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    session_name = f"predict_{BEST_MODEL.stem}_{timestamp}"
    session_dir = RUNS_DIR / "predict" / session_name

    print(f"\nSTARTING INFERENCE | Images: {len(files)} | Session: {session_name} | TTA: {tta}\n")

    all_results = []
    for img_p in files:
        res = run_inference(img_p, model, session_dir, preprocess=preprocess, tta=tta)
        if res:
            all_results.append(res)

    print(f"\nInference Complete.")
    print(f"Successfully processed: {len(all_results)} / {len(files)}")
    print(f"Results saved in: {session_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="Path to image or directory")
    parser.add_argument("--no_preprocess", action="store_false", dest="preprocess", help="Disable enhancement")
    parser.add_argument("--no_tta", action="store_false", dest="tta", help="Disable Test-Time Augmentation")
    parser.set_defaults(preprocess=True, tta=True)

    args = parser.parse_args()

    try:
        run_pipeline(args.image, preprocess=args.preprocess, tta=args.tta)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()