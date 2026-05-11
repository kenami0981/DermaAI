"""
IMAGE ENHANCEMENT PIPELINE

Run this script to improve the quality of images before training or inference.
It creates a copy of your input directory with a '_preprocessed' suffix.

What it does:

* Boosts local contrast (CLAHE) to highlight subtle lesions
* Sharpens details (Unsharp Masking) for clearer textures
* Maintains directory structure and copies existing labels

Usage:

1. Configure IMAGES_DIR or DATA_DIR in config.py.
2. Select the target folder in the main block.
3. Run: python Models/image_enhancement.py

Output:

* Creates: [input_folder]_preprocessed/
* Original images remain untouched.

Requirements:

* Input folder must contain images (.jpg, .jpeg, .png).
* Associated labels (if any) will be copied automatically.

"""

import cv2
import numpy as np
import os
import shutil
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from config import DATA_DIR, IMAGES_DIR

def enhance_details(img):
    """
    Applies enhancement to highlight acne lesions.

    """
    # Contrast Enhancement using CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    enhanced_img = cv2.merge((l_enhanced, a, b))
    enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_LAB2BGR)


    # Sharpening using Unsharp Masking (USM)
    blur = cv2.GaussianBlur(enhanced_img, (0, 0), 3.0)
    # result = original * (1 + amount) + blurred * (-amount)
    # amount=0.6 - clean sharpen without halos
    enhanced_img = cv2.addWeighted(enhanced_img, 1.6, blur, -0.6, 0)
    
    return enhanced_img

def process_single_image(file_info):

    image_path, output_path = file_info
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        return False

    # Apply enhancement
    final_img = enhance_details(img)

    # Save with high quality 
    cv2.imwrite(str(output_path), final_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return True

def run_pipeline(input_folder):
    """
    Main orchestration logic 
    """

    input_path = Path(input_folder)
    output_path = input_path.parent / (input_path.name + "_preprocessed")
    output_path.mkdir(exist_ok=True)

    extensions = ('.png', '.jpg', '.jpeg')
    files = [f for f in input_path.rglob("*") if f.suffix.lower() in extensions]

    if not files:
        print(f"No images found in {input_path}")
        return

    print(f"Starting enhancement for {len(files)} images...")

    tasks = []
    for f in files:
        rel_path = f.relative_to(input_path)
        out_f = output_path / rel_path
        tasks.append((f, out_f))

    # Using ProcessPoolExecutor for CPU-bound image processing
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_single_image, tasks), total=len(tasks)))

    # Synchronize non-image files (eg. labels)
    print("Copying associated label files...")
    for f in input_path.rglob("*"):
        if f.is_file() and f.suffix.lower() not in extensions:
            target = output_path / f.relative_to(input_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)

    print(f"\nEnhancement Complete.\n")
    print(f"Successfully processed: {sum(results)} / {len(files)}")
    print(f"Output folder: {output_path}")

if __name__ == "__main__":

    # run_pipeline(DATA_DIR) # to enhance dataset_final
    run_pipeline(IMAGES_DIR) # to enhance images to predict on 