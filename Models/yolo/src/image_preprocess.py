"""
IMAGE ENHANCEMENT PIPELINE

Run this script to improve the quality of images before training or inference.
It creates a copy of your input directory with a '_preprocessed' suffix.
If applied to dataset it will rename the folder to 'final_dataset_preprocessed.

What it does:

* Boosts local contrast (CLAHE) to highlight subtle lesions
* Sharpening details (Unsharp Masking) - REMOVED IN FINAL V2 (caused noise)
* Maintains directory structure and copies existing labels

Usage:

1. Configure IMAGES_DIR in config.py.
2. Select the target folder in the main block.
3. Run: python Models/yolo/src/image_preprocess.py

Output:

* Creates: final_dataset_preprocessed/
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

def enhance_details(img, clip_l=1.2, clip_a=1.1, grid_size=(8, 8)):
    """
    Applies enhancement to highlight acne lesions.

    """
    # Contrast Enhancement using CLAHE on L and a channels
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Balanced contrast boost
    clahe_l = cv2.createCLAHE(clipLimit=clip_l, tileGridSize=grid_size) 
    l_enhanced = clahe_l.apply(l)
    
    clahe_a = cv2.createCLAHE(clipLimit=clip_a, tileGridSize=grid_size) 
    a_enhanced = clahe_a.apply(a)
    
    enhanced_img = cv2.merge((l_enhanced, a_enhanced, b))
    enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_LAB2BGR)


    return enhanced_img

def process_single_image(file_info):

    image_path, output_path, clip_l, clip_a, grid_size = file_info
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        return False

    # Apply enhancement
    final_img = enhance_details(img, clip_l, clip_a, grid_size)

    # Save with high quality 
    # cv2.imwrite(str(output_path), final_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    cv2.imwrite(str(output_path), final_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return True

def run_pipeline(input_folder, suffix="_preprocessed", clip_l=1.2, clip_a=1.1, grid_size=(8, 8)):
    """
    Main orchestration logic 
    """

    input_path = Path(input_folder)
    output_path = input_path.parent / (input_path.name + suffix)
    output_path.mkdir(exist_ok=True)

    extensions = ('.png', '.jpg', '.jpeg')
    files = [f for f in input_path.rglob("*") if f.suffix.lower() in extensions]

    if not files:
        print(f"No images found in {input_path}")
        return

    print(f"Starting enhancement for {len(files)} images ({suffix.strip('_')})...")

    tasks = []
    for f in files:
        rel_path = f.relative_to(input_path)
        out_f = output_path / rel_path
        tasks.append((f, out_f, clip_l, clip_a, grid_size))

    # Using ProcessPoolExecutor for CPU-bound image processing
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_single_image, tasks), total=len(tasks)))

    # Synchronize non-image files (eg. labels)
    print("Copying associated label files...")
    non_image_files = [f for f in input_path.rglob("*") if f.is_file() and f.suffix.lower() not in extensions]
    for f in tqdm(non_image_files, desc="Copying labels"):
        target = output_path / f.relative_to(input_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)

    print(f"\nEnhancement Complete.\n")
    print(f"Successfully processed: {sum(results)} / {len(files)}")
    print(f"Output folder: {output_path}")

if __name__ == "__main__":

    # Configure before running!!! 
    # dataset to preprocess: 
    dir_to_preprocess = ROOT_DIR / "data" / "Acne04_HD_YOLO_Filtered"

    # dir with images to enhance for prediction:
    # dir_to_preprocess = IMAGES_DIR  
    
    target_suffix = "_preprocessed"

    # Default production run using optimized values
    run_pipeline(
        dir_to_preprocess, 
        suffix=target_suffix, 
        clip_l=1.2, 
        clip_a=1.1, 
        grid_size=(8, 8)
    )

    if dir_to_preprocess == ROOT_DIR / "data" / "Acne04_HD_YOLO_Filtered":
            generated_dir = ROOT_DIR / "data" / "Acne04_HD_YOLO_Filtered_preprocessed"
            final_dataset_dir = ROOT_DIR / "data" / "final_dataset_preprocessed"
            
            if generated_dir.exists():
                print(f"\n[POST-PROCESS] Renaming {generated_dir.name} to {final_dataset_dir.name}...")
                if final_dataset_dir.exists():
                    shutil.rmtree(final_dataset_dir)
                generated_dir.rename(final_dataset_dir)
                print(f"[SUCCESS] Final dataset available at: {final_dataset_dir}")

    # python Models/yolo/src/image_preprocess.py # standard preprocessing


    
    
    
    """
    ======================================================================
    ACCURACY BENCHMARKS: PREPROCESSING IMPACT ON VALIDATION mAP50
    ======================================================================
    
    PHASE 1: INITIAL FILTER SCREENING
    
    [Test 1.1] Model v1.0 (Trained on RAW images)
    PREPROCESSING METHOD       | mAP50    | IMPACT (%)
    Gray World Only            | 0.0559   | +2.68% 
    LAB (L + a*) Only          | 0.0551   | +1.15%
    Full Pipeline Combined     | 0.0448   | -17.84% 

    [Test 1.2] Model v2.0 (Trained on CLAHE images)
    PREPROCESSING METHOD       | mAP50    | IMPACT (%) 
    Original Baseline          | 0.0604   | Reference
    Gray World Only            | 0.0527   | -12.72% .
    LAB (L + a*) Only          | 0.0839   | +38.88% 
    Full Pipeline Combined     | 0.0587   | -2.80% 

    ======================================================================

    PHASE 2: CLAHE PARAMETER GRID OPTIMIZATION
    
    [Test 2.1] Model v1.0 (Trained on RAW images)
    PREPROCESSING METHOD       | mAP50    | IMPACT (%) 
    Original Baseline          | 0.0545   | Reference
    CLAHE Mild (1.0/1.0)       | 0.0576   | +5.72% 
    CLAHE Balanced (1.2/1.1)   | 0.0567   | +4.00% 
    CLAHE Sharp (1.5/1.3 @ 4x4)| 0.0589   | +8.05%

    [Test 2.2] Model v2.0 (Trained on CLAHE images)
    PREPROCESSING METHOD       | mAP50    | IMPACT (%)
    Original Baseline          | 0.0604   | Reference
    CLAHE Mild (1.0/1.0)       | 0.0786   | +30.00% 
    CLAHE Balanced (1.2/1.1)   | 0.0810   | +34.00% 
    CLAHE Sharp (1.5/1.3 @ 4x4)| 0.0760   | +25.82%
    
    """