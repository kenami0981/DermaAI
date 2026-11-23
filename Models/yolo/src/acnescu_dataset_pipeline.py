"""
ACNESCU DATASET PREPARATION PIPELINE

dataset source: https://universe.roboflow.com/detect-4fwbi/acnescu/dataset/1

Transforms raw ACNESCU polygon annotations and images into a clean, YOLO-formatted dataset.
1. Copies raw Roboflow export directory into a working bounding-box directory structure.
2. Converts raw polygon annotations into standardized YOLO bounding boxes (cx, cy, w, h).
3. Gathers all valid images and performs a randomized split into Train (80%), Val (10%), and Test (10%).
4. Generates or updates the required 'data.yaml' configuration file for YOLO training.

PREREQUISITES:
Place the raw Roboflow dataset export directory inside Models/yolo/data/ folder:
-> Path: Models/yolo/data/ACNESCU.v1i.yolo26/

"""

import os
import shutil
import random
from pathlib import Path
import yaml

from config import ROOT

SPLITS = ["train", "valid", "test"]

SRC = ROOT / "data" / "ACNESCU.v1i.yolo26"
DST = ROOT / "data" / "ACNESCU.v1i.yolo26_bbox"

RATIO_TRAIN = 0.8
RATIO_VAL = 0.1
RATIO_TEST = 0.1 

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

DEFAULT_CLASSES = [
    'atrophic_scar', 'closed_comedo', 'hypertrophic_scar', 
    'melasma', 'nevus', 'nodule', 'open_comedo', 
    'other', 'papule', 'pustule'
]

def polygon_to_bbox(parts):
    cls = parts[0]
    coords = list(map(float, parts[1:]))

    xs = coords[0::2]
    ys = coords[1::2]

    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)

    xc = (xmin + xmax) / 2
    yc = (ymin + ymax) / 2
    w = xmax - xmin
    h = ymax - ymin

    return f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def convert_polygons_to_bboxes():
    print(f"[STEP 1] Copying and converting polygons to bboxes in: {DST}")
    if not SRC.exists():
        raise FileNotFoundError(f"Source ACNESCU directory not found at: {SRC}")
    
    shutil.copytree(SRC, DST, dirs_exist_ok=True)

    for split in SPLITS:
        labels_dir = DST / split / "labels"
        if not labels_dir.exists():
            continue

        for txt_file in labels_dir.glob("*.txt"):
            converted = []

            with open(txt_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    # If it already has 5 elements, it is a bbox
                    if len(parts) == 5:
                        converted.append(line)
                        continue

                    converted.append(polygon_to_bbox(parts))

            with open(txt_file, "w", encoding="utf-8") as f:
                f.write("\n".join(converted))

    print("[SUCCESS] Annotation conversion completed successfully.")


def prepare_dataset_splits():
    print(f"\n[STEP 2] Analyzing and splitting dataset in: {DST}")
    
    image_paths = []
    for img_file in DST.rglob("*"):
        if img_file.suffix.lower() in VALID_EXTS and "images" in img_file.parts:
            image_paths.append(img_file)
            
    image_paths = list(set(image_paths))
    
    if not image_paths:
        print("[ERROR] No images found inside the 'images' directory structure!")
        return

    print(f"Found {len(image_paths)} images in total. Gathering image-label pairs...")

    dataset_pairs = []
    for img_path in image_paths:
        lbl_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            lbl_path = img_path.with_suffix(".txt")
            
        dataset_pairs.append((img_path, lbl_path if lbl_path.exists() else None))

    random.seed(42)  
    random.shuffle(dataset_pairs)

    total = len(dataset_pairs)
    train_end = int(total * RATIO_TRAIN)
    val_end = train_end + int(total * RATIO_VAL)

    splits = {
        "train": dataset_pairs[:train_end],
        "valid": dataset_pairs[train_end:val_end],
        "test": dataset_pairs[val_end:]
    }

    if RATIO_TEST > 0 and not splits["test"] and len(dataset_pairs) >= 3:
        splits["test"].append(splits["valid"].pop())

    for split_name, pairs in splits.items():
        if not pairs:
            continue
            
        split_img_dir = DST / split_name / "images"
        split_lbl_dir = DST / split_name / "labels"
        
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Moving {len(pairs)} files to split: {split_name}...")
        
        for img_path, lbl_path in pairs:
            dest_img = split_img_dir / img_path.name
            
            if img_path.resolve() != dest_img.resolve():
                shutil.move(str(img_path), str(dest_img))
                
            if lbl_path and lbl_path.exists():
                dest_lbl = split_lbl_dir / lbl_path.name
                if lbl_path.resolve() != dest_lbl.resolve():
                    shutil.move(str(lbl_path), str(dest_lbl))

    # Generate data.yaml configuration file
    yaml_path = DST / "data.yaml"
    
    classes = DEFAULT_CLASSES
    if yaml_path.exists():
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if "names" in data:
                    classes = data["names"]
        except Exception:
            pass

    yaml_data = {
        "path": str(DST.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "nc": len(classes),
        "names": classes
    }
    
    if splits["test"]:
        yaml_data["test"] = "test/images"

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, sort_keys=False, default_flow_style=False)

    print(f"[SUCCESS] The data.yaml file has been generated at: {yaml_path}")
    print("Dataset splitting completed successfully.")


if __name__ == "__main__":
    print("ACNESCU DATASET PREPARATION PIPELINE START")
    convert_polygons_to_bboxes()
    prepare_dataset_splits()
    print("\n[SUCCESS] The entire ACNESCU preparation process completed successfully.")