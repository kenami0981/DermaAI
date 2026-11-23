import os
import shutil
import random
from pathlib import Path
import yaml

from config import ROOT

SPLITS = ["train", "valid", "test"]

import shutil

DATASET_ROOT = ROOT / "data" / "ACNESCU.v1i.yolo26_bbox_preprocessed"


RATIO_TRAIN = 0.8
RATIO_VAL = 0.1
RATIO_TEST = 0.1 


VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

DEFAULT_CLASSES = [
    'atrophic_scar', 'closed_comedo', 'hypertrophic_scar', 
    'melasma', 'nevus', 'nodule', 'open_comedo', 
    'other', 'papule', 'pustule'
]

def prepare_dataset_splits():
    print(f"Analizowanie datasetu w: {DATASET_ROOT}")
    
    image_paths = []
    
    for img_file in DATASET_ROOT.rglob("*"):
        if img_file.suffix.lower() in VALID_EXTS and "images" in img_file.parts:
            image_paths.append(img_file)
            
    image_paths = list(set(image_paths))
    
    if not image_paths:
        print("Nie znaleziono żadnych obrazów w strukturze folderu 'images'!")
        return

    print(f"Znaleziono łącznie {len(image_paths)} obrazów. Trwa zbieranie par obraz-etykieta...")

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
            
        split_img_dir = DATASET_ROOT / split_name / "images"
        split_lbl_dir = DATASET_ROOT / split_name / "labels"
        
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Przenoszenie {len(pairs)} plików do splitu: {split_name}...")
        
        for img_path, lbl_path in pairs:
            dest_img = split_img_dir / img_path.name
            
            if img_path.resolve() != dest_img.resolve():
                shutil.move(str(img_path), str(dest_img))
                
            if lbl_path and lbl_path.exists():
                dest_lbl = split_lbl_dir / lbl_path.name
                if lbl_path.resolve() != dest_lbl.resolve():
                    shutil.move(str(lbl_path), str(dest_lbl))

    for subfolder in ["train", "valid", "test"]:
        old_dir = DATASET_ROOT / subfolder
        
    yaml_path = DATASET_ROOT / "data.yaml"
    
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
        "train": "train/images",
        "val": "valid/images",
        "nc": len(classes),
        "names": classes
    }
    
    if splits["test"]:
        yaml_data["test"] = "test/images"

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, sort_keys=False, default_flow_style=False)

    print(f"Sukces! Plik data.yaml został zaktualizowany w: {yaml_path}")
    print("Podział datasetu zakończony pomyślnie.")

if __name__ == "__main__":
    prepare_dataset_splits()