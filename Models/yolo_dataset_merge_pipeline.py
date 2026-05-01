import shutil
import random
import sys
import yaml
from pathlib import Path

"""
===========================================================
WARNING: DESTRUCTIVE OPERATION
===========================================================

Run this script to build dataset_final from datasets listed below.
It will recreate dataset_final from scratch and include ONLY those datasets.
To add data: update DATASETS list and run again.

IMPORTANT !!!
If dataset_final already exists, it will be completely DELETED before rebuilding.

What it does:
- Merges multiple YOLO datasets
- Keeps train / valid / test structure
- Converts all classes -> class 0
- Creates data.yaml
- Prevents filename collisions

Requirements:
- Each dataset must follow YOLO structure:

    dataset/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    ├── test/
    │   ├── images/
    │   └── labels/

- Labels must be YOLO format (class x y w h, normalized 0-1)

"""

# CONFIGURATION

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT / "Models" / "data"

# List of datasets to merge 
# (remember it HAS TO contain ALL datasets you want to have in dataset_final, not only new ones!)
DATASETS = [
    BASE_DIR / "Acne.v21i.yolov8",
    BASE_DIR / "acne.v3i.yolov8",
    BASE_DIR / "acne yolo.v1-yolov5-v-0.1.yolov8",
    BASE_DIR / "cubeai-acne-detection-for-yolov8",
    # BASE_DIR / "some_new_dataset"
]

# Output directory (WILL BE DELETED EACH RUN)
OUTPUT_DIR = BASE_DIR / "dataset_final"

SPLITS = ["train", "valid", "test"]


# VALIDATION FUNCTIONS

def check_yolo_dataset(dataset_path: Path, sample_check=100):
    """
    Validate the integrity of a YOLO dataset before merging.
    Checks for missing files, empty labels, and coordinate range.
    """

    ds = Path(dataset_path)
    print(f"DATASET CHECK: {ds}")

    splits = ["train", "valid", "test"]
    found_any = False
    global_ok = True

    for split in splits:
        img_dir = ds / split / "images"
        lbl_dir = ds / split / "labels"

        # Skip if split directory doesn't exist
        if not img_dir.exists() or not lbl_dir.exists():
            continue

        found_any = True
        print(f"\nSPLIT: {split.upper()}")

        images = list(img_dir.glob("*.*"))
        labels = list(lbl_dir.glob("*.txt"))

        print(f"Images: {len(images)}")
        print(f"Labels: {len(labels)}")

        # Check for 1:1 mapping between images and labels
        missing_labels = [img.name for img in images if not (lbl_dir / f"{img.stem}.txt").exists()]
        missing_images = [lbl.name for lbl in labels if not any((img_dir / f"{lbl.stem}{ext}").exists() for ext in [".jpg", ".png", ".jpeg"])]

        print(f"Missing labels: {len(missing_labels)}")
        print(f"Missing images: {len(missing_images)}")

        bad_format = 0
        bad_values = 0
        empty_labels = 0

        # Random sampling for performance optimization
        sample_size = min(len(labels), sample_check)
        sample_labels = random.sample(labels, sample_size) if sample_size > 0 else []

        for lbl in sample_labels:
            try:
                lines = lbl.read_text().strip().splitlines()
                if len(lines) == 0:
                    empty_labels += 1
                    continue

                for line in lines:
                    parts = line.split()
                    if len(parts) != 5:
                        bad_format += 1
                        continue

                    # Validate YOLO coordinates (must be normalized 0-1)
                    _, x, y, w, h = parts
                    x, y, w, h = float(x), float(y), float(w), float(h)
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        bad_values += 1
            except Exception:
                bad_format += 1

        print(f"Checked samples: {len(sample_labels)}")
        print(f"Empty labels: {empty_labels}")
        print(f"Bad format: {bad_format}")
        print(f"Out-of-range: {bad_values}")

        split_ok = (len(missing_labels) == 0 and len(missing_images) == 0 and bad_format == 0 and bad_values == 0)
        
        if split_ok:
            print("SPLIT OK")
        else:
            print("SPLIT BROKEN")
            global_ok = False

    if not found_any:
        print("No valid YOLO structure found (train/valid/test missing)")
        return False

    return global_ok


# MERGE FUNCTIONS

def safe_copy(src: Path, dst: Path):
    """
    Copy file without overwriting existing ones.
    Adds numeric suffix if filename collision occurs.
    """

    if not dst.exists():
        shutil.copy2(src, dst)
        return dst

    stem = dst.stem
    suffix = dst.suffix
    i = 1

    while True:
        new_dst = dst.with_name(f"{stem}_{i}{suffix}")
        if not new_dst.exists():
            shutil.copy2(src, new_dst)
            return new_dst
        i += 1

def convert_labels_to_single_class(label_path: Path):
    """
    Convert YOLO all labels to single-class format.
    Input: class x y w h -> Output: 0 x y w h
    """
    with open(label_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue  # skip malformed lines
        
        parts[0] = "0" # Overwrite class to 0
        new_lines.append(" ".join(parts))

    return "\n".join(new_lines)

def write_label(dst_path: Path, content: str):
    """
    Save the modified label content to the new path.
    """

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "w") as f:
        f.write(content)

def merge_datasets():
    """
    Merge all listed datasets into a unified YOLO dataset.
    - Preserves splits (train/valid/test)
    - Copies images safely using prefix to avoid name clashes
    - Converts all labels to single class (0)
    """

    total_images = 0

    for split in SPLITS:
        print(f"\nPROCESSING SPLIT: {split}")

        out_img_dir = OUTPUT_DIR / split / "images"
        out_lbl_dir = OUTPUT_DIR / split / "labels"

        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        for ds in DATASETS:
            split_dir = ds / split

            if not split_dir.exists():
                nested = list(ds.glob("*"))
                if len(nested) == 1 and nested[0].is_dir():
                    split_dir = nested[0] / split

            img_dir = split_dir / "images"
            lbl_dir = split_dir / "labels"

            if not img_dir.exists() or not lbl_dir.exists():
                print(f"[SKIP] {ds} -> missing split: {split}")
                continue

            images = list(img_dir.glob("*.*"))

            for img_path in images:
                label_path = lbl_dir / (img_path.stem + ".txt")

                if not label_path.exists():
                    continue

                # Generate a unique name based on source dataset to prevent overwrites
                dataset_name = ds.name.replace(".", "_")
                new_name = f"{dataset_name}_{img_path.name}"

                new_img_path = safe_copy(img_path, out_img_dir / new_name)

                new_label_path = out_lbl_dir / (new_img_path.stem + ".txt")
                converted = convert_labels_to_single_class(label_path)
                write_label(new_label_path, converted)

                total_images += 1

    print(f"\nDONE: {total_images} samples merged")

def create_yaml():
    """Generate YOLO data.yaml configuration file for training."""

    yaml_path = OUTPUT_DIR / "data.yaml"

    data = {
        "train": str((OUTPUT_DIR / "train" / "images").resolve()),
        "val": str((OUTPUT_DIR / "valid" / "images").resolve()),
        "test": str((OUTPUT_DIR / "test" / "images").resolve()),
        "nc": 1,
        "names": ["acne_lesion"]
    }

    with open(yaml_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)

    print(f"YAML saved: {yaml_path}")





if __name__ == "__main__":
    
    # 1. RUN PRE-MERGE VALIDATION
    print("PHASE 1: STARTING VALIDATION")
    all_ok = True
    
    for ds_to_check in DATASETS:
        if not ds_to_check.exists():
            print(f"CRITICAL ERROR: Path does not exist: {ds_to_check}")
            all_ok = False
            continue
            
        print("\n" + "="*40)
        if not check_yolo_dataset(ds_to_check):
            all_ok = False

    if not all_ok:
        print("\n" + "!"*40)
        print("VALIDATION FAILED. Merge cancelled to prevent data corruption.")
        print("Please fix the datasets listed above.")
        print("!"*40)
        sys.exit(1)

    print("\n" + "="*40)
    print("VALIDATION SUCCESSFUL! Starting merge process...")
    print("="*40 + "\n")

    # 2. CLEANUP (Safety Check prevents accidental deletion of wrong directory)
    if OUTPUT_DIR.exists():
        if "dataset_final" not in str(OUTPUT_DIR):
            raise ValueError("Refusing to delete non-target directory")

        print(f"Removing existing dataset: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    # 3. EXECUTE CORE PIPELINE
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("BASE_DIR:", BASE_DIR)

    merge_datasets()
    create_yaml()
    
    print("\nPROCESSED COMPLETED SUCCESSFULLY.")