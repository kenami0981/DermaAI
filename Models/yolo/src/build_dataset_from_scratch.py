"""
ACNE DATASET PREPARATION PIPELINE

Transforms raw acne images and JSON annotations into a YOLO formated dataset.
1. Splits image files into Train (80%), Val (10%), and Test (10%).
2. Converts raw pixel annotations (circles) into relative YOLO bounding boxes.
3. Fixes potential typos in dataset file names (e.g., "levele" -> "levle").
4. Isolates unannotated or missing images into a separate workspace folder.
5. Generates the required 'data.yaml' configuration file for YOLO training.

PREREQUISITES:
Place these components inside Models/data/ folder:
1. JSON annotation file (Download from GitHub repository)
   -> URL: https://github.com/AIpourlapeau/acne04v2
   -> Put into: Models/data/Acne04-v2_annotations.json
2. Raw images directory (Download ZIP Archive from Kaggle)
   -> URL: https://www.kaggle.com/datasets/karmagames/acne04-v2
   -> Download the zip file from Kaggle and extract its contents directly into 
      'Models/data/' directory. The archive natively contains the nested structure 
      'archive/img_data/img_data/'.

The whole structure should look like that:
Models/
└── data/
    ├── Acne04-v2_annotations.json
    └── archive/
        └── img_data/
            └── img_data/
                ├── levle0_0.jpg
                ├── levle1_1.jpg ...

"""

import json
import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
JSON_PATH = ROOT / "data" / "Acne04-v2_annotations.json"
SRC_IMAGES_DIR = ROOT / "data" / "archive" / "img_data" / "img_data"

FILTERED_DATASET_DIR = ROOT / "data" / "Acne04_HD_YOLO_Filtered"
NO_ANNOTATIONS_DIR = ROOT / "data" / "Acne04_HD_YOLO_No_Annotations"


def get_dir_size_mb(directory: Path) -> float:
    """Calculates total directory footprint in MB."""
    if not directory.exists():
        return 0.0
    all_files = [f for f in directory.rglob('*') if f.is_file()]
    total_size = sum(
        f.stat().st_size for f in tqdm(all_files, desc=f"Calculating size of {directory.name}", leave=False)
    )
    return total_size / (1024 * 1024)


def count_files_in_split(base_dir: Path, split: str) -> tuple:
    """Counts active image and label files inside a specific split folder."""
    img_dir = base_dir / split / "images"
    lbl_dir = base_dir / split / "labels"
    img_count = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
    lbl_count = len(list(lbl_dir.glob("*"))) if lbl_dir.exists() else 0
    return img_count, lbl_count


def load_json_data() -> dict:
    """Loads and returns the source JSON annotation metadata."""
    print(f"[INFO] Reading dataset metadata from: {JSON_PATH}")
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"JSON annotation file not found at: {JSON_PATH}")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def print_json_stats(data: dict):
    """Parses JSON data structures to analyze the distribution of bounding boxes."""
    images = data.get("images", [])
    annotations = data.get("annotations", [])

    classes = set()
    ann_per_image = {}
    
    for ann in tqdm(annotations, desc="Analyzing JSON annotations structure", leave=False):
        img_id = ann.get("image_id")
        ann_per_image[img_id] = ann_per_image.get(img_id, 0) + 1
        if "class_name" in ann:
            classes.add(ann["class_name"])
        elif "category_id" in ann:
            classes.add(ann["category_id"])

    annotated_images_count = len(ann_per_image)
    unannotated_images_count = len(images) - annotated_images_count

    print("\nDETAILED JSON DATA ANALYSIS")
    print(f"Total Images listed in JSON:     {len(images)}")
    print(f"Total Individual BBox Objects:   {len(annotations)}")
    print(f"Images WITH Annotations in JSON: {annotated_images_count}")
    print(f"Images BLANK in JSON:            {unannotated_images_count}\n")


def process_and_split_dataset(data: dict):
    """Runs the core data pipeline: scans disk assets, tracks maps, splits, and formats metadata."""
    print(f"[INFO] Starting disk-first scanning pipeline...")
    if not SRC_IMAGES_DIR.exists():
        raise FileNotFoundError(f"Source images directory not found at: {SRC_IMAGES_DIR}")

    valid_extensions = {".jpg", ".jpeg", ".png"}
    
    print("[INFO] Scanning source directory for physical image assets...")
    raw_disk_files = list(SRC_IMAGES_DIR.glob("*"))
    all_disk_images = [
        f for f in tqdm(raw_disk_files, desc="Filtering valid image extensions", leave=False) 
        if f.suffix.lower() in valid_extensions
    ]
    
    total_disk_count = len(all_disk_images)
    print(f"[INFO] Found {total_disk_count} physical images on disk.")

    images_json = data.get("images", [])
    annotations_json = data.get("annotations", [])

    unique_classes = sorted(list(set(
        ann["class_name"] for ann in annotations_json if "class_name" in ann
    )))
    class_to_id = {cls: idx for idx, cls in enumerate(unique_classes)}
    class_mapping_yaml = {idx: cls for idx, cls in enumerate(unique_classes)}

    filename_to_json_meta = {}
    for img in tqdm(images_json, desc="Indexing image metadata maps", leave=False):
        filename_to_json_meta[img["file_name"]] = img
        
    ann_map = {}
    for ann in tqdm(annotations_json, desc="Indexing target annotation nodes", leave=False):
        ann_map.setdefault(ann["image_id"], []).append(ann)

    random.seed(42)
    random.shuffle(all_disk_images)

    train_end = int(total_disk_count * 0.80)
    val_end = train_end + int(total_disk_count * 0.10)

    split_assignments = {
        "train": all_disk_images[:train_end],
        "valid": all_disk_images[train_end:val_end],
        "test": all_disk_images[val_end:]
    }

    metrics = {
        "moved_to_filtered": 0,
        "moved_to_no_annotations": 0
    }

    for split, img_list in split_assignments.items():
        filt_img_dir = FILTERED_DATASET_DIR / split / "images"
        filt_lbl_dir = FILTERED_DATASET_DIR / split / "labels"
        no_ann_img_dir = NO_ANNOTATIONS_DIR / split / "images"
        no_ann_lbl_dir = NO_ANNOTATIONS_DIR / split / "labels"

        filt_img_dir.mkdir(parents=True, exist_ok=True)
        filt_lbl_dir.mkdir(parents=True, exist_ok=True)
        no_ann_img_dir.mkdir(parents=True, exist_ok=True)
        no_ann_lbl_dir.mkdir(parents=True, exist_ok=True)

        for src_image_path in tqdm(img_list, desc=f"Processing partition segment '{split}'"):
            raw_file_name = src_image_path.name
            base_name = src_image_path.stem
            norm_base_name = base_name.replace("levele", "levle")
            img_ext = src_image_path.suffix

            json_meta = filename_to_json_meta.get(raw_file_name)
            img_annotations = []
            width, height = 0.0, 0.0
            
            if json_meta:
                img_id = json_meta["id"]
                width = float(json_meta["width"])
                height = float(json_meta["height"])
                img_annotations = ann_map.get(img_id, [])

            if json_meta and img_annotations:
                target_image_path = filt_img_dir / f"{norm_base_name}{img_ext}"
                target_label_path = filt_lbl_dir / f"{norm_base_name}.txt"

                shutil.copy2(src_image_path, target_image_path)

                yolo_lines = []
                for ann in img_annotations:
                    class_name = ann.get("class_name")
                    yolo_class_id = class_to_id.get(class_name, 0)

                    if "coordinates" in ann and "radius" in ann:
                        cx, cy = ann["coordinates"][0], ann["coordinates"][1]
                        radius = float(ann["radius"])

                        x_center = cx / width
                        y_center = cy / height
                        w_norm = (radius * 2) / width
                        h_norm = (radius * 2) / height

                        x_center = min(max(x_center, 0.0), 1.0)
                        y_center = min(max(y_center, 0.0), 1.0)
                        w_norm = min(max(w_norm, 0.0), 1.0)
                        h_norm = min(max(h_norm, 0.0), 1.0)

                        yolo_lines.append(
                            f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
                        )

                with open(target_label_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(yolo_lines))
                metrics["moved_to_filtered"] += 1
            else:
                target_image_path = no_ann_img_dir / f"{norm_base_name}{img_ext}"
                target_label_path = no_ann_lbl_dir / f"{norm_base_name}.txt"

                shutil.copy2(src_image_path, target_image_path)
                target_label_path.write_text("", encoding="utf-8")
                metrics["moved_to_no_annotations"] += 1

    yaml_lines = [
        f"path: {FILTERED_DATASET_DIR.resolve()}",
        "train: train/images",
        "val: valid/images",
        "test: test/images\n",
        "names:"
    ]
    for idx, cls in class_mapping_yaml.items():
        yaml_lines.append(f"  {idx}: {cls}")

    with open(FILTERED_DATASET_DIR / "data.yaml", "w", encoding="utf-8") as y_f:
        y_f.write("\n".join(yaml_lines) + "\n")

    filtered_size = get_dir_size_mb(FILTERED_DATASET_DIR)
    unannotated_size = get_dir_size_mb(NO_ANNOTATIONS_DIR)

    print("\nPIPELINE OUTPUT METRICS & ANALYSIS")
    print(f"Total Physical Images Processed: {total_disk_count}")
    print(f"Moved to Filtered Dataset:        {metrics['moved_to_filtered']} images")
    print(f" -> Storage Footprint:            {filtered_size:.2f} MB")
    print(f"Moved to No Annotations Folder:   {metrics['moved_to_no_annotations']} images")
    print(f" -> Storage Footprint:            {unannotated_size:.2f} MB")
    print("\nDETAILED SUBSTRUCTURE VERIFICATION")
    for s in ["train", "valid", "test"]:
        f_img, f_lbl = count_files_in_split(FILTERED_DATASET_DIR, s)
        u_img, u_lbl = count_files_in_split(NO_ANNOTATIONS_DIR, s)
        print(f"Partition [{s.upper()}]:")
        print(f"  -> Filtered Dataset (Annotated):  {f_img} Images | {f_lbl} Label Files")
        print(f"  -> No Annotations (To Label):     {u_img} Images | {u_lbl} Label Files")


def main():
    """Main execution orchestrator."""
    print("ACNE DATASET PREPARATION PIPELINE")
    
    try:
        shared_json_data = load_json_data()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    print("\n[RUNNING STEP] -> print_json_stats")
    print_json_stats(shared_json_data)
    
    print("\n[RUNNING STEP] -> process_and_split_dataset")
    process_and_split_dataset(shared_json_data)

    print("\n[SUCCESS] Entire data management pipeline completed successfully.")


if __name__ == "__main__":
    main()