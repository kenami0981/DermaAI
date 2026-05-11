"""
===========================================================
WARNING: DESTRUCTIVE OPERATION
===========================================================

Run this script to build dataset_final from DATASETS listed in config.py.
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
- Checks visual deduplicats via pHash

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


import shutil
import random
import sys
import yaml
import imagehash
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# Path setup to ensure local imports work
ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from config import DATA_DIR, DATASETS


# Output directory (WILL BE DELETED EACH RUN)
OUTPUT_DIR = DATA_DIR

SPLITS = ["train", "valid", "test"]
PHASH_THRESHOLD = 2 



# VISUAL ANALYSIS FUNCTIONS

def get_visual_hash(path: Path):
    try:
        with Image.open(path) as img:
            return imagehash.phash(img)
    except Exception:
        return None

def analyze_visual_duplicates():
    """
    Builds a registry of visually similar images and simulates scenarios

    """
    registry = {}
    all_tasks = []

    for ds_path in DATASETS:
        for split in SPLITS:
            img_dir = ds_path / split / "images"
            if not img_dir.exists(): continue
            for img_p in img_dir.glob("*.*"):
                if img_p.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    all_tasks.append({'path': img_p, 'ds': ds_path.name, 'split': split})

    print(f"\nANALYZING {len(all_tasks)} IMAGES FOR VISUAL DUPLICATES (pHash)...")
    for item in tqdm(all_tasks, desc="Hashing images"):
        v_hash = get_visual_hash(item['path'])
        if v_hash is None: continue

        found_key = None
        for stored_hash in registry.keys():
            if v_hash - stored_hash <= PHASH_THRESHOLD:
                found_key = stored_hash
                break
        
        if found_key:
            registry[found_key].append(item)
        else:
            registry[v_hash] = [item]

    # Stats Simulation
    stats = {'initial': {s: 0 for s in SPLITS}, 'A': {s: 0 for s in SPLITS}, 'B': {s: 0 for s in SPLITS}}
    for entries in registry.values():
        for e in entries: stats['initial'][e['split']] += 1
        s_list = [e['split'] for e in entries]
        
        # Scenario A
        if 'train' in s_list: stats['A']['train'] += 1
        elif 'valid' in s_list: stats['A']['valid'] += 1
        else: stats['A']['test'] += 1

        # Scenario B
        has_train = 'train' in s_list
        stats['B']['train'] += s_list.count('train')
        if not has_train:
            stats['B']['valid'] += s_list.count('valid')
            stats['B']['test'] += s_list.count('test')

    print("\nVISUAL ANALYSIS STATISTICS:")
    print(f"{'Split':<10} | {'Current':<10} | {'Scenario A':<10} | {'Scenario B':<10}")
    print("-" * 50)
    for s in SPLITS:
        print(f"{s:<10} | {stats['initial'][s]:<10} | {stats['A'][s]:<10} | {stats['B'][s]:<10}")
    
    return registry

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
        print("No valid YOLO structure found")
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

def merge_datasets(registry, scenario):
    """
    Merge all listed datasets into a unified YOLO dataset based on chosen scenario.
    - scenario '0': Original (Keep everything)
    - scenario 'A': Unique only (Strict deduplication)
    - scenario 'B': Leak-free (Keep all Train, remove leaks from Val/Test)
    """

    total_images = 0

    for h, entries in registry.items():
        to_process = []
        if scenario == '0':
            to_process = entries
        elif scenario == 'A':
            # Priority: train > valid > test
            train_e = [e for e in entries if e['split'] == 'train']
            valid_e = [e for e in entries if e['split'] == 'valid']
            test_e = [e for e in entries if e['split'] == 'test']
            if train_e: to_process = [train_e[0]]
            elif valid_e: to_process = [valid_e[0]]
            else: to_process = [test_e[0]]
        elif scenario == 'B':
            train_e = [e for e in entries if e['split'] == 'train']
            if train_e: to_process = train_e
            else: to_process = entries

        for item in to_process:
            split = item['split']
            out_img_dir = OUTPUT_DIR / split / "images"
            out_lbl_dir = OUTPUT_DIR / split / "labels"
            out_img_dir.mkdir(parents=True, exist_ok=True)
            out_lbl_dir.mkdir(parents=True, exist_ok=True)

            dataset_name = item['ds'].replace(".", "_")
            new_name = f"{dataset_name}_{item['path'].name}"
            new_img_path = safe_copy(item['path'], out_img_dir / new_name)

            # Process label
            old_label_path = item['path'].parent.parent / "labels" / f"{item['path'].stem}.txt"
            converted_content = convert_labels_to_single_class(old_label_path)
            
            with open(out_lbl_dir / (new_img_path.stem + ".txt"), "w") as f:
                f.write(converted_content)
            total_images += 1

    print(f"\nDONE: {total_images} samples merged")

def create_yaml():
    """
    Generate YOLO data.yaml configuration file for training.
    """

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

    # 2. RUN VISUAL ANALYSIS
    visual_registry = analyze_visual_duplicates()

    # 3. USER CHOICE
    print("\nCHOOSE MERGE STRATEGY:")
    print(" [0] Standard Merge (Keep all duplicates)")
    print(" [A] Scenario A (Strict uniqueness, priority TRAIN)")
    print(" [B] Scenario B (Keep all TRAIN, remove leaks from VAL/TEST)")
    
    choice = input("\nSelect strategy (0/A/B) or 'Q' to quit: ").strip().upper()
    if choice == 'Q' or choice not in ['0', 'A', 'B']:
        print("Operation cancelled.")
        sys.exit(0)

    # 4. CLEANUP
    if OUTPUT_DIR.exists():
        print(f"\nRemoving existing dataset: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    # 5. EXECUTE MERGE
    print(f"Starting merge process (Scenario {choice})...")
    merge_datasets(visual_registry, choice)
    create_yaml()
    
    print("\nPROCESSED COMPLETED SUCCESSFULLY.")