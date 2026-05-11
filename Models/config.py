from pathlib import Path
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# CORE PATHS
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "dataset_final"
WEIGHTS_DIR = ROOT / "weights"
RUNS_DIR = ROOT / "runs"

# VERSION CONTROL
CURRENT_VERSION = "v2"

# DYNAMIC CONFIGURATION 
TRAINING_RUN_NAME = f"acne_train_production_{CURRENT_VERSION}"
BEST_MODEL = Path(WEIGHTS_DIR / f"best_{CURRENT_VERSION}.pt")
HPARAM_SEARCH_RESULT = Path(RUNS_DIR / f"acne_hparam_study_{CURRENT_VERSION}" / f"acne_hparam_search_{CURRENT_VERSION}_best.yaml")

# CONSTANTS 
DATA_YAML = DATA_DIR / "data.yaml"
IMAGES_DIR = ROOT / "images" / "raw"
BASE_MODEL = WEIGHTS_DIR /"yolov8s.pt"
PRETRAINED_WEIGHTS = Path(WEIGHTS_DIR / BASE_MODEL)

IMG_SIZE = 640
CONF_THRESHOLD = 0.15

DATASETS = [
    DATA_DIR / "acne yolo.v13-original-dataset.yolov8",
    DATA_DIR / "acne-detection-yolo.v1i.yolov8",
    DATA_DIR / "acne.v3i.yolov8",
    DATA_DIR / "Acne.v21i.yolov8",
    DATA_DIR / "cubeai-acne-detection-for-yolov8",
    DATA_DIR / "yolov8-acne-detection.v4i.yolov8",
]