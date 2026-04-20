from pathlib import Path
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Root folder = Models/
ROOT = Path(__file__).resolve().parent

# Data
DATA_DIR = ROOT / "data"
DATA_YAML = DATA_DIR / "data.yaml"

# Models / weights
MODELS_DIR = ROOT
YOLO_WEIGHTS = ROOT / "models" / "yolov8n.pt"
BEST_MODEL = ROOT / "runs" / "train" / "weights" / "best.pt"

# Runs output
RUNS_DIR = ROOT / "runs"

# Images
IMAGES_DIR = ROOT / "images"

# Class weights assigned based on clinical acne severity scales.
# Sources:
# GAGS (Global Acne Grading System): https://pubmed.ncbi.nlm.nih.gov/3831839/
# DermNet (Acne classification): https://dermnetnz.org/topics/acne-vulgaris

CLASS_WEIGHTS = {
    "blackheads": 0.15, # Zaskórniki otwarte (niezapalne)
    "whiteheads": 0.15, # Zaskórniki zamknięte (niezapalne)
    "papules": 0.50,  # Grudki (stan zapalny)
    "pustules": 0.75, # Krosty (ropne, zaawansowany stan zapalny)
    "nodules": 1.00,  # Guzki (najcięższe, ryzyko bliznowacenia)
    "dark spot": 0.05 # Przebarwienia (zmiany nieaktywne / pozapalne)
}