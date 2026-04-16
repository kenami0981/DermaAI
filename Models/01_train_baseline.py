from ultralytics import YOLO
from pathlib import Path
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def setup_paths():

    current_path = Path.cwd()
    
    if current_path.name == "Models":
        base_dir = current_path
    else:
        base_dir = current_path / "Models"
    
    if not base_dir.exists():
        base_dir = Path(__file__).resolve().parent
        if base_dir.name != "Models":
            base_dir = base_dir.parent / "Models"

    os.chdir(base_dir)
    
    data_yaml = base_dir / "data" / "data.yaml"
    runs_dir = base_dir / "runs"
    model_path = base_dir / "yolov8n.pt"
    
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing data file: {data_yaml}")
        
    return data_yaml, runs_dir, model_path

def train_baseline(data_path, save_dir, model_path):

    model = YOLO(str(model_path))
    
    model.train(
        data=str(data_path),
        epochs=1,
        imgsz=320,
        batch=4,
        workers=0,
        project=str(save_dir),
        name="acne_baseline_test",
        exist_ok=True,
        plots=True,
        device='cpu'
    )

def main():
    try:
        data_path, runs_path, model_path = setup_paths()
        
        train_baseline(data_path, runs_path, model_path)
        
        print(f"\nSuccess! All files are in: {Path.cwd()}")
    except Exception as e:
        print(f"\nTraining failed: {e}")

if __name__ == "__main__":
    main()