from ultralytics import YOLO
from pathlib import Path
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def setup_validation():

    current_path = Path.cwd()
    
    if current_path.name == "Models":
        base_dir = current_path
    else:
        base_dir = current_path / "Models"
        
    if not base_dir.exists():
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    os.chdir(base_dir)
    
    model_path = base_dir / "runs" / "acne_baseline_test" / "weights" / "best.pt"
    data_yaml = base_dir / "data" / "data.yaml"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found. Run 01_train_baseline.py first.")
        
    return model_path, data_yaml

def main():
    try:
        model_path, data_path = setup_validation()
        model = YOLO(str(model_path))
        
        print(f"Starting validation for: {model_path.name}...")
        metrics = model.val(
            data=str(data_path),
            imgsz=320,
            split='val',
            device='cpu'
        )
        
        print("\n" + "="*30)
        print(f"Precision: {metrics.results_dict['metrics/precision(B)']:.3f}")
        print(f"Recall: {metrics.results_dict['metrics/recall(B)']:.3f}")
        print(f"mAP50: {metrics.results_dict['metrics/mAP50(B)']:.3f}")
        print(f"mAP50-95: {metrics.results_dict['metrics/mAP50-95(B)']:.3f}")
        print("="*30)

    except Exception as e:
        print(f"Validation Error: {e}")

if __name__ == "__main__":
    main()



# PRECISION (0.543): "How many detections were correct?"
# Range: 0.0 (all wrong) to 1.0 (perfectly clean detections).
# Status: When the model finds a spot, there is a 54% chance it is right.
# Half of detections are "False Alarms" (False Positives).

# RECALL (0.113): "How many actual objects were found?"
# Range: 0.0 (found nothing) to 1.0 (found every single object).
# Status: The model only detects ~11% of the acne in your data.
# 89% of skin changes are ignored (False Negatives).

# mAP50 (0.030): "Overall accuracy at 50% overlap (IoU)."
# Range: 0.0 to 1.0. This is the primary benchmark for YOLO.
# Status: A score of 0.03 means the model is barely functioning. 
# For medical tasks, we should aim for > 0.60.

# mAP50-95 (0.008): "Average accuracy across strict overlap thresholds."
# Range: 0.0 to 1.0. Measures how perfectly the boxes fit the objects.
# Status: At 0.008, the model has no "spatial precision" yet.