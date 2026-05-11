# ACNE DETECTION - MODEL REGISTRY


Central registry for tracking all trained model versions, datasets, preprocessing pipelines, and benchmark performance.

**GOALS:**
* Experiment reproducibility
* Version traceability
* Deployment consistency
* Objective comparison

> **IMPORTANT** 
> Every production-grade run must be logged here

---
## TABLE OF CONTENTS

* [1. PERFORMANCE METRICS](#1-performance-metrics)
* [2. CURRENT PRODUCTION MODEL](#2-current-production-model)
* [3. EXPERIMENT HISTORY](#3-experiment-history)
    * [Model Version v1.0](#model-version-v10)
    * [Model Version v2.0](#model-version-v20)
* [4. VERSIONING POLICY](#4-versioning-policy)
* [5. FUTURE ROADMAP](#5-future-roadmap)
* [6. ROUGH IDEAS](#6-rough-ideas)

---
# 1. PERFORMANCE METRICS


| Version | Set | Epoch | Precision | Recall | mAP@50 | mAP@50-95 | Inference (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **v1** | Training | 149 | 0.59346 | 0.45272 | 0.44992 | 0.18746 | - |
| **v1** | Validation | 149 | 0.48673 | 0.38801 | 0.37981 | 0.13265 | - |
| **v2** | Training | 85 | 0.47108 | 0.45831 | 0.39397 | 0.13288 | - |
| **v2** | Validation | 85 | 0.397 | 0.312 | 0.280 | 0.095 | - |

### 1.1 EVALUATION METRICS GUIDE

**PRECISION (Correctness)**
Measures the ratio of correctly identified lesions to the total number of detections. 
High precision = low "false alarm" rate.

**RECALL (Completeness)**
Measures the ratio of correctly identified lesions to the total number of actual lesions. 
High recall = minimal missed lesions.

**mAP@50 (General Quality)**
Mean Average Precision at 0.50 IoU. The primary indicator of the model's balance between finding objects and being correct.

**mAP@50-95 (Geometric Precision)**
Average precision across multiple IoU thresholds. Reflects how accurately bounding boxes fit the actual boundaries.

---
# 2. CURRENT PRODUCTION MODEL

Since we haven't launched the production version yet, we are using our most successful experiment as the current standard: 

* **VERSION:** v1.0
* **RUN ID:** `acne_train_production_v1`
* **STATUS:** Not deployed 
* **ARCHITECTURE:** YOLOv8n (640x640)
* **CLASSES:** 0 (`acne-lesion`)
* **WHY THIS ONE?** Even though v2 is newer, v1 is more reliable and generalizes new images better

---
# 3. EXPERIMENT HISTORY


## 3.1 MODEL VERSION V1.0

### **3.1.1 METADATA:**
* **Date:** 2026-05-04
* **HPO Study:** `acne_hparam_search_v1` (12 Trials)
* **Best Trial:** `acne_hparam_search_v1_trial_3` (49 training epoch)
* **Hardware:** Google Colab GPU T4 (`yolo_train_colab.ipynb`)
* **Datasets Used:** 
    * https://universe.roboflow.com/kritsakorn/acne-kbm0q/dataset/21
    * https://universe.roboflow.com/dermafind/acne-zqozl/dataset/3
    * https://universe.roboflow.com/ance-yolo/acne-yolo/dataset/1
    * https://www.kaggle.com/datasets/cubeai/acne-detection-for-yolov8
* **Preprocessing:** Raw data, no preprocessing applied 
* **Data Integrity:** Image duplication check not performed (Scenario 0), class mapping to single-class format (class 0)


### **3.1.2 BASE PARAMETERS**
```yaml
epochs: 150
patience: 30
imgsz: 640
batch: -1
workers: 4
device: GPU
cos_lr: true
fliplr: 0.5
save_period: 1
```

### **3.1.3 OPTIMIZED HYPERPARAMETERS**
```yaml
optimizer: AdamW
lr0: 0.000938
weight_decay: 2.409e-05
mosaic: 0.525313
mixup: 0.036816
hsv_s: 0.269225
hsv_v: 0.360626
scale: 0.203803
```


### **3.1.4 STRENGTHS:**
* **Good Learning:** The model learned steadily without crashing (Source: **Loss Curves**).
* **High Precision:** When the model finds a spot, it is usually right (Source: **Validation Metrics**).
* **Final Boost:** The model improved significantly in the last few epochs (Source: **Loss Curves**).

### **3.1.5 WEAKNESSES:**
* **Knowledge Gap:** The model performs much better on training data than on tests (Source: **Generalization Gap**).
* **Duplicate Data:** Unremoved duplicate images in the dataset confused the model and hurt its learning (Source: **Dataset Metadata**).
* **Low Recall:** The model is too careful and misses many actual acne spots (Source: **Validation Metrics**).
* **Inaccurate Boxes:** The model finds the spot but doesn't draw the box perfectly around it (Source: **Validation Metrics**).

---

## 3.2 MODEL VERSION V2.0

### **3.2.1 METADATA:**
* **Date:** 9.05.26
* **HPO Study:** `acne_hparam_search_v2` (14 Trials)
* **Best Trial:** `acne_hparam_search_v2_trial_0` (94 training epoch - best epoch 85)
* **Hardware:** Google Colab GPU T4 (`yolo_train_colab.ipynb`)
* **Datasets Used:** 
    * https://universe.roboflow.com/ance-yolo/acne-yolo/dataset/13
    * https://universe.roboflow.com/acne-severity/acne-detection-yolo/dataset/1
    * https://universe.roboflow.com/dermafind/acne-zqozl/dataset/3
    * https://universe.roboflow.com/kritsakorn/acne-kbm0q/dataset/21
    * https://universe.roboflow.com/osman-kagan-kurnaz/skin-detection-uvj1f/dataset/8
    * https://universe.roboflow.com/dermatologiaestoril/yolov8-acne-detection/dataset/4
* **Preprocessing:** enhancement using CLAHE for local contrast and Unsharp Masking for lesion sharpening
* **Data Integrity:** visual deduplication using pHash (threshold=2) all training images are kept and duplicates are removed from val and test sets (Scenario B), class mapping to single-class format (class 0)


### **3.2.2 BASE PARAMETERS**
```yaml
epochs: 150
patience: 30
imgsz: 640
batch: -1
workers: 4
device: GPU
cos_lr: true
fliplr: 0.5
save_period: 10
```

### **3.2.3 OPTIMIZED HYPERPARAMETERS**
```yaml
degrees: 1.465218
hsv_s: 0.078545
hsv_v: 0.240186
lr0: 0.0019593
mixup: 0.027834
mosaic: 0.532219
scale: 0.511213
weight_decay: 0.002331
```


### **3.2.4 STRENGTHS:**

* **Rapid Initial Learning:** The model reaches a high level of performance very quickly, with metrics stabilizing as early as epoch 40 (Source: **Validation Metrics**).
* **Effective Learning Rate Management:** The implementation of a warm-up phase followed by a cosine decay effectively stabilized the training process (Source: **Learning Rate Curves**).
* **Consistent Recall:** Despite lower overall precision, the model maintains a recall level comparable to v1, meaning it is still capable of identifying a similar percentage of actual lesions (Source: **Performance Table**).

### **3.2.5 WEAKNESSES:**

* **Significant Overfitting:** There is a massive and growing difference between training and validation loss. The model is "memorizing" the training data rather than learning general features (Source: **Overfitting Indicator**).
* **Critical Generalization Gap:** The box loss on the validation set starts increasing after epoch 20, while the training loss continues to drop. This indicates the model loses its ability to accurately locate lesions on new images over time (Source: **Loss Analysis**).
* **Performance Regression:** Despite data preprocessing (CLAHE, Unsharp Masking) and deduplication, this version performs significantly worse than v1 across all key metrics (mAP, Precision, Recall) (Source: **Performance Table**).
* **Low Geometric Accuracy:** A very low mAP@50-95 (0.095 on Val) suggests that the predicted bounding boxes are poorly aligned with the ground truth (Source: **Validation Metrics**).

---


# 4. VERSIONING POLICY

**PATCH (v1.0.X)** -> Script/metadata fixes. No retraining.

**MINOR (v1.X)** -> New HParams, dataset expansion, preprocessing. Same architecture.

**MAJOR (vX.0)** -> Architecture change (e.g., v8n -> v8s, v8 -> v11), new classes etc.

---

# 5. FUTURE ROADMAP

* **v2.0:** Migration to YOLOv8s architecture & implementation of data enhancement (done)
* **v1.1** ???

---
# 6. ROUGH IDEAS

* **Transfer Learning** - We could start from weights pre-trained on medical datasets (like ISIC) rather than generic COCO objects to better recognize skin textures.
* **Early Stopping** - We should implement a better trigger (more like 30-50) to kill training the moment Validation mAP stops growing to prevent the massive overfitting seen in v2.
* **Image Tiling** - We could try "Slicing Aided Hyper Inference" (SAHI) or training on small crops to help the model find tiny lesions that get lost in 640x640.
* **Stain Normalization** - We could normalize the skin tones across different datasets to make the model more robust to lighting and camera changes.
* **Hair & Noise Removal** - We could use specialized filters to remove hair or background clutter so the model focuses strictly on the acne.
* **Architecture Scaling** - We could test YOLOv26 to see if a larger "brain" captures complex lesion features more accurately. It is better suited for small objects.
* **Loss Function Tweaking** - We could try different loss functions (like Focal Loss) to help the model focus on hard examples it keeps missing.
* **Pseudo-labeling** - We could use our best model to label raw, unlabeled skin images and then retrain on this larger dataset.
* **Metadata Integration** - We could try feeding patient age or skin type into the model as extra info to see if it improves diagnostic accuracy.
* **Specialized Testing** - We should create separate test sets for "Severe" vs "Mild" cases to pinpoint exactly where the model fails.
* **Color Augmentation** - We could push the HSV (hue, saturation, value) limits even further to simulate different skin conditions and lighting environments.
* **Ensemble Scoring** - We could combine the predictions of v1 and v2 to see if their combined opinion is better than single model.