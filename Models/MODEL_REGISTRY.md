# ACNE DETECTION - MODEL REGISTRY


## Central registry for tracking all trained model versions, datasets, preprocessing pipelines, and benchmark performance.

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
    * [3.1 Model Version v1.0 (YOLOv8n)](#31-model-version-v10-yolov8n)
    * [3.2 Model Version v2.0 (YOLOv8s)](#32-model-version-v20-yolov8s)
    * [3.3 Model Versions v3.0 & v3.1 (YOLO26s)](#33-model-versions-v30--v31-yolo26s)
    * [3.4 Model Version v3.2 (YOLO26s)](#34-model-version-v32-yolo26s)
* [4. VERSIONING POLICY](#4-versioning-policy)
* [5. FUTURE ROADMAP](#5-future-roadmap)
* [6. ROUGH IDEAS](#6-rough-ideas)

---

# 1. PERFORMANCE METRICS


| Version | Set | Epoch | Precision | Recall | mAP@50 | mAP@50-95 | Inference (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **v1** | Validation | 149 | 0.593 | 0.453 | 0.450 | 0.187 | - |
| **v2** | Validation | 85 | 0.473 | 0.450 | 0.385 | 0.129 | - |
| **v3.2** | Validation | 39 | 0.500 | 0.464 | 0.444 | 0.172 | - |

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

* **VERSION:** v3.2
* **RUN ID:** `acne_train_production_v3.2`
* **STATUS:** -
* **ARCHITECTURE:** YOLO26s (1280x1280)
* **CLASSES:** 0 (`acne-lesion`)
* **WHY THIS ONE?** Unlike previous versions, v3.2 completely resolves the generalization gap using verified, clean dermatological annotations from the Acne04-v2 dataset. By scaling the resolution to 1280 px and utilizing the stable MuSGD optimizer, it ensures reliable real-world performance on new images without memorizing noise.
---
# 3. EXPERIMENT HISTORY


## 3.1 MODEL VERSION V1.0 (YOLOv8n)

### **3.1.1 METADATA:**

* **Base Model:** YOLOv8n
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

## 3.2 MODEL VERSION V2.0 (YOLOv8s)

### **3.2.1 METADATA:**

* **Base Model:** YOLOv8s
* **Date:** 2026-05-09
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

## 3.3 MODEL VERSIONS V3.0 & V3.1 (YOLO26s)

### **3.3.1 MODEL VERSION V3.0 (REJECTED) METADATA & PARAMETERS:**

* **Base Model:** YOLO26s | **Date:** 2026-05-31
* **Hardware:** Google Colab GPU T4
* **Dataset:** (~10,000 images)
* **Status:** **REJECTED**
* **Preliminary Results:** Extremely slow. After the full 40 epochs, the model achieved an unsatisfactory **mAP50 = 0.25428** and **mAP50-95 = 0.07767**.
* **Reason for Termination:** **Critical performance bottleneck.** The massive scale of the dataset (10k images) combined with the heavy YOLO26s architecture and optimizer overhead drastically inflated the epoch duration (over 300s). Training a single trial took over 3.5 hours, making the planned HPO (12 trials) impossible to complete within Colab's runtime limits.

### **3.3.2 MODEL VERSION V3.1 (REJECTED) METADATA & PARAMETERS:**

* **Base Model:** YOLO26s | **Date:** 2026-06-02
* **Dataset:** Downscaled subset from V3.0 (aimed at accelerating iteration speed).
* **Status:** **REJECTED**
* **Preliminary Results:** Metric progression completely plateaued. By epoch 35, the model reached a mere **mAP50 = 0.20271** and **mAP50-95 = 0.05341** and stopped because of early stopping (patience=8)
* **Reason for Termination:** **Data Integrity & Labeling Issues.** While reducing the data volume successfully cut down epoch times (90–100s / epoch), the metrics refused to scale. A data audit revealed that the unified dataset contained shifted, inaccurate, or flat-out corrupted bounding boxes. The model was learning pure noise, forcing an immediate halt and a total overhaul of the data pipeline.

---

## 3.4 MODEL VERSION V3.2 (YOLO26s)

### **3.4.1 PHASE 1: RESEARCH & HPO MID-STUDY INTERVENTION**

* **Base Model:** YOLO26s | **Date:** 2026-06-05
* **Annotations (`JSON_PATH`):** [AIpourlapeau GitHub](https://github.com/AIpourlapeau/acne04v2/blob/main/Acne04-v2_annotations.json)
* **Source Images (`SRC_IMAGES_DIR`):** [Kaggle Dataset](https://www.kaggle.com/datasets/karmagames/acne04-v2)

#### **Trials 0-9:**

* **AdamW Volatility:** Trial 6 reached the highest mAP50 ($0.43984$), but the training curves fluctuated heavily near the end. This final peak score was likely an unstable, lucky epoch.
* **MuSGD Stability:** Trial 0 achieved a strong $0.42697$ mAP50 with highly consistent growth and a stable plateau at the end. Due to this reliability, MuSGD is selected as the definitive optimizer for YOLO26s.

#### **Hot-Fix & Search Space Refinement:**

To eliminate unstable AdamW runs and optimize MuSGD performance, the search space was refined after Trial 9:

* **`optimizer`:** Fixed to `"MuSGD"`.
* **`lr0`:** Raised to `[1e-3, 4e-3]` (log scale) - MuSGD requires a higher learning rate to converge within 50 epochs (Trial 8 failed because 0.00059 was too low).
* **`weight_decay`:** Updated to `weight_decay` `[1e-3, 6e-3]` to manage the higher learning rate.
* **Augmentations:** Moderated (`mosaic`: `[0.3, 0.55]`, `mixup`: `[0.0, 0.08]`) to ensure faster convergence under the NMS-free `end2end: True` head.

#### **Key Performance Comparison (V3.2 vs. Predecessors):**

| Model Version | Status | Epochs | Training Time | Best mAP50 | Best mAP50-95 | Primary Cause / Outcome |
| --- | --- | --- | --- | --- | --- | --- |
| **V3.0** | REJECTED | 40 / 40 (Trial 1) | ~3.5h (Too slow) | 0.25428 | 0.07767 | Severe CPU/GPU bottleneck on 10k images. |
| **V3.1** | REJECTED | 35 / 40 (Trial 1) | ~1.0h (Normal) | 0.20271 | 0.05341 | Corrupted dataset annotation quality. |
| **V3.2** | **SUCCESS** | **50 / 50** (Trial 1) | **~1.6h** | **0.43061** | **0.16430** | **Clean data (Filtered), excellent convergence.** |

---

### **3.4.2 PHASE 2: PRODUCTION RUN METADATA**

* **Run ID:** `acne_train_production_v3.2`
* **Status:** SUCCESSFUL
* **Date:** 2026-06-06
* **Hardware:** Kaggle GPU
* **Dataset:** Acne04-v2 filtered dataset (~1000 high-resolution images)
* **Data Integrity:** Corrupted samples removed, coordinate validation performed, single-class mapping (`acne-lesion`)
* **Preprocessing:** Verified annotations extracted from official Acne04-v2 JSON labels, enhancement using CLAHE for local contrast and Unsharp Masking for lesion sharpening

#### **BASE PARAMETERS**

```yaml
epochs: 100
patience: 12
imgsz: 1280
batch: 4
workers: 4
device: GPU
cos_lr: true
save_period: 1
plots: true
close_mosaic: 4
```

#### **OPTIMIZED HYPERPARAMETERS**

```yaml
lr0: 0.0012578355505042395
weight_decay: 0.005157769895932762
mosaic: 0.20548245350470415
mixup: 0.07473415023930467
translate: 0.09528151387192053
hsv_s: 0.0012
hsv_v: 0.0187
scale: 0.10775188997103605
degrees: 1.5876795049155819
erasing: 0.25
```

#### **PERFORMANCE**

| Set | Epoch | Precision | Recall | mAP@50 | mAP@50-95 |
| --- | --- | --- | --- | --- | --- |
| **Validation** | 39 | **0.500** | **0.464** | **0.444** | **0.172** |

**Best Epoch:** 39

```text
Precision     = 0.50025
Recall        = 0.46383
mAP@50        = 0.44438
mAP@50-95     = 0.17223
```

Here is the simplified, concise version in English:

---

### **STRENGTHS**

* **No Overfitting & Stable Learning:** Loss curves decreased consistently, and the generalization gap stayed close to zero.
* **Clean Dataset:** Transitioning to the verified **Acne04-v2** dataset eliminated the broken and shifted bounding boxes that blocked V3.1.
* **1280 px Resolution:** Moving to a higher resolution successfully allowed the model to capture the micro-details of skin lesions.
* **Real-World Localization Progress:** Bounding boxes are placed much more accurately on acne during actual performance, proving its real-world viability.

### **WEAKNESSES**

* **Low Recall (0.464):** The model still misses more than half of the actual skin lesions.
* **Low mAP@50-95 (0.172):** Bounding box precision under stricter thresholds is still insufficient for use.
* **Data Volume Constraint:** ~1000 images are not enough to fully exploit the architectural capacity of **YOLO26s**.
* **Early Plateau:** The model peaked at **epoch 39**, meaning subsequent training epochs yielded no further improvement.
* **Small Object Challenge:** Acne lesions occupy a tiny percentage of the image, making them easy for the model to overlook.

---

### **CONCLUSION**

Model **V3.2** is the best and most stable version produced in the project so far. While its metrics are slightly lower than V1.0, these results are 100% honest - v1.0 was heavily overfitted to noisy data. V3.2 generalizes well on new images and serves as a solid baseline for next steps.
---

# 4. VERSIONING POLICY

**PATCH (v1.0.X)** -> Script/metadata fixes. No retraining.

**MINOR (v1.X)** -> New HParams, dataset expansion, preprocessing. Same architecture.

**MAJOR (vX.0)** -> Architecture change (e.g., v8n -> v8s, v8 -> v11), new classes etc.

---

# 5. FUTURE ROADMAP

* **v2.0:** Migration to YOLOv8s architecture & implementation of data enhancement (done)
* **v3.0** Migration to YOLO26 architecture (done)

* **v4.0:** Dataset expansion (control images, severe cases), SAHI/higher resolution benchmarks, and image stability validation (planned)

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