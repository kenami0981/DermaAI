# DermaAI – Acne Detection

## Table of Contents
* [1. Project Overview](#1-dermaai--acne-detection)
    * [Dataset](#dataset)
    * [Setup For YOLO](#setup-for-yolo)
    * [Setup For TensorFlow CNN](#setup-for-tensorflow-cnn)
* [2. Conda Environment Setup](#2-conda-environment-setup)
    * [Create environment](#create-environment)
    * [Activate environment](#activate-environment)
    * [Update environment](#update-environment-optional)
    * [Remove environment](#remove-environment-optional)
* [3. VS Code Setup](#3-vs-code-setup-interpreter-selection)
    * [Select Python interpreter](#select-python-interpreter)
    * [Important notes](#important-notes)
* [4. YOLOv8 Windows + Conda + VS Code – Troubleshooting](#4-yolov8-windows--conda--vs-code--troubleshooting)
    * [1. Wrong Python Version](#1-wrong-python-version-eg-313-instead-of-310)
    * [2. ModuleNotFoundError](#2-modulenotfounderror-ultralytics)
    * [3. Training Doesn't Start](#3-training-doesnt-start-no-epoch-1)
    * [4. OMP Error #15](#4-omp-error-15-crash)
* [5. Google Colab](#5-google-colab)

---

# 1. DermaAI – Acne Detection

This part focuses on detecting acne from facial images using:
- YOLO 
- CNN 

TBD

---

## Dataset

The dataset is NOT stored in the repository.

### Download dataset:
[https://universe.roboflow.com/kritsakorn/acne-kbm0q/dataset/21](https://universe.roboflow.com/kritsakorn/acne-kbm0q/dataset/21)

---

## Setup For YOLO

1. Open the link above  
2. Select `YOLOv8` format  
3. Download dataset (ZIP file)  
4. Extract the ZIP on your computer  
5. Place the dataset into `Models/data/`

---

## Setup For TensorFlow CNN

TBD

---

# 2. Conda Environment Setup

You can use Conda to manage dependencies for the ML part of the project.

---

## Create environment

Make sure you are where `Models/` is located, then in the terminal:

```bash
conda env create -f environment.yml
```
---
## Activate environment
```bash
conda activate acne-detection
```
---
## Update environment (optional)

If `environment.yml` changes:

```bash
conda env update -f environment.yml --prune
```
---
## Remove environment (optional)
```bash
conda remove --name derma-models --all
```
---

# 3. VS Code Setup (Interpreter Selection)

To ensure the project runs correctly, VS Code must use the Conda environment created for this project.

---

## Select Python interpreter
1. Press `Ctrl+Shift+P`.
2. Type `Python: Select Interpreter`.
3. Choose the one labeled `acne-detection`.

---

## Important notes
1. **RUNNING**: Always run scripts from the Terminal.
2. **INTERPRETER**: Ensure the correct Conda/Env interpreter is active in VS Code. If imports fail, your terminal is likely using the wrong Python version (you need to type `conda activate acne-detection` in terminal). 
3. **STRUCTURE**: This script MUST include a `if __name__ == "__main__": main()` block to handle multiprocessing and path initialization correctly.
4. **GIT/REPO**: 
    - DO NOT upload datasets to the repository.
    - DO NOT upload trained models (.pt, .h5) to GitHub (they are too large) we will solve it later.

---

# 4. YOLOv8 Windows + Conda + VS Code – Troubleshooting

---

### 1. Wrong Python Version (eg. 3.13 instead of 3.10)
**Symptom:** you tried `conda activate` and it runs, but `python --version` shows system Python.
**Fix (PowerShell):**
```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
conda init powershell
# Restart VS Code
```

---

### 2. ModuleNotFoundError (ultralytics)
**Symptom:** `import YOLO` fails.
**Fix (In active env):**
```bash
python -m pip install ultralytics
python -c "from ultralytics import YOLO; print('OK')"
```

---

### 3. Training Doesn't Start (No Epoch 1)
**Symptom:** Logs stop at `Plotting labels...`.
**Cause:** Wrong execution method or missing `if __name__ == "__main__":`.
**Fix:** Run manually in terminal: `python my_script.py` NOT with vs code "Run Python File".

---

### 4. OMP Error #15 (Crash)
**Symptom:** Immediate crash after starting training.
**Fix:** Add this to the very top of your `.py` file:
```python
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```

---

# 5. Google Colab 

If local hardware is too weak for the dataset size, training can be done in Google Colab using high-performance T4 GPUs.


All intructions, training scripts, environment setup, and evaluation tools are located in the following notebook:
> `Models/yolo_training_colab.ipynb`