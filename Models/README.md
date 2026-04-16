# DermaAI – Acne Detection

## Table of Contents
* [1. Project Overview](#dermaai--acne-detection)
    * [Dataset](#dataset)
    * [Setup For YOLO](#setup-for-yolo)
    * [Setup For TensorFlow CNN](#setup-for-tensorflow-cnn)
* [2. Conda Environment Setup](#conda-environment-setup)
    * [Create environment](#create-environment)
    * [Activate environment](#activate-environment)
    * [Update environment](#update-environment-if-needed)
    * [Remove environment](#remove-environment-optional)
* [3. VS Code Setup](#vs-code-setup-interpreter-selection)
    * [Select Python interpreter](#select-python-interpreter-in-vs-code)
    * [Important notes](#important-notes)
* [4. Troubleshooting Guide](#yolov8-windows--conda--vs-code--troubleshooting)
    * [1. Wrong Python Version](#1-wrong-python-version-313-instead-of-310)
    * [2. VS Code Interpreter Mismatch](#2-vs-code-interpreter-mismatch)
    * [3. ModuleNotFoundError](#3-modulenotfounderror-ultralytics)
    * [4. Training Doesn't Start](#4-training-doesnt-start-no-epoch-1)
    * [5. OMP Error #15](#5-omp-error-15-crash)
    * [6. Universal Workflow](#6-summary-universal-5-minute-workflow)

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
https://universe.roboflow.com/kritsakorn/acne-kbm0q/dataset/21

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
## Update environment (if needed)

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

## Select Python interpreter in VS Code

1. Open VS Code in the project folder  
2. Press:
```
Ctrl + Shift + P
```
3. Type:
```
Python: Select Interpreter
```
4. Choose the interpreter that contains:
```
acne-detection
```
If it does not appear automatically, select:
```
Enter interpreter path...
```
and locate (for Windows):

```
C:\Users\<username>\anaconda3\envs\acne-detection\python.exe
```

## Important notes
1. **RUNNING**: Always run scripts from the Terminal.
2. **INTERPRETER**: Ensure the correct Conda/Env interpreter is active in VS Code. If imports fail, your terminal is likely using the wrong Python version.
3. **STRUCTURE**: This script MUST include a `if __name__ == "__main__": main()` block to handle multiprocessing and path initialization correctly.
4. **GIT/REPO**: 
    - DO NOT upload datasets to the repository (use .gitignore).
    - DO NOT upload trained models (.pt, .h5) to Git - they are too large.

---

# 4. YOLOv8 Windows + conda + VS Code – Troubleshooting

---

### 1. Wrong Python Version (3.13 instead of 3.10)
**Symptom:** `conda activate` runs, but `python --version` shows system Python (3.13).
**Fix (PowerShell):**
```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
conda init powershell
# Restart VS Code
```

---

### 2. VS Code Interpreter Mismatch
**Symptom:** Terminal is correct, but VS Code logic/linting uses the wrong Python.
**Fix (GUI):**
1.  `Ctrl + Shift + P` -> **Python: Select Interpreter**.
2.  Choose: `conda env: acne-detection (Python 3.10)`.

---

### 3. ModuleNotFoundError (ultralytics)
**Symptom:** `import YOLO` fails.
**Fix (In active env):**
```bash
python -m pip install ultralytics
python -c "from ultralytics import YOLO; print('OK')"
```

---

### 4. Training Doesn't Start (No Epoch 1)
**Symptom:** Logs stop at `Plotting labels...`.
**Cause:** Wrong execution method or missing `if __name__ == "__main__":`.
**Fix:** Run manually in terminal: `python my_script.py`.

---

### 5. OMP Error #15 (Crash)
**Symptom:** Immediate crash after starting training.
**Fix:** Add this to the very top of your `.py` file:
```python
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```

---

### 6. Summary: Universal 5-Minute Workflow
1.  **Activate:** `conda activate acne-detection`
2.  **Verify:** `where python` (must show your `envs` path).
3.  **Run in terminal:** `python my_script.py`

---

Oto gotowa sekcja poświęcona uruchamianiu projektu w **Google Colab**, sformatowana w Markdown:

---

# 5. Google Colab 

If local hardware is insufficient for the dataset size, training can be offloaded to Google Colab.


TBD 
