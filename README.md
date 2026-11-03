# Hand Gesture Recognition SOTA & Explainability (Assignment 2)

**Course:** Computer Vision, Astana IT University (AITU)  
**Gestures:** `like` (👍), `palm` (✋), `peace` (✌️)  
**Repository:** [github.com/hulahup4ikk/hand-gesture-recognition-sota](https://github.com/hulahup4ikk/hand-gesture-recognition-sota)

---

## 1. Project Overview
This repository implements the upgraded Hand Gesture Recognition pipeline for Assignment 2:
- **SOTA Vision Architecture**: Benchmarked modern **ConvNeXt-Tiny** against the lightweight **MobileNetV2** baseline from Assignment 1.
- **Dataset Scaling**: Integrated the custom Assignment 1 dataset with a curated subset of the **HaGRID (HAnd Gesture Recognition Image Dataset)** (350 balanced samples/class), expanding training data to 1,048 images.
- **Strict Benchmark Split**: Evaluated strictly on **Subject 4 (Damir, 64 images)** — completely unseen during training, ensuring zero data leakage and real-world generalization evaluation.
- **Visual Explainability (XAI)**: Implemented native PyTorch **Grad-CAM** to inspect model attention and ensure decisions are grounded in hand landmarks rather than background correlations.
- **Interactive Web Demo**: Streamlit application supporting image upload and real-time webcam inference with latency benchmarking and Grad-CAM overlays.

---

## 2. Experimental Benchmark Results

Evaluated on the exact same unseen test set (Subject 4, 64 samples):

| Metric | Baseline: MobileNetV2 | SOTA: ConvNeXt-Tiny | Delta |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | **93.75%** (60/64) | **98.44%** (63/64) | **+4.69%** |
| **Macro-F1 Score** | **0.9419** | **0.9853** | **+0.0434** |
| `like` F1-Score | 0.9091 (P=0.83, R=1.00) | **0.9756** (P=0.95, R=1.00) | +0.0665 |
| `palm` F1-Score | 1.0000 (P=1.00, R=1.00) | **1.0000** (P=1.00, R=1.00) | Parity |
| `peace` F1-Score | 0.9167 (P=1.00, R=0.85) | **0.9804** (P=1.00, R=0.96) | +0.0637 |
| **Mean Latency (CPU)** | 26.81 ms (~37.3 FPS) | 38.45 ms (~26.0 FPS) | Real-time capable |

---

## 3. Project Directory Structure

```text
├── data/
│   ├── raw/                 # Original Assignment 1 images
│   ├── public_sample/       # Filtered HaGRID 384p samples
│   ├── split/               # Train / Val / Test sets (test is 100% Subject 4)
│   └── prepare_dataset.py   # Automated data downloader & split assembler
├── src/
│   ├── __init__.py
│   ├── dataset.py           # PyTorch Dataset, transforms & loaders
│   ├── model.py             # ConvNeXt-Tiny & MobileNetV2 architecture definitions
│   ├── gradcam.py           # Native PyTorch Grad-CAM implementation
│   ├── evaluate.py          # Benchmark metrics & latency timing
│   ├── train.py             # Training loop with Cosine Annealing
│   └── generate_report_assets.py # Generates figures for the report
├── weights/                 # Model checkpoints (baseline & SOTA)
├── app/
│   ├── __init__.py
│   ├── app.py               # Interactive Streamlit web application
│   └── utils.py             # Preprocessing & inference pipeline
├── report/
│   ├── figures/             # Confusion matrices & Grad-CAM visual samples
│   └── metrics_summary.json # Raw benchmark metrics
├── requirements.txt         # Project dependencies
└── README.md
```

---

## 4. Quickstart Guide

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation
```bash
python data/prepare_dataset.py
```

### 3. Evaluate Models
```bash
# Baseline MobileNetV2
python src/evaluate.py --model mobilenet --weights weights/baseline_mobilenet.pth

# SOTA ConvNeXt-Tiny
python src/evaluate.py --model convnext --weights weights/sota_convnext.pth
```

### 4. Run Interactive Web Demo
```bash
streamlit run app/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.
