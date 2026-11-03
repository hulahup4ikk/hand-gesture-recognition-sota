# Hand Gesture Recognition SOTA (Assignment 2)

Course: Computer Vision, Astana IT University (AITU)  
Gestures: `like`, `peace`, `palm`

## Project Overview
This repository contains the upgraded Hand Gesture Recognition pipeline:
- **SOTA Computer Vision Architectures**: Benchmarking modern vision backbones (e.g., ConvNeXt / Swin Transformer) against the Assignment 1 baseline (MobileNetV2).
- **Explainability (XAI)**: Grad-CAM visual explanations to interpret model decisions and check for spurious correlations.
- **Interactive Web Demo**: Streamlit application supporting image upload and real-time webcam inference with latency benchmarking and Grad-CAM overlays.

## Project Structure
```text
├── data/
│   ├── raw/                 # Original A1 images
│   ├── public_sample/       # Filtered open-source gesture samples (HaGRID)
│   └── split/               # Train / Val / Test sets
├── src/
│   ├── __init__.py
│   ├── dataset.py           # PyTorch Dataset and augmentations
│   ├── model.py             # SOTA model definition and baseline wrappers
│   ├── train.py             # Training loop with validation and checkpointing
│   ├── gradcam.py           # Grad-CAM heatmap extraction
│   └── evaluate.py          # Benchmark metrics and latency evaluation
├── weights/                 # Model checkpoints (baseline and trained SOTA)
├── app/
│   ├── __init__.py
│   ├── app.py               # Streamlit interactive application
│   └── utils.py             # Preprocessing & Grad-CAM visualization helpers
├── requirements.txt         # Python dependencies
└── README.md
```

## Quickstart

1. **Environment Setup**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Interactive Web App**:
   ```bash
   streamlit run app/app.py
   ```
