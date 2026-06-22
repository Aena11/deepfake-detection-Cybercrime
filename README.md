# Deepfake-detection-Cybercrime
# DEEPFAKE1: EfficientNet-B4 on FaceForensics++

## Overview

This directory contains the implementation of **EfficientNet-B4** for deepfake image detection, evaluated on the **FaceForensics++ (FF++)** dataset. This is one component of the DEEPFAKE1 group project for INSE 6610 (W26), where we combine and compare multiple deepfake detection algorithms across benchmark datasets.

## Algorithm: EfficientNet-B4

EfficientNet-B4 uses compound scaling to jointly optimize network depth, width, and resolution. It was the winning architecture in the Facebook Deepfake Detection Challenge (DFDC) and has demonstrated strong performance across multiple deepfake benchmarks.

**Why EfficientNet-B4?**
- Won the DFDC Challenge with top accuracy
- Better parameter efficiency than XceptionNet due to compound scaling
- Reports ~96.8% accuracy on FF++ and ~93.5% AUC on DFDC in published results
- Well-supported in PyTorch via the `timm` library with pretrained ImageNet weights

**Key references:**
- Tan, M. & Le, Q. (2019). "EfficientNet: Rethinking Model Scaling for CNNs." ICML 2019.
- Rössler, A. et al. (2019). "FaceForensics++: Learning to Detect Manipulated Facial Images." ICCV 2019.
- naram92/DeepFakeDetection : EfficientNet-B4 implementation (Laval University, GLO-4030)

## Dataset: FaceForensics++ (FF++)

FaceForensics++ is a large-scale face manipulation benchmark containing:

| Split | Count | Description |
|-------|-------|-------------|
| Original | 1,000 videos | Real YouTube videos with frontal faces |
| Deepfakes | 1,000 videos | Face swap using autoencoder-based method |
| Face2Face | 1,000 videos | Facial reenactment |
| FaceSwap | 1,000 videos | Graphics-based face swap |
| NeuralTextures | 1,000 videos | Neural rendering-based manipulation |

**Compression levels used:** c23 (light compression : standard in literature)

**Access:** Requires academic access request via the FaceForensics GitHub page.

> **Note:** The dataset is NOT stored in this repository. It is kept locally and shared via the team's shared storage. Use the FaceForensics download script to obtain it.

## Directory Structure

```
src/team2/
├── README.md                        # This file
├── efficientnet_deepfake.ipynb      # Main notebook (training, evaluation, results)
├── utils/
│   ├── dataset.py                   # DeepfakeDataset class and data loading
│   ├── transforms.py                # Data augmentation pipeline
│   └── metrics.py                   # Evaluation metrics (accuracy, AUC, F1, MiFID)
└── results/
    ├── results.csv                  # Cross-validation results
    └── plots/                       # Visualizations and comparison charts
```

## Setup

### Dependencies

```bash
pip install torch torchvision timm
pip install opencv-python-headless albumentations
pip install scikit-learn matplotlib seaborn pandas
pip install facenet-pytorch    # MTCNN face detection for frame extraction
```

### Dataset Preparation

1. Request access at the [FaceForensics++ GitHub](https://github.com/ondyari/FaceForensics)
2. Download using the provided script:
   ```bash
   python download-FaceForensics_v3.py /path/to/download -t all -c c23
   ```
3. Extract face crops from video frames:
   ```bash
   python utils/dataset.py --extract-faces --input /path/to/ff++ --output /path/to/faces
   ```
4. Update the dataset path in the notebook config.

### Training

Open `efficientnet_deepfake.ipynb` and run all cells. The notebook performs:
- 5-fold stratified cross-validation
- Training with cosine annealing LR schedule
- Evaluation on held-out fold with accuracy, AUC, F1, log loss

## Evaluation Metrics

| Metric | Purpose |
|--------|---------|
| AUC-ROC | Discrimination ability across thresholds |
| Accuracy | Overall correctness |
| F1-Score | Balance of precision and recall |
| Log Loss | Prediction confidence calibration |

## Actual Results (Trained on Speed HPC Cluster)

| Metric | Value |
|--------|-------|
| Test Accuracy | 86.78% |
| AUC-ROC | 95.19% |
| Real Precision | 86% |
| Fake Precision | 87% |
| Training Epochs | 10 |
| Training Data | 3000 real + 3000 fake frames |
| GPU | Tesla V100-PCIE-32GB |

Results plots are in `efficientnet_ffpp/` folder.

## Team

- **Course:** INSE 6610 - Concordia University (W26)
- **Project:** DEEPFAKE1
- **Repository:** inse6610-w26-team2/DEEPFAKE1
