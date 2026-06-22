# EfficientNet-B4 Cross-Dataset Testing

**Contributor:** Aena (ae_verma)  
**Course:** INSE 6610 | Concordia University | Winter 2026  
**Algorithm:** EfficientNet-B4  
**Training Dataset:** FaceForensics++ (FF++) at C23 compression  

---

## Overview

This folder contains the cross-dataset evaluation of the EfficientNet-B4 deepfake detection model.
The model was trained on FF++ and tested on four datasets to measure generalization across different deepfake types.

---

## Files

| File | Description |
|---|---|
| `test.py` | Cross-dataset testing script |
| `test_job.sh` | SLURM batch job script to run test.py on Speed HPC |
| `cross_dataset_results.png` | Bar chart comparing Accuracy and AUC-ROC across all 4 datasets |
| `cm_FF++.png` | Confusion matrix for FF++ |
| `cm_CelebDF_V2.png` | Confusion matrix for CelebDF V2 |
| `cm_Sarra_Dataset.png` | Confusion matrix for Sarra Dataset (UADFV) |
| `cm_CIFAKE.png` | Confusion matrix for CIFAKE |

---

## How to Run

1. SSH into Concordia Speed HPC cluster
2. Make sure your model file exists at `/speed-scratch/ae_verma/best_efficientnet_b4.pth`
3. Make sure all datasets are in `/speed-scratch/inse6610team2/`
4. Submit the job:

```bash
sbatch /speed-scratch/ae_verma/test_job.sh
```

5. Monitor progress:

```bash
tail -f /speed-scratch/ae_verma/test_output.log
```

---

## Datasets Tested

| Dataset | Location on Speed | Images | Contributor |
|---|---|---|---|
| FF++ | `/speed-scratch/inse6610team2/ff_faces/` | 30,000 | Aena (ae_verma) |
| CelebDF V2 | `/speed-scratch/inse6610team2/Celeb_V2/Test/` | 10,103 | Lilas (g_lilas) |
| Sarra Dataset (UADFV) | `/speed-scratch/inse6610team2/dataset_images/` | 946 | Sarra (s_benred) |
| CIFAKE | `/speed-scratch/inse6610team2/CIFAKE-ResNet-50/test/` | 20,000 | Bhumi (bh_son) |

---

## Results

| Dataset | Accuracy | AUC-ROC |
|---|---|---|
| FF++ (own dataset) | 69.62% | 76.56% |
| CelebDF V2 | 53.57% | 56.73% |
| Sarra Dataset (UADFV) | 58.03% | 63.98% |
| CIFAKE | 52.12% | 52.28% |

---

## Key Findings

- **FF++** performed best as the model was trained on this dataset and is familiar with its specific compression artifacts and face swap style.
- **CelebDF V2** performed poorly due to higher quality face swaps with better blending and different visual characteristics compared to FF++.
- **Sarra Dataset (UADFV)** performed slightly better than CelebDF because it uses older simpler face swap techniques somewhat similar to FF++.
- **CIFAKE** performed worst at near random guessing because it contains AI-generated images from diffusion models, completely outside the model's training domain.
- The overall drop in performance across all non-FF++ datasets confirms the **domain shift** problem, a well-known challenge in deepfake detection where models trained on one dataset do not generalize well to others.