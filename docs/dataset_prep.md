# Dataset Preparation and Verification

**Project:** Hybrid CNN–Transformer Model for Explainable Brain Tumour Classification using MRI Scans  
**Author:** Riya Basak (22089065)  
**Supervisor:** Dr Kheng Lee Koay  
**Date:** November 2025  

---

## 1. Dataset Source

The dataset used in this project is the **Brain Tumour MRI Dataset** released on Kaggle by Nickparvar (2024).  
It contains **7,023 anonymised, T1-weighted contrast-enhanced MRI images** categorised into four classes:

| Class | Description |
|-------|--------------|
| Glioma | Malignant glial tumour |
| Meningioma | Tumour of meninges |
| Pituitary | Pituitary gland tumour |
| No Tumour | Healthy control MRI |

Each image is anonymised, ensuring no ethical risks.

---

## 2. Verification and Cleaning

- **Total images scanned:** 7,023  
- **Duplicate removal:** 297 exact-match duplicates removed using SHA-1 hashes  
- **Unique total:** 6,726 MRI images  

Duplicate removal prevents *data leakage* between training and testing sets.

---

## 3. Preprocessing Pipeline

| Step | Action | Rationale |
|------|---------|-----------|
| 1 | Resize → 224 × 224 px | Standard input for ResNet50V2 & RViT |
| 2 | Normalise → [0, 1] (at training time) | Improves convergence stability |
| 3 | Augmentation | Minor rotation (±12°), horizontal flip, brightness × 1.2 |
| 4 | Stratified split → 70 / 15 / 15 (train/val/test) | Preserves class balance |
| 5 | Verification | Histogram + augmentation previews + CSV summary |

Processing scripts:
- `scripts/dataset_prep.py`
- `scripts/plot_class_distribution.py`
- `scripts/preview_augmentations.py`

Outputs stored under `results/` include:
- `dataset_summary.csv`
- `class_distribution_histogram.png`
- `augmentation_preview.png`

---

## 4. Dataset Summary

| Split | Glioma | Meningioma | Pituitary | No Tumour | Total |
|:------|:-------:|:-----------:|:----------:|:-----------:|:-------:|
| Train | 1134 | 1144 | 1218 | 1212 | 4708 |
| Val | 243 | 245 | 261 | 260 | 1009 |
| Test | 243 | 246 | 261 | 259 | 1009 |
| **Total (Unique)** | 1620 | 1635 | 1740 | 1731 | **6726** |

---

## 5. Literature Alignment

| Reference | Technique Used Here |
|------------|-------------------|
| **Rasool et al. (2024)** | ResNet50V2-based preprocessing pipeline; 70/15/15 split and 224×224 input |
| **Krishnan et al. (2024)** | Rotation-Invariant ViT: small-angle rotation augmentations; patch embedding from 224×224 inputs |
| **Nickparvar (2024)** | Dataset structure and labels identical |

This ensures that our dataset preprocessing exactly matches the experimental protocol used in hybrid CNN–Transformer studies.

---

## 6. Planned Online Transformations (at Training Stage)

These augmentations will be applied dynamically in the dataloaders:

```python
transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
