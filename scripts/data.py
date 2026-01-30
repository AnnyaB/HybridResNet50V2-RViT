# scripts/data.py
#
# Dataset loader for CSV splits produced by dataset_prep.py:
#   data/splits/tightcrop/{train,val,test}.csv
# CSV columns:
#   - image_path
#   - class  (glioma/meningioma/pituitary/notumor)

import os
from pathlib import Path
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class AddGaussianNoise:
    def __init__(self, std=0.02, p=0.5):
        self.std = std
        self.p = p

    def __call__(self, x):
        if torch.rand(1).item() > self.p:
            return x
        noise = torch.randn_like(x) * self.std
        x = x + noise
        return torch.clamp(x, 0.0, 1.0)


def _resolve_path(p, project_root):
    # Your CSV may contain absolute paths from a different machine.
    # This resolver makes the project portable for Colab/Kaggle.
    if os.path.exists(p):
        return p

    p2 = str(p).replace("\\", "/")
    key = "/data/processed/"
    idx = p2.find(key)
    if idx != -1:
        rel = p2[idx + 1:]  # remove leading '/'
        cand = str(Path(project_root) / rel)
        if os.path.exists(cand):
            return cand

    # fallback: try relative to project_root directly
    cand2 = str(Path(project_root) / p2)
    if os.path.exists(cand2):
        return cand2

    return p  # let it fail loudly in __getitem__ for debugging


def build_transforms(train, mean, std):
    ops = []
    if train:
        ops.extend([
            T.RandomRotation(degrees=15),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        ])

    ops.extend([T.ToTensor()])

    if train:
        ops.append(AddGaussianNoise(std=0.02, p=0.5))

    ops.append(T.Normalize(mean=mean, std=std))
    return T.Compose(ops)


class BrainMRICSV(Dataset):
    def __init__(self, csv_path, class_names, transform, project_root):
        self.df = pd.read_csv(csv_path)
        self.class_names = class_names
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.transform = transform
        self.project_root = project_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = _resolve_path(row["image_path"], self.project_root)
        y_str = row["class"]
        y = self.class_to_idx[y_str]

        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        return x, y, img_path
