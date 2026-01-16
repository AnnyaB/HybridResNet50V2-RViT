# scripts/data.py
import os
import csv
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def _read_csv_rows(csv_path):
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]
    return rows


def _infer_cols(rows):
    cols = set(rows[0].keys())
    path_col = None
    label_col = None
    for c in ["path", "filepath", "file", "image", "img_path"]:
        if c in cols:
            path_col = c
            break
    for c in ["label", "class", "target", "y"]:
        if c in cols:
            label_col = c
            break
    return path_col, label_col


class BrainMRIDataset(Dataset):
    def __init__(self, root_dir, csv_path=None, class_to_idx=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        if csv_path is not None and os.path.exists(csv_path):
            rows = _read_csv_rows(csv_path)
            if len(rows) == 0:
                raise RuntimeError(f"CSV is empty: {csv_path}")

            path_col, label_col = _infer_cols(rows)
            if path_col is None or label_col is None:
                raise RuntimeError(f"CSV must have path+label columns. Found: {list(rows[0].keys())}")

            # build class_to_idx if needed
            if class_to_idx is None:
                labels = sorted(list({r[label_col] for r in rows}))
                # if labels are numeric strings, sort numerically
                try:
                    labels_sorted = sorted(labels, key=lambda s: int(s))
                    labels = labels_sorted
                except Exception:
                    pass
                class_to_idx = {lab: i for i, lab in enumerate(labels)}

            for r in rows:
                rel = r[path_col]
                lab = r[label_col]
                img_path = rel if os.path.isabs(rel) else os.path.join(root_dir, rel)
                if lab not in class_to_idx:
                    # allow numeric labels
                    try:
                        li = int(lab)
                        self.samples.append((img_path, li))
                        continue
                    except Exception:
                        raise RuntimeError(f"Label '{lab}' not in class_to_idx keys: {list(class_to_idx.keys())}")
                self.samples.append((img_path, class_to_idx[lab]))

            self.class_to_idx = class_to_idx
            self.idx_to_class = {v: k for k, v in class_to_idx.items()}
            return

        # Fallback: scan root_dir expecting root_dir/<class>/*.png|jpg
        classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        if class_to_idx is None:
            class_to_idx = {c: i for i, c in enumerate(classes)}

        for c in classes:
            cdir = os.path.join(root_dir, c)
            for fn in os.listdir(cdir):
                if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                    self.samples.append((os.path.join(cdir, fn), class_to_idx[c]))

        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(y, dtype=torch.long), path


def build_transforms(norm_mode="0.5"):
    # norm_mode: "0.5" => mean=std=0.5; "imagenet" => ImageNet stats
    if norm_mode == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.RandomRotation(degrees=10)], p=0.25),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_tf, eval_tf


def make_loaders(data_root, splits_dir, batch_size=32, num_workers=2, norm_mode="0.5"):
    train_tf, eval_tf = build_transforms(norm_mode=norm_mode)

    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")

    train_csv = os.path.join(splits_dir, "train.csv")
    val_csv = os.path.join(splits_dir, "val.csv")
    test_csv = os.path.join(splits_dir, "test.csv")

    train_ds = BrainMRIDataset(train_dir, csv_path=train_csv, transform=train_tf)
    class_to_idx = train_ds.class_to_idx

    val_ds = BrainMRIDataset(val_dir, csv_path=val_csv, class_to_idx=class_to_idx, transform=eval_tf)
    test_ds = BrainMRIDataset(test_dir, csv_path=test_csv, class_to_idx=class_to_idx, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_ds.idx_to_class
