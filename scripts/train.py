# scripts/train.py
#
# Training + evaluation + plots for HybridResNet50V2–RViT.
# Produces:
# - best_model.pt
# - training_curves.png (loss/acc)
# - confusion_matrix.png
# - metrics.json (ACC, Precision/Recall/F1 macro + per-class, Specificity macro, Kappa, MCC)
# - history.csv
#
# Early stopping monitors validation macro F1.

import os
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    matthews_corrcoef,
)

from models.hybrid_model_pfd_gste import HybridResNet50V2_RViT
from scripts.data import BrainMRICSV, build_transforms


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


@torch.no_grad()
def evaluate(model, loader, device, class_names):
    model.eval()
    all_y = []
    all_pred = []
    all_prob = []
    total = 0
    correct = 0
    loss_sum = 0.0
    ce = nn.CrossEntropyLoss()

    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)
        logits, _ = model(x)
        loss = ce(logits, y)

        prob = torch.softmax(logits, dim=1)
        pred = prob.argmax(dim=1)

        total += y.size(0)
        correct += (pred == y).sum().item()
        loss_sum += loss.item() * y.size(0)

        all_y.append(y.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
        all_prob.append(prob.cpu().numpy())

    all_y = np.concatenate(all_y)
    all_pred = np.concatenate(all_pred)
    all_prob = np.concatenate(all_prob)

    acc = correct / max(total, 1)
    avg_loss = loss_sum / max(total, 1)

    # macro F1 from classification_report
    rep = classification_report(all_y, all_pred, target_names=class_names, output_dict=True, zero_division=0)
    macro_f1 = rep["macro avg"]["f1-score"]

    return {
        "loss": float(avg_loss),
        "acc": float(acc),
        "macro_f1": float(macro_f1),
        "y_true": all_y,
        "y_pred": all_pred,
        "y_prob": all_prob,
        "report": rep,
    }


def specificity_macro(cm):
    # one-vs-rest specificity for each class then macro average
    n = cm.shape[0]
    specs = []
    for c in range(n):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        spec = tn / (tn + fp + 1e-12)
        specs.append(spec)
    return float(np.mean(specs)), [float(s) for s in specs]


def plot_training(history, out_path):
    epochs = list(range(1, len(history["train_loss"]) + 1))

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_accuracy(history, out_path):
    epochs = list(range(1, len(history["train_acc"]) + 1))

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_confusion(cm, class_names, out_path):
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (Test)")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", type=str, default="data/splits/tightcrop")
    ap.add_argument("--out_dir", type=str, default="results/run_hybrid")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    project_root = str(Path(__file__).resolve().parents[1])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    class_names = ["glioma", "meningioma", "pituitary", "notumor"]

    # Normalization: computed during training in many medical pipelines.
    # Default kept simple and stable for RGB-replicated MRI: mean=0.5, std=0.5 per channel.
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    train_tf = build_transforms(train=True, mean=mean, std=std)
    eval_tf = build_transforms(train=False, mean=mean, std=std)

    train_ds = BrainMRICSV(os.path.join(args.csv_dir, "train.csv"), class_names, train_tf, project_root)
    val_ds = BrainMRICSV(os.path.join(args.csv_dir, "val.csv"), class_names, eval_tf, project_root)
    test_ds = BrainMRICSV(os.path.join(args.csv_dir, "test.csv"), class_names, eval_tf, project_root)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = HybridResNet50V2_RViT(
        num_classes=4,
        patch_size=16,
        embed_dim=140,
        depth=10,
        heads=10,
        mlp_dim=480,
        attn_dropout=0.1,
        vit_dropout=0.1,
        cnn_dropout=0.05,
        fusion_dim=256,
        fusion_dropout=0.5,
        rotations=(0, 1, 2, 3),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    ce = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_macro_f1": [],
        "epoch_time_sec": [],
    }

    best_f1 = -1.0
    best_epoch = -1
    bad_epochs = 0

    start_train = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()

        total = 0
        correct = 0
        loss_sum = 0.0

        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x)
            loss = ce(logits, y)
            loss.backward()
            optimizer.step()

            prob = torch.softmax(logits, dim=1)
            pred = prob.argmax(dim=1)

            total += y.size(0)
            correct += (pred == y).sum().item()
            loss_sum += loss.item() * y.size(0)

        train_loss = loss_sum / max(total, 1)
        train_acc = correct / max(total, 1)

        val_stats = evaluate(model, val_loader, device, class_names)
        val_loss = val_stats["loss"]
        val_acc = val_stats["acc"]
        val_f1 = val_stats["macro_f1"]

        scheduler.step()

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))
        history["val_macro_f1"].append(float(val_f1))
        history["epoch_time_sec"].append(float(time.time() - t0))

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macroF1={val_f1:.4f}"
        )

        # Early stopping on macro F1
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            bad_epochs = 0

            ckpt = {
                "model_state": model.state_dict(),
                "class_names": class_names,
                "mean": mean,
                "std": std,
                "args": vars(args),
            }
            torch.save(ckpt, out_dir / "best_model.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping triggered at epoch {epoch} (best epoch: {best_epoch}, best macroF1: {best_f1:.4f})")
                break

    total_train_time = time.time() - start_train
    print(f"Total training time (sec): {total_train_time:.1f}")

    # Save history
    import pandas as pd
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    # Plots
    plot_training(history, out_dir / "loss_curves.png")
    plot_accuracy(history, out_dir / "acc_curves.png")

    # Test evaluation using best model
    best = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best["model_state"])
    test_stats = evaluate(model, test_loader, device, class_names)

    y_true = test_stats["y_true"]
    y_pred = test_stats["y_pred"]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plot_confusion(cm, class_names, out_dir / "confusion_matrix.png")

    # Metrics
    rep = test_stats["report"]
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    spec_macro, spec_per_class = specificity_macro(cm)

    metrics = {
        "test_loss": test_stats["loss"],
        "test_acc": test_stats["acc"],
        "per_class": {c: rep[c] for c in class_names},
        "macro_avg": rep["macro avg"],
        "weighted_avg": rep["weighted avg"],
        "specificity_macro": spec_macro,
        "specificity_per_class": {class_names[i]: spec_per_class[i] for i in range(len(class_names))},
        "cohens_kappa": float(kappa),
        "mcc_multiclass": float(mcc),
        "confusion_matrix": cm.tolist(),
        "num_parameters": int(sum(p.numel() for p in model.parameters())),
        "epoch_time_sec_mean": float(np.mean(history["epoch_time_sec"])) if len(history["epoch_time_sec"]) else None,
        "total_training_time_sec": float(total_train_time),
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved outputs to:", str(out_dir))
    print("Best epoch:", best_epoch, "Best val macroF1:", best_f1)


if __name__ == "__main__":
    main()
