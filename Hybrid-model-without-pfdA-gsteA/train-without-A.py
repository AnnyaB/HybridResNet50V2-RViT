# THIS IS THE TRAINING FILE FOR HYBRID B ABLATION (WITHOUT PFB-GSTE VARIANT B)

# Libraries I needed
import sys
from pathlib import Path

# PROJECT_ROOT points to the project folder (one level above this script file).
# This allows importing models/... and scripts/... as modules.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# - os for file paths
# - json for saving metrics
# - time for timing epochs / total training
# - Path again (duplicated import, but kept exactly as-is)

import os
import json
import time
from pathlib import Path



# - numpy for concatenation and statistics
# - torch for training/inference
# - nn for losses / modules
# - DataLoader for batching the dataset

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# Plotting utilities for curves and confusion matrix figures

import matplotlib.pyplot as plt


# Metrics used for evaluation reporting (test-time)

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    matthews_corrcoef,
)


# - HybridResNet50V2_RViT is the hybrid CNN-Transformer architecture
# - BrainMRICSV loads images using CSV split files
# - build_transforms builds train/eval preprocessing and augmentation

from models.hybrid_model import HybridResNet50V2_RViT
from scripts.data import BrainMRICSV, build_transforms



# Reproducibility helper

def set_seed(seed):
    # Set PyTorch RNG seed for CPU operations
    torch.manual_seed(seed)
    # Set NumPy RNG seed for any NumPy randomness
    np.random.seed(seed)
    # Set CUDA seeds for all available GPUs (even if training uses one)
    torch.cuda.manual_seed_all(seed)
    # Allow nondeterministic but faster CUDA kernels
    torch.backends.cudnn.deterministic = False
    # Enable cuDNN benchmark for performance (selects fastest kernels for input shapes)
    torch.backends.cudnn.benchmark = True



# Evaluation routine (no gradients)

@torch.no_grad()
def evaluate(model, loader, device, class_names):
    # Put model in eval mode (disables dropout, uses BN running stats)
    model.eval()
    # Lists to collect targets, predictions, and probabilities across batches
    all_y, all_pred, all_prob = [], [], []
    # Running totals for accuracy and average loss
    total, correct, loss_sum = 0, 0, 0.0
    # Plain cross-entropy for evaluation (no label smoothing here)
    ce = nn.CrossEntropyLoss()

    # Iterating over loader batches; dataset yields (image_tensor, label_tensor, path_or_id)
    for x, y, _ in loader:
        # Move batch to device (non_blocking can help when pin_memory=True)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        # Forward pass; model returns (logits, xai_or_none)
        logits, _ = model(x)
        # Compute loss for this batch
        loss = ce(logits, y)

        # Convert logits -> class probabilities
        prob = torch.softmax(logits, dim=1)
        # Predicted class index per sample
        pred = prob.argmax(dim=1)

        # Update totals for accuracy and average loss
        total += y.size(0)
        correct += (pred == y).sum().item()
        loss_sum += loss.item() * y.size(0)

        # Store numpy copies for sklearn metrics later
        all_y.append(y.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
        all_prob.append(prob.cpu().numpy())

    # Concatenate lists into full arrays for whole-set metrics
    all_y = np.concatenate(all_y)
    all_pred = np.concatenate(all_pred)
    all_prob = np.concatenate(all_prob)

    # Accuracy over all samples
    acc = correct / max(total, 1)
    # Mean loss over all samples
    avg_loss = loss_sum / max(total, 1)

    # Classification report provides precision/recall/f1 per class and macro/weighted averages
    rep = classification_report(
        all_y, all_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    # Macro F1 is used as early-stopping metric
    macro_f1 = rep["macro avg"]["f1-score"]

    # Return a bundle of useful evaluation outputs
    return {
        "loss": float(avg_loss),
        "acc": float(acc),
        "macro_f1": float(macro_f1),
        "y_true": all_y,
        "y_pred": all_pred,
        "y_prob": all_prob,
        "report": rep,
    }

# Specificity (macro-average)

def specificity_macro(cm):
    # cm is confusion matrix shape (num_classes, num_classes)
    n = cm.shape[0]
    specs = []
    # Compute per-class specificity: TN / (TN + FP)
    for c in range(n):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        spec = tn / (tn + fp + 1e-12)
        specs.append(spec)
    # Return macro mean and list of per-class values
    return float(np.mean(specs)), [float(s) for s in specs]


# Plot training loss curves

def plot_training(history, out_path):
    # Create x-axis list 1..num_epochs_completed
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


# Plot accuracy curves

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


# Plot confusion matrix (test)

def plot_confusion(cm, class_names, out_path):
    plt.figure()
    # Display matrix as an image
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (Test)")
    plt.colorbar()
    # Tick labels are class names
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)

    # Threshold controls text color for readability
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

# Optimizer parameter grouping

def build_param_groups(model, cnn_lr, vit_lr, weight_decay):
    
    """
    Differential LR and no-weight-decay for biases/norm params.
    """
    # Helper decides which parameters should NOT use weight decay:
    # - 1D params (often LayerNorm/BatchNorm weights)
    # - biases
    # - anything containing bn/norm/ln in its name
    def is_no_decay(name, p):
        if p.ndim == 1:
            return True
        lname = name.lower()
        if lname.endswith(".bias"):
            return True
        if "bn" in lname or "norm" in lname or "ln" in lname:
            return True
        return False

    # Separating parameter lists into CNN vs non-CNN, and decay vs no-decay
    cnn_decay, cnn_no = [], []
    rest_decay, rest_no = [], []

    # Iterating over all named parameters in the model
    for name, p in model.named_parameters():
        # Skipping frozen parameters
        if not p.requires_grad:
            continue
        # CNN parameters are those whose names begin with cnn.
        target_is_cnn = name.startswith("cnn.")
        # Route parameter into correct bucket
        if is_no_decay(name, p):
            (cnn_no if target_is_cnn else rest_no).append(p)
        else:
            (cnn_decay if target_is_cnn else rest_decay).append(p)

    # Building optimizer param groups with different LRs and weight decay settings
    groups = []
    if cnn_decay:
        groups.append({"params": cnn_decay, "lr": cnn_lr, "weight_decay": weight_decay})
    if cnn_no:
        groups.append({"params": cnn_no, "lr": cnn_lr, "weight_decay": 0.0})
    if rest_decay:
        groups.append({"params": rest_decay, "lr": vit_lr, "weight_decay": weight_decay})
    if rest_no:
        groups.append({"params": rest_no, "lr": vit_lr, "weight_decay": 0.0})

    return groups


# Freeze / unfreeze helper

def set_requires_grad(module, flag: bool):
    
    # Enable/disable gradient computation for all parameters in a submodule
    for p in module.parameters():
        p.requires_grad = flag



# Main training entry point

def main():
    import argparse
    # Build CLI arguments so training is configurable from terminal
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", type=str, default="data/splits/tightcrop")
    ap.add_argument("--out_dir", type=str, default="results/run_hybrid")

    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)

    # Differential LR (big impact)
    ap.add_argument("--cnn_lr", type=float, default=1e-4)
    ap.add_argument("--vit_lr", type=float, default=5e-4)

    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=2)

    # Stabilizers
    ap.add_argument("--amp", action="store_true", help="Use mixed precision (CUDA)")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--label_smoothing", type=float, default=0.05)

    # Fine-tuning strategy (Sarada-like usage of pretrained CNN)
    ap.add_argument("--warmup_epochs", type=int, default=5, help="Freeze CNN for first N epochs")
    ap.add_argument("--freeze_cnn_bn", action="store_true", help="Keep CNN BatchNorm frozen")

    # Model config
    ap.add_argument("--cnn_name", type=str, default="resnetv2_50x1_bitm")
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--embed_dim", type=int, default=142)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--heads", type=int, default=10)
    ap.add_argument("--mlp_dim", type=int, default=480)
    ap.add_argument("--attn_dropout", type=float, default=0.1)
    ap.add_argument("--vit_dropout", type=float, default=0.1)
    ap.add_argument("--fusion_dim", type=int, default=256)
    ap.add_argument("--fusion_dropout", type=float, default=0.5)

    # Parse args from command line
    args = ap.parse_args()

    # project_root string is passed into BrainMRICSV so it can resolve relative paths
    project_root = str(Path(__file__).resolve().parents[1])
    # Create output directory for checkpoints/plots/metrics
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Seed all RNGs for repeatable experiments
    set_seed(args.seed)

    # Choose device and whether AMP is active
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = bool(args.amp and device == "cuda")

    # Fixed class ordering used throughout training and evaluation
    class_names = ["glioma", "meningioma", "pituitary", "notumor"]

    # Keep my normalization
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    # Building training transforms (with augmentation) and eval transforms (no augmentation)
    train_tf = build_transforms(train=True, mean=mean, std=std)
    eval_tf = build_transforms(train=False, mean=mean, std=std)

    # Creating datasets from CSV split files
    train_ds = BrainMRICSV(os.path.join(args.csv_dir, "train.csv"), class_names, train_tf, project_root)
    val_ds = BrainMRICSV(os.path.join(args.csv_dir, "val.csv"), class_names, eval_tf, project_root)
    test_ds = BrainMRICSV(os.path.join(args.csv_dir, "test.csv"), class_names, eval_tf, project_root)

    # Build data loaders:
    # - train shuffle=True so batches are randomized each epoch
    # - pin_memory=True helps CPU->GPU transfer speed
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # Building the hybrid model using CLI config values
    model = HybridResNet50V2_RViT(
        num_classes=4,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        attn_dropout=args.attn_dropout,
        vit_dropout=args.vit_dropout,
        fusion_dim=args.fusion_dim,
        fusion_dropout=args.fusion_dropout,
        rotations=(0, 1, 2, 3),
        cnn_name=args.cnn_name,
        cnn_pretrained=True,
    ).to(device)

    # Loss with label smoothing (small but helps multi-class generalisation)
    ce = nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))

    # Warmup: freeze CNN first
    # This keeps the pretrained CNN fixed for the first warmup_epochs epochs.
    if args.warmup_epochs > 0:
        set_requires_grad(model.cnn, False)

    # AdamW optimizer uses parameter groups:
    # - CNN parameters use cnn_lr
    # - the rest uses vit_lr
    # - weight decay is disabled for biases and norm parameters
    optimizer = torch.optim.AdamW(
        build_param_groups(model, cnn_lr=args.cnn_lr, vit_lr=args.vit_lr, weight_decay=args.weight_decay)
    )
    # Cosine schedule for learning rate decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # AMP gradient scaler (enabled only when CUDA + --amp)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # History dict collects values each epoch (later saved to CSV and plotted)
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_macro_f1": [],
        "epoch_time_sec": [],
    }

    # Track best validation macro-F1 for early stopping
    best_f1 = -1.0
    best_epoch = -1
    bad_epochs = 0

    # Wall-clock start time for full training run
    start_train = time.time()


    # Training loop (epoch by epoch)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Unfreeze after warmup
        # When warmup ends, unfreeze CNN and rebuild optimizer/scheduler so CNN params are included properly.
        if epoch == args.warmup_epochs + 1:
            set_requires_grad(model.cnn, True)
            # Rebuild optimizer so CNN params are included with cnn_lr
            optimizer = torch.optim.AdamW(
                build_param_groups(model, cnn_lr=args.cnn_lr, vit_lr=args.vit_lr, weight_decay=args.weight_decay)
            )
            # Restart cosine schedule over remaining epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.epochs - epoch + 1), eta_min=1e-6)
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        # Switch model to training mode (enables dropout etc.)
        model.train()

        # Freeze CNN BN stats if requested (often stabilizes transfer learning)
        if args.freeze_cnn_bn:
            model.freeze_cnn_bn()

        # Running sums for this epoch
        total, correct, loss_sum = 0, 0, 0.0

       
        # Mini-batch loop
   
        for x, y, _ in train_loader:
            # Move data to device
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Clear gradients efficiently
            optimizer.zero_grad(set_to_none=True)

            # Forward pass under autocast if AMP enabled
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, _ = model(x)
                loss = ce(logits, y)

            # Backward pass with scaling (AMP-safe)
            scaler.scale(loss).backward()

            # Grad clip (very important for the val_loss spikes you saw)
            if args.grad_clip and args.grad_clip > 0:
                # Unscale first so clipping works in real gradient units
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))

            # Optimizer step + scaler update (AMP-safe)
            scaler.step(optimizer)
            scaler.update()

            # Compute training accuracy for the batch
            prob = torch.softmax(logits.detach(), dim=1)
            pred = prob.argmax(dim=1)

            # Accumulate epoch totals
            total += y.size(0)
            correct += (pred == y).sum().item()
            loss_sum += float(loss.item()) * y.size(0)

        # Epoch-level train metrics
        train_loss = loss_sum / max(total, 1)
        train_acc = correct / max(total, 1)

        # Validation pass (no gradients)
        val_stats = evaluate(model, val_loader, device, class_names)
        val_loss = val_stats["loss"]
        val_acc = val_stats["acc"]
        val_f1 = val_stats["macro_f1"]

        # Step LR scheduler once per epoch
        scheduler.step()

        # Save epoch metrics into history
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))
        history["val_macro_f1"].append(float(val_f1))
        history["epoch_time_sec"].append(float(time.time() - t0))

        # Console progress log
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macroF1={val_f1:.4f}"
        )

        # Early stopping on macro F1
        # If macro-F1 improves, save checkpoint; otherwise count a "bad epoch".
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            bad_epochs = 0

            # Build checkpoint dict including weights + metadata for reproducibility
            ckpt = {
                "model_state": model.state_dict(),
                "class_names": class_names,
                "mean": mean,
                "std": std,
                "train_args": vars(args),
                "model_cfg": {
                    "num_classes": 4,
                    "patch_size": args.patch_size,
                    "embed_dim": args.embed_dim,
                    "depth": args.depth,
                    "heads": args.heads,
                    "mlp_dim": args.mlp_dim,
                    "attn_dropout": args.attn_dropout,
                    "vit_dropout": args.vit_dropout,
                    "fusion_dim": args.fusion_dim,
                    "fusion_dropout": args.fusion_dropout,
                    "rotations": (0, 1, 2, 3),
                    "cnn_name": args.cnn_name,
                },
            }
            # Save best checkpoint so far
            torch.save(ckpt, out_dir / "best_model.pt")
        else:
            bad_epochs += 1
            # Stop if no improvement for `patience` consecutive epochs
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch}, best macroF1: {best_f1:.4f})")
                break

    # Total training wall-clock time
    total_train_time = time.time() - start_train
    print(f"Total training time (sec): {total_train_time:.1f}")

    # Save history
    # pandas is imported here only for convenient CSV writing
    import pandas as pd
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    # Plots
    # Save loss and accuracy curves based on history CSV data
    plot_training(history, out_dir / "loss_curves.png")
    plot_accuracy(history, out_dir / "acc_curves.png")

    # Test eval using best model
    # Reload the best checkpoint, then evaluate once on the held-out test set.
    best = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best["model_state"])

    test_stats = evaluate(model, test_loader, device, class_names)
    y_true = test_stats["y_true"]
    y_pred = test_stats["y_pred"]

    # Confusion matrix + plot
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plot_confusion(cm, class_names, out_dir / "confusion_matrix.png")

    # Additional test metrics derived from sklearn outputs
    rep = test_stats["report"]
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    spec_macro, spec_per_class = specificity_macro(cm)

    # Metrics JSON collects test results + run metadata for the report
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
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_f1),
    }

    # Write metrics JSON file
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Final prints for quick confirmation in terminal logs
    print("Saved outputs to:", str(out_dir))
    print("Best epoch:", best_epoch, "Best val macroF1:", best_f1)


# Standard Python entry point guard
if __name__ == "__main__":
    main()
