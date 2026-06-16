# AUTHOR: RIYA BASAK
# 

# THIS IS THE TRAINING FILE FOR HYBRID B ABLATION (WITHOUT PFB-GSTE VARIANT B)
# External libraries used here are cited in Appendix A2.3:
# NumPy (Harris et al., 2020); PyTorch (Paszke et al., 2019);
# Matplotlib (Hunter, 2007); scikit-learn (Pedregosa et al., 2011).


# Libraries I needed
import sys
from pathlib import Path

# Compute the project root:
# - __file__ is this script's path
# - parents[1] moves up two levels (e.g., scripts/ -> project_root/)
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Adding project root to Python import search path so models and scripts can be imported.
# -----------------------------
sys.path.append(str(PROJECT_ROOT))

import os
import json
import time
from pathlib import Path

# - numpy for array handling and metrics aggregation
# - torch for training/inference
# - nn for loss functions
# - DataLoader for batching data

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# Matplotlib for saving plots (loss/accuracy curves, confusion matrix).

import matplotlib.pyplot as plt


# scikit-learn metrics for evaluation outputs:
# - confusion matrix
# - classification report (precision/recall/F1)
# - Cohen's kappa and Matthews correlation coefficient

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    matthews_corrcoef,
)


# Project-local imports:
# - Hybrid model architecture
# - Dataset class and transform builder

from models.hybrid_model import HybridResNet50V2_RViT
from scripts.data import BrainMRICSV, build_transforms


# Reproducibility helper:
# Sets random seeds for CPU and CUDA, and configures cuDNN behavior.
# Note:
# - cudnn.deterministic=False and benchmark=True trades strict determinism for speed.

def set_seed(seed):
    
    # Seed PyTorch CPU RNG
    torch.manual_seed(seed)
    # Seed NumPy RNG
    np.random.seed(seed)
    # Seed all CUDA devices (if available)
    torch.cuda.manual_seed_all(seed)
    # Allow non-deterministic algorithms (faster, but slight run-to-run variation possible)
    torch.backends.cudnn.deterministic = False
    # Enable cuDNN benchmarking to pick fastest convolution algorithms for the current shapes
    torch.backends.cudnn.benchmark = True



# Evaluation function (validation/test):
# - Decorated with @torch.no_grad() to disable gradient tracking (faster and less memory).
# - Returns loss, accuracy, macro-F1, raw arrays, and a full classification report dict.

@torch.no_grad()
def evaluate(model, loader, device, class_names):
    
    # Putting model into eval mode (disables dropout, uses running stats in BN, etc.)
    model.eval()
    # Storing ground truth, predictions, and probabilities for full-report metrics later
    all_y, all_pred, all_prob = [], [], []
    # Tracking totals to compute average loss and accuracy
    total, correct, loss_sum = 0, 0, 0.0
    # Use plain CrossEntropyLoss for evaluation (no label smoothing here)
    ce = nn.CrossEntropyLoss()

    # Iterating through evaluation loader batches
    for x, y, _ in loader:
        # Moving inputs/labels to device; non_blocking=True helps when pin_memory=True in DataLoader
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        # Forward pass: model returns logits and a second output (unused here)
        logits, _ = model(x)
        # Computing CE loss for this batch
        loss = ce(logits, y)

        # Converting logits to probabilities
        prob = torch.softmax(logits, dim=1)
        # Predicteding class index per sample
        pred = prob.argmax(dim=1)

        # Updating running totals for accuracy and loss
        total += y.size(0)
        correct += (pred == y).sum().item()
        loss_sum += loss.item() * y.size(0)

        # Collecting arrays on CPU for sklearn metrics
        all_y.append(y.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
        all_prob.append(prob.cpu().numpy())

    # Concatenating batch-wise arrays into full arrays
    all_y = np.concatenate(all_y)
    all_pred = np.concatenate(all_pred)
    all_prob = np.concatenate(all_prob)

    # Compute accuracy and average loss (safe divide)
    acc = correct / max(total, 1)
    avg_loss = loss_sum / max(total, 1)

    # Building classification report with per-class + macro/weighted averages
    rep = classification_report(
        all_y, all_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    # Macro-F1 is used for early stopping
    macro_f1 = rep["macro avg"]["f1-score"]

    # Returning everything needed downstream (plots, metrics.json, early stopping, etc.)
    return {
        "loss": float(avg_loss),
        "acc": float(acc),
        "macro_f1": float(macro_f1),
        "y_true": all_y,
        "y_pred": all_pred,
        "y_prob": all_prob,
        "report": rep,
    }

# Specificity calculation:
# - Computes per-class specificity (TN / (TN + FP)) from confusion matrix
# - Returns macro average specificity and list of per-class specificities

def specificity_macro(cm):
    # Number of classes
    n = cm.shape[0]
    specs = []
    # Compute specificity for each class using one-vs-rest confusion components
    for c in range(n):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        spec = tn / (tn + fp + 1e-12)
        specs.append(spec)
        
    # Returning macro mean and per-class list (as python floats)
    return float(np.mean(specs)), [float(s) for s in specs]



# Plot training/validation loss curves and save as an image.

def plot_training(history, out_path):
    
    # Epoch numbers start from 1 for plotting
    epochs = list(range(1, len(history["train_loss"]) + 1))
    plt.figure()
    # Plot training and validation losses
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()



# Plot training/validation accuracy curves and save as an image.

def plot_accuracy(history, out_path):
    
    # Epoch numbers start from 1 for plotting
    epochs = list(range(1, len(history["train_acc"]) + 1))
    plt.figure()
    # Plot training and validation accuracies
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()



# Plot confusion matrix (test) and save as an image.
# - Adds class labels on axes and cell counts in each square.

def plot_confusion(cm, class_names, out_path):
    
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (Test)")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)

    # Threshold used to choose white/black text for readability
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

# Building optimizer parameter groups:
# - Uses different learning rates for CNN vs Transformer/other parts (cnn_lr vs vit_lr)
# - Applies weight decay selectively:
#   - No weight decay for biases, norm layers, and 1D parameters (typical best practice)

def build_param_groups(model, cnn_lr, vit_lr, weight_decay):
    
    """
    Differential LR and no-weight-decay for biases/norm params.
    """
    # Helper to detect parameters that should NOT receive weight decay
    def is_no_decay(name, p):
        # 1D parameters often correspond to LayerNorm/BatchNorm weights, biases, etc.
        if p.ndim == 1:
            return True
        lname = name.lower()
        # Bias parameters should not decay
        if lname.endswith(".bias"):
            return True
        # Norm layers (BN/LN/Norm) should not decay
        if "bn" in lname or "norm" in lname or "ln" in lname:
            return True
        return False

    # Separating parameter lists for:
    # - CNN: decay vs no-decay
    # - Rest (Transformer/fusion/etc.): decay vs no-decay
    cnn_decay, cnn_no = [], []
    rest_decay, rest_no = [], []

    # Iterating over all named parameters in the model
    for name, p in model.named_parameters():
        # Skip frozen parameters
        if not p.requires_grad:
            continue
        # Identify CNN parameters by name prefix "cnn."
        target_is_cnn = name.startswith("cnn.")
        # Routing parameter into decay/no-decay lists based on rules
        if is_no_decay(name, p):
            (cnn_no if target_is_cnn else rest_no).append(p)
        else:
            (cnn_decay if target_is_cnn else rest_decay).append(p)

    # Building optimizer groups with appropriate lr and weight_decay
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


# Utility to freeze/unfreeze an entire module by toggling requires_grad.
# Used for the warmup strategy where the CNN is frozen for the first N epochs.

def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag



# Main training routine:
# - Parses CLI args
# - Builds datasets/dataloaders
# - Builds model
# - Trains with optional AMP, label smoothing, grad clipping
# - Warmup: freeze CNN then unfreeze and rebuild optimizer/scheduler
# - Early stopping based on validation macro-F1
# - Saves best checkpoint, history.csv, plots and metrics.json

def main():
    
    # argparse is imported locally inside main
    import argparse
    ap = argparse.ArgumentParser()
    # Dataset CSV directory containing train.csv, val.csv, test.csv
    ap.add_argument("--csv_dir", type=str, default="data/splits/tightcrop")
    # Output directory for checkpoints, plots, history, metrics
    ap.add_argument("--out_dir", type=str, default="results/run_hybrid")

    # Training length and batch size
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)

    # Differential LR (big impact)
    ap.add_argument("--cnn_lr", type=float, default=1e-4)
    ap.add_argument("--vit_lr", type=float, default=5e-4)

    # Regularization, early stopping, reproducibility and DataLoader workers
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=2)

    # Stabilizers
    ap.add_argument("--amp", action="store_true", help="Use mixed precision (CUDA)")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--label_smoothing", type=float, default=0.05)

    # Fine-tuning strategy (usage of pretrained CNN)
    ap.add_argument("--warmup_epochs", type=int, default=5, help="Freeze CNN for first N epochs")
    ap.add_argument("--freeze_cnn_bn", action="store_true", help="Keep CNN BatchNorm frozen")

    # Model config (Transformer and fusion hyperparameters)
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

    # Parse all CLI args into args namespace
    args = ap.parse_args()

    # Resolving project root path as string (used by dataset to resolve image paths)
    project_root = str(Path(__file__).resolve().parents[1])
    # Preparing output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Setting RNG seeds and cuDNN settings
    set_seed(args.seed)

    # Choosing device and deciding whether AMP is enabled
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = bool(args.amp and device == "cuda")

    # Fixed class order used throughout training and evaluation
    class_names = ["glioma", "meningioma", "pituitary", "notumor"]

    # Keep my normalization 
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    # Building torchvision-style transforms (train has augmentation, eval is deterministic)
    train_tf = build_transforms(train=True, mean=mean, std=std)
    eval_tf = build_transforms(train=False, mean=mean, std=std)

    # Building datasets from CSV split files
    train_ds = BrainMRICSV(os.path.join(args.csv_dir, "train.csv"), class_names, train_tf, project_root)
    val_ds = BrainMRICSV(os.path.join(args.csv_dir, "val.csv"), class_names, eval_tf, project_root)
    test_ds = BrainMRICSV(os.path.join(args.csv_dir, "test.csv"), class_names, eval_tf, project_root)

    # Building DataLoaders:
    # - train: shuffle=True for SGD, drop_last=True to keep batch sizes consistent
    # - val/test: shuffle=False for stable evaluation
    # - pin_memory=True for faster H2D transfer on CUDA
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # Instantiate the hybrid CNN-Transformer model and move it to device
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
    if args.warmup_epochs > 0:
        set_requires_grad(model.cnn, False)

    # Build AdamW optimizer using parameter groups (differential LR and selective weight decay)
    optimizer = torch.optim.AdamW(
        build_param_groups(model, cnn_lr=args.cnn_lr, vit_lr=args.vit_lr, weight_decay=args.weight_decay)
    )
    # Cosine annealing schedule across the full epoch range initially
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # AMP gradient scaler (enabled only when use_amp=True)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Store epoch-wise metrics for CSV and plots
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_macro_f1": [],
        "epoch_time_sec": [],
    }

    # Early stopping trackers
    best_f1 = -1.0
    best_epoch = -1
    bad_epochs = 0

    # Measure overall training time
    start_train = time.time()

    # Main epoch loop
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Unfreeze after warmup
        if epoch == args.warmup_epochs + 1:
            # Allow CNN params to receive gradients again
            set_requires_grad(model.cnn, True)
            # Rebuild optimizer so CNN params are included with cnn_lr
            optimizer = torch.optim.AdamW(
                build_param_groups(model, cnn_lr=args.cnn_lr, vit_lr=args.vit_lr, weight_decay=args.weight_decay)
            )
            # Rebuild cosine scheduler for the remaining epochs (T_max = remaining steps)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.epochs - epoch + 1), eta_min=1e-6)
            # Recreate scaler (keeps AMP behavior consistent after optimizer rebuild)
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        # Set model to training mode
        model.train()

        # Freezing CNN BN stats if requested (often stabilizes transfer learning)
        if args.freeze_cnn_bn:
            model.freeze_cnn_bn()

        # Accumulators for training epoch
        total, correct, loss_sum = 0, 0, 0.0

        # Batch loop
        for x, y, _ in train_loader:
            # Move data to device
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Clear gradients (set_to_none=True can improve performance/memory)
            optimizer.zero_grad(set_to_none=True)

            # Forward and loss under autocast for AMP if enabled
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, _ = model(x)
                loss = ce(logits, y)

            # Backprop with GradScaler (handles scaling safely under AMP)
            scaler.scale(loss).backward()

            # Grad clip (very important for the val_loss spikes you saw)
            if args.grad_clip and args.grad_clip > 0:
                # Unscale gradients before clipping so clipping threshold is meaningful
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))

            # Optimizer step via scaler (AMP-safe)
            scaler.step(optimizer)
            scaler.update()

            # Compute predictions for accuracy (detach logits to avoid autograd tracking)
            prob = torch.softmax(logits.detach(), dim=1)
            pred = prob.argmax(dim=1)

            # Update running totals
            total += y.size(0)
            correct += (pred == y).sum().item()
            loss_sum += float(loss.item()) * y.size(0)

        # Computing epoch-level training loss/accuracy
        train_loss = loss_sum / max(total, 1)
        train_acc = correct / max(total, 1)

        # Running validation evaluation (no grads)
        val_stats = evaluate(model, val_loader, device, class_names)
        val_loss = val_stats["loss"]
        val_acc = val_stats["acc"]
        val_f1 = val_stats["macro_f1"]

        # Step LR scheduler once per epoch
        scheduler.step()

        # Record epoch metrics for plots/CSV
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))
        history["val_macro_f1"].append(float(val_f1))
        history["epoch_time_sec"].append(float(time.time() - t0))

        # Console log for progress monitoring
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macroF1={val_f1:.4f}"
        )

        # Early stopping on macro F1
        if val_f1 > best_f1:
            # Update best trackers
            best_f1 = val_f1
            best_epoch = epoch
            bad_epochs = 0

            # Build checkpoint dict:
            # - model weights
            # - class names and normalization (for consistent inference)
            # - args used for training
            # - minimal model config to rebuild model for prediction scripts
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
            # No improvement this epoch -> count toward patience
            bad_epochs += 1
            if bad_epochs >= args.patience:
                # Stop training when patience exceeded
                print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch}, best macroF1: {best_f1:.4f})")
                break

    # Compute and print total wall-clock training time
    total_train_time = time.time() - start_train
    print(f"Total training time (sec): {total_train_time:.1f}")

    # Save history
    # pandas is used here to write the history dict into a CSV file.
    import pandas as pd
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    # Plots
    # Save loss curves and accuracy curves from training history.
    plot_training(history, out_dir / "loss_curves.png")
    plot_accuracy(history, out_dir / "acc_curves.png")

    # Test eval using best model
    # Reload the best checkpoint and restore weights before running test evaluation.
    best = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best["model_state"])

    # Evaluate on test set and get arrays for confusion matrix and other metrics.
    test_stats = evaluate(model, test_loader, device, class_names)
    y_true = test_stats["y_true"]
    y_pred = test_stats["y_pred"]

    # Build confusion matrix in fixed label order
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    # Save confusion matrix plot
    plot_confusion(cm, class_names, out_dir / "confusion_matrix.png")

    # Extract classification report and compute additional metrics
    rep = test_stats["report"]
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    spec_macro, spec_per_class = specificity_macro(cm)

    # Prepare a single metrics dictionary to write to metrics.json
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

    # Write metrics.json to disk
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Final console messages summarizing output locations and best validation F1
    print("Saved outputs to:", str(out_dir))
    print("Best epoch:", best_epoch, "Best val macroF1:", best_f1)



# Standard script entrypoint:
# This ensures main() only runs when the file is executed directly.

if __name__ == "__main__":
    main()
