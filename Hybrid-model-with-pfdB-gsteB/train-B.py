# THIS IS THE TRAINING FILE FOR HYBRID B WITH PFB-GSTE VARIANT B

# Libraries I needed
import sys
from pathlib import Path

# Adding the project root (parent of this file’s folder) to Python’s import path 
# This allows imports like from models... and from scripts... to work when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# utilities for filesystem paths, saving JSON, and timing 
import os
import json
import time
from pathlib import Path

# Numerical and deep learning stack 
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Plotting 
import matplotlib.pyplot as plt

#  Evaluation metrics (classification report, confusion matrix, agreement and correlation scores) 
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    matthews_corrcoef,
)

#  Project model and dataset/transform utilities 
from models.hybrid_model import HybridResNet50V2_RViT
from scripts.data import BrainMRICSV, build_transforms


def set_seed(seed):
    
    # Set PyTorch RNG seed (CPU)
    torch.manual_seed(seed)
    # Set NumPy RNG seed
    np.random.seed(seed)
    # Set PyTorch RNG seed (all CUDA devices) for GPU reproducibility where possible
    torch.cuda.manual_seed_all(seed)
    # Allow non-deterministic CuDNN algorithms (often faster; not bitwise reproducible)
    torch.backends.cudnn.deterministic = False
    # Enable CuDNN auto-tuner for best performance on current hardware shapes
    torch.backends.cudnn.benchmark = True


@torch.no_grad()
def evaluate(model, loader, device, class_names):
    
    # Switch model to eval mode (disables dropout, uses BN running stats)
    model.eval()
    # Collect ground-truth labels, predicted labels, and predicted probabilities across the whole loader
    all_y, all_pred, all_prob = [], [], []
    # Running totals to compute average loss and accuracy
    total, correct, loss_sum = 0, 0, 0.0
    # Cross-entropy for evaluation loss (no label smoothing here)
    ce = nn.CrossEntropyLoss()

    # Iterate through batches produced by the DataLoader
    for x, y, _ in loader:
        # Move input batch to GPU/CPU device; non_blocking speeds transfer when pin_memory=True
        x = x.to(device, non_blocking=True)
        # Move labels to device as well
        y = y.to(device, non_blocking=True)
        # Forward pass: model returns (logits, extra) where extra may be None here
        logits, _ = model(x)
        # Compute batch CE loss
        loss = ce(logits, y)

        # Converting logits -> probabilities
        prob = torch.softmax(logits, dim=1)
        # Predicteding class = argmax probability
        pred = prob.argmax(dim=1)

        # Updating totals for accuracy/loss averaging
        total += y.size(0)
        correct += (pred == y).sum().item()
        loss_sum += loss.item() * y.size(0)

        # Saving arrays for metric computation on full set (move to CPU, convert to numpy)
        all_y.append(y.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
        all_prob.append(prob.cpu().numpy())

    # Concatenating list-of-batches into full arrays
    all_y = np.concatenate(all_y)
    all_pred = np.concatenate(all_pred)
    all_prob = np.concatenate(all_prob)

    # Accuracy = total correct / total samples (safe-guard against divide-by-zero)
    acc = correct / max(total, 1)
    # Average loss = sum(loss * batch_size) / total samples
    avg_loss = loss_sum / max(total, 1)

    # Build a full classification report (precision/recall/F1 per class and macro/weighted averages)
    rep = classification_report(
        all_y, all_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    # Extract the macro-average F1 score (used for early stopping)
    macro_f1 = rep["macro avg"]["f1-score"]

    # Return a dictionary of all evaluation outputs (metrics + arrays + report)
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
    
    # cm is expected to be an NxN confusion matrix (N = number of classes)
    n = cm.shape[0]
    specs = []
    # Compute specificity per class using one-vs-rest definition
    for c in range(n):
        # True positives for class c
        tp = cm[c, c]
        # False positives = predicted as c but actually not c
        fp = cm[:, c].sum() - tp
        # False negatives = actually c but predicted as not c
        fn = cm[c, :].sum() - tp
        # True negatives = everything else
        tn = cm.sum() - (tp + fp + fn)
        # Specificity = TN / (TN + FP); small epsilon prevents divide-by-zero
        spec = tn / (tn + fp + 1e-12)
        specs.append(spec)
    # Return macro-average specificity and the per-class list
    return float(np.mean(specs)), [float(s) for s in specs]


def plot_training(history, out_path):
    
    # X-axis epochs are 1..num_epochs_recorded
    epochs = list(range(1, len(history["train_loss"]) + 1))
    # Create a new figure for loss curves
    plt.figure()
    # Plot training loss per epoch
    plt.plot(epochs, history["train_loss"], label="train_loss")
    # Plot validation loss per epoch
    plt.plot(epochs, history["val_loss"], label="val_loss")
    # Label axes
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # Add legend for the two curves
    plt.legend()
    # Tight layout avoids label cutoffs in saved image
    plt.tight_layout()
    # Saving plot image to disk
    plt.savefig(out_path, dpi=200)
    # Close figure to free memory/resources
    plt.close()


def plot_accuracy(history, out_path):
    # X-axis epochs are 1..num_epochs_recorded
    epochs = list(range(1, len(history["train_acc"]) + 1))
    # Create a new figure for accuracy curves
    plt.figure()
    # Plot training accuracy per epoch
    plt.plot(epochs, history["train_acc"], label="train_acc")
    # Plot validation accuracy per epoch
    plt.plot(epochs, history["val_acc"], label="val_acc")
    # Label axes
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    # Add legend
    plt.legend()
    # Keep layout neat
    plt.tight_layout()
    # Saving plot image to disk
    plt.savefig(out_path, dpi=200)
    # Close the figure
    plt.close()


def plot_confusion(cm, class_names, out_path):
    
    # Creating a new figure for the confusion matrix
    plt.figure()
    # Showing confusion matrix as an image
    plt.imshow(cm, interpolation="nearest")
    # Title indicates this is the test confusion matrix
    plt.title("Confusion Matrix (Test)")
    # Adding colorbar to interpret intensity
    plt.colorbar()
    # Tick positions correspond to class indices
    ticks = np.arange(len(class_names))
    # X-axis labels are predicted classes
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    # Y-axis labels are true classes
    plt.yticks(ticks, class_names)

    # Threshold used to decide text color for readability against background
    thresh = cm.max() / 2.0
    # Draw the count value in each cell
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    # Axis labels
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    # Avoid cutting off labels
    plt.tight_layout()
    # Save image to disk
    plt.savefig(out_path, dpi=200)
    # Close figure
    plt.close()


def build_param_groups(model, cnn_lr, vit_lr, weight_decay):
    
    """
    Differential LR and no-weight-decay for biases/norm params.
    """
    # Helper to decide which params should NOT get weight decay
    def is_no_decay(name, p):
        
        # 1D parameters are typically biases or norm scale vectors -> no decay
        if p.ndim == 1:
            return True
        # Lowercase name for robust matching
        lname = name.lower()
        # Explicit bias parameters -> no decay
        if lname.endswith(".bias"):
            return True
        # BatchNorm/LayerNorm/Norm parameters -> no decay
        if "bn" in lname or "norm" in lname or "ln" in lname:
            return True
        return False

    # Split CNN parameters into decay vs no-decay
    cnn_decay, cnn_no = [], []
    # Split "rest of model" parameters into decay vs no-decay
    rest_decay, rest_no = [], []

    # Walk through all named parameters so we can separate cnn.* from others
    for name, p in model.named_parameters():
        # Skip frozen parameters (requires_grad=False)
        if not p.requires_grad:
            continue
        # CNN branch parameters are prefixed by "cnn."
        target_is_cnn = name.startswith("cnn.")
        # Route param to appropriate bucket depending on decay rule and branch
        if is_no_decay(name, p):
            (cnn_no if target_is_cnn else rest_no).append(p)
        else:
            (cnn_decay if target_is_cnn else rest_decay).append(p)

    # Build optimizer param groups with different LRs and decays
    groups = []
    if cnn_decay:
        groups.append({"params": cnn_decay, "lr": cnn_lr, "weight_decay": weight_decay})
    if cnn_no:
        groups.append({"params": cnn_no, "lr": cnn_lr, "weight_decay": 0.0})
    if rest_decay:
        groups.append({"params": rest_decay, "lr": vit_lr, "weight_decay": weight_decay})
    if rest_no:
        groups.append({"params": rest_no, "lr": vit_lr, "weight_decay": 0.0})

    # Return list ready to pass into AdamW
    return groups


def set_requires_grad(module, flag: bool):
    
    # Toggle whether parameters in a module should get gradients
    for p in module.parameters():
        p.requires_grad = flag


def main():
    
    # argparse is used so this script can be run from terminal with configurable options
    import argparse
    ap = argparse.ArgumentParser()
    # Location of CSV split files (train.csv/val.csv/test.csv)
    ap.add_argument("--csv_dir", type=str, default="data/splits/tightcrop")
    # Output folder where checkpoint, plots and metrics are written
    ap.add_argument("--out_dir", type=str, default="results/run_hybrid")

    # Training duration and mini-batch size
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)

    # Differential LR (big impact)
    ap.add_argument("--cnn_lr", type=float, default=1e-4)
    ap.add_argument("--vit_lr", type=float, default=5e-4)

    # Regularization and training control knobs
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
    ap.add_argument("--freeze_cnn_bn", action="store_true", help="Keep CNN BatchNorm frozen (recommended)")

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

    # Parse CLI args into a namespace
    args = ap.parse_args()

    # Resolve project root path string (used by BrainMRICSV to resolve image paths)
    project_root = str(Path(__file__).resolve().parents[1])
    # Creating output directory if missing
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fix seeds for repeatability (as far as possible under CuDNN benchmark=True)
    set_seed(args.seed)

    # Choosing device; AMP only makes sense on CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = bool(args.amp and device == "cuda")

    # Fixed class ordering used everywhere (datasets, reports, confusion matrix)
    class_names = ["glioma", "meningioma", "pituitary", "notumor"]

    # Keeping my normalization
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    # Build training-time transforms (augmentation and normalization)
    train_tf = build_transforms(train=True, mean=mean, std=std)
    # Build evaluation transforms (no augmentation, only normalization)
    eval_tf = build_transforms(train=False, mean=mean, std=std)

    # Creating datasets from CSV splits, mapping class names -> labels and loading images from project_root
    train_ds = BrainMRICSV(os.path.join(args.csv_dir, "train.csv"), class_names, train_tf, project_root)
    val_ds = BrainMRICSV(os.path.join(args.csv_dir, "val.csv"), class_names, eval_tf, project_root)
    test_ds = BrainMRICSV(os.path.join(args.csv_dir, "test.csv"), class_names, eval_tf, project_root)

    # DataLoader for training: shuffle=True for SGD-style training; drop_last=True keeps batch size consistent
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    # Validation DataLoader: no shuffle
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    # Test DataLoader: no shuffle
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # Build the hybrid CNN–Transformer model using CLI config; CNN is pretrained at init
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

    # Warmup: freeze CNN first (only transformer and head update early)
    if args.warmup_epochs > 0:
        set_requires_grad(model.cnn, False)

    # AdamW optimizer with differential learning rates and selective weight decay
    optimizer = torch.optim.AdamW(
        build_param_groups(model, cnn_lr=args.cnn_lr, vit_lr=args.vit_lr, weight_decay=args.weight_decay)
    )
    # Cosine annealing schedule over the full training length (T_max=epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Gradient scaler for AMP (enabled only when use_amp=True)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Training log storage (saved later to CSV and used for plots)
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_macro_f1": [],
        "epoch_time_sec": [],
    }

    # Track best validation macro-F1 for early stopping and checkpointing
    best_f1 = -1.0
    best_epoch = -1
    bad_epochs = 0

    # Measure total training runtime
    start_train = time.time()

    # Main epoch loop (1-indexed for nicer logs)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Unfreeze after warmup
        if epoch == args.warmup_epochs + 1:
            # Allow CNN backbone parameters to receive gradients
            set_requires_grad(model.cnn, True)
            # Rebuild optimizer so CNN params are included with cnn_lr
            optimizer = torch.optim.AdamW(
                build_param_groups(model, cnn_lr=args.cnn_lr, vit_lr=args.vit_lr, weight_decay=args.weight_decay)
            )
            # Rebuild cosine schedule for the remaining epochs only
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.epochs - epoch + 1), eta_min=1e-6)
            # Reset scaler (optional safety; keeps AMP state consistent)
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        # Switch model to training mode (enables dropout, BN updates unless frozen)
        model.train()

        # Freeze CNN BN stats if requested (often stabilizes transfer learning)
        if args.freeze_cnn_bn:
            model.freeze_cnn_bn()

        # Running totals for this epoch’s training metrics
        total, correct, loss_sum = 0, 0, 0.0

        # Mini-batch training loop
        for x, y, _ in train_loader:
            # Move batch to device
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Clear old gradients efficiently
            optimizer.zero_grad(set_to_none=True)

            # Forward pass under autocast if AMP is enabled
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, _ = model(x)
                loss = ce(logits, y)

            # Backward pass (scaled if AMP enabled)
            scaler.scale(loss).backward()

            # Grad clip (very important for the val_loss spikes you saw)
            if args.grad_clip and args.grad_clip > 0:
                # Unscale grads before clipping so clipping is done in true scale
                scaler.unscale_(optimizer)
                # Clip global norm across all parameters
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))

            # Optimizer step via scaler (handles skipping on inf/nan)
            scaler.step(optimizer)
            # Update scale factor for next iteration
            scaler.update()

            # Computing predictions for accuracy (detach logits to avoid graph usage)
            prob = torch.softmax(logits.detach(), dim=1)
            pred = prob.argmax(dim=1)

            # Updating totals for epoch-level loss/accuracy
            total += y.size(0)
            correct += (pred == y).sum().item()
            loss_sum += float(loss.item()) * y.size(0)

        # Epoch-level train metrics
        train_loss = loss_sum / max(total, 1)
        train_acc = correct / max(total, 1)

        # Validation evaluation (no-grad) to compute val loss/acc/macro-F1
        val_stats = evaluate(model, val_loader, device, class_names)
        val_loss = val_stats["loss"]
        val_acc = val_stats["acc"]
        val_f1 = val_stats["macro_f1"]

        # Step LR schedule once per epoch
        scheduler.step()

        # Store metrics for plots/CSV
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))
        history["val_macro_f1"].append(float(val_f1))
        history["epoch_time_sec"].append(float(time.time() - t0))

        # Print progress line for this epoch
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macroF1={val_f1:.4f}"
        )

        # Early stopping on macro F1
        if val_f1 > best_f1:
            # New best validation macro-F1 found
            best_f1 = val_f1
            best_epoch = epoch
            bad_epochs = 0

            # Package checkpoint with weights and metadata needed for inference/reproducibility
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
            # Saving best checkpoint to disk
            torch.save(ckpt, out_dir / "best_model.pt")
        else:
            # No improvement this epoch -> count toward patience
            bad_epochs += 1
            if bad_epochs >= args.patience:
                # Stop training once patience is exceeded
                print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch}, best macroF1: {best_f1:.4f})")
                break

    # Total runtime for training loop (may stop early)
    total_train_time = time.time() - start_train
    print(f"Total training time (sec): {total_train_time:.1f}")

    # Saving history
    import pandas as pd
    # Write epoch-by-epoch metrics to CSV for reporting/plotting later
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    # Plots
    # Save loss curve figure
    plot_training(history, out_dir / "loss_curves.png")
    # Save accuracy curve figure
    plot_accuracy(history, out_dir / "acc_curves.png")

    # Test eval using best model
    # Reload best checkpoint (ensures test metrics correspond to best val macro-F1)
    best = torch.load(out_dir / "best_model.pt", map_location=device)
    # Load best weights into the current model instance
    model.load_state_dict(best["model_state"])

    # Run evaluation on the held-out test set
    test_stats = evaluate(model, test_loader, device, class_names)
    y_true = test_stats["y_true"]
    y_pred = test_stats["y_pred"]

    # Confusion matrix in fixed class order [0..num_classes-1]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    # Save confusion matrix plot image
    plot_confusion(cm, class_names, out_dir / "confusion_matrix.png")

    # Extract full classification report dict from evaluation output
    rep = test_stats["report"]
    # Cohen’s kappa measures agreement beyond chance
    kappa = cohen_kappa_score(y_true, y_pred)
    # MCC is a correlation-like score (extended here via sklearn’s multiclass handling)
    mcc = matthews_corrcoef(y_true, y_pred)
    # Compute macro and per-class specificity from confusion matrix
    spec_macro, spec_per_class = specificity_macro(cm)

    # Collect final metrics into a JSON-serializable dict for the report

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

    # Write metrics.json to output directory
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Print where outputs were saved + best validation summary
    print("Saved outputs to:", str(out_dir))
    print("Best epoch:", best_epoch, "Best val macroF1:", best_f1)


# Standard script entry point so main() runs only when file is executed directly
if __name__ == "__main__":
    main()
