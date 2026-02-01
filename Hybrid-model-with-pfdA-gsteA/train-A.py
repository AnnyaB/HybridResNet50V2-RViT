

# scripts/train.py

#   Train,  validate and test the HybridResNet50V2_RViT model (PFDA-GSTEA style),
#   then save the best checkpoint (by val macro-F1) plus plots and metrics.
#
# Key ideas in this script:
#   - i trained using train.csv, validate using val.csv, and final test using test.csv.
#   - did early stopping based on validation macro-F1 (not just accuracy).
#   - did transfer learning: CNN starts pretrained, and we can freeze it for warmup epochs.
#   - used differential learning rates: smaller LR for CNN, bigger LR for the new transformer/fusion parts.
#   - saved training curves, confusion matrix, and a metrics.json report.
#
# Result (what we get after running):
#   In out_dir/ you will typically see:
#     - best_model.pt           (best checkpoint based on val macro-F1)
#     - history.csv             (epoch-by-epoch loss/acc/f1/time)
#     - loss_curves.png         (train vs val loss curve)
#     - acc_curves.png          (train vs val accuracy curve)
#     - confusion_matrix.png    (test confusion matrix)
#     - metrics.json            (final test metrics + extras like MCC/Kappa/specificity)
#
# How to run (eg demo):
#   python scripts/train.py --csv_dir data/splits/tightcrop --out_dir results/run_hybrid

import sys  # i need to edit sys.path so imports work when running as a script
from pathlib import Path  # convenient path handling

# Finding project root (two levels above this file: .../project/scripts/train.py -> .../project/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Added project root to Python import path so from models... works from anywhere
sys.path.append(str(PROJECT_ROOT))

import os  # OS utilities (path join, etc.)
import json  # to save metrics/config nicely
import time  # for epoch timing and total training time

import numpy as np  # numerical arrays and concatenation for metrics
import torch  # PyTorch core
import torch.nn as nn  # PyTorch layers, losses
from torch.utils.data import DataLoader  # mini-batch loading

import matplotlib.pyplot as plt  # plotting curves and confusion matrix

# sklearn metrics for detailed evaluation
from sklearn.metrics import (
    confusion_matrix,          # confusion matrix on test set
    classification_report,      # per-class precision/recall/F1
    cohen_kappa_score,          # agreement score
    matthews_corrcoef,          # MCC (useful robust metric)
)

# Importing my hybrid model (CNN-Transformer)
from models.hybrid_model import HybridResNet50V2_RViT

# Importing my dataset class and transform builder
from scripts.data import BrainMRICSV, build_transforms


def set_seed(seed):
    
    """
    Make experiments more repeatable by setting random seeds.
    Note: cuDNN benchmark=True trades exact determinism for speed (common practice).
    """
    torch.manual_seed(seed)             # seed PyTorch CPU RNG
    np.random.seed(seed)                # seed NumPy RNG
    torch.cuda.manual_seed_all(seed)    # seed all CUDA devices (if using GPU, I used GPU)

    # Deterministic=False allows faster kernels; Benchmark=True picks best kernels for my shapes
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


@torch.no_grad()
def evaluate(model, loader, device, class_names):
    
    """
    Run evaluation on a DataLoader (val or test) without gradients.
    Returns loss, accuracy, macro-F1, plus arrays for y_true/y_pred/y_prob.
    """
    model.eval()  # set eval mode: dropout off, BN uses running stats

    # I'll collect outputs across all batches so we can compute full-dataset metrics
    all_y, all_pred, all_prob = [], [], []

    # Running totals for scalar metrics
    total, correct, loss_sum = 0, 0, 0.0

    # Plain cross entropy for evaluation (no label smoothing here)
    ce = nn.CrossEntropyLoss()

    # Iterating over batches produced by the loader
    for x, y, _ in loader:
        # Move tensors to GPU/CPU device. non_blocking=True helps if pin_memory=True in DataLoader
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Forward pass: model returns (logits, xai_dict_or_none)
        logits, _ = model(x)

        # Computing loss for this batch
        loss = ce(logits, y)

        # Converting logits -> probabilities for predictions/metrics
        prob = torch.softmax(logits, dim=1)

        # Predicted class is argmax probability
        pred = prob.argmax(dim=1)

        # Updating running counts
        total += y.size(0)
        correct += (pred == y).sum().item()
        loss_sum += loss.item() * y.size(0)  # multiply by batch size so average later is correct

        # Storing batch outputs on CPU as numpy for sklearn
        all_y.append(y.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
        all_prob.append(prob.cpu().numpy())

    # Concatenating lists of arrays into single arrays
    all_y = np.concatenate(all_y)
    all_pred = np.concatenate(all_pred)
    all_prob = np.concatenate(all_prob)

    # Computing accuracy and average loss over dataset
    acc = correct / max(total, 1)
    avg_loss = loss_sum / max(total, 1)

    # classification_report gives per-class and macro/weighted averages
    rep = classification_report(
        all_y, all_pred,
        target_names=class_names,
        output_dict=True,     # dict so we can read values programmatically
        zero_division=0       # avoid warnings if a class has no predictions
    )

    # Macro F1 averages F1 equally over classes (good when classes may be imbalanced)
    macro_f1 = rep["macro avg"]["f1-score"]

    # Returning everything in a nice dict
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
    
    """
    Computing macro specificity from a confusion matrix.
    Specificity for class c = TN / (TN + FP).
    I computeD per-class specificity and then macro-average it.
    """
    n = cm.shape[0]  # number of classes
    specs = []       # collect per-class specificities

    for c in range(n):
        # TP is diagonal entry
        tp = cm[c, c]

        # FP: predicted as class c but actually other classes
        fp = cm[:, c].sum() - tp

        # FN: actually class c but predicted as other classes
        fn = cm[c, :].sum() - tp

        # TN: everything else
        tn = cm.sum() - (tp + fp + fn)

        # Added tiny epsilon to avoid divide-by-zero
        spec = tn / (tn + fp + 1e-12)
        specs.append(spec)

    # Returning macro average and per-class list
    return float(np.mean(specs)), [float(s) for s in specs]


def plot_training(history, out_path):
    
    """
    Plot train vs val loss curve across epochs and save to out_path.
    """
    epochs = list(range(1, len(history["train_loss"]) + 1))  # x-axis: 1..num_epochs_logged

    plt.figure()  # start a fresh figure
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)  # save a crisp image
    plt.close()  # close to free memory


def plot_accuracy(history, out_path):
    
    """
    Plot train vs val accuracy curve across epochs and save to out_path.
    """
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
    
    """
    Plot and save a confusion matrix heatmap (for test set).
    """
    plt.figure()
    plt.imshow(cm, interpolation="nearest")  # show matrix as an image
    plt.title("Confusion Matrix (Test)")
    plt.colorbar()

    # Tick labels are class names
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)

    # Decide text color threshold: bright text on dark cells, dark text on light cells
    thresh = cm.max() / 2.0

    # Put numbers inside each cell
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def build_param_groups(model, cnn_lr, vit_lr, weight_decay):
    
    """
    Build optimizer parameter groups:
      - CNN params get cnn_lr
      - everything else gets vit_lr
      - biases and normalization params get NO weight decay (weight_decay=0)
        (this is a common transformer/CNN fine-tuning trick)
    """
    def is_no_decay(name, p):
        # Rule 1: 1D parameters are usually biases or norm scales -> no decay
        if p.ndim == 1:
            return True

        lname = name.lower()

        # Rule 2: explicit bias parameters -> no decay
        if lname.endswith(".bias"):
            return True

        # Rule 3: batchnorm/layernorm/etc -> no decay
        if "bn" in lname or "norm" in lname or "ln" in lname:
            return True

        return False

    # I'll separate CNN parameters and non-CNN parameters
    cnn_decay, cnn_no = [], []
    rest_decay, rest_no = [], []

    # Walk all trainable parameters with their names
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue  # skip frozen params

        # Identify CNN params by name prefix (because in Hybrid model CNN is stored as model.cnn)
        target_is_cnn = name.startswith("cnn.")

        # Put parameter into correct bucket based on decay rule
        if is_no_decay(name, p):
            (cnn_no if target_is_cnn else rest_no).append(p)
        else:
            (cnn_decay if target_is_cnn else rest_decay).append(p)

    # Build param group dicts for AdamW
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


def set_requires_grad(module, flag: bool):
    
    """
    Freeze/unfreeze a whole module by setting requires_grad on all its parameters.
    """
    for p in module.parameters():
        p.requires_grad = flag


def main():
    # argparse lets me run this script with different settings from the command line
    import argparse
    ap = argparse.ArgumentParser()

    # Where my CSV splits live and where to save outputs
    ap.add_argument("--csv_dir", type=str, default="data/splits/tightcrop")
    ap.add_argument("--out_dir", type=str, default="results/run_hybrid")

    # Basic training parameters
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)

    # Differential learning rates:
    # - CNN is pretrained, so I move it gently (smaller LR)
    # - Transformer/fusion/PFD are newer, so I let them learn faster (bigger LR)
    ap.add_argument("--cnn_lr", type=float, default=1e-4)
    ap.add_argument("--vit_lr", type=float, default=5e-4)

    # Regularization, early stopping and reproducibility
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=10)  # how many no improvement epochs before stopping
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=2)  # dataloader worker processes

    # Stabilizers / training tricks
    ap.add_argument("--amp", action="store_true", help="Use mixed precision (CUDA)")
    ap.add_argument("--grad_clip", type=float, default=1.0)  # clip gradients to avoid exploding updates
    ap.add_argument("--label_smoothing", type=float, default=0.05)  # smoother targets -> often better generalization

    # Fine-tuning strategy
    ap.add_argument("--warmup_epochs", type=int, default=5, help="Freeze CNN for first N epochs")
    ap.add_argument("--freeze_cnn_bn", action="store_true", help="Keep CNN BatchNorm frozen (recommended)")

    # Model hyperparameters (according to my HybridResNet50V2_RViT constructor)
    ap.add_argument("--cnn_name", type=str, default="resnetv2_50x1_bitm")
    ap.add_argument("--patch_size", type=int, default=16)   # stored knob (PFDA doesn't patchify raw image)
    ap.add_argument("--embed_dim", type=int, default=142)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--heads", type=int, default=10)
    ap.add_argument("--mlp_dim", type=int, default=480)
    ap.add_argument("--attn_dropout", type=float, default=0.1)
    ap.add_argument("--vit_dropout", type=float, default=0.1)
    ap.add_argument("--fusion_dim", type=int, default=256)
    ap.add_argument("--fusion_dropout", type=float, default=0.5)

    # Parse CLI args
    args = ap.parse_args()

    # Computed project root and ensure output directory exists
    project_root = str(Path(__file__).resolve().parents[1])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Seed everything
    set_seed(args.seed)

    # Choose device (GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # AMP only makes sense on CUDA
    use_amp = bool(args.amp and device == "cuda")

    # Class order used in training/eval (must match dataset labels)
    class_names = ["glioma", "meningioma", "pituitary", "notumor"]

    # Normalization used in transforms
    # (I kept (0.5,0.5,0.5) which maps [0,1] -> [-1,1] roughly)
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    # Build transforms:
    # - train_tf includes augmentation and noise
    # - eval_tf has no augmentation, just ToTensor and Normalize
    train_tf = build_transforms(train=True, mean=mean, std=std)
    eval_tf = build_transforms(train=False, mean=mean, std=std)

    # Build datasets from CSV files
    train_ds = BrainMRICSV(os.path.join(args.csv_dir, "train.csv"), class_names, train_tf, project_root)
    val_ds   = BrainMRICSV(os.path.join(args.csv_dir, "val.csv"),   class_names, eval_tf,  project_root)
    test_ds  = BrainMRICSV(os.path.join(args.csv_dir, "test.csv"),  class_names, eval_tf,  project_root)

    # Wrap datasets in dataloaders to get mini-batches
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,                   # shuffle for SGD training
        num_workers=args.num_workers,
        pin_memory=True                 # speeds host->GPU transfer
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,                  # no shuffle for evaluation
        num_workers=args.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Instantiate the hybrid model
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
        rotations=(0, 1, 2, 3),         # 4 rotations used inside the model forward
        cnn_name=args.cnn_name,
        cnn_pretrained=True,            # start from pretrained CNN weights
    ).to(device)                        # move model to GPU/CPU

    # Cross entropy loss for training with label smoothing
    ce = nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))

    # Warmup strategy: freeze CNN at start, so new modules learn "on top of fixed features"
    if args.warmup_epochs > 0:
        set_requires_grad(model.cnn, False)

    # AdamW optimizer with differential LR and selective weight decay
    optimizer = torch.optim.AdamW(
        build_param_groups(model, cnn_lr=args.cnn_lr, vit_lr=args.vit_lr, weight_decay=args.weight_decay)
    )

    # Cosine LR schedule: gradually reduces LR over time
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    # AMP scaler (only active if use_amp=True)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # History dict for plots and CSV logging
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_macro_f1": [],
        "epoch_time_sec": [],
    }

    # Early stopping trackers
    best_f1 = -1.0        # best validation macro-F1 so far
    best_epoch = -1       # epoch number where best_f1 happened
    bad_epochs = 0        # how many epochs since last improvement

    start_train = time.time()  # start total timer

    # =========================
    # Main training loop
    # =========================
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()  # timer for this epoch

        # If warmup is over, unfreeze CNN at the right moment
        if epoch == args.warmup_epochs + 1:
            set_requires_grad(model.cnn, True)

            # Important: rebuild optimizer so CNN params are now included with cnn_lr groups
            optimizer = torch.optim.AdamW(
                build_param_groups(model, cnn_lr=args.cnn_lr, vit_lr=args.vit_lr, weight_decay=args.weight_decay)
            )

            # Reset scheduler to count remaining epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=(args.epochs - epoch + 1),
                eta_min=1e-6
            )

            # Recreate scaler (not strictly required, but fine)
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        model.train()  # training mode: dropout on, BN updates stats (unless we freeze BN)

        # Optionally freeze CNN BatchNorm stats (stabilizes transfer learning)
        if args.freeze_cnn_bn:
            model.freeze_cnn_bn()

        # Running totals for training stats this epoch
        total, correct, loss_sum = 0, 0, 0.0

        # Iterate over mini-batches
        for x, y, _ in train_loader:
            # Move batch to device
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Clear old gradients
            optimizer.zero_grad(set_to_none=True)

            # Forward pass under autocast if AMP is enabled
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, _ = model(x)     # forward -> class logits
                loss = ce(logits, y)     # compute training loss

            # Backprop: scale loss first if AMP is enabled
            scaler.scale(loss).backward()

            # Gradient clipping: prevent huge updates (stability)
            if args.grad_clip and args.grad_clip > 0:
                # Unscale before clipping so clip acts on real gradient values
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(args.grad_clip)
                )

            # Optimizer step + scaler update (AMP-safe)
            scaler.step(optimizer)
            scaler.update()

            # Compute accuracy for this batch (detach to avoid graph)
            prob = torch.softmax(logits.detach(), dim=1)
            pred = prob.argmax(dim=1)

            # Update epoch totals
            total += y.size(0)
            correct += (pred == y).sum().item()
            loss_sum += float(loss.item()) * y.size(0)

        # Average training metrics for this epoch
        train_loss = loss_sum / max(total, 1)
        train_acc = correct / max(total, 1)

        # Validation step (no gradients)
        val_stats = evaluate(model, val_loader, device, class_names)
        val_loss = val_stats["loss"]
        val_acc  = val_stats["acc"]
        val_f1   = val_stats["macro_f1"]

        # Step the LR scheduler after the epoch
        scheduler.step()

        # Save history for plots/CSV
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))
        history["val_macro_f1"].append(float(val_f1))
        history["epoch_time_sec"].append(float(time.time() - t0))

        # Print quick epoch summary
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macroF1={val_f1:.4f}"
        )

        # =========================
        # Early stopping logic (macro-F1)
        # =========================
        if val_f1 > best_f1:
            # Improvement! Save best checkpoint
            best_f1 = val_f1
            best_epoch = epoch
            bad_epochs = 0

            # Build checkpoint dict (weights + metadata so inference knows classes/normalization/config)
            ckpt = {
                "model_state": model.state_dict(),   # the actual weights
                "class_names": class_names,          # label order
                "mean": mean,                        # normalization used
                "std": std,
                "train_args": vars(args),            # training CLI args
                "model_cfg": {                       # model hyperparams for reconstruction at inference time
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

            # Save checkpoint to disk
            torch.save(ckpt, out_dir / "best_model.pt")
        else:
            # No improvement this epoch
            bad_epochs += 1

            # If we have not improved for "patience" epochs, stop training
            if bad_epochs >= args.patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(best epoch: {best_epoch}, best macroF1: {best_f1:.4f})"
                )
                break

    # Total training time for the whole run
    total_train_time = time.time() - start_train
    print(f"Total training time (sec): {total_train_time:.1f}")

    # =========================
    # Save history as CSV
    # =========================
    import pandas as pd  # imported here so pandas is only required when saving history
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    # =========================
    # Save plots
    # =========================
    plot_training(history, out_dir / "loss_curves.png")
    plot_accuracy(history, out_dir / "acc_curves.png")

    # =========================
    # Test evaluation using best checkpoint
    # =========================
    best = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best["model_state"])

    test_stats = evaluate(model, test_loader, device, class_names)
    y_true = test_stats["y_true"]
    y_pred = test_stats["y_pred"]

    # Confusion matrix on test set
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plot_confusion(cm, class_names, out_dir / "confusion_matrix.png")

    # Pull more metrics from report
    rep = test_stats["report"]
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    spec_macro, spec_per_class = specificity_macro(cm)

    # Build a clean metrics dict to save as JSON
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

    # Save metrics.json for citing results cleanly in my report later
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Print summary paths and best epoch
    print("Saved outputs to:", str(out_dir))
    print("Best epoch:", best_epoch, "Best val macroF1:", best_f1)


# Standard Python entry point: run main() only when called as a script
if __name__ == "__main__":
    main()
