# scripts/train.py
import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
from tqdm import tqdm

from scripts.utils import set_seed, get_device, AvgMeter, ensure_dir, count_parameters, now
from scripts.data import make_loaders
from scripts.metrics import confusion_matrix, summarize_metrics
from scripts.plots import plot_curves, plot_confusion_matrix

from models.hybrid import HybridResNet50V2_RViT_PFD_GSTE


def run_one_epoch(model, loader, optimizer, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    loss_meter = AvgMeter()
    correct = 0
    total = 0

    ce = nn.CrossEntropyLoss()

    for x, y, _paths in tqdm(loader, leave=False):
        x = x.to(device)
        y = y.to(device)

        with torch.set_grad_enabled(train):
            logits, _probs, _conf = model(x, collect_attn=False)
            loss = ce(logits, y)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        loss_meter.update(loss.item(), k=x.size(0))
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(x.size(0))

    acc = correct / max(1, total)
    return loss_meter.avg, acc


@torch.no_grad()
def evaluate_full(model, loader, device, num_classes):
    model.eval()
    all_t = []
    all_p = []

    for x, y, _paths in tqdm(loader, leave=False):
        x = x.to(device)
        logits, _probs, _conf = model(x, collect_attn=False)
        pred = logits.argmax(dim=1).cpu().numpy()
        all_p.append(pred)
        all_t.append(y.numpy())

    y_true = np.concatenate(all_t)
    y_pred = np.concatenate(all_p)

    cm = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    metrics = summarize_metrics(cm)
    return cm, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/processed/tightcrop")
    ap.add_argument("--splits_dir", type=str, default="data/splits/tightcrop")
    ap.add_argument("--out_dir", type=str, default="results")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)          # Krishnan LR
    ap.add_argument("--weight_decay", type=float, default=1e-2) # Krishnan WD
    ap.add_argument("--seed", type=int, default=22089065)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--norm_mode", type=str, default="0.5", choices=["0.5", "imagenet"])
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()

    ensure_dir(args.out_dir)
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    ensure_dir(ckpt_dir)

    train_loader, val_loader, test_loader, idx_to_class = make_loaders(
        data_root=args.data_root,
        splits_dir=args.splits_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        norm_mode=args.norm_mode
    )
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    num_classes = len(class_names)

    model = HybridResNet50V2_RViT_PFD_GSTE(
        num_classes=num_classes,
        embed_dim=142,
        depth=10,
        heads=10,
        mlp_dim=480,
        attn_dropout=0.1,
        patch_size_px=16,
        feat_stride=16,
        fusion_hidden=512,
        dropout_p=0.5
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("Device:", device)
    print("Classes:", class_names)
    print("#Params:", count_parameters(model))

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    best_val_f1 = -1.0
    best_path = os.path.join(ckpt_dir, "best.pt")

    t0 = now()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_one_epoch(model, train_loader, opt, device, train=True)
        va_loss, va_acc = run_one_epoch(model, val_loader, opt, device, train=False)

        cm_val, m_val = evaluate_full(model, val_loader, device, num_classes=num_classes)
        val_f1 = float(m_val["macro_f1"])

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} macroF1 {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "model_state": model.state_dict(),
                "class_names": class_names,
                "args": vars(args),
                "epoch": epoch
            }, best_path)

    total_train_time = now() - t0
    print("Total training time (sec):", round(total_train_time, 2))

    plot_curves(history, args.out_dir)

    # Load best and evaluate test
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    cm_test, m_test = evaluate_full(model, test_loader, device, num_classes=num_classes)

    print("\n=== TEST METRICS ===")
    print("ACC:", round(m_test["acc"], 4))
    print("Macro Precision:", round(m_test["macro_precision"], 4))
    print("Macro Recall/Sensitivity:", round(m_test["macro_recall"], 4))
    print("Macro F1:", round(m_test["macro_f1"], 4))
    print("Macro Specificity:", round(m_test["macro_specificity"], 4))
    print("Cohen Kappa:", round(m_test["kappa"], 4))
    print("MCC:", round(m_test["mcc"], 4))

    # Save confusion matrix plot
    plot_confusion_matrix(cm_test, class_names, os.path.join(args.out_dir, "confusion_matrix_test.png"))

    # Save metrics as text
    with open(os.path.join(args.out_dir, "test_metrics.txt"), "w") as f:
        f.write("Classes:\n" + str(class_names) + "\n\n")
        f.write("Confusion Matrix:\n" + str(cm_test) + "\n\n")
        for k, v in m_test.items():
            if isinstance(v, np.ndarray):
                f.write(f"{k}: {v.tolist()}\n")
            else:
                f.write(f"{k}: {v}\n")
        f.write(f"\n#Params: {count_parameters(model)}\n")
        f.write(f"Total training time (sec): {total_train_time}\n")

    print("\nSaved outputs to:", args.out_dir)


if __name__ == "__main__":
    main()
