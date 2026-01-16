# scripts/evaluate.py
import os
import time
import argparse
import numpy as np

import torch
from tqdm import tqdm

from scripts.utils import get_device, count_parameters, ensure_dir
from scripts.data import make_loaders
from scripts.metrics import confusion_matrix, summarize_metrics
from scripts.plots import plot_confusion_matrix

from models.hybrid import HybridResNet50V2_RViT_PFD_GSTE


@torch.no_grad()
def inference_latency_ms(model, device, input_shape=(1, 3, 224, 224), iters=100):
    x = torch.randn(*input_shape, device=device)
    model.eval()

    # Warmup
    for _ in range(10):
        model(x, collect_attn=False)

    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.time()
    for _ in range(iters):
        model(x, collect_attn=False)
    torch.cuda.synchronize() if device.type == "cuda" else None
    t1 = time.time()

    ms = (t1 - t0) * 1000.0 / iters
    return ms


@torch.no_grad()
def eval_loader(model, loader, device, num_classes):
    model.eval()
    yt, yp = [], []
    for x, y, _p in tqdm(loader, leave=False):
        x = x.to(device)
        logits, _probs, _conf = model(x, collect_attn=False)
        pred = logits.argmax(dim=1).cpu().numpy()
        yp.append(pred)
        yt.append(y.numpy())
    y_true = np.concatenate(yt)
    y_pred = np.concatenate(yp)
    cm = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    return cm, summarize_metrics(cm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/processed/tightcrop")
    ap.add_argument("--splits_dir", type=str, default="data/splits/tightcrop")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="results_eval")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--norm_mode", type=str, default="0.5", choices=["0.5", "imagenet"])
    ap.add_argument("--latency", action="store_true")
    args = ap.parse_args()

    device = get_device()
    ensure_dir(args.out_dir)

    train_loader, val_loader, test_loader, idx_to_class = make_loaders(
        args.data_root, args.splits_dir, batch_size=args.batch_size, num_workers=args.num_workers, norm_mode=args.norm_mode
    )
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    num_classes = len(class_names)

    model = HybridResNet50V2_RViT_PFD_GSTE(num_classes=num_classes).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    cm, m = eval_loader(model, test_loader, device, num_classes=num_classes)
    plot_confusion_matrix(cm, class_names, os.path.join(args.out_dir, "confusion_matrix.png"))

    with open(os.path.join(args.out_dir, "metrics.txt"), "w") as f:
        f.write("Classes: " + str(class_names) + "\n\n")
        f.write("CM:\n" + str(cm) + "\n\n")
        for k, v in m.items():
            if isinstance(v, np.ndarray):
                f.write(f"{k}: {v.tolist()}\n")
            else:
                f.write(f"{k}: {v}\n")
        f.write(f"\n#Params: {count_parameters(model)}\n")

        if args.latency:
            ms = inference_latency_ms(model, device)
            f.write(f"Inference latency (ms) per forward (B=1): {ms}\n")

    print("Saved to:", args.out_dir)


if __name__ == "__main__":
    main()
