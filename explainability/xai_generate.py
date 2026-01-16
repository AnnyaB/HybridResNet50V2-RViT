# scripts/xai_generate.py
import os
import argparse
import random

import torch

from scripts.utils import get_device, ensure_dir, set_seed
from scripts.data import make_loaders
from models.hybrid import HybridResNet50V2_RViT_PFD_GSTE
from explainability.xai import GradCAMPlusPlus, AttentionRollout, upsample_token_map, overlay_heatmap_on_image


@torch.no_grad()
def pick_samples(loader, n=6):
    batch = next(iter(loader))
    x, y, paths = batch
    idxs = list(range(x.size(0)))
    random.shuffle(idxs)
    idxs = idxs[:min(n, len(idxs))]
    return x[idxs], y[idxs], [paths[i] for i in idxs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/processed/tightcrop")
    ap.add_argument("--splits_dir", type=str, default="data/splits/tightcrop")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="results_xai")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=22089065)
    ap.add_argument("--norm_mode", type=str, default="0.5", choices=["0.5", "imagenet"])
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    ensure_dir(args.out_dir)

    train_loader, val_loader, test_loader, idx_to_class = make_loaders(
        args.data_root, args.splits_dir, batch_size=32, num_workers=2, norm_mode=args.norm_mode
    )
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    num_classes = len(class_names)

    model = HybridResNet50V2_RViT_PFD_GSTE(num_classes=num_classes).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    campp = GradCAMPlusPlus(model)
    rollout = AttentionRollout()

    x, y, paths = pick_samples(test_loader, n=args.n)
    for i in range(x.size(0)):
        xi = x[i:i+1].to(device)

        # Grad-CAM++ on CNN/PFD side
        heat_cam, cls_cam = campp(xi, target_class=None)

        # Attention rollout on transformer side
        tok_map, logits = rollout(model, xi)
        heat_attn = upsample_token_map(tok_map, out_hw=(224, 224))

        pred = int(logits.argmax(dim=1).item())
        title_cam = f"Grad-CAM++ | pred={class_names[pred]}"
        title_attn = f"Attention Rollout | pred={class_names[pred]}"

        overlay_heatmap_on_image(xi.cpu(), heat_cam, os.path.join(args.out_dir, f"{i:02d}_gradcampp.png"), title_cam)
        overlay_heatmap_on_image(xi.cpu(), heat_attn, os.path.join(args.out_dir, f"{i:02d}_attnrollout.png"), title_attn)

    print("Saved XAI images to:", args.out_dir)


if __name__ == "__main__":
    main()
