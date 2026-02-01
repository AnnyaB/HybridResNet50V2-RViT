

# scripts/predict_xai_without_pfd_gste.py
#
# STRICT "WITHOUT PFD+GSTE" version:
# - Forces use_pfd_gste=False when rebuilding model.
# - Grad-CAM++ from CNN backbone feature map output (model.cnn output).
# - Attention rollout uses returned attention list; token grid is fixed 14x14 (224/16).

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.hybrid_model import HybridResNet50V2_RViT


def preprocess_pil(img, mean, std):
    img = img.convert("RGB")
    x = torch.from_numpy(np.array(img)).float() / 255.0
    x = x.permute(2, 0, 1).contiguous()
    mean_t = torch.tensor(mean, dtype=x.dtype).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=x.dtype).view(3, 1, 1)
    x = (x - mean_t) / std_t
    return x.unsqueeze(0)


def normalize_map(h):
    h = h - h.min()
    h = h / (h.max() + 1e-8)
    return h


def gradcam_pp_from_activations(A, grads):
    if A is None:
        raise RuntimeError("Activation A is None (hook failed).")
    if grads is None:
        raise RuntimeError("Gradients missing. Ensure enable_grad + backward ran.")

    grad_1 = grads
    grad_2 = grad_1 ** 2
    grad_3 = grad_2 * grad_1

    spatial_sum = (A * grad_3).sum(dim=(2, 3), keepdim=True)
    denom = 2.0 * grad_2 + spatial_sum + 1e-8
    alpha = grad_2 / denom

    w = (alpha * F.relu(grad_1)).sum(dim=(2, 3), keepdim=True)
    cam = (w * A).sum(dim=1, keepdim=True)
    cam = F.relu(cam).squeeze(0).squeeze(0)
    return normalize_map(cam)


def attention_rollout(attn_list, side=14, eps=1e-6):
    if not attn_list:
        raise RuntimeError("No attention matrices captured. Ensure return_xai=True.")

    mats = []
    for attn in attn_list:
        # attn: (B, H, N, N)
        a = attn.mean(dim=1)  # (B, N, N)
        n = a.shape[-1]
        a = a + torch.eye(n, device=a.device).unsqueeze(0)
        a = a / (a.sum(dim=-1, keepdim=True) + eps)
        mats.append(a)

    R = mats[0]
    for i in range(1, len(mats)):
        R = R @ mats[i]

    # no CLS: importance per token = average over query tokens
    r = R.mean(dim=1).squeeze(0)  # (N,)
    r = normalize_map(r)

    N = int(r.shape[0])
    side = int(side)
    if side * side != N:
        raise RuntimeError(f"Token count mismatch: side={side} => {side*side}, but N={N}")

    return r.reshape(side, side)


def overlay_and_save(img_rgb, heat, out_path, title):
    plt.figure(figsize=(5, 5))
    plt.imshow(img_rgb)
    plt.imshow(heat, alpha=0.45)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=200)
    plt.close()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="results/xai_without_pfd_gste")
    ap.add_argument("--mc_samples", type=int, default=20)
    ap.add_argument("--target_class", type=int, default=-1)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device)
    class_names = ckpt.get("class_names", ["glioma", "meningioma", "pituitary", "notumor"])
    mean = ckpt.get("mean", [0.5, 0.5, 0.5])
    std = ckpt.get("std", [0.5, 0.5, 0.5])
    cfg = ckpt.get("model_cfg", {})

    # FORCE "without PFD+GSTE" for this script
    model = HybridResNet50V2_RViT(
        num_classes=cfg.get("num_classes", len(class_names)),
        img_size=cfg.get("img_size", 224),
        patch_size=cfg.get("patch_size", 16),
        embed_dim=cfg.get("embed_dim", 142),
        depth=cfg.get("depth", 10),
        heads=cfg.get("heads", 10),
        mlp_dim=cfg.get("mlp_dim", 480),
        attn_dropout=cfg.get("attn_dropout", 0.1),
        vit_dropout=cfg.get("vit_dropout", 0.1),
        fusion_dim=cfg.get("fusion_dim", 256),
        fusion_dropout=cfg.get("fusion_dropout", 0.5),
        rotations=tuple(cfg.get("rotations", (0, 1, 2, 3))),
        cnn_name=cfg.get("cnn_name", "resnetv2_50x1_bitm"),
        cnn_pretrained=False,
        use_pfd_gste=False,   # <- strict ablation
    ).to(device)

    state = ckpt.get("model_state") or ckpt.get("state_dict")
    if state is None:
        raise KeyError("Checkpoint missing weights ('model_state' or 'state_dict').")

    try:
        model.load_state_dict(state)
    except Exception as e:
        raise RuntimeError(
            "Failed to load checkpoint into WITHOUT-PFD/GSTE model.\n"
            "This usually means the checkpoint was trained WITH PFD/GSTE.\n"
            "Use your 'predict_xai_with_pfd_gste.py' script for that checkpoint, "
            "or train/save an ablation checkpoint."
        ) from e

    model.eval()

    img = Image.open(args.image).convert("RGB").resize((224, 224))
    x = preprocess_pil(img, mean, std).to(device)

    hook_cache = {}

    def _cnn_hook(module, inputs, output):
        # output is feature map (B,C,7,7) from ResNet50V2 timm features_only stage
        feat = output
        feat.retain_grad()
        hook_cache["cnn_feat"] = feat

    h = model.cnn.register_forward_hook(_cnn_hook)
    try:
        with torch.enable_grad():
            logits, xai = model(x, return_xai=True)

        prob = torch.softmax(logits, dim=1).squeeze(0)
        pred_idx = int(prob.argmax().item())
        conf = float(prob[pred_idx].item())
        target = pred_idx if args.target_class < 0 else int(args.target_class)

        model.zero_grad(set_to_none=True)
        logits[0, target].backward()

        A = hook_cache.get("cnn_feat", None)
        if A is None or A.grad is None:
            raise RuntimeError("Failed to capture CNN feature map gradients for Grad-CAM++.")
    finally:
        h.remove()

    # Grad-CAM++ heatmap
    cam_small = gradcam_pp_from_activations(A, A.grad)
    cam = F.interpolate(
        cam_small.unsqueeze(0).unsqueeze(0),
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    ).squeeze()
    cam = cam.detach().cpu().numpy()

    # Attention rollout (fixed 14x14 tokens for 224/16 when no GSTE)
    if not isinstance(xai, dict):
        raise RuntimeError("Expected xai dict from model when return_xai=True.")

    attn_list = xai.get("attn") or xai.get("attn_list")
    if attn_list is None:
        raise KeyError("XAI dict missing attention list: expected 'attn' or 'attn_list'.")

    attn_tok = attention_rollout(attn_list, side=14)
    attn_img = F.interpolate(
        attn_tok.unsqueeze(0).unsqueeze(0),
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    ).squeeze()
    attn_img = attn_img.detach().cpu().numpy()

    # MC Dropout uncertainty
    with torch.no_grad():
        mu, var = model.mc_dropout_predict(x, mc_samples=args.mc_samples)
        mu = mu.squeeze(0).cpu().numpy()
        var = var.squeeze(0).cpu().numpy()
    mu_pred = int(np.argmax(mu))
    mu_conf = float(mu[mu_pred])
    mu_var = float(var[mu_pred])

    img_rgb = np.array(img)
    overlay_and_save(img_rgb, cam, out_dir / "xai_gradcampp.png",
                     f"Grad-CAM++ (CNN feat, target={class_names[target]})")
    overlay_and_save(img_rgb, attn_img, out_dir / "xai_attn_rollout.png",
                     "Attention Rollout (no GSTE)")

    print("Prediction (single pass):")
    print(f"  class = {class_names[pred_idx]} (idx={pred_idx})")
    print(f"  confidence = {conf:.4f}")
    print("\nMC Dropout:")
    print(f"  mean-pred class = {class_names[mu_pred]} (idx={mu_pred})")
    print(f"  mean confidence = {mu_conf:.4f}")
    print(f"  predictive variance = {mu_var:.6f}")
    print("\nSaved XAI:")
    print(out_dir / "xai_gradcampp.png")
    print(out_dir / "xai_attn_rollout.png")


if __name__ == "__main__":
    main()
