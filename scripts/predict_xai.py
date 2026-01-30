# scripts/predict_xai.py
#
# Single-image inference + confidence + MC-dropout uncertainty + XAI:
# - Grad-CAM++ on CNN/PFD feature map (captured via hook)
# - Attention Rollout on transformer attention
#
# Saves:
#   <out_dir>/xai_gradcampp_overlay.png
#   <out_dir>/xai_gradcampp_heat.png
#   <out_dir>/xai_attn_rollout_overlay.png
#   <out_dir>/xai_attn_rollout_heat.png

import sys
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

# --- make imports work no matter where you run from ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.hybrid_model_pfd_gste import HybridResNet50V2_RViT  # noqa: E402


def preprocess_pil(img, mean, std):
    img = img.convert("RGB")
    x = torch.from_numpy(np.array(img)).float() / 255.0  # (H,W,3)
    x = x.permute(2, 0, 1).contiguous()                  # (3,H,W)
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    x = (x - mean) / std
    return x.unsqueeze(0)  # (1,3,H,W)


def normalize_map(h):
    h = h - h.min()
    h = h / (h.max() + 1e-8)
    return h


class ActivationCatcher:
    """Stores the last activation tensor (and retains grads) from a module hook."""
    def __init__(self):
        self.A = None

    def hook(self, module, inp, out):
        # PFD returns (gated_feat, mask). We want gated_feat for CAM.
        if isinstance(out, (tuple, list)):
            A = out[0]
        else:
            A = out
        self.A = A
        # Needed so A.grad is populated after backward()
        self.A.retain_grad()


def gradcam_pp_from_activation(A):
    """
    Grad-CAM++ (practical form) computed from activation A and its gradients A.grad.
    A: (1,C,h,w), A.grad: (1,C,h,w)
    """
    grads = A.grad
    if grads is None:
        raise RuntimeError(
            "No gradients on activation. This usually means the chosen activation "
            "is not connected to the target score, or retain_grad() wasn't called."
        )

    grad_1 = grads
    grad_2 = grad_1 ** 2
    grad_3 = grad_2 * grad_1

    # Sum over spatial for each channel
    spatial_sum = (A * grad_3).sum(dim=(2, 3), keepdim=True)

    denom = 2.0 * grad_2 + spatial_sum + 1e-8
    alpha = grad_2 / denom

    w = (alpha * F.relu(grad_1)).sum(dim=(2, 3), keepdim=True)  # (1,C,1,1)
    cam = (w * A).sum(dim=1, keepdim=True)                      # (1,1,h,w)
    cam = F.relu(cam)

    cam = cam.squeeze(0).squeeze(0)  # (h,w)
    cam = normalize_map(cam)
    return cam


def attention_rollout(attn_list, eps=1e-6):
    """
    Attention rollout:
      - average heads
      - add identity (residual)
      - row-normalize
      - multiply across layers
      - token relevance = mean over query tokens
    attn_list: list of (B,heads,N,N)
    returns: (N,) relevance
    """
    if not attn_list:
        raise RuntimeError("No attention matrices captured (attn_list empty).")

    A = []
    for attn in attn_list:
        a = attn.mean(dim=1)  # (B,N,N)
        n = a.shape[-1]
        a = a + torch.eye(n, device=a.device).unsqueeze(0)
        a = a / (a.sum(dim=-1, keepdim=True) + eps)
        A.append(a)

    R = A[0]
    for i in range(1, len(A)):
        R = R @ A[i]  # (B,N,N)

    r = R.mean(dim=1).squeeze(0)  # (N,)
    r = normalize_map(r)
    return r


def save_heat_and_overlay(img_rgb, heat, out_heat, out_overlay, title):
    # heat expected in [0,1], shape (H,W)
    heat = np.clip(heat, 0.0, 1.0)

    # heat-only
    plt.figure()
    plt.imshow(heat)
    plt.title(title + " (heat)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_heat, dpi=200)
    plt.close()

    # overlay
    plt.figure()
    plt.imshow(img_rgb)
    plt.imshow(heat, alpha=0.45)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_overlay, dpi=200)
    plt.close()


def build_model_from_ckpt(ckpt, device):
    # Build using cfg if present to guarantee correct dims
    cfg = ckpt.get("model_cfg", {})
    class_names = ckpt.get("class_names", ["glioma", "meningioma", "pituitary", "notumor"])

    # Only pass kwargs that exist in your HybridResNet50V2_RViT __init__
    import inspect
    sig = inspect.signature(HybridResNet50V2_RViT.__init__)
    allowed = set(sig.parameters.keys())

    kwargs = dict(
        num_classes=cfg.get("num_classes", len(class_names)),
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
        cnn_pretrained=False,  # load from checkpoint
    )
    kwargs = {k: v for k, v in kwargs.items() if k in allowed}

    model = HybridResNet50V2_RViT(**kwargs).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="results/xai")
    ap.add_argument("--mc_samples", type=int, default=20)
    ap.add_argument("--target_class", type=int, default=-1, help="class index, or -1 for predicted class")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device)
    class_names = ckpt["class_names"]
    mean = ckpt["mean"]
    std = ckpt["std"]

    model = build_model_from_ckpt(ckpt, device)

    # --- Hook to capture PFD feature maps for Grad-CAM++ ---
    catcher = ActivationCatcher()
    if not hasattr(model, "pfd"):
        raise RuntimeError("Model has no attribute 'pfd'. Cannot hook PFD feature map for CAM.")
    h = model.pfd.register_forward_hook(catcher.hook)

    # Load image
    img = Image.open(args.image).convert("RGB").resize((224, 224))
    x = preprocess_pil(img, mean, std).to(device)

    # Forward with XAI enabled (to capture attention list)
    logits, xai = model(x, return_xai=True)

    prob = torch.softmax(logits, dim=1).squeeze(0)
    pred_idx = int(torch.argmax(prob).item())
    conf = float(prob[pred_idx].item())

    # Target for explanation
    target = pred_idx if args.target_class is None or args.target_class < 0 else int(args.target_class)

    # --- Grad-CAM++ backward ---
    model.zero_grad(set_to_none=True)
    score = logits[0, target]
    score.backward(retain_graph=True)

    if catcher.A is None:
        raise RuntimeError("Hook did not capture activation. Ensure model.pfd is used in forward().")
    cam_small = gradcam_pp_from_activation(catcher.A)  # (h,w) e.g., (7,7)

    cam = F.interpolate(
        cam_small.unsqueeze(0).unsqueeze(0),
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    ).squeeze().detach().cpu().numpy()

    # --- Attention rollout ---
    if xai is None:
        raise RuntimeError("Model did not return an xai dict even with return_xai=True.")

    attn_list = xai.get("attn_list", None)
    if attn_list is None:
        attn_list = xai.get("attn", None)  # common alternative key
    if attn_list is None:
        raise KeyError("xai dict missing attention list (expected 'attn_list' or 'attn').")

    r = attention_rollout(attn_list)  # (N,)

    # infer token grid
    N = int(r.numel())
    side = int(math.sqrt(N))
    if side * side != N:
        # fallback: best-effort rectangle
        ht = side
        wt = max(1, N // max(side, 1))
    else:
        ht, wt = side, side

    attn_tok = r.reshape(ht, wt)
    attn_img = F.interpolate(
        attn_tok.unsqueeze(0).unsqueeze(0),
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    ).squeeze().detach().cpu().numpy()

    # --- MC Dropout uncertainty (optional) ---
    with torch.no_grad():
        mu, var = model.mc_dropout_predict(x, mc_samples=args.mc_samples)
        mu = mu.squeeze(0).cpu().numpy()
        var = var.squeeze(0).cpu().numpy()
    mu_pred = int(np.argmax(mu))
    mu_conf = float(mu[mu_pred])
    mu_var = float(var[mu_pred])

    # Save images
    img_rgb = np.array(img)  # uint8
    save_heat_and_overlay(
        img_rgb, cam,
        out_dir / "xai_gradcampp_heat.png",
        out_dir / "xai_gradcampp_overlay.png",
        f"Grad-CAM++ (target={class_names[target]})"
    )
    save_heat_and_overlay(
        img_rgb, attn_img,
        out_dir / "xai_attn_rollout_heat.png",
        out_dir / "xai_attn_rollout_overlay.png",
        "Attention Rollout (token relevance)"
    )

    # Cleanup hook
    h.remove()

    # Print outputs
    print("Prediction (single pass):")
    print(f"  class = {class_names[pred_idx]} (idx={pred_idx})")
    print(f"  confidence = {conf:.4f}")

    print("\nMC Dropout (uncertainty-aware):")
    print(f"  mean-pred class = {class_names[mu_pred]} (idx={mu_pred})")
    print(f"  mean confidence = {mu_conf:.4f}")
    print(f"  predictive variance (pred class) = {mu_var:.6f}")

    print("\nSaved XAI:")
    print(f"  {str(out_dir / 'xai_gradcampp_overlay.png')}")
    print(f"  {str(out_dir / 'xai_attn_rollout_overlay.png')}")
    print(f"  {str(out_dir / 'xai_gradcampp_heat.png')}")
    print(f"  {str(out_dir / 'xai_attn_rollout_heat.png')}")


if __name__ == "__main__":
    main()