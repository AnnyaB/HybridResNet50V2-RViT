
# scripts/predict_xai.py
#
# Single-image inference + confidence + MC-dropout uncertainty + XAI:
# - Grad-CAM++ on a captured CNN feature map (PFD output if present, else CNN backbone feature map)
# - Attention Rollout on transformer attention (from returned xai dict)
#
# Saves:
#   <out_dir>/xai_gradcampp.png
#   <out_dir>/xai_attn_rollout.png
#
# Usage (terminal):
#   python scripts/predict_xai.py --checkpoint results/run_hybrid/best_model.pt --image path/to/img.jpg --out_dir results/xai
#
# NOTE:
# - DO NOT run `python ...` inside a Jupyter cell without `!` in front.
# - This file is meant for terminal/VSCode terminal.

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# --- IMPORTANT: avoid Mac/Flask thread crash by forcing non-GUI backend ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image

# ---- Fix "No module named 'models'" when running from scripts/ ----
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.hybrid_model import HybridResNet50V2_RViT


# -------------------------
# Preprocess (match training)
# -------------------------
def preprocess_pil(img, mean, std):
    img = img.convert("RGB")
    x = torch.from_numpy(np.array(img)).float() / 255.0  # (H,W,3)
    x = x.permute(2, 0, 1).contiguous()                  # (3,H,W)
    mean_t = torch.tensor(mean, dtype=x.dtype).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=x.dtype).view(3, 1, 1)
    x = (x - mean_t) / std_t
    return x.unsqueeze(0)  # (1,3,H,W)


def normalize_map(h):
    h = h - h.min()
    h = h / (h.max() + 1e-8)
    return h


# -------------------------
# Grad-CAM++ (activation-based)
# -------------------------
def gradcam_pp_from_activations(A, grads):
    """
    Grad-CAM++:
      alpha = grad^2 / (2*grad^2 + sum(A*grad^3) + eps)
      w_k = sum(alpha * relu(grad))
      cam = relu(sum(w_k * A_k))

    A:     (1,C,h,w)
    grads: (1,C,h,w)
    """
    if A is None:
        raise RuntimeError("Activation map A is None (hook failed).")
    if grads is None:
        raise RuntimeError(
            "Gradients not found for CNN feature map. Ensure forward was NOT inside torch.no_grad "
            "and backward() ran."
        )

    grad_1 = grads
    grad_2 = grad_1 ** 2
    grad_3 = grad_2 * grad_1

    spatial_sum = (A * grad_3).sum(dim=(2, 3), keepdim=True)
    denom = 2.0 * grad_2 + spatial_sum + 1e-8
    alpha = grad_2 / denom

    w = (alpha * F.relu(grad_1)).sum(dim=(2, 3), keepdim=True)  # (1,C,1,1)
    cam = (w * A).sum(dim=1, keepdim=True)                      # (1,1,h,w)
    cam = F.relu(cam)

    cam = cam.squeeze(0).squeeze(0)  # (h,w)
    return normalize_map(cam)


# -------------------------
# Attention rollout
# -------------------------
def attention_rollout(attn_list, eps=1e-6):
    """
    Attention rollout (no CLS assumed):
      - average heads
      - add identity
      - row-normalize
      - multiply across layers
      - token relevance = mean over query tokens

    attn_list: list of (B,heads,N,N)
    returns: token map (ht, wt) torch tensor in [0,1]
    """
    if not attn_list or len(attn_list) == 0:
        raise RuntimeError("No attention matrices captured. Ensure model(x, return_xai=True) returns them.")

    mats = []
    for attn in attn_list:
        a = attn.mean(dim=1)  # (B,N,N)
        n = a.shape[-1]
        a = a + torch.eye(n, device=a.device).unsqueeze(0)
        a = a / (a.sum(dim=-1, keepdim=True) + eps)
        mats.append(a)

    R = mats[0]
    for i in range(1, len(mats)):
        R = R @ mats[i]

    r = R.mean(dim=1).squeeze(0)  # (N,)
    r = normalize_map(r)

    N = int(r.shape[0])
    ht = int(round(float(np.sqrt(N))))
    ht = max(ht, 1)
    wt = max(N // ht, 1)
    if ht * wt != N:
        # fallback: single row
        ht, wt = 1, N

    return r.reshape(ht, wt)


# -------------------------
# Overlay saver
# -------------------------
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
    ap.add_argument("--out_dir", type=str, default="results/xai")
    ap.add_argument("--mc_samples", type=int, default=20)
    ap.add_argument("--target_class", type=int, default=-1, help="class index, or -1 = explain predicted class")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load checkpoint ----
    ckpt = torch.load(args.checkpoint, map_location=device)
    class_names = ckpt.get("class_names", ["glioma", "meningioma", "pituitary", "notumor"])
    mean = ckpt.get("mean", [0.5, 0.5, 0.5])
    std = ckpt.get("std", [0.5, 0.5, 0.5])
    cfg = ckpt.get("model_cfg", {})

    # ---- Build model exactly from cfg ----
    model = HybridResNet50V2_RViT(
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
        cnn_pretrained=False,  # weights come from checkpoint
    ).to(device)

    state = ckpt.get("model_state", None) or ckpt.get("state_dict", None)
    if state is None:
        raise KeyError("Checkpoint missing weights. Expected key 'model_state' (or 'state_dict').")
    model.load_state_dict(state)
    model.eval()

    # ---- Load and preprocess image ----
    img = Image.open(args.image).convert("RGB").resize((224, 224))
    x = preprocess_pil(img, mean, std).to(device)

    # ---- Hook to capture feature map for Grad-CAM++ ----
    hook_cache = {}

    def _feat_hook(module, inputs, output):
        # output can be:
        # - PFD version: (feat_path, mask_feat)
        # - no-PFD version: feat tensor
        feat_map = output[0] if isinstance(output, (tuple, list)) else output
        feat_map.retain_grad()
        hook_cache["cnn_feat"] = feat_map

    # Prefer PFD output if present, else use CNN backbone feature map
    if hasattr(model, "pfd"):
        hook_handle = model.pfd.register_forward_hook(_feat_hook)
    else:
        hook_handle = model.cnn.register_forward_hook(_feat_hook)

    try:
        # Forward with grads enabled (needed for Grad-CAM++)
        with torch.enable_grad():
            logits, xai = model(x, return_xai=True)

        prob = torch.softmax(logits, dim=1).squeeze(0)
        pred_idx = int(torch.argmax(prob).item())
        conf = float(prob[pred_idx].item())
        target = pred_idx if (args.target_class is None or args.target_class < 0) else int(args.target_class)

        # Backprop for Grad-CAM++
        model.zero_grad(set_to_none=True)
        logits[0, target].backward(retain_graph=True)

        A = hook_cache.get("cnn_feat", None)
        if A is None:
            raise RuntimeError("Failed to capture CNN feature map via hook (cnn_feat missing).")

    finally:
        hook_handle.remove()

    # ---- Grad-CAM++ map -> image size ----
    cam_small = gradcam_pp_from_activations(A, A.grad)  # (h,w) torch
    cam = F.interpolate(cam_small.unsqueeze(0).unsqueeze(0), size=(224, 224),
                        mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
    cam = cam.detach().cpu().numpy()

    # ---- Attention rollout ----
    if not isinstance(xai, dict):
        raise RuntimeError("Model did not return XAI dict. Ensure forward returns (logits, dict) when return_xai=True.")

    # Your model returns xai keys as {"attn": attn_list} OR could be {"attn_list": ...}
    attn_list = xai.get("attn", None)
    if attn_list is None:
        attn_list = xai.get("attn_list", None)
    if attn_list is None:
        raise KeyError("XAI dict missing attention list. Expected key 'attn' (or 'attn_list').")

    attn_tok = attention_rollout(attn_list)  # (ht,wt)
    attn_img = F.interpolate(attn_tok.unsqueeze(0).unsqueeze(0), size=(224, 224),
                             mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
    attn_img = attn_img.detach().cpu().numpy()

    # ---- MC dropout uncertainty (optional) ----
    with torch.no_grad():
        mu, var = model.mc_dropout_predict(x, mc_samples=args.mc_samples)
        mu = mu.squeeze(0).cpu().numpy()
        var = var.squeeze(0).cpu().numpy()
    mu_pred = int(np.argmax(mu))
    mu_conf = float(mu[mu_pred])
    mu_var = float(var[mu_pred])

    # ---- Save overlays ----
    img_rgb = np.array(img)
    overlay_and_save(img_rgb, cam, out_dir / "xai_gradcampp.png",
                     f"Grad-CAM++ (target={class_names[target]})")
    overlay_and_save(img_rgb, attn_img, out_dir / "xai_attn_rollout.png",
                     "Attention Rollout (token relevance)")

    # ---- Print summary ----
    print("Prediction (single pass):")
    print(f"  class = {class_names[pred_idx]} (idx={pred_idx})")
    print(f"  confidence = {conf:.4f}")

    print("\nMC Dropout (uncertainty-aware):")
    print(f"  mean-pred class = {class_names[mu_pred]} (idx={mu_pred})")
    print(f"  mean confidence = {mu_conf:.4f}")
    print(f"  predictive variance (pred class) = {mu_var:.6f}")

    print("\nSaved XAI:")
    print(f"  {str(out_dir / 'xai_gradcampp.png')}")
    print(f"  {str(out_dir / 'xai_attn_rollout.png')}")


if __name__ == "__main__":
    main()