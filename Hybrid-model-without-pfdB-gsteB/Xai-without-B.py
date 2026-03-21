# AUTHOR: RIYA BASAK
# 22089065

# THIS IS THE XAI SCRIPT FOR HYBRID B ABLATION (WITHOUT PFD-GSTE VARIANT B):
# External libraries used here are cited in Appendix A2.3:
# NumPy (Harris et al., 2020); PyTorch (Paszke et al., 2019);
# Matplotlib (Hunter, 2007); Pillow (Clark and contributors, 2024).


# Libraries I needd
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


# Making local project imports work when running this file directly:
# - ROOT is set to the folder containing this script.
# - That folder is added to sys.path so from models... resolves reliably.

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Local project model import:
# - HybridResNet50V2_RViT is my hybrid CNN-Transformer model class.
# - This script will force it to run in the without PFD-GSTE configuration.

from models.hybrid_model import HybridResNet50V2_RViT

# Preprocessing a PIL image into a normalized torch tensor batch:
# - Ensures RGB
# - Converts to float in [0,1]
# - Changes layout from HWC -> CHW
# - Normalizes using checkpoint mean/std
# - Adds a batch dimension (1, C, H, W)

def preprocess_pil(img, mean, std):
    
    # Ensuring 3-channel RGB (some files could be grayscale or RGBA)
    img = img.convert("RGB")
    # Converting PIL -> numpy -> torch float tensor, scale to [0,1]
    x = torch.from_numpy(np.array(img)).float() / 255.0
    # Reordering dimensions from (H, W, C) to (C, H, W)
    x = x.permute(2, 0, 1).contiguous()
    # Building broadcastable mean/std tensors shaped (3,1,1)
    mean_t = torch.tensor(mean, dtype=x.dtype).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=x.dtype).view(3, 1, 1)
    # Normalizing per-channel
    x = (x - mean_t) / std_t
    # Adding batch dimension -> (1, 3, H, W)
    return x.unsqueeze(0)



# Normalize any heatmap/vector to [0,1] safely:
# - Shift by min
# - Divide by max (with small epsilon to avoid divide-by-zero)

def normalize_map(h):
    h = h - h.min()
    h = h / (h.max() + 1e-8)
    return h



# Computing Grad-CAM++ given:
# - A: activations from a target CNN layer (feature map)
# - grads: gradients of target logit w.r.t. those activations
#
# Output:
# - a 2D normalized heatmap (still at feature-map resolution, e.g., 7x7)

def gradcam_pp_from_activations(A, grads):
    
    # Validating the forward hook captured the activation tensor
    if A is None:
        raise RuntimeError("Activation A is None (hook failed).")
    # Validating a backward pass produced gradients
    if grads is None:
        raise RuntimeError("Gradients missing. Ensure enable_grad + backward ran.")

    # Grad-CAM++ uses first, second, and third powers of gradients
    grad_1 = grads
    grad_2 = grad_1 ** 2
    grad_3 = grad_2 * grad_1

    # Spatial sum term used in the Grad-CAM++ alpha denominator
    spatial_sum = (A * grad_3).sum(dim=(2, 3), keepdim=True)
    denom = 2.0 * grad_2 + spatial_sum + 1e-8
    # Per-pixel weighting coefficients (alpha)
    alpha = grad_2 / denom

    # Channel weights: sum over spatial positions of alpha * ReLU(grad)
    w = (alpha * F.relu(grad_1)).sum(dim=(2, 3), keepdim=True)
    # Weighted sum across channels to get CAM, then ReLU
    cam = (w * A).sum(dim=1, keepdim=True)
    cam = F.relu(cam).squeeze(0).squeeze(0)
    # Normalizing final heatmap to [0,1]
    return normalize_map(cam)


# Transformer attention rollout:
# - attn_list is expected to be a list of attention tensors per layer
# - Each attn tensor: (B, H, N, N)
#   B=batch, H=heads, N=tokens
#
# Process:
# - Average heads
# - Add identity (residual connection approximation)
# - Row-normalize
# - Multiply matrices layer-by-layer to accumulate token-to-token relevance
# - Output a (side, side) map, where side=14 corresponds to 224/16 tokens
#   (no CLS token assumed here; it expects N == side*side)


def attention_rollout(attn_list, side=14, eps=1e-6):
    # Ensuring the model actually returned attention matrices
    if not attn_list:
        raise RuntimeError("No attention matrices captured. Ensure return_xai=True.")

    mats = []
    for attn in attn_list:
        # attn: (B, H, N, N)
        # Average across heads -> (B, N, N)
        a = attn.mean(dim=1)  # (B, N, N)
        # Token count N
        n = a.shape[-1]
        # Adding identity to mimic residual/skip behavior
        a = a + torch.eye(n, device=a.device).unsqueeze(0)
        # Row-normalizing so each row sums to ~1
        a = a / (a.sum(dim=-1, keepdim=True) + eps)
        # Collecting per-layer normalized attention
        mats.append(a)

    # Starting rollout with first layer
    R = mats[0]
    # Composing attentions through layers (matrix multiply)
    for i in range(1, len(mats)):
        R = R @ mats[i]

    # no CLS: importance per token = average over query tokens
    # R: (B, N, N) -> mean over query dimension -> (B, N) -> squeeze batch -> (N,)
    r = R.mean(dim=1).squeeze(0)  # (N,)
    # Normalizing token relevance to [0,1]
    r = normalize_map(r)

    # Validating token grid size matches side*side
    N = int(r.shape[0])
    side = int(side)
    if side * side != N:
        raise RuntimeError(f"Token count mismatch: side={side} => {side*side}, but N={N}")

    # Reshaping 1D token relevance into a 2D grid
    return r.reshape(side, side)


# Overlaying a heatmap on the original RGB image and saving to disk:
# - First draw the base image
# - Then draw the heatmap with alpha blending
# - Save using a non-interactive backend (Agg)

def overlay_and_save(img_rgb, heat, out_path, title):
    plt.figure(figsize=(5, 5))
    plt.imshow(img_rgb)
    plt.imshow(heat, alpha=0.45)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=200)
    plt.close()



# Main CLI entry:
# - Loads checkpoint and rebuilds without PFD-GSTE variant B model
# - Runs a forward pass with return_xai=True
# - Uses a forward hook and backward pass to compute Grad-CAM++ from CNN features
# - Computes attention rollout from transformer attentions
# - Estimates uncertainty via MC Dropout
# - Saves overlay images and prints summary

def main():
    
    # Importing argparse locally (keeps global imports minimal)
    import argparse
    # Building command-line interface
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="results/xai_without_pfd_gste")
    ap.add_argument("--mc_samples", type=int, default=20)
    ap.add_argument("--target_class", type=int, default=-1)
    args = ap.parse_args()

    # Choosing CUDA if available, else CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Loading checkpoint to chosen device
    ckpt = torch.load(args.checkpoint, map_location=device)
    # Pulling metadata (fallback defaults if missing)
    class_names = ckpt.get("class_names", ["glioma", "meningioma", "pituitary", "notumor"])
    mean = ckpt.get("mean", [0.5, 0.5, 0.5])
    std = ckpt.get("std", [0.5, 0.5, 0.5])
    cfg = ckpt.get("model_cfg", {})

    # forcing without PFD-GSTE variant B for this script
    # Rebuild model using saved config values (or defaults)
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

    # Support either checkpoint key name for the weights dict
    state = ckpt.get("model_state") or ckpt.get("state_dict")
    if state is None:
        raise KeyError("Checkpoint missing weights ('model_state' or 'state_dict').")

    # Try loading weights into the without PFD-GSTE variant B architecture
    # If the checkpoint was trained WITH PFD+GSTE, shapes/keys won't match and this will error
    try:
        model.load_state_dict(state)
    except Exception as e:
        raise RuntimeError(
            "Failed to load checkpoint into WITHOUT-PFD/GSTE model.\n"
            "This usually means the checkpoint was trained WITH PFD/GSTE.\n"
            "Use your 'predict_xai_with_pfd_gste.py' script for that checkpoint, "
            "or train/save an ablation checkpoint."
        ) from e

    # Switching to evaluation mode (disables training-time behaviors like Dropout unless explicitly used)
    model.eval()

    # Loading the input image, enforce RGB, and resize to the model's expected 224x224
    img = Image.open(args.image).convert("RGB").resize((224, 224))
    # Converting to normalized tensor batch and move to device
    x = preprocess_pil(img, mean, std).to(device)

    # Hook cache is a simple dict to store the activation tensor from the CNN
    hook_cache = {}

    # Forward hook function:
    # - Captures the CNN feature map output and retains gradient for Grad-CAM++
    def _cnn_hook(module, inputs, output):
        # output is feature map (B,C,7,7) from ResNet50V2 timm features_only stage
        feat = output
        feat.retain_grad()
        hook_cache["cnn_feat"] = feat

    # Registering hook on the CNN module so we can grab its feature maps during forward
    h = model.cnn.register_forward_hook(_cnn_hook)
    try:
        # Enabling gradients for Grad-CAM++ (even though model.eval() was set)
        with torch.enable_grad():
            # Forwarding pass requesting XAI outputs (for instance., attentions)
            logits, xai = model(x, return_xai=True)

        # Converting logits to probabilities
        prob = torch.softmax(logits, dim=1).squeeze(0)
        # Predicted class index (argmax)
        pred_idx = int(prob.argmax().item())
        # Confidence of the predicted class
        conf = float(prob[pred_idx].item())
        # Choosing target class:
        # - default uses predicted class
        # - user can override with --target_class
        target = pred_idx if args.target_class < 0 else int(args.target_class)

        # Clearing old gradients before backward
        model.zero_grad(set_to_none=True)
        # Backpropagating only the chosen class logit to get grads w.r.t. CNN features
        logits[0, target].backward()

        # Retrieving the CNN feature map and its gradients
        A = hook_cache.get("cnn_feat", None)
        if A is None or A.grad is None:
            raise RuntimeError("Failed to capture CNN feature map gradients for Grad-CAM++.")
    finally:
        # Always remove hook to avoid side effects in later runs
        h.remove()

    # Grad-CAM++ heatmap
    # Compute Grad-CAM++ at CNN feature-map resolution (e.g., 7x7)
    cam_small = gradcam_pp_from_activations(A, A.grad)
    # Upsample CAM to 224x224 for overlay visualization
    cam = F.interpolate(
        cam_small.unsqueeze(0).unsqueeze(0),
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    ).squeeze()
    # Moving CAM to CPU numpy for matplotlib
    cam = cam.detach().cpu().numpy()

    # Attention rollout (fixed 14x14 tokens for 224/16 when no GSTE)
    # Validate xai is the expected dictionary container
    if not isinstance(xai, dict):
        raise RuntimeError("Expected xai dict from model when return_xai=True.")

    # Support either key name for attention list
    attn_list = xai.get("attn") or xai.get("attn_list")
    if attn_list is None:
        raise KeyError("XAI dict missing attention list: expected 'attn' or 'attn_list'.")

    # Rollout attention to a token-grid heatmap (14x14)
    attn_tok = attention_rollout(attn_list, side=14)
    # Upsample token-grid heatmap to image size for overlay
    attn_img = F.interpolate(
        attn_tok.unsqueeze(0).unsqueeze(0),
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    ).squeeze()
    # Moving attention heatmap to CPU numpy for matplotlib
    attn_img = attn_img.detach().cpu().numpy()

    # MC Dropout uncertainty
    # Running MC dropout prediction without gradient tracking
    with torch.no_grad():
        mu, var = model.mc_dropout_predict(x, mc_samples=args.mc_samples)
        mu = mu.squeeze(0).cpu().numpy()
        var = var.squeeze(0).cpu().numpy()
    # Predicted class from mean probabilities
    mu_pred = int(np.argmax(mu))
    # Mean confidence and predictive variance for the predicted class
    mu_conf = float(mu[mu_pred])
    mu_var = float(var[mu_pred])

    # Converting PIL image to RGB numpy array for plotting
    img_rgb = np.array(img)
    # Saving Grad-CAM++ overlay
    overlay_and_save(img_rgb, cam, out_dir / "xai_gradcampp.png",
                     f"Grad-CAM++ (CNN feat, target={class_names[target]})")
    # Saving attention rollout overlay
    overlay_and_save(img_rgb, attn_img, out_dir / "xai_attn_rollout.png",
                     "Attention Rollout (no GSTE)")

    # Printing a readable summary to terminal
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

# Standard Python script entrypoint:
# - Ensures main() runs only when executing this file directly

if __name__ == "__main__":
    main()
