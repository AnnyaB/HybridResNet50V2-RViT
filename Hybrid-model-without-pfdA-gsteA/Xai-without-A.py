# AUTHOR: RIYA BASAK
# 22089065

# THIS IS THE XAI SCRIPT FOR HYBRID A ABLATION (WITHOUT PFD-GSTE VARIANT A):
# External libraries used here are cited in Appendix A2.3:
# NumPy (Harris et al., 2020); PyTorch (Paszke et al., 2019);
# Matplotlib (Hunter, 2007); Pillow (Clark and contributors, 2024).


# Libraries I needed 
import sys
from pathlib import Path


# - numpy for array utilities and sqrt for reshaping tokens
# - torch for inference, gradients, hooks, checkpoint loading
# - F for interpolate/relu and other tensor ops

import numpy as np
import torch
import torch.nn.functional as F


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Pillow is used to open/convert/resize the input image.
from PIL import Image


# ROOT points to the directory containing this script (scripts/).
# By adding ROOT to sys.path, Python can resolve imports like models.hybrid_model, 
# (I HAD ERRORS AND SPEND QUITE SOME TIME TO FIGURE THIS OUT)
# even when the script is run from inside the scripts folder.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Importing the hybrid model definition from the project.
from models.hybrid_model import HybridResNet50V2_RViT



# Preprocess 

# Converts a PIL image into a normalized torch tensor:
# - ensures RGB
# - scales pixel values to [0,1]
# - converts to CHW layout
# - normalizes using mean/std (same style as training)
# - adds batch dimension (1,3,H,W)

def preprocess_pil(img, mean, std):
    
    img = img.convert("RGB")
    x = torch.from_numpy(np.array(img)).float() / 255.0  # (H,W,3)
    x = x.permute(2, 0, 1).contiguous()                  # (3,H,W)
    mean_t = torch.tensor(mean, dtype=x.dtype).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=x.dtype).view(3, 1, 1)
    x = (x - mean_t) / std_t
    return x.unsqueeze(0)  # (1,3,H,W)


# Normalizes a heatmap/tensor to [0,1] for consistent visualization.
def normalize_map(h):
    h = h - h.min()
    h = h / (h.max() + 1e-8)
    return h

# Grad-CAM++ (activation-based)

# Computes Grad-CAM++ from:
# - A: activation feature map captured by a forward hook (shape: 1,C,h,w)
# - grads: gradients of the target logit w.r.t. A (shape: 1,C,h,w)
def gradcam_pp_from_activations(A, grads):
    
    """
    Grad-CAM++:
      alpha = grad^2 / (2*grad^2 + sum(A*grad^3) + eps)
      w_k = sum(alpha * relu(grad))
      cam = relu(sum(w_k * A_k))

    A:     (1,C,h,w)
    grads: (1,C,h,w)
    """
    # Guard: hook must have captured activations
    if A is None:
        raise RuntimeError("Activation map A is None (hook failed).")
    # Guard: backward must have produced gradients
    if grads is None:
        raise RuntimeError(
            "Gradients not found for CNN feature map. Ensure forward was NOT inside torch.no_grad "
            "and backward() ran."
        )

    # First, second, third derivative-like terms used by Grad-CAM++
    grad_1 = grads
    grad_2 = grad_1 ** 2
    grad_3 = grad_2 * grad_1

    # Spatial sum term: sum over (h,w) of A * grad^3 for each channel
    spatial_sum = (A * grad_3).sum(dim=(2, 3), keepdim=True)
    # Denominator for alpha weights (stabilized by eps)
    denom = 2.0 * grad_2 + spatial_sum + 1e-8
    # Alpha coefficients per pixel
    alpha = grad_2 / denom

    # Channel weights: sum over spatial of alpha * relu(grad)
    w = (alpha * F.relu(grad_1)).sum(dim=(2, 3), keepdim=True)  # (1,C,1,1)
    # Weighted sum of activations across channels -> raw CAM
    cam = (w * A).sum(dim=1, keepdim=True)                      # (1,1,h,w)
    # Keep only positive influence
    cam = F.relu(cam)

    # Remove batch and channel dims -> (h,w), then normalize to [0,1]
    cam = cam.squeeze(0).squeeze(0)  # (h,w)
    return normalize_map(cam)

# Attention rollout

# Builds a token-relevance map from transformer attention matrices:
# - averages across heads
# - adds identity (residual)
# - row-normalizes
# - multiplies matrices across layers
# - reduces to a single relevance score per token (mean across query tokens)

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
    # Guard: must have captured at least one attention matrix
    if not attn_list or len(attn_list) == 0:
        raise RuntimeError("No attention matrices captured. Ensure model(x, return_xai=True) returns them.")

    mats = []
    # Process each layer attention tensor
    for attn in attn_list:
        # Average over heads -> (B,N,N)
        a = attn.mean(dim=1)  # (B,N,N)
        n = a.shape[-1]
        # Add identity to account for residual connections
        a = a + torch.eye(n, device=a.device).unsqueeze(0)
        # Row-normalize so rows sum to 1
        a = a / (a.sum(dim=-1, keepdim=True) + eps)
        mats.append(a)

    # Multiply attention matrices across layers to get joint attention
    R = mats[0]
    for i in range(1, len(mats)):
        R = R @ mats[i]

    # Token importance: average relevance over all query tokens
    r = R.mean(dim=1).squeeze(0)  # (N,)
    r = normalize_map(r)

    # Infer a 2D grid shape from token count N
    N = int(r.shape[0])
    ht = int(round(float(np.sqrt(N))))
    ht = max(ht, 1)
    wt = max(N // ht, 1)
    if ht * wt != N:
        # fallback: single row if factoring isn't perfect
        ht, wt = 1, N

    # Return relevance reshaped to (ht,wt)
    return r.reshape(ht, wt)

# Overlay saver

# Saves an overlay plot:
# - base RGB image
# - heatmap overlaid with alpha transparency
def overlay_and_save(img_rgb, heat, out_path, title):
    plt.figure(figsize=(5, 5))
    plt.imshow(img_rgb)
    plt.imshow(heat, alpha=0.45)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=200)
    plt.close()

# Entry-point routine

def main():
    import argparse

    # CLI argument parser for terminal usage
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="results/xai")
    ap.add_argument("--mc_samples", type=int, default=20)
    ap.add_argument("--target_class", type=int, default=-1, help="class index, or -1 = explain predicted class")
    args = ap.parse_args()

    # Select device (CUDA if available, else CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Creating output directory if it doesn't exist
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Loading checkpoint
    # Load trained weights and metadata (class names, mean/std, model config).
    ckpt = torch.load(args.checkpoint, map_location=device)
    class_names = ckpt.get("class_names", ["glioma", "meningioma", "pituitary", "notumor"])
    mean = ckpt.get("mean", [0.5, 0.5, 0.5])
    std = ckpt.get("std", [0.5, 0.5, 0.5])
    cfg = ckpt.get("model_cfg", {})

    # Building model exactly from cfg
    # Reconstruct the architecture with the same hyperparameters used during training.
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

    # Extracting weights dict from checkpoint (supports both key names).
    state = ckpt.get("model_state", None) or ckpt.get("state_dict", None)
    if state is None:
        raise KeyError("Checkpoint missing weights. Expected key 'model_state' (or 'state_dict').")
    # Loading weights into model and switch to eval mode for deterministic inference
    model.load_state_dict(state)
    model.eval()

    # Load and preprocess image
    # Open input image, force RGB, resize to model input size, then normalize.
    img = Image.open(args.image).convert("RGB").resize((224, 224))
    x = preprocess_pil(img, mean, std).to(device)

    # Hook to capture feature map for Grad-CAM++
    # hook_cache stores the feature map tensor so we can use it after forward/backward.
    hook_cache = {}

    def _feat_hook(module, inputs, output):
        
        # output can be:
        # - PFD version: (feat_path, mask_feat)
        # - no-PFD version: feat tensor
        # Choosing the feature tensor in both cases.
        feat_map = output[0] if isinstance(output, (tuple, list)) else output
        # Ensuring gradients will be retained for this tensor so A.grad is available after backward().
        feat_map.retain_grad()
        # Storing for Grad-CAM++
        hook_cache["cnn_feat"] = feat_map

    # Prefer PFD output if present, else use CNN backbone feature map
    # If model has pfd, we hook there so Grad-CAM++ uses the gated (pathology-focused) feature map.
    
    if hasattr(model, "pfd"):
        hook_handle = model.pfd.register_forward_hook(_feat_hook)
    else:
        hook_handle = model.cnn.register_forward_hook(_feat_hook)

    try:
        # Forward with grads enabled (needed for Grad-CAM++)
        # return_xai=True requests attention matrices etc. from the transformer.
        with torch.enable_grad():
            logits, xai = model(x, return_xai=True)

        # Converting logits to probabilities and pick predicted class
        prob = torch.softmax(logits, dim=1).squeeze(0)
        pred_idx = int(torch.argmax(prob).item())
        conf = float(prob[pred_idx].item())
        # Choosing which class to explain: predicted class by default, else user-specified target_class
        target = pred_idx if (args.target_class is None or args.target_class < 0) else int(args.target_class)

        # Backpropagating for Grad-CAM++
        model.zero_grad(set_to_none=True)
        # Differentiating the target logit w.r.t. the captured feature map
        logits[0, target].backward(retain_graph=True)

        # Retrieving captured feature map for Grad-CAM++
        A = hook_cache.get("cnn_feat", None)
        if A is None:
            raise RuntimeError("Failed to capture CNN feature map via hook (cnn_feat missing).")

    finally:
        # Always remove hook handle to avoid leaking hooks across runs
        hook_handle.remove()

    # Grad-CAM++ map -> image size 
    # Computing CAM on feature resolution, then upsample to 224x224 for overlay.
    cam_small = gradcam_pp_from_activations(A, A.grad)  # (h,w) torch
    cam = F.interpolate(cam_small.unsqueeze(0).unsqueeze(0), size=(224, 224),
                        mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
    cam = cam.detach().cpu().numpy()

    # Attention rollout 
    # Ensure XAI output is a dict and contains attention matrices.
    if not isinstance(xai, dict):
        raise RuntimeError("Model did not return XAI dict. Ensure forward returns (logits, dict) when return_xai=True.")

    # my model returns xai keys as {"attn": attn_list} OR could be {"attn_list": ...}
    attn_list = xai.get("attn", None)
    if attn_list is None:
        attn_list = xai.get("attn_list", None)
    if attn_list is None:
        raise KeyError("XAI dict missing attention list. Expected key 'attn' (or 'attn_list').")

    # Converting attention matrices into a token relevance heatmap, then upsample to image size.
    attn_tok = attention_rollout(attn_list)  # (ht,wt)
    attn_img = F.interpolate(attn_tok.unsqueeze(0).unsqueeze(0), size=(224, 224),
                             mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
    attn_img = attn_img.detach().cpu().numpy()

    # MC dropout uncertainty (optional)
    # Uses the model's mc_dropout_predict helper:
    # - runs multiple stochastic forward passes with dropout enabled
    # - returns mean and variance over predicted probabilities
    with torch.no_grad():
        mu, var = model.mc_dropout_predict(x, mc_samples=args.mc_samples)
        mu = mu.squeeze(0).cpu().numpy()
        var = var.squeeze(0).cpu().numpy()
    mu_pred = int(np.argmax(mu))
    mu_conf = float(mu[mu_pred])
    mu_var = float(var[mu_pred])

    # Save overlays 
    # Overlay and save Grad-CAM++ and attention rollout heatmaps.
    img_rgb = np.array(img)
    overlay_and_save(img_rgb, cam, out_dir / "xai_gradcampp.png",
                     f"Grad-CAM++ (target={class_names[target]})")
    overlay_and_save(img_rgb, attn_img, out_dir / "xai_attn_rollout.png",
                     "Attention Rollout (token relevance)")

    # Print summary 
    # Report single-pass prediction and confidence.
    print("Prediction (single pass):")
    print(f"  class = {class_names[pred_idx]} (idx={pred_idx})")
    print(f"  confidence = {conf:.4f}")

    # Report MC-dropout mean prediction and predictive variance as uncertainty signal.
    print("\nMC Dropout (uncertainty-aware):")
    print(f"  mean-pred class = {class_names[mu_pred]} (idx={mu_pred})")
    print(f"  mean confidence = {mu_conf:.4f}")
    print(f"  predictive variance (pred class) = {mu_var:.6f}")

    # Showing where images were saved.
    print("\nSaved XAI:")
    print(f"  {str(out_dir / 'xai_gradcampp.png')}")
    print(f"  {str(out_dir / 'xai_attn_rollout.png')}")


# Standard Python entry point guard: run main() only when executed directly.
if __name__ == "__main__":
    main()
