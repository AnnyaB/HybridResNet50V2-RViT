# ==================================================================================================
# scripts/predict_xai_with_pfd_gste.py   (XAI B)
# ==================================================================================================
# This script is an explain one image runner for the full Hybrid B model (PFD and GSTE ON).
#
# What I implemented myself in this script:
# - A strict XAI execution path that always runs the full model mode so mask and attention exist.
# - Grad-CAM++ built from the PFD-gated CNN activation map and its gradients.
# - Attention rollout that chains attention across transformer blocks to get token importance.
# - Simple overlay utilities that save explanation images (CAM and attention) to disk.
# - MC-dropout inference wrapper usage to report mean prediction and predictive variance.
#
# What I import from libraries:
# - sys, pathlib.Path: clean path and import handling so the script runs from anywhere.
# - numpy: image array conversion, sqrt/argmax helpers, and lightweight numeric ops.
# - torch and torch.nn.functional: tensors, autograd, interpolation, and activations.
# - matplotlib: saving overlays to files (Agg backend so it works without a display).
# - PIL.Image: reading/resizing input images.
# - HybridResNet50V2_RViT: the model class that returns (logits, xai_payload).
# ==================================================================================================


import sys  # I use sys.path edits so local imports work no matter my current working directory.
from pathlib import Path  # I use Path for cross-platform filesystem paths and clean joins.

import numpy as np  # I use NumPy for PIL->array conversion and small numeric utilities.

import torch  # I use torch for tensors, autograd, device placement, and model execution.
import torch.nn.functional as F  # I use F for relu/interpolate/pooling without creating new modules.

import matplotlib  # use matplotlib to write overlay images to disk.
matplotlib.use("Agg")  # force a headless backend so saving works on servers/CI (no GUI needed).

import matplotlib.pyplot as plt  # use pyplot to draw and save overlay figures.

from PIL import Image  # use PIL to load the input image file reliably.


# I define ROOT as the folder containing this script file.
# That gives me a stable “project-relative” anchor for imports and output paths.
ROOT = Path(__file__).resolve().parent  # resolve() follows symlinks; parent picks the containing folder.

# If my project root isn’t importable yet, I push it onto sys.path.
# This prevents ModuleNotFoundError when the script is run from a different directory.
if str(ROOT) not in sys.path:  # check as string because sys.path is a list of strings.
    sys.path.insert(0, str(ROOT))  # put it first so local modules win over similarly named installed ones.

# Now I can import the model class from the project package.
from models.hybrid_model import HybridResNet50V2_RViT  # this model returns logits and optional XAI payload.


def preprocess_pil(img, mean, std):
    # --- Goal: turn a PIL image into a normalized torch tensor shaped (1,3,224,224). ---

    img = img.convert("RGB")  # I force 3 channels so the model always sees consistent input shape.

    # Convert PIL -> NumPy -> float tensor scaled into [0,1].
    # At this moment, the model would “see” real-valued pixel intensities instead of uint8.
    x = torch.from_numpy(np.array(img)).float() / 255.0  # (H,W,3) -> float32 and scaled.

    # PIL/NumPy uses HWC layout; PyTorch CNNs expect CHW layout.
    x = x.permute(2, 0, 1).contiguous()  # (H,W,C) -> (C,H,W), contiguous for safe downstream ops.

    # I convert mean/std lists into tensors shaped (3,1,1) so they broadcast across H and W.
    mean_t = torch.tensor(mean, dtype=x.dtype).view(3, 1, 1)  # channel-wise mean for normalization.
    std_t = torch.tensor(std, dtype=x.dtype).view(3, 1, 1)  # channel-wise std for normalization.

    # Standardization: the model sees inputs in the same scale distribution as training.
    x = (x - mean_t) / std_t  # (C,H,W) normalized per channel.

    # Add batch dimension because the model expects (B,C,H,W).
    return x.unsqueeze(0)  # (C,H,W) -> (1,C,H,W)


def normalize_map(h):
    # --- Goal: map any heatmap-like tensor to [0,1] so overlays behave consistently. ---

    h = h - h.min()  # shift so smallest value becomes 0.
    h = h / (h.max() + 1e-8)  # scale so largest value becomes 1; eps avoids divide-by-zero.
    return h  # return normalized heatmap.


def gradcam_pp_from_activations(A, grads):
    # --- Goal: compute Grad-CAM++ heatmap from an activation map A and its gradients grads. ---

    if A is None:  # if A is missing, the forward hook didn’t capture what I expected.
        raise RuntimeError("Activation A is None (hook failed).")

    if grads is None:  # if grads are missing, backward didn’t flow into A.
        raise RuntimeError("Gradients missing. Ensure enable_grad + backward ran.")

    # Grad-CAM++ uses 1st/2nd/3rd powers of gradients to get stable per-location weights.
    grad_1 = grads  # first-order gradients: d(logit)/dA
    grad_2 = grad_1 ** 2  # squared gradients: emphasize strong gradient magnitudes.
    grad_3 = grad_2 * grad_1  # cubed gradients: used in the alpha denominator term.

    # This term captures how activations interact with third-order grads across space.
    spatial_sum = (A * grad_3).sum(dim=(2, 3), keepdim=True)  # sum over H,W, keep shape for broadcasting.

    # Denominator stabilizes alpha so weights don’t blow up when gradients are small.
    denom = 2.0 * grad_2 + spatial_sum + 1e-8  # eps prevents division by zero.

    # Alpha is a per-location scalar telling how much each pixel-location should contribute.
    alpha = grad_2 / denom  # same shape as grads: (B,C,H,W)

    # Channel weights: sum over spatial locations, using positive gradients as helpful evidence.
    w = (alpha * F.relu(grad_1)).sum(dim=(2, 3), keepdim=True)  # (B,C,1,1)

    # CAM: weighted sum of activation channels produces a single heatmap.
    cam = (w * A).sum(dim=1, keepdim=True)  # (B,1,H,W)

    # ReLU keeps only regions that increase the target logit (positive evidence).
    cam = F.relu(cam).squeeze(0).squeeze(0)  # (B,1,H,W) -> (H,W) for single-image batch.

    return normalize_map(cam)  # normalize to [0,1] for clean overlay.


def attention_rollout(attn_list, side=None, eps=1e-6):
    # --- Goal: turn a list of attention matrices into a single token-importance map. ---

    # The model returns attention per block only when return_xai=True.
    if not attn_list:
        raise RuntimeError("No attention matrices captured. Ensure return_xai=True.")

    mats = []  # I’ll store one normalized attention matrix per transformer layer.

    for attn in attn_list:  # iterate each block’s attention tensor.
        # attn shape is expected to be (B, heads, N, N).
        a = attn.mean(dim=1)  # average over heads -> (B, N, N) so I get one graph per layer.

        n = a.shape[-1]  # N tokens; when GSTE shrinks, N = side*side (smaller than 196).

        # I add identity so each token keeps some self-signal (like residual connections).
        a = a + torch.eye(n, device=a.device).unsqueeze(0)  # (1,N,N) broadcasts across batch.

        # Row-normalize so each query token distributes a total mass of 1.0.
        a = a / (a.sum(dim=-1, keepdim=True) + eps)  # eps prevents divide-by-zero.

        mats.append(a)  # store this layer’s normalized attention.

    R = mats[0]  # rollout starts from the first layer’s attention graph.

    # Multiply graphs across layers to model information flow through depth.
    for i in range(1, len(mats)):  # start at second layer.
        R = R @ mats[i]  # chain attentions: (B,N,N) @ (B,N,N) -> (B,N,N)

    # No CLS token here, so I summarize token importance by averaging over query tokens.
    # That produces one importance score per key token.
    r = R.mean(dim=1).squeeze(0)  # (B,N,N) -> (N,) for single-image batch.

    r = normalize_map(r)  # normalize scores into [0,1].

    N = int(r.shape[0])  # token count as a Python int (needed for reshape logic).

    if side is not None:
        side = int(side)  # enforce integer side length.

        # If side is provided, I treat it as authoritative and sanity-check the reshape.
        if side * side != N:
            raise RuntimeError(f"gste_side mismatch: side={side} but N={N}")

        return r.reshape(side, side)  # reshape into a 2D token grid.

    # If side wasn’t provided, I try to infer a near-square grid shape.
    ht = int(round(float(np.sqrt(N))))  # rough square root guess.
    ht = max(ht, 1)  # avoid zero height.
    wt = max(N // ht, 1)  # compute width.

    # If this inference doesn’t exactly match N, I fall back to a simple strip.
    if ht * wt != N:
        ht, wt = 1, N  # reshape as 1xN so nothing crashes.

    return r.reshape(ht, wt)  # return inferred 2D map (or strip).


def overlay_and_save(img_rgb, heat, out_path, title):
    # --- Goal: save an overlay visualization (base image + heatmap) to disk. ---

    plt.figure(figsize=(5, 5))  # create a new figure so previous plots don’t contaminate this one.
    plt.imshow(img_rgb)  # draw the original image first (what we are explaining).
    plt.imshow(heat, alpha=0.45)  # draw the heatmap on top with transparency.
    plt.title(title)  # label the method or target class so the output is self-describing.
    plt.axis("off")  # remove axes for clean visual output.
    plt.tight_layout()  # keep the saved crop tight and neat.
    plt.savefig(str(out_path), dpi=200)  # write to disk; dpi improves readability.
    plt.close()  # close figure to free memory (important for batch runs).


def main():
    # I keep argparse inside main so importing this file doesn’t run CLI setup automatically.
    import argparse  # local import keeps script import-time light.

    ap = argparse.ArgumentParser()  # create CLI parser.

    ap.add_argument("--checkpoint", type=str, required=True)  # trained weights + metadata.
    ap.add_argument("--image", type=str, required=True)  # image path to explain.
    ap.add_argument("--out_dir", type=str, default="results/xai_with_pfd_gste")  # output folder path.
    ap.add_argument("--mc_samples", type=int, default=20)  # number of MC-dropout passes.
    ap.add_argument("--target_class", type=int, default=-1)  # -1 => explain predicted class.

    args = ap.parse_args()  # parse CLI args.

    device = "cuda" if torch.cuda.is_available() else "cpu"  # prefer GPU if available.

    out_dir = Path(args.out_dir)  # convert output dir to Path for clean joins.
    out_dir.mkdir(parents=True, exist_ok=True)  # ensure output directory exists.

    ckpt = torch.load(args.checkpoint, map_location=device)  # load checkpoint on chosen device.

    # Class names: used only for printing + nicer titles.
    class_names = ckpt.get("class_names", ["glioma", "meningioma", "pituitary", "notumor"])

    # Normalization stats: these must match training preprocessing if available.
    mean = ckpt.get("mean", [0.5, 0.5, 0.5])  # fallback if checkpoint didn’t store them.
    std = ckpt.get("std", [0.5, 0.5, 0.5])  # fallback if checkpoint didn’t store them.

    cfg = ckpt.get("model_cfg", {})  # optional model config dict saved alongside weights.

    # ----------------------------------------------------------------------------------------------
    # FORCE FULL MODE: PFD + GSTE ON (XAI B)
    # ----------------------------------------------------------------------------------------------
    # The point of this script is to explain using:
    # - PFD mask (for ROI guidance + visualization)
    # - GSTE dynamic tokens (so attention map aligns to the actually-used token grid)
    model = HybridResNet50V2_RViT(
        num_classes=cfg.get("num_classes", len(class_names)),  # output logits width.
        patch_size=cfg.get("patch_size", 16),  # patch size controls token grid resolution.
        embed_dim=cfg.get("embed_dim", 142),  # token channel width D.
        depth=cfg.get("depth", 10),  # number of transformer blocks.
        heads=cfg.get("heads", 10),  # attention heads.
        mlp_dim=cfg.get("mlp_dim", 480),  # MLP hidden width in each block.
        attn_dropout=cfg.get("attn_dropout", 0.1),  # attention weight dropout.
        vit_dropout=cfg.get("vit_dropout", 0.1),  # transformer dropout.
        fusion_dim=cfg.get("fusion_dim", 256),  # fusion representation width.
        fusion_dropout=cfg.get("fusion_dropout", 0.5),  # dropout for fusion/CNN descriptor.
        rotations=tuple(cfg.get("rotations", (0, 1, 2, 3))),  # rotation set for RViT averaging.
        cnn_name=cfg.get("cnn_name", "resnetv2_50x1_bitm"),  # timm backbone name.

        cnn_pretrained=False,  # I don’t load backbone pretraining because checkpoint provides final weights.

        use_pfd_gste=True,  # critical: full mode must be ON so mask and dynamic tokens exist.
    ).to(device)  # move model parameters to device.

    state = ckpt.get("model_state") or ckpt.get("state_dict")  # support common checkpoint key patterns.
    if state is None:
        raise KeyError("Checkpoint missing weights ('model_state' or 'state_dict').")

    model.load_state_dict(state)  # load trained parameters into model.
    model.eval()  # put model in eval mode (deterministic unless we selectively re-enable dropout).

    # ----------------------------------------------------------------------------------------------
    # Load and preprocess image
    # ----------------------------------------------------------------------------------------------
    img = Image.open(args.image)  # open image file.
    img = img.convert("RGB")  # guarantee 3 channels.
    img = img.resize((224, 224))  # enforce expected size for backbone and patch embed.

    x = preprocess_pil(img, mean, std)  # (1,3,224,224) normalized tensor.
    x = x.to(device)  # move input to same device as model.

    # ----------------------------------------------------------------------------------------------
    # Hooks for Grad-CAM++: capture the PFD output feature map and its gradients
    # ----------------------------------------------------------------------------------------------
    hook_cache = {}  # I store activations here so the hook can pass data back to the main code.

    def _pfd_hook(module, inputs, output):
        # This hook runs during forward() on the PFD module.

        # PFD forward returns (feat_path, mask). I verify structure so failures aren’t silent.
        if not isinstance(output, (tuple, list)) or len(output) < 1:
            raise RuntimeError("PFD hook expected tuple/list output (feat_path, mask).")

        feat_path = output[0]  # gated CNN feature map (the exact thing the CNN branch uses downstream).
        feat_path.retain_grad()  # keep gradients on this tensor so backward() fills feat_path.grad.
        hook_cache["cnn_feat"] = feat_path  # store activation for CAM++ computation.

    h = model.pfd.register_forward_hook(_pfd_hook)  # register hook and keep handle so I can remove it.

    try:
        # CAM++ needs gradients, so I explicitly enable grad even though model is in eval().
        with torch.enable_grad():
            logits, xai = model(x, return_xai=True)  # forward pass and XAI payload.

        # The script expects a dict payload when return_xai=True.
        if not isinstance(xai, dict):
            raise RuntimeError("Expected xai dict from model when return_xai=True.")

        # In full mode, xai["mask"] must exist; if it’s None, something is wrong.
        if xai.get("mask", None) is None:
            raise RuntimeError("mask is None => PFD/GSTE not active, or model forward didn’t return it.")

        prob = torch.softmax(logits, dim=1).squeeze(0)  # convert logits to class probabilities.

        pred_idx = int(prob.argmax().item())  # predicted class index.
        conf = float(prob[pred_idx].item())  # predicted class confidence.

        target = pred_idx if args.target_class < 0 else int(args.target_class)  # which class to explain.

        model.zero_grad(set_to_none=True)  # clear stale gradients for a clean backward pass.

        logits[0, target].backward()  # backprop from target logit into hooked activation map.

        A = hook_cache.get("cnn_feat", None)  # retrieve activation captured by hook.

        # CAM++ needs both activation and its gradient.
        if A is None or A.grad is None:
            raise RuntimeError("Failed to capture gradients for Grad-CAM++ (A or A.grad is missing).")

    finally:
        h.remove()  # always remove hook even if something throws; prevents hook stacking across runs.

    # ----------------------------------------------------------------------------------------------
    # Build Grad-CAM++ heatmap (image-level)
    # ----------------------------------------------------------------------------------------------
    cam_small = gradcam_pp_from_activations(A, A.grad)  # CAM++ at feature map resolution (e.g., 7x7).

    cam = F.interpolate(  # resize CAM to 224x224 so it overlays directly on the input image.
        cam_small.unsqueeze(0).unsqueeze(0),  # (H,W) -> (1,1,H,W) for interpolate.
        size=(224, 224),  # match input resolution.
        mode="bilinear",  # smooth resizing.
        align_corners=False,  # safe bilinear behavior.
    ).squeeze()  # back to (224,224).

    cam = cam.detach().cpu().numpy()  # convert to NumPy for matplotlib plotting.

    # ----------------------------------------------------------------------------------------------
    # Build Attention Rollout heatmap (image-level)
    # ----------------------------------------------------------------------------------------------
    attn_list = xai.get("attn") or xai.get("attn_list")  # support either key name if model changes.
    if attn_list is None:
        raise KeyError("XAI dict missing attention list: expected 'attn' or 'attn_list'.")

    gste_side = xai.get("gste_side", None)  # side length used after GSTE shrink (may be <= 14).

    attn_tok = attention_rollout(attn_list, side=gste_side)  # token-importance map (side x side).

    attn_img = F.interpolate(  # resize token map to 224x224 for overlay on the image.
        attn_tok.unsqueeze(0).unsqueeze(0),  # (side,side) -> (1,1,side,side)
        size=(224, 224),  # image size.
        mode="bilinear",  # smooth upscale.
        align_corners=False,  # safe bilinear behavior.
    ).squeeze()  # back to (224,224)

    attn_img = attn_img.detach().cpu().numpy()  # convert to NumPy for plotting.

    # ----------------------------------------------------------------------------------------------
    # MC Dropout uncertainty (mean + variance)
    # ----------------------------------------------------------------------------------------------
    with torch.no_grad():  # no gradients needed for MC-dropout reporting.
        mu, var = model.mc_dropout_predict(x, mc_samples=args.mc_samples)  # mean prob + variance.

        mu = mu.squeeze(0).cpu().numpy()  # (1,C) -> (C,)
        var = var.squeeze(0).cpu().numpy()  # (1,C) -> (C,)

    mu_pred = int(np.argmax(mu))  # class index with highest mean probability.
    mu_conf = float(mu[mu_pred])  # mean confidence for that class.
    mu_var = float(var[mu_pred])  # predictive variance for that class.

    # ----------------------------------------------------------------------------------------------
    # Save overlays
    # ----------------------------------------------------------------------------------------------
    img_rgb = np.array(img)  # (H,W,3) uint8 image for matplotlib background.

    overlay_and_save(  # save Grad-CAM++ overlay.
        img_rgb,  # base image.
        cam,  # heatmap.
        out_dir / "xai_gradcampp.png",  # output path.
        f"Grad-CAM++ (PFD target={class_names[target]})",  # title string.
    )

    overlay_and_save(  # save attention rollout overlay.
        img_rgb,  # base image.
        attn_img,  # heatmap.
        out_dir / "xai_attn_rollout.png",  # output path.
        "Attention Rollout",  # title string.
    )

    # ----------------------------------------------------------------------------------------------
    # Print summary (console)
    # ----------------------------------------------------------------------------------------------
    print("Prediction (single pass):")  # header for single forward pass.
    print(f"  class = {class_names[pred_idx]} (idx={pred_idx})")  # predicted class label and index.
    print(f"  confidence = {conf:.4f}")  # predicted confidence.

    print("\nMC Dropout:")  # header for MC-dropout results.
    print(f"  mean-pred class = {class_names[mu_pred]} (idx={mu_pred})")  # mean prediction label and index.
    print(f"  mean confidence = {mu_conf:.4f}")  # mean confidence from MC samples.
    print(f"  predictive variance = {mu_var:.6f}")  # variance as a quick uncertainty signal.

    print("\nSaved XAI:")  # header for file outputs.
    print(out_dir / "xai_gradcampp.png")  # path to Grad-CAM++ overlay.
    print(out_dir / "xai_attn_rollout.png")  # path to attention rollout overlay.


if __name__ == "__main__":  # standard entrypoint guard so imports don’t auto-run main().
    main()  # run the script when executed directly.
