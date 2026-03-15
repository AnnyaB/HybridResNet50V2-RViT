# THIS IS THE DEMO APPLICATION CODE 
# IT LETS USER UPLOAD AN MRI IMAGE, THEN RUNS INFERENCE AND XAI ON 4 MODELS DEFINED IN models_registry.json
# THE XAI OVERLAYS ARE COMPUTED USING THE SCRIPTS FOR GRAD-CAM++ AND ATTENTION ROLLOUT IN MY REPO, 
# BUT ADAPTED TO WORK SAFELY IN THIS WEB SERVER CONTEXT.

# External libraries used here are cited in Appendix A2.3:
# Flask (Pallets, n.d.); NumPy (Harris et al., 2020);
# PyTorch (Paszke et al., 2019); Matplotlib (Hunter, 2007);
# Pillow (Clark and contributors, n.d.).


# THESE ARE THE LIBARIES I USED IN THIS APP WHICH COULDN'T BE ALL DONE BY ME FROM SCRATCH  

import os  # OS utilities (env vars, paths, etc.)
os.environ["MPLBACKEND"] = "Agg"  # Forcing Matplotlib to use a non-GUI backend \

import io  # In-memory bytes buffers (BytesIO)
import sys  # Python runtime/system features (sys.path manipulation)
import json  # JSON parsing for the model registry and checkpoint metadata
import base64  # Convert PNG bytes <-> base64 strings for HTML embedding
import math  # Math functions/constants (kept for utility, may be unused)
import inspect  # Introspection (signature/parameters, class detection)
import hashlib  # Hashing (unique module naming)
import importlib.util  # Dynamic import utilities (load modules from file paths)
from pathlib import Path  # Standard library: modern path handling

import numpy as np  # NumPy: array ops for image conversion and probability handling
import torch  # PyTorch: tensors, models, device management, inference and grads
import torch.nn.functional as F  # PyTorch functional ops: relu, interpolate, softmax, etc.

import matplotlib  # Matplotlib core (configured for non-interactive plotting)
matplotlib.use("Agg")  # Explicitly set Matplotlib backend again (belt-and-suspenders safety)
import matplotlib.pyplot as plt  # Plotting API used to create bar charts and overlays

from PIL import Image  # Pillow: image loading, conversion, resizing
from flask import Flask, render_template, request  # Flask: web server, templating and request handling


APP_ROOT = Path(__file__).resolve().parent  # Directory containing this app.py file
PROJECT_ROOT = APP_ROOT.parent  # Parent directory (project root) used for import path fallback
REGISTRY_PATH = APP_ROOT / "models_registry.json"  # JSON file telling the app where models live

DEFAULT_CLASS_NAMES = ["glioma", "meningioma", "pituitary", "notumor"]  # Default label order


# Registry / device

def _read_registry():
    
    # Read and parse models_registry.json so I know which repos/checkpoints to load.
    if not REGISTRY_PATH.exists():  # Guard: fail fast if registry JSON is missing
        raise FileNotFoundError(f"Missing registry: {REGISTRY_PATH}")  # Clear error message
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:  # Open registry file safely
        return json.load(f)  # Parse JSON into a Python dict/list structure


def _resolve_device(device_str):
    
    # Deciding which torch.device to use based on registry setting and CUDA availability.
    s = (device_str or "auto").lower()  # Normalize string; default to auto if None/empty
    if s == "auto":  # auto means: use GPU if available, otherwise CPU
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if s == "cuda" and not torch.cuda.is_available():  # If requested CUDA but it's not there
        return torch.device("cpu")  # Fall back to CPU to avoid crashing
    if s in ("cpu", "cuda"):  # If device string is valid and available
        return torch.device(s)  # Return the requested device
    return torch.device("cpu")  # Safe default for unknown inputs



# dynamic importing

def _unique_module_name(repo_dir, py_file):
    
    # Creating a unique (but stable) module name so importing multiple repos doesn't collide.
    h = hashlib.md5((str(repo_dir) + "|" + str(py_file)).encode("utf-8")).hexdigest()[:10]  # Hash
    return f"mri_mod_{h}"  # Prefix and short hash => unique module name


def _load_module_from_file(repo_dir, py_file):
    
    # Loading a Python file as an imported module *without* requiring it to be installed as a package.
    if not py_file.exists():  # Guard: ensure the file exists before importing
        raise FileNotFoundError(f"Python file not found: {py_file}")  # Helpful error

    added = []  # Tracking which paths we temporarily insert into sys.path
    for p in (str(repo_dir), str(PROJECT_ROOT)):  # Add repo_dir and project root to module search path
        if p not in sys.path:  # Only add if not already present
            sys.path.insert(0, p)  # Inserting at front so it takes precedence for imports
            added.append(p)  # Remembering so I can remove later

    try:
        mod_name = _unique_module_name(repo_dir, py_file)  # Computing collision-safe module name
        spec = importlib.util.spec_from_file_location(mod_name, str(py_file))  # Building import spec
        if spec is None or spec.loader is None:  # Guard: spec must contain a loader to execute
            raise ImportError(f"Could not load module spec from: {py_file}")  # Clearing failure reason
        module = importlib.util.module_from_spec(spec)  # Creating an empty module object
        spec.loader.exec_module(module)  # Executing the file code inside that module object
        return module  # Returning the loaded module for attribute access (classes/functions)
    finally:
        for p in added:  # Always clean up sys.path modifications
            if p in sys.path:  # Only removing if still present
                sys.path.remove(p)  # Restoring original import environment


def _pick_model_ctor(module):
    
    # Choosing which model class (constructor) to instantiate from an imported module.
    if hasattr(module, "HybridResNet50V2_RViT"):  # Preferred: explicit known class name
        return getattr(module, "HybridResNet50V2_RViT")  # Returning that class directly

    candidates = []  # Otherwise, discovering candidate torch.nn.Module classes in that module
    for name in dir(module):  # Iterating over all symbols in the module namespace
        obj = getattr(module, name)  # Fetching the symbol by name
        try:
            # Keeping only classes that subclass torch.nn.Module
            if inspect.isclass(obj) and issubclass(obj, torch.nn.Module):
                # ...and were defined in this file/module (not imported from elsewhere)
                if getattr(obj, "__module__", "") == getattr(module, "__name__", ""):
                    candidates.append(obj)  # Adding as possible model class
        except Exception:
            continue  # If issubclass fails for weird objects, just skipping them safely

    if not candidates:  # If nothing was found, I cannot instantiate any model
        raise AttributeError("No torch.nn.Module class found in the model file.")  # Hard stop

    for c in candidates:  # Prefer classes that look hybrid by name, if multiple exist
        if "hybrid" in c.__name__.lower():  # Heuristic: name contains hybrid
            return c  # Returning the most likely intended model class
    return candidates[0]  # Otherwise returning the first discovered candidate as fallback


def _filter_kwargs_for_signature(ctor, kwargs):
    
    # Filtering a kwargs dict so we only pass parameters the model constructor actually accepts.
    sig = inspect.signature(ctor.__init__)  # Getting the __init__ signature of the class
    allowed = set(sig.parameters.keys())  # Collect parameter names (including self)
    allowed.discard("self")  # Removing self because we never pass it explicitly
    return {k: v for k, v in kwargs.items() if k in allowed}  # Keeping only accepted keys


def _instantiate_model(ctor, cfg, class_names, force_kwargs):
    
    # Building a base config with defaults, then applying any forced overrides from the registry.
    base = {
        "num_classes": cfg.get("num_classes", len(class_names)),  # Outputting classes
        "img_size": cfg.get("img_size", 224),  # Input image size
        "patch_size": cfg.get("patch_size", 16),  # ViT patch size
        "embed_dim": cfg.get("embed_dim", 142),  # Transformer embedding width
        "depth": cfg.get("depth", 10),  # Transformer depth (#blocks)
        "heads": cfg.get("heads", 10),  # Attention heads
        "mlp_dim": cfg.get("mlp_dim", 480),  # MLP hidden size in transformer blocks
        "attn_dropout": cfg.get("attn_dropout", 0.1),  # Attention dropout prob
        "vit_dropout": cfg.get("vit_dropout", 0.1),  # General transformer dropout prob
        "fusion_dim": cfg.get("fusion_dim", 256),  # Fusion MLP hidden size
        "fusion_dropout": cfg.get("fusion_dropout", 0.5),  # Fusion dropout prob
        "rotations": tuple(cfg.get("rotations", (0, 1, 2, 3))),  # Rotations used by RViT logic
        "cnn_name": cfg.get("cnn_name", "resnetv2_50x1_bitm"),  # timm CNN backbone id
        "cnn_pretrained": False,  # Web demo uses checkpoint weights, not ImageNet preload
    }
    if isinstance(force_kwargs, dict):  # Only applying overrides if given as a dict
        base.update(force_kwargs)  # Registry can enforce model-specific settings (for instance., use_pfd_gste)

    safe = _filter_kwargs_for_signature(ctor, base)  # Dropping config keys the ctor doesn't accept
    return ctor(**safe)  # Instantiating the model with compatible kwargs only


def _load_checkpoint(path, device):
    
    # Loading a PyTorch checkpoint from disk to a target device (cpu/cuda), without any absolute paths.
    if not path.exists():  # Guard: checkpoint file must exist
        raise FileNotFoundError(f"Checkpoint not found: {path}")  # Clearing error
    return torch.load(str(path), map_location=device)  # Loading with device mapping (no absolute paths)


def _extract_state_dict(ckpt):
    
    # Extracting the actual weights dict from various possible checkpoint formats.
    keys = ["model_state", "state_dict", "model_state_dict", "model", "net", "weights"]  # Common keys
    for k in keys:  # Trying each key in priority order
        if k in ckpt and isinstance(ckpt[k], dict):  # Must exist and be a dict
            return ckpt[k]  # Returning the contained state dict
    # Alternate format: checkpoint itself is already a state_dict mapping param_name -> Tensor
    if isinstance(ckpt, dict) and ckpt and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt  # Treat entire dict as the state_dict
    raise KeyError("Checkpoint missing weights (expected model_state/state_dict/etc).")  # Nothing usable found



# Image and plotting helpers I created for easy viewing

def preprocess_image(pil_img, mean, std, img_size):

    # Converting PIL -> normalized torch tensor in NCHW format and returning RGB numpy image for plotting.
    img = pil_img.convert("RGB").resize((img_size, img_size))  # Ensuring 3-channel RGB and fixed size
    x = torch.from_numpy(np.array(img)).float() / 255.0  # HWC uint8 -> float32 in [0,1]
    x = x.permute(2, 0, 1).contiguous()  # Reordering HWC -> CHW (PyTorch convention), make memory contiguous
    mean_t = torch.tensor(mean, dtype=x.dtype).view(3, 1, 1)  # Broadcastable mean (3x1x1)
    std_t = torch.tensor(std, dtype=x.dtype).view(3, 1, 1)  # Broadcastable std (3x1x1)
    x = (x - mean_t) / std_t  # Normalizing per-channel: (x-mean)/std
    return x.unsqueeze(0), np.array(img)  # Adding batch dim => 1x3xHxW, plus RGB array for overlay plotting


def normalize_map(h):
    
    # Normalizing a heatmap tensor so its values lie in [0,1].
    h = h - h.min()  # Shifting min to 0
    h = h / (h.max() + 1e-8)  # Scaling max to ~1 (avoiding divide-by-zero)
    return h  # Returning normalized heatmap


def fig_to_base64(fig):

    # Rendering a Matplotlib figure to PNG bytes and returning a base64 string for HTML <img src="...">.
    buf = io.BytesIO()  # In-memory bytes buffer
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")  # Save figure as PNG at decent DPI
    plt.close(fig)  # Closing to free memory (important for web servers)
    buf.seek(0)  # Rewinding buffer to beginning before reading
    return base64.b64encode(buf.read()).decode("utf-8")  # bytes -> base64 -> str


def overlay_base64(img_rgb, heat, title):

    # Creating a figure with the original image and semi-transparent heat overlay.
    fig = plt.figure(figsize=(4, 4))  # Creating a 4x4 inch figure
    plt.imshow(img_rgb)  # Drawing the base RGB image
    plt.imshow(heat, alpha=0.45)  # Drawing heatmap on top with transparency
    plt.title(title)  # Adding plot title
    plt.axis("off")  # Hiding axes for cleaner display
    plt.tight_layout()  # Reducing whitespace padding
    return fig_to_base64(fig)  # Converting the figure to base64 PNG


def barplot_base64(labels, probs, title):
    
    # Making a class-probability bar chart and returning it as base64.
    fig = plt.figure(figsize=(5, 3))  # Creating a 5x3 inch figure
    plt.bar(labels, probs)  # Bar chart: x=labels, y=probabilities
    plt.ylim(0.0, 1.0)  # Fix y-axis to probability range
    plt.title(title)  # Adding title
    plt.xticks(rotation=20, ha="right")  # Rotating labels so they fit better
    plt.tight_layout()  # Reducing whitespace
    return fig_to_base64(fig)  # Rendering and base64-encoding


def tumorprob_base64(tumor_p, notumor_p, title):
    
    # Making a 2-bar chart for Tumor (any) vs No tumor and returning it as base64.
    fig = plt.figure(figsize=(4.5, 3))  # Creating a 4.5x3 inch figure
    plt.bar(["Tumor (any)", "No tumor"], [tumor_p, notumor_p])  # Two bars with the given values
    plt.ylim(0.0, 1.0)  # Probability range
    plt.title(title)  # Adding title
    plt.tight_layout()  # Reducing whitespace
    return fig_to_base64(fig)  # Converting figure to base64 PNG



# XAI 

def _pick_last_4d_tensor(obj):
    # timm CNN often returns list/tuple of features; picking last stage (used downstream)
    if isinstance(obj, torch.Tensor) and obj.ndim == 4:  # If output is already a 4D feature map: N,C,H,W
        return obj  # Using it directly
    if isinstance(obj, (list, tuple)):  # If output is a list/tuple of intermediate feature maps
        for t in reversed(obj):  # Searching from the end (deepest stage first)
            if isinstance(t, torch.Tensor) and t.ndim == 4:  # Accepting only 4D tensors
                return t  # Returingn the deepest valid feature map
    return None  # If nothing matches, signal failure


def _safe_forward_return_xai(model, x):
    
    # Trying to call model(x, return_xai=True); if unsupported, just call model(x).
    try:
        out = model(x, return_xai=True)  # Preferred: models that support returning extra XAI artifacts
    except TypeError:
        return model(x), None  # Fallback: standard forward returns logits only

    if isinstance(out, (tuple, list)):  # If model returns multiple outputs (logits, xai)
        logits = out[0]  # First item is assumed to be logits
        xai = out[1] if len(out) > 1 else None  # Second item (if present) is XAI dict/structure
        return logits, xai  # Returning logits and xai
    return out, None  # If it wasn’t a tuple/list, treat it as logits only


def _gradcam_pp(A, grads, xai_module):
    
    # Prefer your XAI module’s Grad-CAM++ if it exists, else use internal formula.
    if xai_module is not None and hasattr(xai_module, "gradcam_pp_from_activations"):  # Checking helper exists
        return xai_module.gradcam_pp_from_activations(A, grads)  # Delegating to my project’s implementation

    grad_1 = grads  # First-order gradients dY/dA
    grad_2 = grad_1 ** 2  # Square of gradients
    grad_3 = grad_2 * grad_1  # Cube of gradients
    spatial_sum = (A * grad_3).sum(dim=(2, 3), keepdim=True)  # Sum across spatial dims (H,W)
    denom = 2.0 * grad_2 + spatial_sum + 1e-8  # Denominator for alpha weights (epsilon avoids div-by-zero)
    alpha = grad_2 / denom  # Grad-CAM++ alpha coefficients per pixel
    w = (alpha * F.relu(grad_1)).sum(dim=(2, 3), keepdim=True)  # Channel weights (sum over H,W)
    cam = (w * A).sum(dim=1, keepdim=True)  # Weighted sum over channels => 1x1xH xW
    cam = F.relu(cam).squeeze(0).squeeze(0)  # ReLU and drop batch+channel dims => HxW
    return normalize_map(cam)  # Normalize to [0,1] for visualization


def _attention_rollout(attn_list, xai_dict, xai_module):
    
    # Computing attention rollout via your xai module, adapting side if needed.
    if xai_module is None or not hasattr(xai_module, "attention_rollout"):  # Ensuring function exists
        return None  # If missing, we cannot compute rollout

    fn = xai_module.attention_rollout  # Grabbing rollout function
    sig = inspect.signature(fn)  # Inspecting parameters so we can pass only what it accepts
    kwargs = {}  # Collecting optional kwargs

    # Some of my scripts accept side=... (GSTE) or side=14 (no GSTE).
    if "side" in sig.parameters:  # Only passing side if the function supports it
        if isinstance(xai_dict, dict) and xai_dict.get("gste_side", None) is not None:  # If GSTE provides side
            kwargs["side"] = xai_dict.get("gste_side")  # Using the model-reported token grid side length
        else:
            # safe default for 224/16 when no GSTE
            kwargs["side"] = 14  # 224px / 16px patches = 14 patches per side

    return fn(attn_list, **kwargs)  # Running rollout and return its output (typically a token/heat map)


def _compute_xai_overlays(model, x, img_rgb, img_size, pred_idx, xai_module, force_use_pfd_gste):
    
    """
    Trying PFD hook first (if active), else CNN hook.
    This is suitable for my scripts and avoids the ablation-B Grad-CAM failure.
    """
    overlays = {"gradcam": None, "rollout": None}  # Outputting structure initialized to not available

    def run_with_hook(hook_kind):
        hook_cache = {}  # Storage for captured activations ("A") from the forward hook

        def _pfd_hook(_m, _inp, out):
            # PFD returns (feat_path, mask_feat)
            if isinstance(out, (tuple, list)) and len(out) >= 1:  # PFD might output tuple/list
                feat = out[0]  # First item is the feature tensor used for Grad-CAM++
            else:
                feat = None  # If unexpected output structure, mark as None
            if isinstance(feat, torch.Tensor):  # Only proceed if we captured a tensor
                feat.retain_grad()  # Ensuring gradients for this activation are retained during backward()
                hook_cache["A"] = feat  # Storing activation for later Grad-CAM++ computation

        def _cnn_hook(_m, _inp, out):
            feat = _pick_last_4d_tensor(out)  # Extracting a 4D tensor from CNN outputs (handles list/tuple)
            if isinstance(feat, torch.Tensor):  # Proceeding only if valid tensor
                feat.retain_grad()  # Retaining activation gradients
                hook_cache["A"] = feat  # Caching activation for Grad-CAM++

        target_mod = None  # The module we will attach the hook to (either model.pfd or model.cnn)
        hook_fn = None  # Which hook function to register

        if hook_kind == "pfd":  # If we are targeting PFD features
            if hasattr(model, "pfd"):  # Only if model actually has PFD module
                target_mod = model.pfd  # Hook PFD module
                hook_fn = _pfd_hook  # Using PFD-specific hook logic
        else:  # Otherwise, target the CNN backbone
            if hasattr(model, "cnn"):  # Only if model has a CNN submodule
                target_mod = model.cnn  # Hook CNN module
                hook_fn = _cnn_hook  # Using CNN hook logic

        if target_mod is None:  # If module not present, we cannot hook this path
            return None, None, None  # Returning no result

        h = target_mod.register_forward_hook(hook_fn)  # Registering forward hook to capture activations
        try:
            with torch.enable_grad():  # Enabling gradients (needed for Grad-CAM backward pass)
                logits, xai = _safe_forward_return_xai(model, x)  # Running forward and (maybe) get XAI dict
                if isinstance(logits, (tuple, list)):  # Defensive: if logits wrapped again
                    logits = logits[0]  # Extracting logits

                model.zero_grad(set_to_none=True)  # Clearing gradients from previous passes
                logits[0, pred_idx].backward(retain_graph=True)  # Backpropagating predicted-class score

                A = hook_cache.get("A", None)  # Retrieving captured activation tensor
                if not (isinstance(A, torch.Tensor) and A.grad is not None and A.ndim == 4):  # Validating grads
                    return logits, xai, None  # If activation/grads missing, Grad-CAM cannot be computed

                cam_small = _gradcam_pp(A, A.grad, xai_module)  # Computing Grad-CAM++ at feature-map resolution
                cam = F.interpolate(
                    cam_small.unsqueeze(0).unsqueeze(0),  # Adding batch+channel dims => 1x1xH xW
                    size=(img_size, img_size),  # Upsampling to input image resolution
                    mode="bilinear",  # Smooth interpolation
                    align_corners=False,  # Standard safe setting for bilinear interpolation
                ).squeeze().detach().cpu().numpy()  # Removing dims, detaching, moving to CPU, converting to NumPy

                return logits, xai, cam  # Return logits, XAI dict, and full-size CAM heatmap
        finally:
            h.remove()  # Always removing hook to prevent leaks or double-hooking on later calls

    # Deciding whether to even try PFD first:
    # If force_use_pfd_gste is explicitly False (ablation B), skipping PFD.
    try_pfd = (force_use_pfd_gste is not False) and hasattr(model, "pfd")  # PFD only if allowed and exists

    logits = None  # Placeholder for logits (may be set in hook run)
    xai = None  # Placeholder for xai dict (may be set in hook run)
    cam = None  # Placeholder for Grad-CAM heatmap

    if try_pfd:  # If I should attempt PFD hook path first
        logits, xai, cam = run_with_hook("pfd")  # Running forward+backward with PFD activation capture

    if cam is None:  # If PFD failed or was skipped, try CNN hook path
        logits, xai, cam = run_with_hook("cnn")  # Running forward and backward with CNN activation capture

    if cam is not None:  # If I successfully computed a heatmap
        overlays["gradcam"] = overlay_base64(img_rgb, cam, "Grad-CAM++")  # Converting overlay to base64 PNG

    # attention rollout
    if isinstance(xai, dict):  # Rollout requires an XAI dict with attention matrices
        attn_list = xai.get("attn") or xai.get("attn_list")  # Accepting either key name
        if attn_list:  # Only proceeding if we have attention data
            tok = _attention_rollout(attn_list, xai, xai_module)  # Computing rollout token map
            if isinstance(tok, torch.Tensor):  # Ensuring we got a tensor back
                tok_img = F.interpolate(
                    tok.unsqueeze(0).unsqueeze(0),  # Adding batch and channel dims
                    size=(img_size, img_size),  # Upsampling to image size
                    mode="bilinear",  # Smoothing interpolation
                    align_corners=False,  # Safe default
                ).squeeze().detach().cpu().numpy()  # Converting to NumPy heatmap
                overlays["rollout"] = overlay_base64(img_rgb, tok_img, "Attention rollout")  # Base64 overlay

    return overlays  # Returning dict with (maybe) gradcam + rollout overlays


def run_inference_with_xai(loaded, pil_img):
    
    device = loaded["device"]  # Torch device to run model on (cpu/cuda)
    model = loaded["model"]  # The instantiated PyTorch model
    class_names = loaded["class_names"]  # Class labels for the model outputs
    xai_module = loaded.get("xai_module", None)  # Optional module containing attention_rollout/gradcam helpers

    img_size = int(loaded["cfg"].get("img_size", 224))  # Model input image size (default 224)
    x, img_rgb = preprocess_image(pil_img, loaded["mean"], loaded["std"], img_size)  # Preprocess to tensor and RGB
    x = x.to(device)  # Moving input batch to correct device

    with torch.no_grad():  # Disabling gradient tracking for the main probability forward pass
        out = model(x)  # Running model forward
        logits = out[0] if isinstance(out, (tuple, list)) else out  # Extracting logits if model returns tuple/list
        probs_t = torch.softmax(logits, dim=1).squeeze(0)  # Converting logits -> probabilities, drop batch dim
        probs = probs_t.detach().cpu().numpy()  # Moving to CPU and converting to NumPy array for easier handling

    pred_idx = int(np.argmax(probs))  # Index of the highest-probability class
    conf = float(probs[pred_idx])  # Confidence = max probability

    if "notumor" in class_names:  # If notumor exists, compute tumor probability as sum of other classes
        notumor_idx = class_names.index("notumor")  # Finding the notumor class index
        tumor_prob = float(np.sum([p for i, p in enumerate(probs) if i != notumor_idx]))  # Sum of non-notumor
        notumor_prob = float(probs[notumor_idx])  # Probability of notumor
    else:
        tumor_prob = float(np.sum(probs[:3]))  # Fallback: assume first 3 are tumor types
        notumor_prob = float(1.0 - tumor_prob)  # Remaining mass treated as notumor

    overlays = _compute_xai_overlays(
        model=model,  # Pass model for hook-based Grad-CAM++
        x=x,  # Input batch tensor (needs gradients inside XAI function)
        img_rgb=img_rgb,  # RGB image array for overlay display
        img_size=img_size,  # Target upsample size for heatmaps
        pred_idx=pred_idx,  # Class index to backprop for Grad-CAM++
        xai_module=xai_module,  # Optional helpers module
        force_use_pfd_gste=loaded.get("force_use_pfd_gste", None),  # Allow skipping PFD in certain ablations
    )

    probs_chart_b64 = barplot_base64(class_names, probs, "Class probabilities")  # Base64 class-probability plot
    tumor_chart_b64 = tumorprob_base64(tumor_prob, notumor_prob, "Tumor vs No-tumor")  # Base64 tumor vs no

    return {
        "model_id": loaded["id"],  # Registry model id
        "model_name": loaded["name"],  # Friendly model name
        "pred_class": class_names[pred_idx],  # Predicted label string
        "pred_idx": pred_idx,  # Predicted label index
        "confidence": conf,  # Max class probability
        "probs": [float(p) for p in probs],  # Full probability vector as JSON-friendly floats
        "class_names": class_names,  # Label list (for front-end display)
        "tumor_prob": tumor_prob,  # Aggregate tumor probability
        "notumor_prob": notumor_prob,  # Notumor probability
        "chart_probs": probs_chart_b64,  # Base64 PNG for class bar chart
        "chart_tumor": tumor_chart_b64,  # Base64 PNG for tumor vs no-tumor chart
        "xai_gradcam": overlays["gradcam"],  # Base64 PNG for Grad-CAM++ overlay (or None)
        "xai_rollout": overlays["rollout"],  # Base64 PNG for attention rollout overlay (or None)
    }



# Load all models and their XAI modules

def load_all_models():
    reg = _read_registry()  # Load JSON registry dict
    device = _resolve_device(reg.get("device", "auto"))  # Decide device based on registry and availability

    loaded = []  # List that will hold loaded model entries
    models = reg.get("models", [])  # List of model configurations from registry
    if not models:  # Guard: registry must define at least one model
        raise RuntimeError("No models defined in models_registry.json")  # Hard fail

    for m in models:  # Iterating over each model config entry
        repo_dir = (APP_ROOT / m["repo_dir"]).resolve()  # Resolving repo directory path relative to app
        ckpt_path = (repo_dir / m.get("checkpoint", "best_model.pt")).resolve()  # Resolving checkpoint file

        model_file = (repo_dir / m["model_file"]).resolve()  # Resolving path to Python model definition file
        xai_file = (repo_dir / m.get("xai_file", "")).resolve() if m.get("xai_file") else None  # Optional XAI file

        model_module = _load_module_from_file(repo_dir, model_file)  # Dynamically import model code
        ctor = _pick_model_ctor(model_module)  # Picking which torch.nn.Module class to instantiate

        ckpt = _load_checkpoint(ckpt_path, device)  # Loading checkpoint dict onto chosen device

        class_names = ckpt.get("class_names", DEFAULT_CLASS_NAMES)  # Loading class labels from checkpoint (fallback)
        mean = ckpt.get("mean", [0.5, 0.5, 0.5])  # Loading normalization mean from checkpoint (fallback)
        std = ckpt.get("std", [0.5, 0.5, 0.5])  # Loading normalization std from checkpoint (fallback)
        cfg = ckpt.get("model_cfg", {}) or {}  # Loading model config dict (ensure not None)
        force_kwargs = m.get("force_kwargs", {}) or {}  # Registry-enforced kwargs overrides

        model = _instantiate_model(ctor, cfg, class_names, force_kwargs)  # Building model with safe kwargs
        state = _extract_state_dict(ckpt)  # Extracting weights dictionary from checkpoint

        missing, unexpected = model.load_state_dict(state, strict=False)  # Loading weights, allow mismatch keys
        if missing or unexpected:  # If there are keys that didn't match, warn but keep running
            print(f"[WARN] {m.get('id')} missing: {len(missing)}, unexpected: {len(unexpected)}")  # Log counts

        model.to(device)  # Moving model parameters to the chosen device
        model.eval()  # Switching to inference mode (disables dropout, uses running stats, etc.)

        xai_module = None  # Placeholder for optional loaded XAI module
        if xai_file is not None and xai_file.exists():  # Only attempt XAI import if path exists
            try:
                xai_module = _load_module_from_file(repo_dir, xai_file)  # Dynamically import XAI helpers
            except Exception as e:
                print(f"[WARN] Could not load XAI module for {m.get('id')}: {repr(e)}")  # Warn on failure
                xai_module = None  # Ensuring xai_module stays None if import fails

        loaded.append({
            "id": m.get("id", ""),  # Model id used in UI/logging
            "name": m.get("name", ""),  # Friendly model name for UI
            "repo_dir": str(repo_dir),  # Repo directory (string for JSON friendliness)
            "model": model,  # Instantiated model object
            "device": device,  # Device to run on
            "class_names": list(class_names),  # Ensuring list type
            "mean": list(mean),  # Ensuring list type
            "std": list(std),  # Ensuring list type
            "cfg": dict(cfg),  # Ensuring dict type
            "xai_module": xai_module,  # Optional XAI module
            "force_use_pfd_gste": force_kwargs.get("use_pfd_gste", None),  # Remembering if registry forces ablation
        })

    return loaded  # Returning list of loaded model entries



# Flask app

app = Flask(__name__)  # Creating Flask application instance
LOADED_MODELS = load_all_models()  # Loading all models once at startup (avoids reloading per request)


@app.route("/", methods=["GET", "POST"])  # Defining the home route for both viewing and submitting an image
def index():
    results = None  # Will hold list of per-model inference outputs for template rendering
    uploaded_preview = None  # Base64 PNG preview for the uploaded image (224x224)
    error = None  # Erroring message string (if any)

    if request.method == "POST":  # If user submitted the form (uploaded an image)
        file = request.files.get("image")  # Getting the uploaded file by name= image from the HTML form
        if not file or file.filename == "":  # Validating that a file was actually provided
            error = "No image selected."  # Setting user-friendly error message
            return render_template("index.html", results=results, error=error, uploaded_preview=uploaded_preview)  # Render

        try:
            raw = file.read()  # Reading uploaded file bytes
            pil = Image.open(io.BytesIO(raw)).convert("RGB")  # Decoding image bytes into PIL and force RGB

            prev = pil.resize((224, 224))  # Creating small preview copy for the UI
            buf = io.BytesIO()  # In-memory buffering for PNG encoding
            prev.save(buf, format="PNG")  # Saving preview image as PNG into buffer
            uploaded_preview = base64.b64encode(buf.getvalue()).decode("utf-8")  # Buffer bytes -> base64 string

            results = []  # Initializing results list
            for lm in LOADED_MODELS:  # Running the same uploaded image through each loaded model
                results.append(run_inference_with_xai(lm, pil))  # Appending one dict per model for template display

        except Exception as e:
            error = f"Failed to process image or run inference: {repr(e)}"  # Catching-all error for robustness

    return render_template("index.html", results=results, error=error, uploaded_preview=uploaded_preview)  # Render page


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)  # Running local dev server on localhost:5000 with debug on
