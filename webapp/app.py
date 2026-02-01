# webapp/app.py
# ------------------------------------------------------------
# Multi-model MRI demo web app:
# - Upload ONE image
# - Run 4 models
# - Show prediction + confidence + probability charts
# - XAI overlays:
#     * Grad-CAM++ (PFD output if active, else CNN last-stage feature map)
#     * Attention rollout (from xai["attn"] or xai["attn_list"])
#
# Goals:
# - No absolute paths
# - model locations via models_registry.json (relative paths allowed)
# - isolate imports to avoid collisions
# - minimal imports (no typing, no dataclasses)
# ------------------------------------------------------------

import os
os.environ["MPLBACKEND"] = "Agg"

import io
import sys
import json
import base64
import math
import inspect
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from flask import Flask, render_template, request


APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent
REGISTRY_PATH = APP_ROOT / "models_registry.json"

DEFAULT_CLASS_NAMES = ["glioma", "meningioma", "pituitary", "notumor"]


# -----------------------------
# Registry / device
# -----------------------------
def _read_registry():
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Missing registry: {REGISTRY_PATH}")
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_device(device_str):
    s = (device_str or "auto").lower()
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if s == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if s in ("cpu", "cuda"):
        return torch.device(s)
    return torch.device("cpu")


# -----------------------------
# Safe dynamic importing
# -----------------------------
def _unique_module_name(repo_dir, py_file):
    h = hashlib.md5((str(repo_dir) + "|" + str(py_file)).encode("utf-8")).hexdigest()[:10]
    return f"mri_mod_{h}"


def _load_module_from_file(repo_dir, py_file):
    if not py_file.exists():
        raise FileNotFoundError(f"Python file not found: {py_file}")

    added = []
    for p in (str(repo_dir), str(PROJECT_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
            added.append(p)

    try:
        mod_name = _unique_module_name(repo_dir, py_file)
        spec = importlib.util.spec_from_file_location(mod_name, str(py_file))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module spec from: {py_file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for p in added:
            if p in sys.path:
                sys.path.remove(p)


def _pick_model_ctor(module):
    if hasattr(module, "HybridResNet50V2_RViT"):
        return getattr(module, "HybridResNet50V2_RViT")

    candidates = []
    for name in dir(module):
        obj = getattr(module, name)
        try:
            if inspect.isclass(obj) and issubclass(obj, torch.nn.Module):
                if getattr(obj, "__module__", "") == getattr(module, "__name__", ""):
                    candidates.append(obj)
        except Exception:
            continue

    if not candidates:
        raise AttributeError("No torch.nn.Module class found in the model file.")

    for c in candidates:
        if "hybrid" in c.__name__.lower():
            return c
    return candidates[0]


def _filter_kwargs_for_signature(ctor, kwargs):
    sig = inspect.signature(ctor.__init__)
    allowed = set(sig.parameters.keys())
    allowed.discard("self")
    return {k: v for k, v in kwargs.items() if k in allowed}


def _instantiate_model(ctor, cfg, class_names, force_kwargs):
    base = {
        "num_classes": cfg.get("num_classes", len(class_names)),
        "img_size": cfg.get("img_size", 224),
        "patch_size": cfg.get("patch_size", 16),
        "embed_dim": cfg.get("embed_dim", 142),
        "depth": cfg.get("depth", 10),
        "heads": cfg.get("heads", 10),
        "mlp_dim": cfg.get("mlp_dim", 480),
        "attn_dropout": cfg.get("attn_dropout", 0.1),
        "vit_dropout": cfg.get("vit_dropout", 0.1),
        "fusion_dim": cfg.get("fusion_dim", 256),
        "fusion_dropout": cfg.get("fusion_dropout", 0.5),
        "rotations": tuple(cfg.get("rotations", (0, 1, 2, 3))),
        "cnn_name": cfg.get("cnn_name", "resnetv2_50x1_bitm"),
        "cnn_pretrained": False,
    }
    if isinstance(force_kwargs, dict):
        base.update(force_kwargs)

    safe = _filter_kwargs_for_signature(ctor, base)
    return ctor(**safe)


def _load_checkpoint(path, device):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(str(path), map_location=device)


def _extract_state_dict(ckpt):
    keys = ["model_state", "state_dict", "model_state_dict", "model", "net", "weights"]
    for k in keys:
        if k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k]
    if isinstance(ckpt, dict) and ckpt and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt
    raise KeyError("Checkpoint missing weights (expected model_state/state_dict/etc).")


# -----------------------------
# Image + plotting helpers
# -----------------------------
def preprocess_image(pil_img, mean, std, img_size):
    img = pil_img.convert("RGB").resize((img_size, img_size))
    x = torch.from_numpy(np.array(img)).float() / 255.0
    x = x.permute(2, 0, 1).contiguous()
    mean_t = torch.tensor(mean, dtype=x.dtype).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=x.dtype).view(3, 1, 1)
    x = (x - mean_t) / std_t
    return x.unsqueeze(0), np.array(img)


def normalize_map(h):
    h = h - h.min()
    h = h / (h.max() + 1e-8)
    return h


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def overlay_base64(img_rgb, heat, title):
    fig = plt.figure(figsize=(4, 4))
    plt.imshow(img_rgb)
    plt.imshow(heat, alpha=0.45)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    return fig_to_base64(fig)


def barplot_base64(labels, probs, title):
    fig = plt.figure(figsize=(5, 3))
    plt.bar(labels, probs)
    plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    return fig_to_base64(fig)


def tumorprob_base64(tumor_p, notumor_p, title):
    fig = plt.figure(figsize=(4.5, 3))
    plt.bar(["Tumor (any)", "No tumor"], [tumor_p, notumor_p])
    plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.tight_layout()
    return fig_to_base64(fig)


# -----------------------------
# XAI core (matches your scripts)
# -----------------------------
def _pick_last_4d_tensor(obj):
    # KEY FIX: timm CNN often returns list/tuple of features; pick LAST stage (used downstream)
    if isinstance(obj, torch.Tensor) and obj.ndim == 4:
        return obj
    if isinstance(obj, (list, tuple)):
        for t in reversed(obj):
            if isinstance(t, torch.Tensor) and t.ndim == 4:
                return t
    return None


def _safe_forward_return_xai(model, x):
    try:
        out = model(x, return_xai=True)
    except TypeError:
        return model(x), None

    if isinstance(out, (tuple, list)):
        logits = out[0]
        xai = out[1] if len(out) > 1 else None
        return logits, xai
    return out, None


def _gradcam_pp(A, grads, xai_module):
    # Prefer your XAI module’s Grad-CAM++ if it exists, else use internal formula.
    if xai_module is not None and hasattr(xai_module, "gradcam_pp_from_activations"):
        return xai_module.gradcam_pp_from_activations(A, grads)

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


def _attention_rollout(attn_list, xai_dict, xai_module):
    if xai_module is None or not hasattr(xai_module, "attention_rollout"):
        return None

    fn = xai_module.attention_rollout
    sig = inspect.signature(fn)
    kwargs = {}

    # Some of your scripts accept side=... (GSTE) or side=14 (no GSTE).
    if "side" in sig.parameters:
        if isinstance(xai_dict, dict) and xai_dict.get("gste_side", None) is not None:
            kwargs["side"] = xai_dict.get("gste_side")
        else:
            # safe default for 224/16 when no GSTE
            kwargs["side"] = 14

    return fn(attn_list, **kwargs)


def _compute_xai_overlays(model, x, img_rgb, img_size, pred_idx, xai_module, force_use_pfd_gste):
    """
    Try PFD hook first (if active), else CNN hook.
    This mirrors your scripts and avoids the ablation-B Grad-CAM failure.
    """
    overlays = {"gradcam": None, "rollout": None}

    def run_with_hook(hook_kind):
        hook_cache = {}

        def _pfd_hook(_m, _inp, out):
            # PFD returns (feat_path, mask_feat)
            if isinstance(out, (tuple, list)) and len(out) >= 1:
                feat = out[0]
            else:
                feat = None
            if isinstance(feat, torch.Tensor):
                feat.retain_grad()
                hook_cache["A"] = feat

        def _cnn_hook(_m, _inp, out):
            feat = _pick_last_4d_tensor(out)
            if isinstance(feat, torch.Tensor):
                feat.retain_grad()
                hook_cache["A"] = feat

        target_mod = None
        hook_fn = None

        if hook_kind == "pfd":
            if hasattr(model, "pfd"):
                target_mod = model.pfd
                hook_fn = _pfd_hook
        else:
            if hasattr(model, "cnn"):
                target_mod = model.cnn
                hook_fn = _cnn_hook

        if target_mod is None:
            return None, None, None

        h = target_mod.register_forward_hook(hook_fn)
        try:
            with torch.enable_grad():
                logits, xai = _safe_forward_return_xai(model, x)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]

                model.zero_grad(set_to_none=True)
                logits[0, pred_idx].backward(retain_graph=True)

                A = hook_cache.get("A", None)
                if not (isinstance(A, torch.Tensor) and A.grad is not None and A.ndim == 4):
                    return logits, xai, None

                cam_small = _gradcam_pp(A, A.grad, xai_module)
                cam = F.interpolate(
                    cam_small.unsqueeze(0).unsqueeze(0),
                    size=(img_size, img_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze().detach().cpu().numpy()

                return logits, xai, cam
        finally:
            h.remove()

    # Decide whether to even try PFD first:
    # If force_use_pfd_gste is explicitly False (ablation B), skip PFD.
    try_pfd = (force_use_pfd_gste is not False) and hasattr(model, "pfd")

    logits = None
    xai = None
    cam = None

    if try_pfd:
        logits, xai, cam = run_with_hook("pfd")

    if cam is None:
        logits, xai, cam = run_with_hook("cnn")

    if cam is not None:
        overlays["gradcam"] = overlay_base64(img_rgb, cam, "Grad-CAM++")

    # attention rollout
    if isinstance(xai, dict):
        attn_list = xai.get("attn") or xai.get("attn_list")
        if attn_list:
            tok = _attention_rollout(attn_list, xai, xai_module)
            if isinstance(tok, torch.Tensor):
                tok_img = F.interpolate(
                    tok.unsqueeze(0).unsqueeze(0),
                    size=(img_size, img_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze().detach().cpu().numpy()
                overlays["rollout"] = overlay_base64(img_rgb, tok_img, "Attention rollout")

    return overlays


def run_inference_with_xai(loaded, pil_img):
    device = loaded["device"]
    model = loaded["model"]
    class_names = loaded["class_names"]
    xai_module = loaded.get("xai_module", None)

    img_size = int(loaded["cfg"].get("img_size", 224))
    x, img_rgb = preprocess_image(pil_img, loaded["mean"], loaded["std"], img_size)
    x = x.to(device)

    with torch.no_grad():
        out = model(x)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        probs_t = torch.softmax(logits, dim=1).squeeze(0)
        probs = probs_t.detach().cpu().numpy()

    pred_idx = int(np.argmax(probs))
    conf = float(probs[pred_idx])

    if "notumor" in class_names:
        notumor_idx = class_names.index("notumor")
        tumor_prob = float(np.sum([p for i, p in enumerate(probs) if i != notumor_idx]))
        notumor_prob = float(probs[notumor_idx])
    else:
        tumor_prob = float(np.sum(probs[:3]))
        notumor_prob = float(1.0 - tumor_prob)

    overlays = _compute_xai_overlays(
        model=model,
        x=x,
        img_rgb=img_rgb,
        img_size=img_size,
        pred_idx=pred_idx,
        xai_module=xai_module,
        force_use_pfd_gste=loaded.get("force_use_pfd_gste", None),
    )

    probs_chart_b64 = barplot_base64(class_names, probs, "Class probabilities")
    tumor_chart_b64 = tumorprob_base64(tumor_prob, notumor_prob, "Tumor vs No-tumor")  

    return {
        "model_id": loaded["id"],
        "model_name": loaded["name"],
        "pred_class": class_names[pred_idx],
        "pred_idx": pred_idx,
        "confidence": conf,
        "probs": [float(p) for p in probs],
        "class_names": class_names,
        "tumor_prob": tumor_prob,
        "notumor_prob": notumor_prob,
        "chart_probs": probs_chart_b64,
        "chart_tumor": tumor_chart_b64,
        "xai_gradcam": overlays["gradcam"],
        "xai_rollout": overlays["rollout"],
    }


# -----------------------------
# Load all models + their XAI modules
# -----------------------------
def load_all_models():
    reg = _read_registry()
    device = _resolve_device(reg.get("device", "auto"))

    loaded = []
    models = reg.get("models", [])
    if not models:
        raise RuntimeError("No models defined in models_registry.json")

    for m in models:
        repo_dir = (APP_ROOT / m["repo_dir"]).resolve()
        ckpt_path = (repo_dir / m.get("checkpoint", "best_model.pt")).resolve()

        model_file = (repo_dir / m["model_file"]).resolve()
        xai_file = (repo_dir / m.get("xai_file", "")).resolve() if m.get("xai_file") else None

        model_module = _load_module_from_file(repo_dir, model_file)
        ctor = _pick_model_ctor(model_module)

        ckpt = _load_checkpoint(ckpt_path, device)

        class_names = ckpt.get("class_names", DEFAULT_CLASS_NAMES)
        mean = ckpt.get("mean", [0.5, 0.5, 0.5])
        std = ckpt.get("std", [0.5, 0.5, 0.5])
        cfg = ckpt.get("model_cfg", {}) or {}
        force_kwargs = m.get("force_kwargs", {}) or {}

        model = _instantiate_model(ctor, cfg, class_names, force_kwargs)
        state = _extract_state_dict(ckpt)

        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[WARN] {m.get('id')} missing: {len(missing)}, unexpected: {len(unexpected)}")

        model.to(device)
        model.eval()

        xai_module = None
        if xai_file is not None and xai_file.exists():
            try:
                xai_module = _load_module_from_file(repo_dir, xai_file)
            except Exception as e:
                print(f"[WARN] Could not load XAI module for {m.get('id')}: {repr(e)}")
                xai_module = None

        loaded.append({
            "id": m.get("id", ""),
            "name": m.get("name", ""),
            "repo_dir": str(repo_dir),
            "model": model,
            "device": device,
            "class_names": list(class_names),
            "mean": list(mean),
            "std": list(std),
            "cfg": dict(cfg),
            "xai_module": xai_module,
            "force_use_pfd_gste": force_kwargs.get("use_pfd_gste", None),
        })

    return loaded


# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)
LOADED_MODELS = load_all_models()


@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    uploaded_preview = None
    error = None

    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            error = "No image selected."
            return render_template("index.html", results=results, error=error, uploaded_preview=uploaded_preview)

        try:
            raw = file.read()
            pil = Image.open(io.BytesIO(raw)).convert("RGB")

            prev = pil.resize((224, 224))
            buf = io.BytesIO()
            prev.save(buf, format="PNG")
            uploaded_preview = base64.b64encode(buf.getvalue()).decode("utf-8")

            results = []
            for lm in LOADED_MODELS:
                results.append(run_inference_with_xai(lm, pil))

        except Exception as e:
            error = f"Failed to process image or run inference: {repr(e)}"

    return render_template("index.html", results=results, error=error, uploaded_preview=uploaded_preview)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
