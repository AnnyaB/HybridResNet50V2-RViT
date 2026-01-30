# app.py
import os
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")

import sys
import io
import math
import base64
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from flask import Flask, render_template, request

import matplotlib.pyplot as plt


# ============================================================
# ✅ FIX #1: Make Python able to import your model package
# Your trained project lives in: ./Hybrid-model-with-pfdA-gsteA/
# and inside it there is: ./Hybrid-model-with-pfdA-gsteA/models/...
# So we must add ./Hybrid-model-with-pfdA-gsteA to sys.path.
# ============================================================
APP_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = APP_ROOT / "Hybrid-model-with-pfdA-gsteA"  # contains "models/" folder
if not MODEL_ROOT.exists():
    raise FileNotFoundError(
        f"MODEL_ROOT not found: {MODEL_ROOT}\n"
        f"Expected folder structure:\n"
        f"  {APP_ROOT}/Hybrid-model-with-pfdA-gsteA/models/..."
    )
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

# ============================================================
# ✅ FIX #2: Robust import of the PFDA–GSTEA model
# (Try common filenames; fail with a helpful message if none match.)
# ============================================================
HybridResNet50V2_RViT = None
import_errors = []

for modname in (
    "models.hybrid_model",            # most common in your repo
    "models.hybrid_model_pfd_gste",   # your alternative name
    "models.hybrid_model_pfd_gstea",  # another common naming
):
    try:
        module = __import__(modname, fromlist=["HybridResNet50V2_RViT"])
        HybridResNet50V2_RViT = getattr(module, "HybridResNet50V2_RViT")
        break
    except Exception as e:
        import_errors.append(f"{modname}: {repr(e)}")

if HybridResNet50V2_RViT is None:
    raise ImportError(
        "Could not import HybridResNet50V2_RViT from any expected module.\n"
        "Tried:\n"
        + "\n".join(f"  - {x}" for x in import_errors)
        + "\n\nCheck which file exists inside:\n"
        f"  {MODEL_ROOT}/models/\n"
        "and ensure it defines: class HybridResNet50V2_RViT(nn.Module)"
    )


app = Flask(__name__)

device = torch.device("cpu")  # change to torch.device("cuda") if you want GPU and have it available


# -------------------------
# Load checkpoint
# -------------------------
CKPT_PATH = MODEL_ROOT / "best_model.pt"
if not CKPT_PATH.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

ckpt = torch.load(str(CKPT_PATH), map_location=device)

CLASSES = ckpt.get("class_names", ["glioma", "meningioma", "pituitary", "notumor"])
MEAN = ckpt.get("mean", [0.485, 0.456, 0.406])
STD = ckpt.get("std", [0.229, 0.224, 0.225])
cfg = ckpt.get("model_cfg", {})

model = HybridResNet50V2_RViT(
    num_classes=cfg.get("num_classes", len(CLASSES)),
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
    cnn_pretrained=False,  # checkpoint provides weights
).to(device)

state = ckpt.get("model_state", None)
if state is None:
    state = ckpt.get("state_dict", None)
if state is None:
    raise KeyError("Checkpoint missing model weights. Expected 'model_state' or 'state_dict'.")

model.load_state_dict(state, strict=True)
model.eval()


# -------------------------
# Transform: SAME normalization as training
# -------------------------
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])


# -------------------------
# XAI helpers
# -------------------------
def normalize_map(h: torch.Tensor) -> torch.Tensor:
    h = h - h.min()
    h = h / (h.max() + 1e-8)
    return h


class ActivationCatcher:
    """
    Hook-capture PFD output feature map (gated features) for Grad-CAM++.
    PFD returns: (feat_path, mask_feat)
    """
    def __init__(self):
        self.A = None

    def hook(self, module, inp, out):
        A = out[0] if isinstance(out, (tuple, list)) else out
        self.A = A
        self.A.retain_grad()


def gradcam_pp_from_activation(A: torch.Tensor) -> torch.Tensor:
    """
    Grad-CAM++ from activation A and gradients A.grad.
    A: (1,C,h,w)
    returns cam: (h,w) in [0,1]
    """
    grads = A.grad
    if grads is None:
        raise RuntimeError("Gradients missing on activation. Did backward() run with grads enabled?")

    grad_1 = grads
    grad_2 = grad_1 ** 2
    grad_3 = grad_2 * grad_1

    spatial_sum = (A * grad_3).sum(dim=(2, 3), keepdim=True)
    denom = 2.0 * grad_2 + spatial_sum + 1e-8
    alpha = grad_2 / denom

    w = (alpha * F.relu(grad_1)).sum(dim=(2, 3), keepdim=True)  # (1,C,1,1)
    cam = (w * A).sum(dim=1, keepdim=True)                      # (1,1,h,w)
    cam = F.relu(cam).squeeze(0).squeeze(0)                     # (h,w)
    return normalize_map(cam)


def attention_rollout(attn_list, eps=1e-6) -> torch.Tensor:
    """
    Attention rollout (no CLS):
      - avg heads
      - add identity
      - row-normalize
      - multiply across layers
      - relevance = mean over query tokens
    attn_list: list of (B,heads,N,N)
    returns: (N,) in [0,1]
    """
    if not attn_list:
        raise RuntimeError("No attention matrices captured.")

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
    return normalize_map(r)


def fig_to_base64_png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def make_prob_bar_chart(probs: np.ndarray, classes: list[str]) -> str:
    fig = plt.figure()
    plt.bar(classes, probs)
    plt.ylim(0, 1.0)
    plt.ylabel("Probability")
    plt.title("Class probabilities")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    return fig_to_base64_png(fig)


def make_overlay_base64(img_rgb: np.ndarray, heat: np.ndarray, title: str) -> str:
    fig = plt.figure()
    plt.imshow(img_rgb)
    plt.imshow(heat, alpha=0.45)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    return fig_to_base64_png(fig)


# -------------------------
# Routes
# -------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    pred = conf = None
    probs_b64 = None

    gradcam_b64 = None
    attn_b64 = None
    pfdmask_b64 = None

    err = None

    if request.method == "POST":
        hook_handle = None
        try:
            # ---- Load image ----
            if "image" not in request.files:
                raise RuntimeError("No file uploaded. Make sure your form input name is 'image'.")
            img = Image.open(request.files["image"]).convert("RGB")
            img_resized = img.resize((224, 224))
            x = transform(img).unsqueeze(0).to(device)

            # ---- Hook PFD for Grad-CAM++ ----
            if not hasattr(model, "pfd"):
                raise RuntimeError("Model has no attribute 'pfd' (needed for Grad-CAM++ hook).")

            catcher = ActivationCatcher()
            hook_handle = model.pfd.register_forward_hook(catcher.hook)

            # ---- Forward with grads enabled + XAI enabled ----
            with torch.enable_grad():
                logits, xai = model(x, return_xai=True)

            # ---- Prediction ----
            probs = torch.softmax(logits, dim=1).squeeze(0)
            conf_val, idx = probs.max(dim=0)
            target = int(idx.item())
            pred = CLASSES[target]
            conf = f"{float(conf_val.item()) * 100:.2f}%"

            probs_b64 = make_prob_bar_chart(probs.detach().cpu().numpy(), CLASSES)

            # ---- Backward for Grad-CAM++ (target = predicted class) ----
            model.zero_grad(set_to_none=True)
            logits[0, target].backward(retain_graph=True)

            if catcher.A is None:
                raise RuntimeError("Hook did not capture PFD activation (feat_path).")

            cam_small = gradcam_pp_from_activation(catcher.A)  # (h,w) e.g. 7x7
            cam = F.interpolate(
                cam_small.unsqueeze(0).unsqueeze(0),
                size=(224, 224),
                mode="bilinear",
                align_corners=False
            ).squeeze(0).squeeze(0).detach().cpu().numpy()

            # ---- Attention rollout ----
            if not isinstance(xai, dict):
                raise RuntimeError("Model did not return XAI dict. Ensure return_xai=True is implemented.")

            attn_list = xai.get("attn", None)
            if attn_list is None:
                attn_list = xai.get("attn_list", None)
            if attn_list is None:
                raise KeyError("xai dict missing attention list (expected 'attn' or 'attn_list').")

            r = attention_rollout(attn_list)  # (N,)
            N = int(r.numel())

            # PFDA–GSTEA typically N=49 => 7x7
            side = int(round(math.sqrt(N)))
            ht = max(side, 1)
            wt = max(N // ht, 1)
            if ht * wt != N:
                ht, wt = 1, N

            attn_tok = r.reshape(ht, wt)
            attn_img = F.interpolate(
                attn_tok.unsqueeze(0).unsqueeze(0),
                size=(224, 224),
                mode="bilinear",
                align_corners=False
            ).squeeze(0).squeeze(0).detach().cpu().numpy()

            # ---- PFD mask overlay (directly from model output) ----
            # hybrid_model.py returns "mask" already upsampled to 224x224
            mask_img = xai.get("mask", None)
            if mask_img is not None:
                mask_img = mask_img.squeeze(0).squeeze(0).detach().cpu().numpy()  # (224,224)
                mask_img = (mask_img - mask_img.min()) / (mask_img.max() - mask_img.min() + 1e-8)

            # ---- Build overlays ----
            img_rgb = np.array(img_resized)
            gradcam_b64 = make_overlay_base64(img_rgb, cam, f"Grad-CAM++ (target={pred})")
            attn_b64 = make_overlay_base64(img_rgb, attn_img, "Attention rollout (token relevance)")
            if mask_img is not None:
                pfdmask_b64 = make_overlay_base64(img_rgb, mask_img, "PFD mask (pathology gate)")

        except Exception as e:
            err = str(e)

        finally:
            if hook_handle is not None:
                hook_handle.remove()

    return render_template(
        "index.html",
        pred=pred,
        conf=conf,
        probs_b64=probs_b64,
        gradcam_b64=gradcam_b64,
        attn_b64=attn_b64,
        pfdmask_b64=pfdmask_b64,
        err=err
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, threaded=True)
