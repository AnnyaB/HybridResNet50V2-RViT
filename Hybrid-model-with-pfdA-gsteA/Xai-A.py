
# scripts/predict_xai.py
#
# This script runs single-image inference with extras:
# - normal prediction and confidence
# - MC-dropout uncertainty (mean and variance over multiple stochastic passes)
# - XAI:
#     * Grad-CAM++ using a hook on the CNN/PFD feature map (no dependency on xai["cnn_feat"])
#     * Attention Rollout using transformer attention matrices from xai["attn"]
#
#   xai_gradcampp.png      (CNN/PFD explanation)
#   xai_attn_rollout.png   (Transformer explanation)

import sys  # it lets us edit Python import paths (sys.path)
from pathlib import Path  # robust path handling (OS-independent so that can run on Windows/Mac/Linux)

import numpy as np  # for arrays and small numeric helpers
import torch  # PyTorch tensor and autograd engine
import torch.nn.functional as F  # functional ops (softmax, interpolate, relu, pooling, etc.)
import matplotlib.pyplot as plt  # plotting and saving heatmap overlays
from PIL import Image  # image reading and basic image ops

# ---- Fix No module named 'models' when running from scripts/ ----
ROOT = Path(__file__).resolve().parent # project root = two folders above scripts/
if str(ROOT) not in sys.path:  # if root isn't already in Python import search path
    sys.path.insert(0, str(ROOT))  # add it so `from models...` works

from models.hybrid_model import HybridResNet50V2_RViT  # imports my PFDA-GSTEA hybrid model


def preprocess_pil(img, mean, std):  # convert PIL image into a normalized torch tensor batch
    img = img.convert("RGB")  # ensure 3 channels (RGB)
    x = torch.from_numpy(np.array(img)).float() / 255.0  # convert to float tensor in [0,1], shape (H,W,3)
    x = x.permute(2, 0, 1).contiguous()                  # change layout to (3,H,W) for PyTorch convnets
    mean_t = torch.tensor(mean, dtype=x.dtype).view(3, 1, 1)  # reshape mean to broadcast per-channel
    std_t = torch.tensor(std, dtype=x.dtype).view(3, 1, 1)    # reshape std to broadcast per-channel
    x = (x - mean_t) / std_t  # normalize like training did
    return x.unsqueeze(0)  # add batch dimension -> (1,3,H,W)


def normalize_map(h):  # scale any heatmap tensor into [0,1] for visualization
    h = h - h.min()  # shift so minimum becomes 0
    h = h / (h.max() + 1e-8)  # scale so maximum becomes ~1 (eps avoids divide-by-zero)
    return h  # return normalized heatmap


def gradcam_pp_from_activations(A, grads):  # compute Grad-CAM++ heatmap given activations and gradients
    """
    Grad-CAM++ idea (high level):
      - Use gradients (and higher-order terms) to compute per-channel weights
      - Weighted sum channels of activation map -> heatmap
      - ReLU and normalize -> final saliency map

    Inputs expected shapes:
      A     : (1, C, h, w)    activation map
      grads : (1, C, h, w)    gradients d(score)/d(A)
    """
    if grads is None:  # if gradients aren't present, Grad-CAM++ can't work
        raise RuntimeError(  # throw a clear error to help debugging
            "Gradients not found for CNN feature map. Ensure forward pass was not inside torch.no_grad "
            "and backward() ran."
        )

    grad_1 = grads  # first-order grads
    grad_2 = grad_1 ** 2  # squared grads
    grad_3 = grad_2 * grad_1  # cubed grads

    spatial_sum = (A * grad_3).sum(dim=(2, 3), keepdim=True)  # sum over spatial dims (h,w), keep shape (1,C,1,1)
    denom = 2.0 * grad_2 + spatial_sum + 1e-8  # denominator for alpha (eps for stability)
    alpha = grad_2 / denom  # Grad-CAM++ alpha coefficients, shape (1,C,h,w)

    w = (alpha * F.relu(grad_1)).sum(dim=(2, 3), keepdim=True)  # channel weights, shape (1,C,1,1)
    cam = (w * A).sum(dim=1, keepdim=True)                      # weighted sum over channels -> (1,1,h,w)
    cam = F.relu(cam)  # clamp negative values (only keep positive evidence)

    cam = cam.squeeze(0).squeeze(0)  # drop batch and channel dims -> (h,w)
    return normalize_map(cam)  # normalize into [0,1] for overlay


def attention_rollout(attn_list, eps=1e-6):  # compute token relevance map from attention layers
    """
    Attention rollout (no CLS assumed):
      1) average attention across heads
      2) add identity (skip connection style)
      3) row-normalize
      4) multiply attention matrices across layers
      5) get a per-token relevance score (here: mean over query tokens)

    attn_list: list of attention tensors, each shaped (B, heads, N, N)
    returns: token relevance map reshaped to (ht, wt) in [0,1]
    """
    if not attn_list or len(attn_list) == 0:  # must have at least one layer of attention
        raise RuntimeError("No attention matrices captured. Run model with return_xai=True.")  # guidance

    mats = []  # we will store processed attention matrices per layer

    for attn in attn_list:  # loop over transformer layers
        a = attn.mean(dim=1)  # average heads -> (B, N, N)
        n = a.shape[-1]  # number of tokens N

        a = a + torch.eye(n, device=a.device).unsqueeze(0)  # add identity so tokens keep some self-information
        a = a / (a.sum(dim=-1, keepdim=True) + eps)  # row-normalize (each row sums ~1)
        mats.append(a)  # store this layer’s normalized attention

    R = mats[0]  # start rollout product with first layer
    for i in range(1, len(mats)):  # multiply through remaining layers
        R = R @ mats[i]  # matrix multiply to propagate attention through depth

    r = R.mean(dim=1).squeeze(0)  # mean over query tokens -> (N,) and drop batch -> (N,)
    r = normalize_map(r)  # normalize relevance scores to [0,1]

    N = r.shape[0]  # number of tokens
    ht = int(round(float(np.sqrt(N))))  # guess grid height as sqrt(N) (works for 49 -> 7)
    ht = max(ht, 1)  # safety clamp
    wt = max(N // ht, 1)  # compute width from height
    if ht * wt != N:  # if reshape won't fit perfectly, fall back to 1 x N
        ht, wt = 1, N  # safe fallback that always matches token count

    return r.reshape(ht, wt)  # return token-grid heatmap (ht, wt)


def overlay_and_save(img_rgb, heat, out_path, title):  # helper: overlay heatmap onto image and save
    plt.figure(figsize=(5, 5))  # create a new figure of fixed size
    plt.imshow(img_rgb)  # show the base RGB image
    plt.imshow(heat, alpha=0.45)  # overlay the heatmap with transparency
    plt.title(title)  # add a title
    plt.axis("off")  # hide axes ticks/frames
    plt.tight_layout()  # reduce padding/margins
    plt.savefig(str(out_path), dpi=200)  # save to disk as PNG (or whatever extension)
    plt.close()  # close figure to avoid memory buildup


def main():  # CLI entry point for the script
    import argparse  # standard library for command-line argument parsing

    ap = argparse.ArgumentParser()  # create argument parser
    ap.add_argument("--checkpoint", type=str, required=True)  # path to .pt checkpoint file
    ap.add_argument("--image", type=str, required=True)  # path to image file
    ap.add_argument("--out_dir", type=str, default="results/xai")  # output folder for saved PNGs
    ap.add_argument("--mc_samples", type=int, default=20)  # number of MC-dropout samples
    ap.add_argument(  # argument for target explanation class
        "--target_class", type=int, default=-1,
        help="set to class index, or -1 to explain predicted class"
    )
    args = ap.parse_args()  # parse CLI args into `args`

    device = "cuda" if torch.cuda.is_available() else "cpu"  # pick GPU if available else CPU
    out_dir = Path(args.out_dir)  # convert to Path object for convenience
    out_dir.mkdir(parents=True, exist_ok=True)  # create output folder if missing

    ckpt = torch.load(args.checkpoint, map_location=device)  # load checkpoint onto device safely
    class_names = ckpt.get("class_names", ["glioma", "meningioma", "pituitary", "notumor"])  # label names (fallback default)
    mean = ckpt.get("mean", [0.485, 0.456, 0.406])  # normalization mean (fallback ImageNet)
    std = ckpt.get("std", [0.229, 0.224, 0.225])  # normalization std (fallback ImageNet)
    cfg = ckpt.get("model_cfg", {})  # model configuration dict (fallback empty)

    model = HybridResNet50V2_RViT(  # rebuild the model using checkpoint config
        num_classes=cfg.get("num_classes", len(class_names)),  # number of classes
        patch_size=cfg.get("patch_size", 16),  # stored in model (PFDA tokenises feature map)
        embed_dim=cfg.get("embed_dim", 142),  # transformer embedding dim
        depth=cfg.get("depth", 10),  # number of transformer blocks
        heads=cfg.get("heads", 10),  # attention heads
        mlp_dim=cfg.get("mlp_dim", 480),  # MLP hidden dim in blocks
        attn_dropout=cfg.get("attn_dropout", 0.1),  # attention dropout
        vit_dropout=cfg.get("vit_dropout", 0.1),  # transformer dropout
        fusion_dim=cfg.get("fusion_dim", 256),  # fusion hidden dim
        fusion_dropout=cfg.get("fusion_dropout", 0.5),  # fusion dropout
        rotations=tuple(cfg.get("rotations", (0, 1, 2, 3))),  # rotations for feature-map token averaging
        cnn_name=cfg.get("cnn_name", "resnetv2_50x1_bitm"),  # timm backbone name
        cnn_pretrained=False,  # don't load pretrained weights here (checkpoint will load trained weights)
    ).to(device)  # move model to device

    state = ckpt.get("model_state", None)  # try preferred key for weights
    if state is None:  # if not found,
        state = ckpt.get("state_dict", None)  # try alternate common key name
    if state is None:  # if still missing,
        raise KeyError("Checkpoint missing model weights. Expected 'model_state' or 'state_dict'.")  # hard fail
    model.load_state_dict(state)  # load weights into model
    model.eval()  # eval mode (dropout off, BN uses running stats)

    img = Image.open(args.image).convert("RGB").resize((224, 224))  # load and standardize image size
    x = preprocess_pil(img, mean, std).to(device)  # preprocess -> tensor batch (1,3,224,224)

    # ---- Capture CNN/PFD feature map via hook (no xai['cnn_feat']) ----
    hook_cache = {}  # dict to store activation tensors captured by the hook

    def _pfd_hook(module, inputs, output):  # forward hook function called during model forward
        # output from model.pfd is expected to be (feat_path, mask_feat)
        feat_path = output[0]  # get gated feature map (the thing we want Grad-CAM++ on)
        feat_path.retain_grad()  # tell PyTorch to keep gradients for this non-leaf tensor
        hook_cache["cnn_feat"] = feat_path  # store it so we can use it after forward/backward

    if not hasattr(model, "pfd"):  # safety check: ensure this model actually has a PFD module
        raise RuntimeError("Model has no attribute 'pfd' to hook for Grad-CAM++.")  # fail early

    hook_handle = model.pfd.register_forward_hook(_pfd_hook)  # attach hook to capture activation during forward

    # ---- Forward with grads enabled ----
    with torch.enable_grad():  # ensure gradients are tracked even though model is in eval()
        logits, xai = model(x, return_xai=True)  # forward pass; also request attention maps

    hook_handle.remove()  # remove hook so it doesn't keep capturing on future calls

    prob = torch.softmax(logits, dim=1).squeeze(0)  # convert logits -> probabilities, drop batch -> (C,)
    pred_idx = int(torch.argmax(prob).item())  # predicted class index
    conf = float(prob[pred_idx].item())  # confidence of predicted class

    target = pred_idx if (args.target_class is None or args.target_class < 0) else int(args.target_class)  # choose class to explain

    # ---- Backprop for Grad-CAM++ ----
    model.zero_grad(set_to_none=True)  # clear gradients from any previous ops
    logits[0, target].backward(retain_graph=True)  # compute gradients for the chosen class score

    A = hook_cache.get("cnn_feat", None)  # retrieve captured activation map from hook
    if A is None:  # if hook didn't capture anything, something went wrong
        raise RuntimeError("Failed to capture CNN feature map from PFD hook.")  # fail loudly

    cam_small = gradcam_pp_from_activations(A, A.grad)  # compute Grad-CAM++ heatmap at feature-map resolution (h,w)
    cam = F.interpolate(  # upsample heatmap to input image size (224,224)
        cam_small.unsqueeze(0).unsqueeze(0),  # add batch+channel -> (1,1,h,w)
        size=(224, 224),  # desired output resolution
        mode="bilinear",  # smooth interpolation
        align_corners=False  # standard safe choice
    ).squeeze(0).squeeze(0)  # drop batch+channel -> (224,224)
    cam = cam.detach().cpu().numpy()  # move to CPU and convert to numpy for plotting

    # ---- Attention rollout ----
    if not isinstance(xai, dict):  # ensure model returned a dictionary payload
        raise RuntimeError("Model did not return XAI dict; ensure model(x, return_xai=True).")  # guidance

    attn_list = xai.get("attn", None)  # preferred key for attention list
    if attn_list is None:  # fallback if key differs
        attn_list = xai.get("attn_list", None)  # alternate key name
    if attn_list is None:  # if still missing
        raise KeyError("XAI dict missing attention list. Expected key 'attn' (or 'attn_list').")  # hard fail

    attn_tok = attention_rollout(attn_list)  # compute token relevance map (ht,wt)
    attn_img = F.interpolate(  # upsample token map to image resolution
        attn_tok.unsqueeze(0).unsqueeze(0),  # add batch+channel -> (1,1,ht,wt)
        size=(224, 224),  # match image size
        mode="bilinear",  # smooth interpolation
        align_corners=False  # stable setting
    ).squeeze(0).squeeze(0)  # drop batch+channel -> (224,224)
    attn_img = attn_img.detach().cpu().numpy()  # convert to numpy for plotting

    # ---- MC dropout uncertainty ----
    with torch.no_grad():  # no gradients needed for MC-dropout sampling
        mu, var = model.mc_dropout_predict(x, mc_samples=args.mc_samples)  # mean and variance over stochastic passes
        mu = mu.squeeze(0).cpu().numpy()  # drop batch -> (C,) and convert to numpy
        var = var.squeeze(0).cpu().numpy()  # drop batch -> (C,) and convert to numpy

    mu_pred = int(np.argmax(mu))  # class index with highest mean probability
    mu_conf = float(mu[mu_pred])  # mean confidence of that predicted class
    mu_var = float(var[mu_pred])  # predictive variance for that class (uncertainty proxy)

    img_rgb = np.array(img)  # convert PIL image to numpy array (H,W,3) for plotting
    overlay_and_save(  # save Grad-CAM++ overlay
        img_rgb, cam, out_dir / "xai_gradcampp.png",
        f"Grad-CAM++ (target={class_names[target]})"
    )
    overlay_and_save(  # save attention rollout overlay
        img_rgb, attn_img, out_dir / "xai_attn_rollout.png",
        "Attention Rollout (token relevance)"
    )

    print("Prediction (single pass):")  # header line
    print(f"  class = {class_names[pred_idx]} (idx={pred_idx})")  # predicted class name + index
    print(f"  confidence = {conf:.4f}")  # predicted confidence

    print("\nMC Dropout (uncertainty-aware):")  # MC-dropout section header
    print(f"  mean-pred class = {class_names[mu_pred]} (idx={mu_pred})")  # mean predicted class
    print(f"  mean confidence = {mu_conf:.4f}")  # mean predicted confidence
    print(f"  predictive variance (pred class) = {mu_var:.6f}")  # uncertainty number

    print("\nSaved XAI:")  # saved files section header
    print(f"  {str(out_dir / 'xai_gradcampp.png')}")  # path to Grad-CAM++ image
    print(f"  {str(out_dir / 'xai_attn_rollout.png')}")  # path to attention rollout image


if __name__ == "__main__":  # standard Python entry point guard
    main()  # run main() when executed as a script