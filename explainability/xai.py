# explainability/xai.py
# Implements Algorithm 2:
# - Grad-CAM++ (CNN/PFD side) on cached F_path feature maps
# - Attention Rollout (Transformer side) using collected attn_stack from model
#
# References for the *ideas* (not copied code):
# - Grad-CAM++: Chattopadhay et al., 2018.
# - Attention Rollout: Abnar & Zuidema, 2020 (residual + rollout multiplication).

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def _normalize01(x, eps=1e-8):
    x = x - x.min()
    x = x / (x.max() + eps)
    return x


class GradCAMPlusPlus:
    def __init__(self, model):
        self.model = model

    def __call__(self, x, target_class=None):
        """
        Produces Grad-CAM++ heatmap from cached F_path.
        NOTE: Computing exact elementwise 2nd/3rd derivatives is prohibitively expensive.
        This implementation uses the common practical Grad-CAM++ form where grad^2 and grad^3
        are used in the alpha coefficient expression (widely used in research codebases).
        """
        self.model.zero_grad(set_to_none=True)

        logits, _probs, _conf = self.model(x, collect_attn=False)

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        score = logits[:, target_class].sum()
        F_path = self.model._cache["F_path"]  # (B,C,H,W)

        grads = torch.autograd.grad(score, F_path, create_graph=False, retain_graph=True)[0]  # (B,C,H,W)

        # Practical Grad-CAM++ coefficients
        g1 = grads
        g2 = g1 * g1
        g3 = g2 * g1

        # Sum activations per channel
        A = F_path
        sumA = A.sum(dim=(2, 3), keepdim=True)

        eps = 1e-8
        alpha = g2 / (2.0 * g2 + sumA * g3 + eps)

        # Positive gradients only
        weights = (alpha * F.relu(g1)).sum(dim=(2, 3), keepdim=True)  # (B,C,1,1)

        cam = (weights * A).sum(dim=1, keepdim=True)  # (B,1,H,W)
        cam = F.relu(cam)
        cam = cam[0, 0].detach().cpu()

        cam = _normalize01(cam.numpy())
        return cam, target_class


class AttentionRollout:
    def __init__(self, eps=1e-8):
        self.eps = eps

    def __call__(self, model, x):
        """
        Produces attention rollout heatmap in token grid space, then upsample to image.
        Must run model(x, collect_attn=True) to cache attn_stack.
        """
        logits, _probs, _conf = model(x, collect_attn=True)
        attn_stack = model._cache["attn_stack"]  # list of (B,heads,N,N)
        Htok, Wtok = model._cache["tok_hw"]

        # Use batch index 0
        mats = []
        for A in attn_stack:
            A = A[0]  # (heads, N, N)
            A = A.mean(dim=0)  # (N,N) head averaging (16)
            mats.append(A)

        N = mats[0].shape[0]
        I = torch.eye(N, device=mats[0].device)

        # (17)-(18): residual + row normalization
        mats2 = []
        for A in mats:
            A = A + I
            A = A / (A.sum(dim=-1, keepdim=True) + self.eps)
            mats2.append(A)

        # (19): rollout composition
        R = mats2[0]
        for A in mats2[1:]:
            R = R @ A

        # (20): GAP-consistent token relevance (no CLS): mean over rows
        r_tok = R.mean(dim=0)  # (N,)
        H = r_tok.reshape(Htok, Wtok)
        H = H.detach().cpu().numpy()
        H = _normalize01(H)

        return H, logits


def overlay_heatmap_on_image(img_tensor, heatmap, out_path, title=""):
    """
    img_tensor: (1,3,224,224) normalized tensor
    heatmap: (224,224) in [0,1]
    """
    img = img_tensor[0].detach().cpu()

    # Unnormalize for visualization (assumes mean=0.5 std=0.5 by default)
    img = img * 0.5 + 0.5
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.imshow(heatmap, alpha=0.45)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def upsample_token_map(token_map, out_hw=(224, 224)):
    # token_map: (Htok,Wtok) -> (H,W)
    t = torch.tensor(token_map, dtype=torch.float32)[None, None, :, :]
    t = F.interpolate(t, size=out_hw, mode="bilinear", align_corners=False)
    return t[0, 0].numpy()
