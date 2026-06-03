
"""
## Reusable PFD-GSTE guidance modules.

This file implements task-agnostic spatial guidance components for medical image
classification. PFD learns a soft pathology-focused mask from CNN feature maps,
while GSTE uses that mask to reweight feature tokens or image patch tokens before
downstream classification.

The modules are designed to be imported into different CNN, Transformer, or
hybrid classifiers, rather than being tied to one brain-tumour architecture.

## Variant A supports feature-token guidance; Variant B supports patch-token guidance
with optional mask-driven token-grid reduction.

"""


import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _check_tensor(name, value, ndim):
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.ndim != ndim:
        raise ValueError(
            f"{name} must have {ndim} dimensions, got shape {tuple(value.shape)}."
        )


def _check_same_batch(left_name, left, right_name, right):
    if left.shape[0] != right.shape[0]:
        raise ValueError(
            f"{left_name} and {right_name} must have the same batch size, "
            f"got {left.shape[0]} and {right.shape[0]}."
        )


def _infer_square_grid(num_tokens):
    side = int(math.sqrt(num_tokens))
    if side * side != num_tokens:
        raise ValueError(
            "token_hw was not provided and the token count is not square. "
            "Pass token_hw=(height, width) explicitly."
        )
    return side, side


def _resolve_hw(token_hw, num_tokens):
    if token_hw is None:
        return _infer_square_grid(num_tokens)

    if isinstance(token_hw, int):
        height, width = token_hw, token_hw
    else:
        height, width = token_hw

    height = int(height)
    width = int(width)

    if height <= 0 or width <= 0:
        raise ValueError("token_hw must contain positive spatial dimensions.")

    if height * width != num_tokens:
        raise ValueError(
            f"token_hw={token_hw} does not match token count {num_tokens}."
        )

    return height, width


def _resize_mask(mask, size, mode="bilinear"):
    if mode in ("linear", "bilinear", "bicubic", "trilinear"):
        return F.interpolate(mask, size=size, mode=mode, align_corners=False)
    return F.interpolate(mask, size=size, mode=mode)


class PathologyFocusedGate(nn.Module):
    """
    Learns a soft spatial guidance mask from a CNN feature map.

    Input:
        features: (B, C, H, W)

    Output:
        gated_features: (B, C, H, W)
        mask:           (B, 1, H, W)

    The module is task-agnostic. It can be used with spatial CNN feature maps
    from brain, breast, lung, retinal, or other medical image classifiers.
    """

    def __init__(self, in_channels, hidden_channels=0, init_bias=0.0):
        super().__init__()

        in_channels = int(in_channels)
        hidden_channels = int(hidden_channels)

        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")

        if hidden_channels > 0:
            self.mask_head = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=True),
                nn.GELU(),
                nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
            )
            final_layer = self.mask_head[-1]
        else:
            self.mask_head = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)
            final_layer = self.mask_head

        nn.init.xavier_uniform_(final_layer.weight)
        nn.init.constant_(final_layer.bias, float(init_bias))

    def forward(self, features):
        _check_tensor("features", features, 4)

        mask = torch.sigmoid(self.mask_head(features))
        gated_features = features * mask

        return gated_features, mask


class FeatureTokenGuidance(nn.Module):
    """
    GSTE-A feature-token guidance.

    Reweights CNN-derived transformer tokens using a spatial guidance mask.

    Input:
        tokens:   (B, N, D)
        mask:     (B, 1, H, W)
        token_hw: optional (token_h, token_w)

    Output:
        guided_tokens: (B, N, D)
        alpha:         (B, N, 1)
    """

    def __init__(self, eps=1e-6, resize_mode="bilinear"):
        super().__init__()
        self.eps = float(eps)
        self.resize_mode = resize_mode

    def forward(self, tokens, mask, token_hw=None):
        _check_tensor("tokens", tokens, 3)
        _check_tensor("mask", mask, 4)
        _check_same_batch("tokens", tokens, "mask", mask)

        if mask.shape[1] != 1:
            raise ValueError(f"mask must have one channel, got {mask.shape[1]}.")

        token_h, token_w = _resolve_hw(token_hw, tokens.shape[1])

        alpha = _resize_mask(mask, size=(token_h, token_w), mode=self.resize_mode)
        alpha = alpha.flatten(2).transpose(1, 2)
        alpha = alpha / (alpha.mean(dim=1, keepdim=True) + self.eps)

        guided_tokens = tokens * alpha

        return guided_tokens, alpha


class PatchEmbed2d(nn.Module):
    """
    Lightweight ViT patch embedding.

    Input:
        images: (B, C, H, W)

    Output:
        tokens:  (B, N, D)
        token_h: int
        token_w: int

    It can be replaced with similar task-specific patch embedding layer if required.
    """

    def __init__(self, in_channels=3, embed_dim=128, patch_size=16):
        super().__init__()

        patch_size = int(patch_size)

        if patch_size <= 0:
            raise ValueError("patch_size must be positive.")

        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            int(in_channels),
            int(embed_dim),
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

    def forward(self, images):
        _check_tensor("images", images, 4)

        height, width = images.shape[-2:]

        if height < self.patch_size or width < self.patch_size:
            raise ValueError(
                f"image size {(height, width)} is smaller than patch_size={self.patch_size}."
            )

        x = self.proj(images)
        _, _, token_h, token_w = x.shape

        tokens = x.flatten(2).transpose(1, 2)

        return tokens, token_h, token_w


class PatchTokenGuidance(nn.Module):
    """
    GSTE-B patch-token guidance.

    Reweights image patch tokens using a spatial guidance mask and can optionally
    shrink the token grid when the mask is spatially concentrated.

    Input:
        tokens:   (B, N, D)
        mask:     (B, 1, H, W)
        token_hw: optional (token_h, token_w)

    Output:
        guided_tokens: (B, N_out, D)
        alpha:         (B, 1, token_h, token_w)
        output_hw:     (out_h, out_w)
    """

    def __init__(
        self,
        min_side=7,
        std_flat=0.05,
        std_full=0.30,
        max_shrink=0.50,
        eps=1e-6,
        resize_mode="bilinear",
    ):
        super().__init__()

        self.min_side = int(min_side)
        self.std_flat = float(std_flat)
        self.std_full = float(std_full)
        self.max_shrink = float(max_shrink)
        self.eps = float(eps)
        self.resize_mode = resize_mode

        if self.min_side <= 0:
            raise ValueError("min_side must be positive.")
        if not 0.0 <= self.max_shrink < 1.0:
            raise ValueError("max_shrink must be in [0, 1).")
        if self.std_full <= self.std_flat:
            raise ValueError("std_full must be greater than std_flat.")

    def _choose_output_hw(self, alpha, token_h, token_w):
        with torch.no_grad():
            concentration = alpha.std(dim=(2, 3), unbiased=False).mean().item()

            if concentration <= self.std_flat:
                return token_h, token_w

            ratio = (concentration - self.std_flat) / (self.std_full - self.std_flat)
            ratio = max(0.0, min(1.0, ratio))

            shrink = self.max_shrink * ratio

            out_h = int(round(token_h * (1.0 - shrink)))
            out_w = int(round(token_w * (1.0 - shrink)))

            min_h = min(self.min_side, token_h)
            min_w = min(self.min_side, token_w)

            out_h = max(min_h, min(token_h, out_h))
            out_w = max(min_w, min(token_w, out_w))

            return out_h, out_w

    def forward(self, tokens, mask, token_hw=None, shrink=True):
        _check_tensor("tokens", tokens, 3)
        _check_tensor("mask", mask, 4)
        _check_same_batch("tokens", tokens, "mask", mask)

        if mask.shape[1] != 1:
            raise ValueError(f"mask must have one channel, got {mask.shape[1]}.")

        batch_size, num_tokens, embed_dim = tokens.shape
        token_h, token_w = _resolve_hw(token_hw, num_tokens)

        alpha = _resize_mask(mask, size=(token_h, token_w), mode=self.resize_mode)
        alpha = alpha / (alpha.mean(dim=(2, 3), keepdim=True) + self.eps)

        token_map = tokens.transpose(1, 2).reshape(
            batch_size,
            embed_dim,
            token_h,
            token_w,
        )
        token_map = token_map * alpha

        if shrink:
            out_h, out_w = self._choose_output_hw(alpha, token_h, token_w)
        else:
            out_h, out_w = token_h, token_w

        if out_h != token_h or out_w != token_w:
            numerator = F.adaptive_avg_pool2d(token_map, output_size=(out_h, out_w))
            denominator = F.adaptive_avg_pool2d(alpha, output_size=(out_h, out_w))
            token_map = numerator / (denominator + self.eps)

        guided_tokens = token_map.flatten(2).transpose(1, 2)

        return guided_tokens, alpha, (out_h, out_w)


class PFDGSTEVariantA(nn.Module):
    """
    Reusable PFD-GSTE Variant A.

    Variant A is used when transformer tokens are produced from CNN feature maps.

    Pipeline:
        CNN features -> PFD mask -> gated CNN features -> 1x1 token projection
        -> GSTE feature-token reweighting

    This module does not include a classifier. It returns guided tokens that can
    be passed to any transformer, attention block, pooling head, or classifier.
    """

    def __init__(self, in_channels, embed_dim, hidden_channels=0):
        super().__init__()

        self.pfd = PathologyFocusedGate(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
        )

        self.token_projection = nn.Conv2d(
            int(in_channels),
            int(embed_dim),
            kernel_size=1,
            bias=True,
        )

        self.gste = FeatureTokenGuidance()

    def forward(self, features):
        _check_tensor("features", features, 4)

        gated_features, mask = self.pfd(features)

        token_map = self.token_projection(gated_features)
        _, _, token_h, token_w = token_map.shape

        tokens = token_map.flatten(2).transpose(1, 2)
        guided_tokens, alpha = self.gste(tokens, mask, token_hw=(token_h, token_w))

        return guided_tokens, mask, alpha


class PFDGSTEVariantB(nn.Module):
    """
    Reusable PFD-GSTE Variant B.

    Variant B is used when:
        1. PFD guides the CNN feature pathway.
        2. GSTE guides raw-image patch tokens.
        3. The patch-token grid can optionally shrink when the guidance mask is concentrated.

    This module does not include a classifier.

    Output:
        gated_features: CNN features after PFD
        guided_tokens:  patch tokens after GSTE
        mask_image:     image-scale soft mask
        alpha:          token-grid guidance weights
        output_hw:      output token grid after optional shrinking
    """

    def __init__(
        self,
        in_channels,
        embed_dim,
        image_channels=3,
        patch_size=16,
        hidden_channels=0,
        min_side=7,
        max_shrink=0.50,
    ):
        super().__init__()

        self.pfd = PathologyFocusedGate(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
        )

        self.patch_embed = PatchEmbed2d(
            in_channels=image_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )

        self.gste = PatchTokenGuidance(
            min_side=min_side,
            max_shrink=max_shrink,
        )

    def forward(self, images, features, shrink=True):
        _check_tensor("images", images, 4)
        _check_tensor("features", features, 4)
        _check_same_batch("images", images, "features", features)

        gated_features, mask_low = self.pfd(features)

        mask_image = _resize_mask(
            mask_low,
            size=(images.shape[2], images.shape[3]),
            mode="bilinear",
        )

        tokens, token_h, token_w = self.patch_embed(images)

        guided_tokens, alpha, output_hw = self.gste(
            tokens,
            mask_image,
            token_hw=(token_h, token_w),
            shrink=shrink,
        )

        return gated_features, guided_tokens, mask_image, alpha, output_hw


def enable_mc_dropout(model):
    """
    Enables dropout layers during inference while leaving non-dropout layers unchanged.
    Useful for Monte Carlo dropout uncertainty estimation.
    """

    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


@torch.no_grad()
def mc_dropout_predict(model, images, samples=20):
    """
    Runs stochastic Monte Carlo dropout prediction.

    The model may return logits directly or return a tuple whose first item is logits.

    Output:
        mean:     predictive mean probability, shape (B, C)
        variance: predictive probability variance, shape (B, C)
    """

    model.eval()
    enable_mc_dropout(model)

    probabilities = []

    for _ in range(int(samples)):
        output = model(images)
        logits = output[0] if isinstance(output, tuple) else output
        probabilities.append(torch.softmax(logits, dim=1))

    probabilities = torch.stack(probabilities, dim=0)

    mean = probabilities.mean(dim=0)
    variance = probabilities.var(dim=0, unbiased=False)

    return mean, variance
