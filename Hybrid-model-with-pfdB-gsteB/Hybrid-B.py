# ==================================================================================================
# models/hybrid_model.py  (HYBRID B)
# ==================================================================================================
# In this file, I’m building a CNN-Transformer hybrid classifier with optional mask-guided focusing.
#
# Implemented from scratch by me (custom logic, not a library feature):
# - A clean wrapper around a timm ResNet50V2 backbone that always returns the final feature map.
# - PFD: a learned 1x1 conv and sigmoid mask that gates CNN features (pathology-focused gating).
# - ViT-style patch tokenization and rotation handling (rotate image -> patchify -> embed -> average).
# - FlexibleMHSA: attention that still works even when embed_dim isn’t divisible by num_heads.
# - RViTBlock: attention, depthwise conv (local mixing), MLP, with pre-LN and residuals.
# - GSTE: mask-guided token weighting and optional dynamic grid shrink via weighted pooling.
# - Fusion head: merges CNN descriptor and Transformer descriptor and predicts class logits.
# - XAI-friendly outputs: attention maps, upsampled mask and cached CNN features for CAM hooks.
#
# Libraries I import:
# - math: tiny numeric helpers (ceil, sqrt).
# - torch: tensors, autograd, device ops.
# - torch.nn: layer modules (Conv2d, Linear, LayerNorm, Dropout, etc.).
# - torch.nn.functional: functional ops (interpolate, pooling, activations).
# - timm: model factory for pretrained ResNet50V2 (imported inside the backbone wrapper).
#
# ==================================================================================================

import math  # I use this for ceil/sqrt when handling head dims and token-grid inference.
import torch  # I use torch as the tensor and autograd engine for everything in this model.
import torch.nn as nn  # I use nn.Module and layer primitives to build trainable blocks.
import torch.nn.functional as F  # I use functional ops for pooling/interpolation/activations.

# --------------------------------------------------------------------------------------------------
# CNN BACKBONE: pretrained ResNet50V2 via timm
# --------------------------------------------------------------------------------------------------
class ResNet50V2TimmBackbone(nn.Module):
    """
    I wrap timm’s ResNetV2 so the rest of the model always gets a consistent output:
      input  : (B, 3, 224, 224)
      output : (B, C, 7, 7)  for 224x224 input (final ResNet stage feature map)
    """

    def __init__(self, model_name="resnetv2_50x1_bitm", pretrained=True):
        super().__init__()  # I initialize nn.Module so parameters register properly.

        # I import timm here (inside the class) so importing this file doesn’t hard-crash
        # in environments where timm isn’t installed.
        try:
            import timm  # I try to import timm, which is required for the backbone.
        except Exception as e:
            # I raise a clear, direct error so the failure mode is obvious.
            raise ImportError(
                "timm is required for pretrained ResNet50V2. Install timm to use this backbone."
            ) from e

        self.model_name = model_name  # I store the backbone name so runs/checkpoints are reproducible.

        # I request feature maps instead of a classification head:
        # - features_only=True => returns intermediate feature maps as a list
        # - out_indices=(4,)  => only keep the deepest (final) stage output
        import timm  # I import again after the guarded import so I can call it safely.

        self.backbone = timm.create_model(  # I ask timm to build the model.
            model_name,  # this selects the exact architecture + weight recipe.
            pretrained=pretrained,  # if True, the backbone starts with pretrained weights.
            features_only=True,  # I want feature maps, not logits.
            out_indices=(4,),  # I only keep the final stage feature map.
        )

        # I try to read the output channel count from timm’s feature metadata.
        # This keeps later layers (PFD, projections) correctly sized.
        try:
            self.out_ch = int(self.backbone.feature_info.channels()[-1])  # deepest stage channels
        except Exception:
            self.out_ch = 2048  # fallback: common final ResNet stage width.

    def forward(self, x):
        # x is an image batch: (B, 3, H, W). With defaults: (B, 3, 224, 224).
        feats = self.backbone(x)  # the model converts pixels into hierarchical feature maps (list).
        return feats[-1]  # I return the final stage feature map: (B, C, 7, 7) for 224x224.
    
# --------------------------------------------------------------------------------------------------
# PFD: learned spatial gating on CNN feature maps
# --------------------------------------------------------------------------------------------------
class PFD(nn.Module):
    """
    PFD (pathology-focused gating):
    - I predict a 1-channel spatial mask from the CNN feature map using a 1x1 conv.
    - I pass it through sigmoid so each location is a soft weight in [0, 1].
    - I multiply feature_map * mask (broadcast over channels) to emphasize ROI locations.

    What the model sees after this:
    - Same feature map shape, but spatial energy is concentrated where the learned mask is high.
    """

    def __init__(self, in_ch):
        super().__init__()  # I register this as a proper trainable module.

        self.mask_conv = nn.Conv2d(  # I compress C channels -> 1 mask channel.
            in_ch,  # input channels from CNN feature map (e.g., 2048).
            1,  # output channel = 1 (a single spatial mask).
            kernel_size=1,  # 1x1 conv mixes channels per location without changing spatial layout.
            stride=1,  # keep same spatial resolution.
            padding=0,  # no padding needed for 1x1.
            bias=True,  # I allow a bias term so the mask can shift baseline activation.
        )

    def forward(self, feat):
        # feat is the CNN feature map: (B, C, 7, 7) for 224x224 input.
        mask_logits = self.mask_conv(feat)  # raw mask scores per spatial location: (B, 1, 7, 7)
        mask = torch.sigmoid(mask_logits)  # squash to [0, 1] so it acts like a soft gate.
        gated = feat * mask  # broadcast multiply: each (i,j) scales all channels equally.
        return gated, mask  # I return both: gated features for learning + mask for GSTE/XAI.

# --------------------------------------------------------------------------------------------------
# PATCH EMBEDDING: turn an image into patch tokens (ViT-style)
# --------------------------------------------------------------------------------------------------
class ImagePatchEmbed(nn.Module):
    """
    I convert an image into a sequence of patch embeddings.
    With defaults:
      img_size=224, patch_size=16  => 14x14 patches => 196 tokens.
    So the transformer operates on tokens (patch vectors) instead of raw pixels.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=142):
        super().__init__()  # I register the module so parameters are tracked.

        self.img_size = img_size  # I store the input size expectation (helps shape reasoning).
        self.patch_size = patch_size  # I store patch size (defines token grid resolution).

        self.grid_h = img_size // patch_size  # number of patch rows (224//16 = 14).
        self.grid_w = img_size // patch_size  # number of patch cols (224//16 = 14).
        self.num_patches = self.grid_h * self.grid_w  # total tokens (14*14 = 196).

        # This Conv2d is the classic ViT patchify trick:
        # kernel=stride=patch_size => non-overlapping patches + linear projection to embed_dim.
        self.proj = nn.Conv2d(
            in_chans,  # 3 input channels for RGB.
            embed_dim,  # token embedding dimension (D).
            kernel_size=patch_size,  # patch height/width.
            stride=patch_size,  # step equals patch size => no overlap.
            bias=True,  # bias lets the projection shift token baselines.
        )

    def forward(self, x):
        # x is the rotated (or original) image batch: (B, 3, 224, 224) by default.
        x = self.proj(x)  # the model now sees a patch grid: (B, D, 14, 14).

        B, D, H, W = x.shape  # I capture shapes so I can flatten correctly.

        # flatten(2): (B, D, H*W) turns the 2D grid into a token list per channel.
        # transpose(1,2): (B, H*W, D) => the standard transformer token layout (B, N, D).
        tokens = x.flatten(2).transpose(1, 2)  # tokens: (B, N, D), where N=H*W.

        return tokens, H, W  # I return tokens plus token-grid size (H,W) for later steps.


class PositionalAndRotationEmbedding(nn.Module):
    """
    I add:
      - positional embedding: tells the model where each patch came from in the grid
      - rotation embedding: tells the model which rotation produced this token stream

    If the token grid gets shrunk (GSTE), I interpolate the positional grid so it still matches.
    """

    def __init__(self, base_h=14, base_w=14, embed_dim=142, n_rot=4):
        super().__init__()  # I initialize parameters correctly inside nn.Module.

        self.base_h = base_h  # base grid height (usually 14).
        self.base_w = base_w  # base grid width (usually 14).
        self.embed_dim = embed_dim  # token dimension D.
        self.n_rot = int(n_rot)  # number of supported rotation IDs (default 4).

        # One learnable vector per base-grid position: shape (1, base_h*base_w, D).
        self.pos = nn.Parameter(torch.zeros(1, base_h * base_w, embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)  # I initialize like standard ViT practice.

        # One learnable vector per rotation ID: shape (n_rot, D).
        self.rot = nn.Parameter(torch.zeros(self.n_rot, embed_dim))
        nn.init.trunc_normal_(self.rot, std=0.02)  # similar init keeps scales consistent.

    def forward(self, tokens, ht, wt, rot_idx):
        # tokens is (B, N, D) and ideally N == ht*wt.

        if ht == self.base_h and wt == self.base_w:  # if grid is unchanged, I can reuse pos directly.
            pos = self.pos  # (1, base_h*base_w, D)
        else:
            # If grid changed, I reshape pos into an image-like grid so I can interpolate smoothly.
            pos = self.pos.reshape(1, self.base_h, self.base_w, self.embed_dim)  # (1,H,W,D)
            pos = pos.permute(0, 3, 1, 2)  # (1,D,H,W) so interpolate works on spatial dims.
            pos = F.interpolate(  # I resize the positional grid to (ht, wt).
                pos,
                size=(ht, wt),
                mode="bilinear",
                align_corners=False,
            )
            pos = pos.permute(0, 2, 3, 1)  # back to (1,H,W,D)
            pos = pos.reshape(1, ht * wt, self.embed_dim)  # flatten back to (1,N,D)

        r = int(rot_idx) % self.n_rot  # I clamp rotation ID into [0..n_rot-1] safely.

        # The model now sees: content token, position, and rotation identity, all in the same D-dim vector.
        tokens = tokens + pos + self.rot[r].view(1, 1, -1)  # broadcast rot embedding over tokens.

        return tokens  # I return the enriched token sequence.

# --------------------------------------------------------------------------------------------------
# IMPORTANT: flexible MHSA when dim isn't divisible by heads
# --------------------------------------------------------------------------------------------------
class FlexibleMHSA(nn.Module):
    """
    Standard multi-head attention expects dim % heads == 0.
    Here I avoid shape errors by rounding head_dim up, then projecting back down afterward.

    Mechanically:
      head_dim = ceil(dim / heads)
      inner_dim = heads * head_dim   (>= dim)
    Attention runs in inner_dim, then I project back to dim.
    """

    def __init__(self, dim, num_heads, attn_dropout=0.1, proj_dropout=0.1):
        super().__init__()  # I register attention weights as parameters.

        self.dim = dim  # token embedding size D.
        self.num_heads = num_heads  # number of attention heads H.

        head_dim = int(math.ceil(dim / num_heads))  # I round up so each head has enough width.
        inner_dim = num_heads * head_dim  # total width used inside attention (may exceed dim).

        self.inner_dim = inner_dim  # I store for reshaping in forward().
        self.head_dim = head_dim  # I store for scaling and reshaping.

        self.scale = head_dim ** -0.5  # I scale dot-products to keep logits numerically stable.

        self.qkv = nn.Linear(  # I compute Q,K,V together for speed.
            dim,  # input token dim.
            inner_dim * 3,  # concatenated QKV output size.
            bias=True,  # bias improves flexibility in projections.
        )

        self.attn_drop = nn.Dropout(attn_dropout)  # dropout on attention weights (regularization).
        self.proj = nn.Linear(inner_dim, dim, bias=True)  # project from inner_dim back to dim.
        self.proj_drop = nn.Dropout(proj_dropout)  # dropout after projection (regularization).

    def forward(self, x, return_attn=False):
        B, N, D = x.shape  # B=batch size, N=tokens, D=embedding dim.

        qkv = self.qkv(x)  # the model maps each token into Q,K,V vectors in one shot: (B,N,3*inner).

        # I reshape to separate (Q,K,V) and heads:
        # (B, N, 3, H, head_dim) then permute to (3, B, H, N, head_dim).
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        q = qkv[0]  # queries: (B, H, N, head_dim)
        k = qkv[1]  # keys:    (B, H, N, head_dim)
        v = qkv[2]  # values:  (B, H, N, head_dim)

        # Each query token compares against all key tokens, per head.
        attn = (q @ k.transpose(-2, -1)) * self.scale  # attention logits: (B,H,N,N)

        attn = attn.softmax(dim=-1)  # convert logits into probabilities per query over keys.
        attn = self.attn_drop(attn)  # randomly drop some attention edges during training.

        out = attn @ v  # weighted sum of values: (B,H,N,head_dim)

        out = out.transpose(1, 2).reshape(B, N, self.inner_dim)  # merge heads: (B,N,inner_dim)

        out = self.proj(out)  # compress back to original token dim: (B,N,D)
        out = self.proj_drop(out)  # apply dropout after projection.

        if return_attn:  # if I’m doing XAI, I pass attention matrices out for rollout/inspection.
            return out, attn

        return out, None  # otherwise I keep signature stable with attn=None.

# --------------------------------------------------------------------------------------------------
# RViT BLOCK: MHSA -> depth-wise conv -> MLP (pre-LN and residual)
# --------------------------------------------------------------------------------------------------
class RViTBlock(nn.Module):
    """
    One RViT block mixes information in three stages:
      1) MHSA: global mixing (any token can attend to any other token).
      2) DWConv: local mixing (neighbor patches influence each other in the grid).
      3) MLP: per-token nonlinear transform.

    Structure:
      x = x + Attn(LN(x))
      x = x + DWConv(LN(x))   (after reshaping tokens back into (H,W))
      x = x + MLP(LN(x))
    """

    def __init__(self, dim, heads, mlp_dim, attn_dropout=0.1, dropout=0.1, ht=14, wt=14):
        super().__init__()  # I register everything as trainable submodules.

        self.ht = ht  # expected token-grid height (base case).
        self.wt = wt  # expected token-grid width (base case).

        self.ln1 = nn.LayerNorm(dim)  # pre-LN for attention keeps gradients stable.
        self.attn = FlexibleMHSA(  # attention module (handles odd dim/head splits).
            dim,  # token dimension.
            heads,  # number of heads.
            attn_dropout=attn_dropout,  # attention dropout.
            proj_dropout=dropout,  # projection dropout.
        )

        self.ln2 = nn.LayerNorm(dim)  # pre-LN before local conv mixing.

        self.dwconv = nn.Conv2d(  # depthwise conv mixes local neighborhoods per channel.
            dim,  # in_channels = dim (treat token channels as conv channels).
            dim,  # out_channels = dim (same width).
            kernel_size=3,  # 3x3 local neighborhood.
            padding=1,  # keep spatial size unchanged.
            groups=dim,  # groups=dim => each channel has its own filter (depthwise).
            bias=True,  # bias allows local shift per channel.
        )

        self.dw_drop = nn.Dropout(dropout)  # dropout on conv output for regularization.

        self.ln3 = nn.LayerNorm(dim)  # pre-LN before MLP.

        self.mlp = nn.Sequential(  # classic transformer MLP block.
            nn.Linear(dim, mlp_dim),  # expand features.
            nn.GELU(),  # nonlinearity.
            nn.Dropout(dropout),  # dropout in the hidden expansion.
            nn.Linear(mlp_dim, dim),  # project back to token dim.
            nn.Dropout(dropout),  # dropout on output projection.
        )

    def forward(self, x, return_attn=False):
        # x is token sequence: (B, N, D).

        # --- Global mixing via attention ---
        x_norm = self.ln1(x)  # I normalize so attention sees stable token statistics.
        attn_out, attn = self.attn(x_norm, return_attn=return_attn)  # attention update (and maybe attn map).
        x = x + attn_out  # residual add: keep original token info while adding global interactions.

        # --- Local spatial mixing via depthwise conv ---
        y = self.ln2(x)  # normalize before reshaping to grid.

        B, N, D = y.shape  # capture shapes for reshaping.
        ht = self.ht  # start with the configured expected height.
        wt = self.wt  # start with the configured expected width.

        if ht * wt != N:  # if GSTE shrink changed token count, I infer a near-square grid.
            side = int(math.sqrt(N))  # rough square root guess.
            ht = max(side, 1)  # ensure at least 1.
            wt = max(N // ht, 1)  # compute width to match N.
            # (This keeps the conv path usable even when tokens were pooled.)

        y2 = y.transpose(1, 2).reshape(B, D, ht, wt)  # tokens -> spatial grid: (B,D,H,W).
        y2 = self.dwconv(y2)  # the model applies 3x3 local filters per channel.
        y2 = y2.reshape(B, D, ht * wt).transpose(1, 2)  # grid -> tokens back: (B,N,D).
        y2 = self.dw_drop(y2)  # dropout on local-mixed tokens.
        x = x + y2  # residual add: keep attention-mixed tokens and add local neighborhood info.

        # --- Per-token nonlinear transform via MLP ---
        x_norm2 = self.ln3(x)  # normalize for stable MLP input.
        x = x + self.mlp(x_norm2)  # residual add after MLP transform.

        return x, attn  # tokens plus attention matrix (if requested).


# --------------------------------------------------------------------------------------------------
# RViT ENCODER: stack of RViTBlocks
# --------------------------------------------------------------------------------------------------
class RViTEncoder(nn.Module):
    def __init__(self, dim=142, depth=10, heads=10, mlp_dim=480, attn_dropout=0.1, dropout=0.1, ht=14, wt=14):
        super().__init__()  # I register the encoder stack as a module.

        self.blocks = nn.ModuleList([  # I build a depth-sized list of identical blocks.
            RViTBlock(  # each block refines the token representation.
                dim,  # token dimension.
                heads,  # attention heads.
                mlp_dim,  # MLP expansion dimension.
                attn_dropout=attn_dropout,  # attention dropout.
                dropout=dropout,  # dropout in projections/MLP/conv branch.
                ht=ht,  # expected token grid height.
                wt=wt,  # expected token grid width.
            )
            for _ in range(depth)  # repeat for the number of layers.
        ])

        self.ln = nn.LayerNorm(dim)  # final LN keeps output well-scaled.

    def forward(self, x, return_attn=False):
        attn_list = []  # I store attention maps per layer only when XAI asks for them.

        for blk in self.blocks:  # walk through the transformer depth.
            x, attn = blk(x, return_attn=return_attn)  # tokens updated by this layer.

            if return_attn and attn is not None:  # if requested, I keep the attention matrix.
                attn_list.append(attn)

        x = self.ln(x)  # final normalization of tokens.

        return x, attn_list  # return final tokens + list of attention matrices.


# --------------------------------------------------------------------------------------------------
# MAIN HYBRID MODEL (HYBRID B): CNN (PFD) and RViT (GSTE) -> fusion -> classifier
# --------------------------------------------------------------------------------------------------
class HybridResNet50V2_RViT(nn.Module):
    def __init__(
        self,
        num_classes=4,         # I predict 4 classes by default (so logits has width 4).
        img_size=224,          # I assume images are resized to 224x224.
        patch_size=16,         # use 16x16 patches (224/16 => 14 tokens per side).
        embed_dim=142,         # token embedding dimension D (intentionally odd to test flexibility).
        depth=10,              # number of transformer blocks.
        heads=10,              # number of attention heads.
        mlp_dim=480,           # hidden width of the transformer MLP.
        attn_dropout=0.1,      # dropout on attention weights.
        vit_dropout=0.1,       # dropout inside transformer projections/MLP/conv branch.
        fusion_dim=256,        # width used to fuse CNN and ViT descriptors.
        fusion_dropout=0.5,    # dropout used in CNN descriptor and fusion head.
        rotations=(0, 1, 2, 3),# 4 rotations: 0°,90°,180°,270° (encoded as k for rot90).
        cnn_name="resnetv2_50x1_bitm",  # timm backbone identifier.
        cnn_pretrained=True,   # whether to start from pretrained CNN weights.
        use_pfd_gste=True,     # full model (True) vs ablation path (False).
        gste_min_side=7,       # minimum token-grid side after shrinking (prevents over-shrinking).
        gste_std_flat=0.05,    # below this std, mask is treated as “flat” => no shrinking.
        gste_std_full=0.30,    # at/above this std, mask is “concentrated” => allow max shrink.
        gste_max_shrink=0.50,  # maximum shrink fraction (0.50 => up to 50% side reduction).
    ):
        super().__init__()  # I register this model as an nn.Module with trainable parameters.

        self.num_classes = num_classes  # I store class count so other methods can rely on it.
        self.rotations = rotations  #  store which rotations I will iterate over.
        self.patch_size = patch_size  # store patch size because token grid depends on it.
        self.img_size = img_size  # store expected image size for shape logic.
        self.use_pfd_gste = bool(use_pfd_gste)  # I force the flag into a clean boolean.

        # ----------------------------
        # CNN branch: ResNet50V2 feature extractor
        # ----------------------------
        self.cnn = ResNet50V2TimmBackbone(  # I build the CNN backbone wrapper.
            model_name=cnn_name,  # selects which ResNetV2 variant to use.
            pretrained=cnn_pretrained,  # whether to load pretrained weights.
        )

        self.cnn_out_ch = int(self.cnn.out_ch)  # I store CNN output channels (needed downstream).

        # ----------------------------
        # PFD: learned gating mask on CNN feature map
        # ----------------------------
        self.pfd = PFD(in_ch=self.cnn_out_ch)  # I create the mask gate module.

        # ----------------------------
        # CNN descriptor head
        # ----------------------------
        self.cnn_pool = nn.AdaptiveAvgPool2d(1)  # I pool (H,W) -> (1,1) per channel.
        self.cnn_drop = nn.Dropout(p=float(fusion_dropout))  # dropout on CNN descriptor for regularization.
        self.cnn_proj = nn.Linear(self.cnn_out_ch, fusion_dim)  # project CNN channels -> fusion width.
        self.cnn_bn = nn.BatchNorm1d(fusion_dim)  # stabilize fused CNN vector distribution.
        # (ReLU happens in forward so it’s explicit when reading the pipeline.)

        # ----------------------------
        # Transformer branch: patch embed and pos/rot embed + encoder
        # ----------------------------
        grid = img_size // patch_size  # base tokens per side (224//16 = 14).
        self.base_grid = int(grid)  # I store base side length for GSTE decisions.

        self.gste_min_side = int(max(1, min(gste_min_side, self.base_grid)))  # clamp to valid range.
        self.gste_std_flat = float(gste_std_flat)  # threshold where mask is too flat to guide shrink.
        self.gste_std_full = float(gste_std_full)  # threshold where mask is fully concentrated.
        self.gste_max_shrink = float(gste_max_shrink)  # maximum side shrink fraction.

        self.patch = ImagePatchEmbed(  # I build the patchify+project layer.
            img_size=img_size,  # expected input image size.
            patch_size=patch_size,  # patch size.
            in_chans=3,  # RGB input.
            embed_dim=embed_dim,  # token dimension D.
        )

        self.posrot = PositionalAndRotationEmbedding(  # I build position+rotation embedding module.
            base_h=grid,  # base grid height.
            base_w=grid,  # base grid width.
            embed_dim=embed_dim,  # token dimension.
            n_rot=4,  # rotation IDs (0..3).
        )

        self.encoder = RViTEncoder(  # I build the transformer encoder stack.
            dim=embed_dim,  # token dimension.
            depth=depth,  # number of layers.
            heads=heads,  # attention heads.
            mlp_dim=mlp_dim,  # MLP expansion width.
            attn_dropout=attn_dropout,  # attention dropout.
            dropout=vit_dropout,  # general dropout.
            ht=grid,  # base grid height.
            wt=grid,  # base grid width.
        )

        self.vit_proj = nn.Linear(embed_dim, fusion_dim)  # project transformer pooled vector -> fusion width.

        # ----------------------------
        # Fusion and classifier
        # ----------------------------
        self.fuse_fc = nn.Linear(fusion_dim * 2, fusion_dim)  # merge [cnn_vec, vit_vec] -> fusion_dim.
        self.fuse_drop = nn.Dropout(p=float(fusion_dropout))  # dropout on fused vector.
        self.out = nn.Linear(fusion_dim, num_classes)  # final classifier logits.

        # ----------------------------
        # XAI caches (not parameters; just last-forward memory)
        # ----------------------------
        self._last_cnn_feat = None  # I keep the last CNN feature map so Grad-CAM-style tools can hook it.
        self._last_cnn_mask_img = None  # I keep the last upsampled mask for overlay-style explanations.

    def freeze_cnn_bn(self):
        # I optionally freeze BN running stats inside the CNN backbone by setting BN2d to eval().
        for m in self.cnn.modules():  # walk every submodule in the CNN.
            if isinstance(m, nn.BatchNorm2d):  # find 2D batch norms inside the backbone.
                m.eval()  # stop updating running mean/var (useful for small batch sizes).

    def _mask_to_alpha_grid(self, mask_img, out_side):
        """
        mask_img: (B, 1, H, W) at image scale
        out_side: target token side (e.g., 14 for 14x14)

        I downsample mask to token-grid alpha (B,1,out_side,out_side),
        then normalize mean to ~1 so token magnitudes don’t collapse or explode.
        """
        alpha = F.adaptive_avg_pool2d(  # I pool the dense mask into a coarse token-aligned grid.
            mask_img,  # image-scale mask.
            output_size=(out_side, out_side),  # token grid resolution.
        )
        alpha = alpha / (alpha.mean(dim=(2, 3), keepdim=True) + 1e-6)  # normalize mean alpha per sample.
        return alpha  # the model uses this alpha as “importance weights” per token cell.

    def _choose_dynamic_side(self, alpha_base):
        """
        alpha_base: (B,1,g,g) where g=base_grid

        I decide whether to shrink based on how concentrated the mask is.
        The idea:
          - flat mask => no meaningful ROI => keep full grid
          - peaky mask => ROI exists => shrink and focus compute on ROI
        """
        with torch.no_grad():  # I don’t need gradients for this decision logic.
            std = alpha_base.std(dim=(2, 3), keepdim=False)  # std over spatial cells => peakiness.
            s = float(std.mean().item())  # I use batch mean so every sample uses same side.

            if s <= self.gste_std_flat:  # if mask is near-uniform, shrinking would be arbitrary.
                return self.base_grid  # keep full resolution tokens.

            denom = max(self.gste_std_full - self.gste_std_flat, 1e-6)  # avoid division by zero.
            t = (s - self.gste_std_flat) / denom  # map std into a 0..1 interval.
            t = max(0.0, min(1.0, t))  # clamp into [0,1].

            shrink = self.gste_max_shrink * t  # shrink grows smoothly until max_shrink.
            target = int(round(self.base_grid * (1.0 - shrink)))  # convert shrink to a target side length.
            side = max(self.gste_min_side, min(self.base_grid, target))  # clamp side into allowed range.

            return side  # this is the side length used for dynamic token pooling.

    def _gste_dynamic_tokens(self, tokens_base, alpha_base, side):
        """
        tokens_base: (B, g*g, D) where g=base_grid
        alpha_base : (B, 1, g, g) mean-normalized
        side       : chosen side <= g

        What I do:
          1) reshape tokens -> (B,D,g,g)
          2) multiply by alpha (ROI weighting)
          3) if side < g, weighted pooling to (side,side)
          4) flatten back -> (B, side*side, D)
        """
        B, N, D = tokens_base.shape  # read shapes from token tensor.
        g = self.base_grid  # base token side length.

        x = tokens_base.transpose(1, 2).reshape(B, D, g, g)  # token seq -> spatial map.
        w = alpha_base  # weights map aligned with token grid.

        x = x * w  # ROI weighting: boosts token energy where alpha is high.

        if side < g:  # if shrinking is active, I pool tokens down to fewer cells.
            num = F.adaptive_avg_pool2d(x, output_size=(side, side))  # pooled weighted token values.
            den = F.adaptive_avg_pool2d(w, output_size=(side, side)) + 1e-6  # pooled weights sum.
            x = num / den  # proper weighted average so magnitude stays meaningful.

        tokens = x.flatten(2).transpose(1, 2)  # spatial map -> token seq again.
        return tokens, side, side  # return tokens plus the new grid dimensions.

    def forward(self, x, return_xai=False):
        # x is the input image batch: (B,3,224,224) typically.

        # ==========================================================================================
        # 1) CNN backbone forward
        # ==========================================================================================
        feat = self.cnn(x)  # the model extracts a deep feature map: (B,C,7,7).

        # ==========================================================================================
        # 2) PFD gating (optional)
        # ==========================================================================================
        if self.use_pfd_gste:  # full model path uses the learned mask.
            feat_path, mask_feat = self.pfd(feat)  # produce gated CNN features and low-res mask.
            feat_for_cnn = feat_path  # CNN descriptor uses gated features so it also focuses on ROI.

            mask_img = F.interpolate(  # I upsample the 7x7 mask to image scale for guidance and overlays.
                mask_feat,  # (B,1,7,7)
                size=(x.shape[2], x.shape[3]),  # match original image size (H,W).
                mode="bilinear",  # smooth upsampling.
                align_corners=False,  # standard safe setting for bilinear.
            )
        else:  # ablation path disables PFD/GSTE.
            feat_for_cnn = feat  # CNN descriptor uses raw CNN features.
            mask_img = None  # no mask exists in this mode.

        self._last_cnn_feat = feat_for_cnn  # cache for CAM hooks / debugging.
        self._last_cnn_mask_img = mask_img  # cache mask overlay for visualization tools.

        # ==========================================================================================
        # 3) CNN pooled descriptor head
        # ==========================================================================================
        z_cnn = self.cnn_pool(feat_for_cnn).flatten(1)  # (B,C): global average pooling collapses 7x7.
        z_cnn = self.cnn_drop(z_cnn)  # dropout: prevents relying on a small subset of channels.
        z_cnn = self.cnn_proj(z_cnn)  # (B,fusion_dim): project CNN descriptor into fusion width.
        z_cnn = self.cnn_bn(z_cnn)  # batchnorm: stabilize distribution of CNN descriptor.
        z_cnn = F.relu(z_cnn, inplace=True)  # nonlinearity: model keeps positive-biased features.

        # ==========================================================================================
        # 4) GSTE: choose a dynamic token grid size (optional)
        # ==========================================================================================
        if self.use_pfd_gste:  # only decide shrink when guidance exists.
            alpha0 = self._mask_to_alpha_grid(mask_img, self.base_grid)  # (B,1,g,g) guidance grid.
            dyn_side = self._choose_dynamic_side(alpha0)  # choose shrink side based on mask peakiness.
        else:
            dyn_side = self.base_grid  # ablation: keep full token grid.

        # ==========================================================================================
        # 5) Rotation ViT path: rotate -> patchify -> (optional GSTE) -> add embeddings
        # ==========================================================================================
        token_sets = []  # I’ll store one token sequence per rotation.

        for k in self.rotations:  # iterate through rotation IDs.
            rot_id = int(k) % 4  # ensure rot90 uses a valid 0..3 value.

            x_r = torch.rot90(  # rotate the image so transformer sees a different orientation.
                x,  # original image batch.
                k=rot_id,  # number of 90-degree rotations.
                dims=(2, 3),  # rotate over H and W dims.
            )

            tokens_base, ht, wt = self.patch(x_r)  # patchify rotated image: tokens (B,N,D), grid (ht,wt).

            if self.use_pfd_gste:  # if guidance is enabled, align mask to this rotation too.
                m_r = torch.rot90(  # rotate the mask the same way so ROI stays aligned to patches.
                    mask_img,  # image-scale mask (B,1,H,W).
                    k=rot_id,  # same rotation index.
                    dims=(2, 3),  # rotate H/W.
                )

                alpha_base = self._mask_to_alpha_grid(m_r, self.base_grid)  # token-grid alpha weights.

                tokens, ht, wt = self._gste_dynamic_tokens(  # apply weighting + optional shrink.
                    tokens_base,  # base tokens.
                    alpha_base,  # weights aligned to tokens.
                    dyn_side,  # target side length.
                )
            else:
                tokens, ht, wt = tokens_base, ht, wt  # ablation: no token weighting, no shrink.

            tokens = self.posrot(tokens, ht, wt, rot_id)  # inject position and rotation identity into tokens.
            token_sets.append(tokens)  # store this rotation’s token sequence.

        Tavg = torch.stack(token_sets, dim=0).mean(dim=0)  # average across rotations before encoding.

        # ==========================================================================================
        # 6) Transformer encoder: global+local reasoning over patch tokens
        # ==========================================================================================
        Tenc, attn_list = self.encoder(  # encoder refines token relationships layer-by-layer.
            Tavg,  # averaged token set.
            return_attn=return_xai,  # only collect attention maps when XAI is requested.
        )

        z_vit = Tenc.mean(dim=1)  # global average pooling over tokens (no CLS token here).
        z_vit = self.vit_proj(z_vit)  # project transformer summary into fusion_dim.
        z_vit = F.relu(z_vit, inplace=True)  # nonlinearity before fusion.

        # ==========================================================================================
        # 7) Fusion + classifier
        # ==========================================================================================
        z = torch.cat([z_cnn, z_vit], dim=1)  # concatenate descriptors so classifier sees both branches.
        h = self.fuse_fc(z)  # fuse down to fusion_dim.
        h = F.relu(h, inplace=True)  # nonlinearity on fused representation.
        h = self.fuse_drop(h)  # dropout: regularize final classifier input.
        logits = self.out(h)  # final class scores (B,num_classes).

        # ==========================================================================================
        # 8) XAI payload (optional)
        # ==========================================================================================
        if return_xai:
            # - "attn": per-layer attention matrices (lets you do attention rollout / token influence).
            # - "mask": image-scale learned mask (a soft heatmap you can overlay on input image).
            # - "gste_side": the chosen token-grid side length (shows whether model shrank tokens).
            return logits, {
                "attn": attn_list,     # list of (B, heads, N, N) per layer.
                "mask": mask_img,      # (B,1,H,W) soft spatial mask.
                "gste_side": dyn_side  # int side length used for dynamic tokens.
            }

        return logits, None  # default: predictions only.

    @torch.no_grad()
    def mc_dropout_predict(self, x, mc_samples=20):
        # Monte Carlo dropout inference:
        # I intentionally keep dropout stochastic at inference time to estimate uncertainty.

        self.eval()  # set model to eval mode (BN uses running stats; deterministic layers stay stable).

        for m in self.modules():  # walk through all submodules.
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):  # locate dropout modules.
                m.train()  # re-enable dropout randomness even though model is in eval().

        probs = []  # I’ll collect class probability vectors from multiple stochastic forward passes.

        for _ in range(int(mc_samples)):  # repeat forward pass mc_samples times.
            logits, _ = self.forward(x, return_xai=False)  # forward without XAI overhead.
            p = torch.softmax(logits, dim=1)  # convert logits into probabilities.
            probs.append(p)  # store this sample’s probability output.

        probs = torch.stack(probs, dim=0)  # shape: (S, B, C) where S=mc_samples.

        mu = probs.mean(dim=0)  # predictive mean probability per class.
        var = probs.var(dim=0, unbiased=False)  # predictive variance per class (uncertainty signal).

        return mu, var  # return mean + variance as the MC-dropout estimate.


class HybridResNet50V2_RViT_Ablation(HybridResNet50V2_RViT):
    """
    Ablation version:
    - I disable PFD and GSTE so the model runs without mask guidance or dynamic token shrinking.
    - This keeps the rest of the pipeline identical for a fair comparison.
    """

    def __init__(self, *args, **kwargs):
        kwargs["use_pfd_gste"] = False  # I force ablation mode regardless of caller input.
        super().__init__(*args, **kwargs)  # build the parent with guidance disabled.
