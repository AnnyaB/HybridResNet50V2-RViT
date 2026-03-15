# THIS IS THE FILE FOR THE HYBRID MODEL WITHOUT PFD-GSTE VARIANT B (ABLATION).
# External libraries used here are cited in Appendix A2.3:
# PyTorch (Paszke et al., 2019); timm (Wightman, 2019).


# Libraries I needed 

# - math is used for ceil/sqrt and attention scaling helpers
# - torch / nn / F provide tensors, modules, and functional ops

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# CNN: pretrained ResNet50V2 (timm)

# This backbone wraps a timm ResNetV2 model configured to return feature maps only.
# The hybrid model uses these CNN feature maps as one expert branch (CNN pathway).
class ResNet50V2TimmBackbone(nn.Module):
    
    """
    Uses timm pretrained ResNetV2 feature extractor.
    Returns last feature map (B, C, 7, 7) for 224x224 input.
    """
    def __init__(self):
        super().__init__()
        # model_name selects the timm architecture string
        # pretrained controls whether timm loads pretrained weights
    def __init__(self, model_name="resnetv2_50x1_bitm", pretrained=True):
        super().__init__()
        # timm is imported here so the project can still load (with a clear error)
        # even if timm isn't installed in a given environment.
        try:
            import timm
        except Exception as e:
            raise ImportError(
                "timm is required for pretrained ResNet50V2. "
                "Install timm or use an environment (e.g., Kaggle) that includes it."
            ) from e

        # Storing the model name for reference/debugging
        self.model_name = model_name
        # Creating a feature-extractor version of the model:
        # - features_only=True makes timm return intermediate feature maps
        # - out_indices=(4,) requests only the final stage feature map
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(4,),  # last stage
        )

        # Determine the number of channels produced by the requested feature stage.
        # This avoids needing a dummy forward pass.
        try:
            self.out_ch = int(self.backbone.feature_info.channels()[-1])
        except Exception:
            # Fallback to a common ResNet50 feature depth if feature_info is unavailable.
            self.out_ch = 2048  # fallback common value

    # Forward returns the final feature map from timm's features-only backbone.
    def forward(self, x):
        feats = self.backbone(x)  # list length 1
        return feats[-1]



# PFD: pathology mask gating (VARIANT B)

# PFD learns a soft (sigmoid) spatial mask over CNN features and gates them.
# This is only used when use_pfd_gste=True in the hybrid model.
class PFD(nn.Module):
    """
    PFD (VARIANT B): learned pathology gate on CNN feature maps.
    M = sigmoid(conv1x1(F)), F_path = M * F
    """
    # in_ch is the channel count of the CNN feature map (e.g., 2048)
    def __init__(self, in_ch):
        super().__init__()
        # 1x1 conv reduces (B,C,H,W) -> (B,1,H,W) to produce a single-channel mask logits map.
        self.mask_conv = nn.Conv2d(in_ch, 1, kernel_size=1, stride=1, padding=0, bias=True)

    # Returns gated features and the mask itself.
    def forward(self, feat):
        # Converting conv output to [0,1] mask via sigmoid
        mask = torch.sigmoid(self.mask_conv(feat))  # (B,1,h,w) in [0,1]
        # Gating CNN features spatially (broadcast over channels)
        gated = feat * mask                         # (B,C,h,w)
        return gated, mask

# RViT: patchify IMAGE (P=16) -> tokens (B,N,D)


# This module does ViT-style patch embedding using a Conv2d with kernel=stride=patch_size.
# Each patch becomes one token (linear projection).
class ImagePatchEmbed(nn.Module):
    
    """
    Patch embedding as linear projection of flattened patches.
    Conv2d(kernel=P, stride=P) is equivalent to linear patch projection in ViT.
    """
    # img_size and patch_size define the grid; embed_dim is token dimension D
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=142):
        super().__init__()
        # Storing sizes for reference
        self.img_size = img_size
        self.patch_size = patch_size
        # Computing patch grid size (e.g., 224/16 = 14)
        self.grid_h = img_size // patch_size
        self.grid_w = img_size // patch_size
        # Total tokens per image = grid_h * grid_w (e.g., 196)
        self.num_patches = self.grid_h * self.grid_w

        # Patch projection: (B,3,224,224) -> (B,D,14,14)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

    def forward(self, x):
        # x: (B,3,H,W) -> (B,D,gh,gw) -> (B,N,D)
        x = self.proj(x)
        B, D, H, W = x.shape
        # Flattening spatial grid into tokens and transpose to (B,N,D)
        tokens = x.flatten(2).transpose(1, 2)
        # Returning tokens plus grid height/width so later modules know token layout
        return tokens, H, W


# Adds:
# - learnable positional embeddings (for token locations)
# - learnable rotation embeddings (for which rotation was applied)
# Also supports interpolating positional embeddings if token grid is not the base 14x14.
class PositionalAndRotationEmbedding(nn.Module):
    
    """
    Learnable positional embeddings and rotation embeddings (Krishnan-style).
    Default base grid for 224, P=16 is 14x14.
    """
    def __init__(self, base_h=14, base_w=14, embed_dim=142, n_rot=4):
        super().__init__()
        # Storing base grid and embedding config
        self.base_h = base_h
        self.base_w = base_w
        self.embed_dim = embed_dim
        self.n_rot = int(n_rot)

        # Positional embeddings for the base token grid: (1, 196, D)
        self.pos = nn.Parameter(torch.zeros(1, base_h * base_w, embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

        # Rotation embeddings for 4 rotations: (4, D)
        self.rot = nn.Parameter(torch.zeros(self.n_rot, embed_dim))
        nn.init.trunc_normal_(self.rot, std=0.02)

    def forward(self, tokens, ht, wt, rot_idx):
        
        # Interpolating pos if grid differs
        if ht == self.base_h and wt == self.base_w:
            # If grid matches base, use learned pos directly
            pos = self.pos
        else:
            # Reshaping to (1,D,H,W) so we can interpolate in 2D
            pos = self.pos.reshape(1, self.base_h, self.base_w, self.embed_dim).permute(0, 3, 1, 2)
            # Interpolate positional grid to new (ht, wt)
            pos = F.interpolate(pos, size=(ht, wt), mode="bilinear", align_corners=False)
            # Reshaping back to token sequence: (1, ht*wt, D)
            pos = pos.permute(0, 2, 3, 1).reshape(1, ht * wt, self.embed_dim)

        # Map rot_idx to one of n_rot embeddings safely
        r = int(rot_idx) % self.n_rot
        # Adding positional and rotation embedding to each token
        tokens = tokens + pos + self.rot[r].view(1, 1, -1)
        return tokens

# Flexible MHSA (supports dim not divisible by heads)

# Multi-Head Self Attention normally requires dim % heads == 0.
# This implementation avoids dropping dimensions by rounding UP each head dimension.
class FlexibleMHSA(nn.Module):
    
    """
    Flexible MHSA for dim not divisible by heads.
    Uses inner_dim = heads * ceil(dim/heads) so I do NOT drop dimensions.
    Attention runs in inner_dim, then projected back to dim.
    """
    def __init__(self, dim, num_heads, attn_dropout=0.1, proj_dropout=0.1):
        super().__init__()
        # Save config
        self.dim = dim
        self.num_heads = num_heads

        # Use ceil to ensure all original dimensions are representable
        head_dim = int(math.ceil(dim / num_heads))   # ceil, not floor
        inner_dim = num_heads * head_dim

        # Store derived dimensions and scale factor for dot-product attention
        self.inner_dim = inner_dim
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        # Single linear layer generates Q,K,V concatenated
        self.qkv = nn.Linear(dim, inner_dim * 3, bias=True)
        # Dropout on attention probabilities
        self.attn_drop = nn.Dropout(attn_dropout)
        # Project back from inner_dim to original dim
        self.proj = nn.Linear(inner_dim, dim, bias=True)
        # Dropout on projected output
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x, return_attn=False):
        # x is a token sequence: (B,N,D)
        B, N, D = x.shape
        # Compute QKV: (B,N,3*inner_dim)
        qkv = self.qkv(x)  # (B,N,3*inner_dim)
        # Reshape to separate Q,K,V and heads: (3,B,H,N,hd)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # Split out Q,K,V each: (B,H,N,hd)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B,H,N,hd)

        # Scaled dot-product attention: (B,H,N,N)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B,H,N,N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values: (B,H,N,hd)
        out = attn @ v  # (B,H,N,hd)
        # Merge heads back: (B,N,inner_dim)
        out = out.transpose(1, 2).reshape(B, N, self.inner_dim)  # (B,N,inner_dim)
        # Project to model dim: (B,N,dim)
        out = self.proj(out)  # (B,N,dim)
        out = self.proj_drop(out)

        # Optionally return attention matrix for XAI rollout
        if return_attn:
            return out, attn
        return out, None


# One encoder block:
# - Pre-LN + MHSA + residual
# - Pre-LN + depth-wise conv (token->grid->conv->tokens) + residual
# - Pre-LN + MLP + residual
class RViTBlock(nn.Module):
    """
    MHSA -> DWConv -> MLP, each with pre-LN + residual.
    """
    def __init__(self, dim, heads, mlp_dim, attn_dropout=0.1, dropout=0.1, ht=14, wt=14):
        super().__init__()
        # Store expected grid shape (used to reshape tokens for DWConv)
        self.ht = ht
        self.wt = wt

        # Attention sub-layer
        self.ln1 = nn.LayerNorm(dim)
        self.attn = FlexibleMHSA(dim, heads, attn_dropout=attn_dropout, proj_dropout=dropout)

        # Depth-wise convolution sub-layer (token mixing in 2D)
        self.ln2 = nn.LayerNorm(dim)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True)
        self.dw_drop = nn.Dropout(dropout)

        # MLP sub-layer
        self.ln3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, return_attn=False):
        # Pre-LN -> attention -> residual
        attn_out, attn = self.attn(self.ln1(x), return_attn=return_attn)
        x = x + attn_out

        # Pre-LN for conv branch
        y = self.ln2(x)
        B, N, D = y.shape
        # Use stored ht/wt as default grid
        ht, wt = self.ht, self.wt

        # If token count doesn't match ht*wt (for instance., GSTE dynamic tokens),
        # estimate a grid shape so we can still apply 2D DWConv.
        if ht * wt != N:
            side = int(math.sqrt(N))
            ht = max(side, 1)
            wt = max(N // ht, 1)

        # Tokens -> (B,D,ht,wt) -> depth-wise conv -> back to (B,N,D)
        y2 = y.transpose(1, 2).reshape(B, D, ht, wt)
        y2 = self.dwconv(y2)
        y2 = y2.reshape(B, D, ht * wt).transpose(1, 2)
        y2 = self.dw_drop(y2)
        # Residual add for conv branch
        x = x + y2

        # Pre-LN -> MLP -> residual
        x = x + self.mlp(self.ln3(x))
        # Return tokens and (optionally) attention matrix
        return x, attn


# Stacks multiple RViTBlocks and returns:
# - final normalized token sequence
# - attention matrices per block (if requested for XAI)
class RViTEncoder(nn.Module):
    
    def __init__(self, dim=142, depth=10, heads=10, mlp_dim=480, attn_dropout=0.1, dropout=0.1, ht=14, wt=14):
        super().__init__()
        # Creating a list of identical blocks (depth times)
        self.blocks = nn.ModuleList([
            RViTBlock(dim, heads, mlp_dim, attn_dropout=attn_dropout, dropout=dropout, ht=ht, wt=wt)
            for _ in range(depth)
        ])
        # Final LayerNorm after all blocks
        self.ln = nn.LayerNorm(dim)

    def forward(self, x, return_attn=False):
        # Collecting attention matrices for rollout if requested
        attn_list = []
        for blk in self.blocks:
            x, attn = blk(x, return_attn=return_attn)
            if return_attn and attn is not None:
                attn_list.append(attn)
        # Final normalization
        x = self.ln(x)
        return x, attn_list



# Hybrid model (FULL or ABLATION via flag)

# This is the main hybrid architecture:
# - CNN branch: ResNetV2 features -> pooled -> regularised head -> fusion_dim vector
# - Transformer branch: rotate image -> patchify -> (pos+rot) -> average -> encoder -> pooled -> fusion_dim vector
# - Fusion head: concat -> FC -> dropout -> classifier logits

# Optional (use_pfd_gste=True):
# - PFD gating on CNN features produces a mask
# - GSTE uses mask to guide a dynamic token grid (saliency-driven)
class HybridResNet50V2_RViT(nn.Module):
    def __init__(
        self,
        num_classes=4,
        img_size=224,
        patch_size=16,
        embed_dim=142,
        depth=10,
        heads=10,
        mlp_dim=480,
        attn_dropout=0.1,
        vit_dropout=0.1,
        fusion_dim=256,
        fusion_dropout=0.5,
        rotations=(0, 1, 2, 3),
        cnn_name="resnetv2_50x1_bitm",
        cnn_pretrained=True,
        use_pfd_gste=False,
        gste_min_side=7,
        gste_std_flat=0.05,
        gste_std_full=0.30,
        gste_max_shrink=0.50,
    ):
        super().__init__()
        # Storing core config for later use
        self.num_classes = num_classes
        self.rotations = rotations
        self.patch_size = patch_size
        self.img_size = img_size
        self.use_pfd_gste = bool(use_pfd_gste)

        # CNN branch (Sarada starting point: pretrained backbone)
        self.cnn = ResNet50V2TimmBackbone(model_name=cnn_name, pretrained=cnn_pretrained)
        self.cnn_out_ch = int(self.cnn.out_ch)

        # PFD VARINT B
        self.pfd = PFD(in_ch=self.cnn_out_ch)

        # head regularisation (minimal, backbone intact)
        self.cnn_drop = nn.Dropout(p=float(fusion_dropout))
        self.cnn_proj = nn.Linear(self.cnn_out_ch, fusion_dim)
        self.cnn_bn = nn.BatchNorm1d(fusion_dim)
        self.cnn_pool = nn.AdaptiveAvgPool2d(1)

        # Transformer branch: patchify IMAGE
        grid = img_size // patch_size  # 14 for 224/16
        self.base_grid = int(grid)
        self.gste_min_side = int(max(1, min(gste_min_side, self.base_grid)))

        # GSTE selection knobs (robust dynamic side selection)
        self.gste_std_flat = float(gste_std_flat)   # below this => treat mask as flat => keep full grid
        self.gste_std_full = float(gste_std_full)   # at/above this => allow maximum shrink
        self.gste_max_shrink = float(gste_max_shrink)  # fraction of base grid to shrink at most (e.g., 0.50 => down to ~7)

        # Patch embedding AND (pos+rot) embedding AND encoder
        self.patch = ImagePatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        self.posrot = PositionalAndRotationEmbedding(base_h=grid, base_w=grid, embed_dim=embed_dim, n_rot=4)
        self.encoder = RViTEncoder(
            dim=embed_dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            attn_dropout=attn_dropout,
            dropout=vit_dropout,
            ht=grid,
            wt=grid,
        )
        # Project transformer pooled vector to fusion_dim
        self.vit_proj = nn.Linear(embed_dim, fusion_dim)

        # Fusion head
        self.fuse_fc = nn.Linear(fusion_dim * 2, fusion_dim)
        self.fuse_drop = nn.Dropout(p=float(fusion_dropout))
        self.out = nn.Linear(fusion_dim, num_classes)

        # For external XAI hooks (optional)
        self._last_cnn_feat = None
        self._last_cnn_mask_img = None

    # Utility: freeze BatchNorm2d layers inside CNN backbone by setting them to eval()
    def freeze_cnn_bn(self):
        for m in self.cnn.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    # Convert image-scale mask to a token-grid guidance map (alpha) of size out_side x out_side.
    # Then normalize alpha so its mean is ~1 (keeps feature scale stable across samples).
    def _mask_to_alpha_grid(self, mask_img, out_side):
        """
        mask_img: (B,1,H,W) -> adaptive pool to (out_side,out_side)
        Return alpha_grid with mean≈1 per-sample to keep scale stable.
        """
        alpha = F.adaptive_avg_pool2d(mask_img, output_size=(out_side, out_side))  # (B,1,s,s)
        alpha = alpha / (alpha.mean(dim=(2, 3), keepdim=True) + 1e-6)
        return alpha

    # Deciding how much to shrink the token grid based on mask "concentration":
    # - If mask is flat (low std), do NOT shrink (keep full grid to avoid collapse)
    # - If mask is peaked (high std), allow shrink down to gste_min_side (bounded)
    def _choose_dynamic_side(self, alpha_base):
        """
        GSTE dynamic grid side in [gste_min_side .. base_grid], guided by mask concentration.

        Fixes the collapse on flat mask issue:
        - If mask is flat/uncertain (std very low), KEEP full grid.
        - Shrink grid only when mask is truly concentrated/peaked.
        """
        with torch.no_grad():
            # alpha_base mean≈1; use std as concentration proxy
            std = alpha_base.std(dim=(2, 3), keepdim=False)  # (B,1)
            s = float(std.mean().item())

            # flat mask => keep full grid (prevents early-training collapse)
            if s <= self.gste_std_flat:
                return self.base_grid

            # map std in [flat .. full] -> shrink fraction in [0 .. max_shrink]
            denom = max(self.gste_std_full - self.gste_std_flat, 1e-6)
            t = (s - self.gste_std_flat) / denom
            t = max(0.0, min(1.0, t))

            shrink = self.gste_max_shrink * t  # 0..max_shrink
            target = int(round(self.base_grid * (1.0 - shrink)))

            # clamp to allowed range
            side = max(self.gste_min_side, min(self.base_grid, target))
            return side

    # Applying the mask-guidance to tokens and optionally downsample token grid to (side x side).
    # Weighted pooling is used so ROI emphasis remains meaningful after downsampling.
    
    def _gste_dynamic_tokens(self, tokens_base, alpha_base, side):
        
        """
        Mask-guided dynamic token evolution:
        - tokens_base: (B,N,D) with N=base_grid^2
        - alpha_base : (B,1,base_grid,base_grid), mean≈1
        - side: dynamic grid side (<= base_grid)

        Returns tokens_dyn: (B, side*side, D) and (side,side).
        """
        B, N, D = tokens_base.shape
        g = self.base_grid

        x = tokens_base.transpose(1, 2).reshape(B, D, g, g)          # (B,D,g,g)
        w = alpha_base                                               # (B,1,g,g)

        # Always apply guidance (ROI emphasis)
        x = x * w                                                    # broadcast -> (B,D,g,g)

        # If downsampling, use weighted pooling so ROI emphasis is preserved
        if side < g:
            num = F.adaptive_avg_pool2d(x, output_size=(side, side))  # (B,D,side,side)
            den = F.adaptive_avg_pool2d(w, output_size=(side, side)) + 1e-6  # (B,1,side,side)
            x = num / den

        tokens = x.flatten(2).transpose(1, 2)                        # (B, side*side, D)
        return tokens, side, side

    # Forward pass:
    # - Produces logits
    # - Optionally returns XAI dict with attentions, mask, and chosen GSTE side
    
    def forward(self, x, return_xai=False):
        # =========================
        # CNN backbone
        # =========================
        feat = self.cnn(x)  # (B,C,7,7)

        # =========================
        # PFD (optional)
        # =========================
        if self.use_pfd_gste:
            feat_path, mask_feat = self.pfd(feat)  # (B,C,7,7), (B,1,7,7)
            feat_for_cnn = feat_path
            mask_img = F.interpolate(mask_feat, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
        else:
            feat_for_cnn = feat
            mask_img = None

        # Save for possible Grad-CAM++ hooking / debugging
        self._last_cnn_feat = feat_for_cnn
        self._last_cnn_mask_img = mask_img

        # =========================
        # CNN pooled vector (head regularisation)
        # =========================
        z_cnn = self.cnn_pool(feat_for_cnn).flatten(1)   # (B,C)
        z_cnn = self.cnn_drop(z_cnn)
        z_cnn = self.cnn_proj(z_cnn)                     # (B,fusion_dim)
        z_cnn = self.cnn_bn(z_cnn)
        z_cnn = F.relu(z_cnn, inplace=True)

        # =========================
        # GSTE: choose dynamic token grid once (batch-consistent) from mask
        # =========================
        if self.use_pfd_gste:
            alpha0 = self._mask_to_alpha_grid(mask_img, self.base_grid)  # (B,1,14,14)
            dyn_side = self._choose_dynamic_side(alpha0)                 # int in [min..14]
        else:
            dyn_side = self.base_grid

        # =========================
        # RViT rotations: rotate IMAGE, patchify P=16, average embeddings
        # =========================
        token_sets = []
        for k in self.rotations:
            # rotate IMAGE (k in {0,1,2,3} for 0/90/180/270)
            rot_id = int(k) % 4
            x_r = torch.rot90(x, k=rot_id, dims=(2, 3))

            tokens_base, ht, wt = self.patch(x_r)  # (B,14*14,D), ht=wt=14

            if self.use_pfd_gste:
                # rotate mask at IMAGE scale to match rotated image, then pool to base grid
                m_r = torch.rot90(mask_img, k=rot_id, dims=(2, 3))    # (B,1,H,W)
                alpha_base = self._mask_to_alpha_grid(m_r, self.base_grid)  # (B,1,14,14)

                # dynamic token evolution (guided, dynamic grid)
                tokens, ht, wt = self._gste_dynamic_tokens(tokens_base, alpha_base, dyn_side)
            else:
                tokens, ht, wt = tokens_base, ht, wt

            # adding positional and rotation embeddings
            tokens = self.posrot(tokens, ht, wt, rot_id)  # interpolates pos if ht/wt != 14
            token_sets.append(tokens)

        # Average embeddings across rotations BEFORE encoder (Average integration)
        Tavg = torch.stack(token_sets, dim=0).mean(dim=0)  # (B,N,D), N = dyn_side^2 or 196

        # =========================
        # Transformer encoder (MHSA + DWConv + MLP)
        # =========================
        Tenc, attn_list = self.encoder(Tavg, return_attn=return_xai)

        # Token global average pooling
        z_vit = Tenc.mean(dim=1)
        z_vit = F.relu(self.vit_proj(z_vit), inplace=True)

        # =========================
        # Fusion and classifier
        # =========================
        z = torch.cat([z_cnn, z_vit], dim=1)
        h = F.relu(self.fuse_fc(z), inplace=True)
        h = self.fuse_drop(h)
        logits = self.out(h)

        if return_xai:
            return logits, {
                "attn": attn_list,
                "mask": mask_img,
                "gste_side": dyn_side,
            }

        return logits, None

    # MC Dropout:
    # - Keeps model mostly in eval mode
    # - Temporarily turns Dropout layers to train mode to sample stochastic predictions
    # - Returns mean and variance of class probabilities across samples
    @torch.no_grad()
    def mc_dropout_predict(self, x, mc_samples=20):
        self.eval()
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()

        probs = []
        for _ in range(int(mc_samples)):
            logits, _ = self.forward(x, return_xai=False)
            probs.append(torch.softmax(logits, dim=1))

        probs = torch.stack(probs, dim=0)
        mu = probs.mean(dim=0)
        var = probs.var(dim=0, unbiased=False)
        return mu, var


# Convenience subclass that forces ablation behavior:
# - Always disables PFD and GSTE
# - Still keeps RViT rotation and average embedding logic intact
class HybridResNet50V2_RViT_Ablation(HybridResNet50V2_RViT):
    """
    Ablation: Hybrid WITHOUT Krsna extensions (PFD-GSTE).
    Still paper-faithful to Krishnan RViT (rotate IMAGE, patchify P=16, avg embeddings).
    """
    def __init__(self, *args, **kwargs):
        kwargs["use_pfd_gste"] = False
        super().__init__(*args, **kwargs)
