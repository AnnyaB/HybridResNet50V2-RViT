#
# models/hybrid_model.py
# This file defines a hybrid CNN and Transformer model (PFDA-GSTEA).
# It uses a ResNet50V2 backbone to get a feature map, learns a pathology mask (PFD-A),
# tokenises the feature map (not raw image patches), applies static mask weighting (GSTE-A),
# runs a transformer encoder, then fuses CNN and ViT features for classification.

import math  # I used sqrt and other small math helpers
import torch  # core tensor library and autograd
import torch.nn as nn  # PyTorch neural network layers
import torch.nn.functional as F  # functional ops (relu, interpolate, etc.)

# -------------------------
# CNN backbone wrapper (timm ResNet50V2)
# -------------------------
class ResNet50V2TimmBackbone(nn.Module): #nn.Module is the base class for all neural network modules in PyTorch 
    # and inherits its properties and methods.

    """
    Wraps a timm ResNetV2 model configured to return feature maps (not logits).
    For 224x224 input, last stage feature map is typically (B, 2048, 7, 7), since
    it's the final deepest convolutional stage (stage 5 or conv5_x) of Resnet50v2 before global pooling.
    """
    def __init__(self, model_name="resnetv2_50x1_bitm", pretrained=True):
        super().__init__()  # init base nn.Module

        # I imported timm inside the constructor so the file can still be imported
        # even in environments where timm isn't installed (fails only when instantiating).
        try:
            import timm  # timm = model zoo and pretrained weights 
            #(The model zoo is the architecture code and pretrained weights is the knowledge of the saved math)
        except Exception as e:
            # Raised a clear error so later if someone uses this code, they know what to install.
            raise ImportError(
                "timm is required for pretrained ResNet50V2. "
                "Install timm or use an environment (for instance., Kaggle) that includes it."
            ) from e

        self.model_name = model_name  # stored for logging / reproducibility

        # Created the model in features_only mode so forward returns feature maps.
        # out_indices=(4,) means I only kept the last stage feature map.
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(4,),
        )

    def forward(self, x):
        # Forward pass through the backbone.
        # timm returns a list of feature maps, one per selected stage.
        feats = self.backbone(x)
        # I requested only the last stage, so took the last element.
        return feats[-1]

# -------------------------
# PFD-A module: learn a pathology mask and gate CNN features
# -------------------------
class PFD(nn.Module):
    
    """
    PFD: mask = sigmoid(conv1x1(feat)), gated_feat = feat * mask
    Mask is 1-channel and broadcasts across feature channels.
    """
    def __init__(self, in_ch):
        super().__init__()  # init base

        # 1x1 conv turns (B, in_ch, H, W) -> (B, 1, H, W)
        # This produces mask logits before sigmoid.
        self.mask_conv = nn.Conv2d(
            in_ch, 1, kernel_size=1, stride=1, padding=0, bias=True
        )

    def forward(self, feat):
        
        # Converted mask logits into [0,1] weights.
        mask = torch.sigmoid(self.mask_conv(feat))  # (B,1,H,W)

        # Gated the feature map: broadcast multiply mask across all channels.
        gated = feat * mask  # (B,C,H,W)

        # Returned both: gated features for transformer and mask for GSTE/XAI.
        return gated, mask

# -------------------------
# Tokeniser for FEATURE MAPS (not raw-image patchify)
# -------------------------
class FeatureTokenEmbed(nn.Module):
    
    """
    Turns a CNN feature map into transformer tokens:
      (B, C, H, W) -> 1x1 conv -> (B, D, H, W) -> flatten -> (B, N, D)
    where N = H*W (e.g., 7*7 = 49 tokens).
    """
    def __init__(self, in_ch, embed_dim):
        super().__init__()  # init base

        # 1x1 conv to map CNN channels into transformer embedding dimension.
        self.proj = nn.Conv2d(
            in_ch, embed_dim, kernel_size=1, stride=1, padding=0, bias=True
        )

    def forward(self, feat):
        # Project channels: (B,C,H,W) -> (B,D,H,W)
        x = self.proj(feat)

        # Grab dimensions so we can reshape cleanly.
        B, D, H, W = x.shape

        # Flatten spatial grid H*W into token sequence N.
        # flatten(2) gives (B,D,N), then transpose -> (B,N,D).
        tokens = x.flatten(2).transpose(1, 2)

        # Return tokens plus grid size so positional embeddings know H/W.
        return tokens, H, W

class PositionalAndRotationEmbedding(nn.Module):
    
    """
    Adds learnable positional embeddings and rotation embeddings.
    Positional embeddings are stored for a base grid (default 14x14, but PFDA uses 7x7).
    If grid changes, position table is interpolated to match new H/W.
    """
    def __init__(self, base_h=14, base_w=14, embed_dim=142, n_rot=4):
        
        super().__init__()  # init base

        self.base_h = base_h  # base positional grid height
        self.base_w = base_w  # base positional grid width
        self.embed_dim = embed_dim  # token embedding dim

        # Learnable positional table: (1, base_h*base_w, D)
        self.pos = nn.Parameter(torch.zeros(1, base_h * base_w, embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)  # ViT-style init

        # Learnable rotation embedding per rotation index: (n_rot, D)
        self.rot = nn.Parameter(torch.zeros(n_rot, embed_dim))
        nn.init.trunc_normal_(self.rot, std=0.02)

    def forward(self, tokens, ht, wt, rot_idx):
        # If token grid matches base table, just use it.
        if ht == self.base_h and wt == self.base_w:
            pos = self.pos
        else:
            # Otherwise reshape pos into (1,D,base_h,base_w) so we can interpolate.
            pos = self.pos.reshape(1, self.base_h, self.base_w, self.embed_dim).permute(0, 3, 1, 2)

            # Resized to (ht, wt) using bilinear interpolation.
            pos = F.interpolate(pos, size=(ht, wt), mode="bilinear", align_corners=False)

            # Converted back to (1, N, D).
            pos = pos.permute(0, 2, 3, 1).reshape(1, ht * wt, self.embed_dim)

        # Add position + rotation embedding (rotation added to every token).
        tokens = tokens + pos + self.rot[rot_idx].view(1, 1, -1)

        return tokens  # enriched tokens

# -------------------------
# MHSA (Multi-Head Self Attention) that survives dim not divisible by heads since it drops remainder dims and Krishnan et al used 142 dim with 10 heads
# -------------------------
class FlexibleMHSA(nn.Module):
    
    """
    Standard MHSA needs dim divisible by num_heads.
    This version uses inner_dim = num_heads * floor(dim/num_heads),
    runs attention in inner_dim, then projects back to dim.
    """
    def __init__(self, dim, num_heads, attn_dropout=0.1, proj_dropout=0.1):
        super().__init__()  # init base

        self.dim = dim  # model dim
        self.num_heads = num_heads  # number of heads

        # Computed per-head dim using floor division.
        head_dim = dim // num_heads

        # Attention actually runs in inner_dim (might drop remainder dims).
        inner_dim = num_heads * head_dim

        if inner_dim <= 0:
            raise ValueError("embed_dim too small for given number of heads.")

        self.inner_dim = inner_dim
        self.head_dim = head_dim

        # Scale factor for dot-product attention.
        self.scale = head_dim ** -0.5

        # Single linear to get Q,K,V packed together: (B,N,dim) -> (B,N,3*inner_dim)
        self.qkv = nn.Linear(dim, inner_dim * 3, bias=True)

        self.attn_drop = nn.Dropout(attn_dropout)  # dropout on attention weights

        # Project from inner_dim back to dim.
        self.proj = nn.Linear(inner_dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_dropout)  # dropout after projection

    def forward(self, x, return_attn=False):
        # x is tokens: (B,N,D)
        B, N, D = x.shape

        # Compute qkv.
        qkv = self.qkv(x)  # (B,N,3*inner_dim)

        # Reshaped into (3, B, heads, N, head_dim)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        # Splitted out query, key, value.
        q, k, v = qkv[0], qkv[1], qkv[2]  # each is (B,heads,N,head_dim)

        # Attention logits: (B,heads,N,N)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Softmax over keys so each query attends to all tokens.
        attn = attn.softmax(dim=-1)

        # Dropout on attention weights.
        attn = self.attn_drop(attn)

        # Weighted sum of values: (B,heads,N,head_dim)
        out = attn @ v

        # Merge heads back: (B,N,inner_dim)
        out = out.transpose(1, 2).reshape(B, N, self.inner_dim)

        # Project back to dim: (B,N,dim)
        out = self.proj(out)
        out = self.proj_drop(out)

        # Optionally return attention maps for XAI/rollout.
        if return_attn:
            return out, attn

        return out, None

# -------------------------
# RViT-style block: MHSA -> DWConv -> MLP (each with pre-LN + residual)
# -------------------------
class RViTBlock(nn.Module):
    """
    Token block:
      x = x + MHSA(LN(x))
      x = x + DWConv(LN(x))  [DWConv is applied after reshaping tokens into a 2D grid]
      x = x + MLP(LN(x))
    """
    def __init__(self, dim, heads, mlp_dim, attn_dropout=0.1, dropout=0.1, ht=14, wt=14):
        
        super().__init__()  # init base

        self.ht = ht  # expected grid height (might be corrected at runtime)
        self.wt = wt  # expected grid width

        self.ln1 = nn.LayerNorm(dim)  # pre-LN for attention
        self.attn = FlexibleMHSA(dim, heads, attn_dropout=attn_dropout, proj_dropout=dropout)

        self.ln2 = nn.LayerNorm(dim)  # pre-LN for DWConv branch

        # Depthwise conv = per-channel conv over spatial grid; helps local mixing.
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True)
        self.dw_drop = nn.Dropout(dropout)

        self.ln3 = nn.LayerNorm(dim)  # pre-LN for MLP

        # Standard transformer MLP (feed-forward network).
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, return_attn=False):
        # Attention branch (pre-LN and residual).
        attn_out, attn = self.attn(self.ln1(x), return_attn=return_attn)
        x = x + attn_out

        # DWConv branch needs a 2D grid, so we reshape tokens into (B,D,ht,wt).
        y = self.ln2(x)
        B, N, D = y.shape

        ht, wt = self.ht, self.wt

        # If metadata ht/wt doesn't match token count N, infer something reasonable.
        if ht * wt != N:
            side = int(math.sqrt(N))  # try square-ish
            ht = side
            wt = N // max(side, 1)

        # Tokens -> 2D feature map for depthwise conv.
        y2 = y.transpose(1, 2).reshape(B, D, ht, wt)

        # Local mixing via depthwise conv.
        y2 = self.dwconv(y2)

        # Back to tokens.
        y2 = y2.reshape(B, D, ht * wt).transpose(1, 2)
        y2 = self.dw_drop(y2)

        # Residual add for DWConv branch.
        x = x + y2

        # MLP branch (pre-LN + residual).
        x = x + self.mlp(self.ln3(x))

        return x, attn

class RViTEncoder(nn.Module):
    # Stack multiple RViTBlocks and end with a final LayerNorm.
    def __init__(self, dim=142, depth=10, heads=10, mlp_dim=480, attn_dropout=0.1, dropout=0.1, ht=14, wt=14):
        super().__init__()  # init base

        # Create depth identical blocks.
        self.blocks = nn.ModuleList([
            RViTBlock(dim, heads, mlp_dim, attn_dropout=attn_dropout, dropout=dropout, ht=ht, wt=wt)
            for _ in range(depth)
        ])

        self.ln = nn.LayerNorm(dim)  # final normalization

    def forward(self, x, return_attn=False):
        attn_list = []  # stored attention maps for all layers if requested

        for blk in self.blocks:
            x, attn = blk(x, return_attn=return_attn)
            if return_attn and attn is not None:
                attn_list.append(attn)

        x = self.ln(x)  # final LN

        return x, attn_list

# -------------------------
# Main hybrid PFDA-GSTEA model
# -------------------------
class HybridResNet50V2_RViT(nn.Module):
    def __init__(
        self,
        num_classes=4,
        patch_size=16,  # stored knob; PFDA forward doesn't patchify raw images
        embed_dim=142,
        depth=10,
        heads=10,
        mlp_dim=480,
        attn_dropout=0.1,
        vit_dropout=0.1,
        fusion_dim=256,
        fusion_dropout=0.5,
        rotations=(0, 1, 2, 3),  # torch.rot90 k values
        cnn_name="resnetv2_50x1_bitm",
        cnn_pretrained=True,
    ):
        super().__init__()  # init base

        self.num_classes = num_classes  # number of class logits
        self.rotations = rotations  # which rotations we run over
        self.patch_size = patch_size  # stored (used only in helper, not main PFDA tokenisation)

        # Build the CNN feature extractor.
        self.cnn = ResNet50V2TimmBackbone(model_name=cnn_name, pretrained=cnn_pretrained)

        # ResNetV2 last stage commonly outputs 2048 channels.
        self.cnn_out_ch = 2048

        # PFD-A mask gating module on the CNN feature map.
        self.pfd = PFD(in_ch=self.cnn_out_ch)

        # CNN pooled descriptor path: pool -> linear -> fusion_dim.
        self.cnn_proj = nn.Linear(self.cnn_out_ch, fusion_dim)
        self.cnn_pool = nn.AdaptiveAvgPool2d(1)  # global average pooling over H,W

        # Transformer path: tokenise *feature map* (7x7 -> 49 tokens).
        self.patch = FeatureTokenEmbed(in_ch=self.cnn_out_ch, embed_dim=embed_dim)

        # Positional+rotation embeddings sized for a 7x7 token grid.
        self.posrot = PositionalAndRotationEmbedding(base_h=7, base_w=7, embed_dim=embed_dim, n_rot=4)

        # Encoder stack. NOTE: ht/wt here are computed using 7//patch_size,
        # which may become 0; the block has runtime fallback to infer ht/wt from N.
        self.encoder = RViTEncoder(
            dim=embed_dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            attn_dropout=attn_dropout,
            dropout=vit_dropout,
            ht=7 // patch_size,
            wt=7 // patch_size,
        )

        # Project transformer pooled vector into fusion_dim.
        self.vit_proj = nn.Linear(embed_dim, fusion_dim)

        # Fusion head: concatenate CNN+ViT -> compress -> dropout -> logits.
        self.fuse_fc = nn.Linear(fusion_dim * 2, fusion_dim)
        self.fuse_drop = nn.Dropout(p=fusion_dropout)
        self.out = nn.Linear(fusion_dim, num_classes)

    def freeze_cnn_bn(self):
        # Handy finetuning trick: keep BN layers in eval mode so their running stats don't drift.
        for m in self.cnn.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _gste_alpha_from_mask(self, mask_img):
        # Helper (currently unused in forward): downsample an image-scale mask
        # into patch-scale weights and normalize them.
        P = self.patch_size
        alpha = F.avg_pool2d(mask_img, kernel_size=P, stride=P)
        alpha = alpha.flatten(2).transpose(1, 2)
        alpha = alpha / (alpha.mean(dim=1, keepdim=True) + 1e-6)
        return alpha

    def forward(self, x, return_xai=False):
        # Forward returns logits by default.
        # If return_xai=True, also returns a dict with mask + attention maps.

        # -------------------------
        # CNN feature extraction
        # -------------------------
        feat = self.cnn(x)  # (B,2048,7,7)

        # CNN pooled vector: global average pool -> flatten -> linear -> ReLU.
        z_cnn = self.cnn_pool(feat).flatten(1)            # (B,2048)
        z_cnn = F.relu(self.cnn_proj(z_cnn), inplace=True) # (B,fusion_dim)

        # -------------------------
        # PFD-A: mask + gated feature map
        # -------------------------
        feat_path, mask_feat = self.pfd(feat)  # feat_path (B,2048,7,7), mask_feat (B,1,7,7)

        # Upsample mask only for visualization / overlay.
        mask_img = F.interpolate(
            mask_feat,
            size=(x.shape[2], x.shape[3]),  # match input H,W
            mode="bilinear",
            align_corners=False,
        )

        # -------------------------
        # Rotate FEATURE MAPS and GSTE-A weighting per rotation
        # -------------------------
        token_sets = []  # we store tokens for each rotation then average them

        for rot_idx, k in enumerate(self.rotations):
            # Rotate gated features and the mask the same way.
            f_r = torch.rot90(feat_path, k=k, dims=(2, 3))
            m_r = torch.rot90(mask_feat, k=k, dims=(2, 3))

            # Tokenise rotated feature map: (B,2048,7,7) -> (B,49,D)
            tokens, ht, wt = self.patch(f_r)

            # Turn rotated mask into per-token weights: (B,1,7,7) -> (B,49,1)
            alpha = m_r.flatten(2).transpose(1, 2)

            # Normalize so average weight is ~1 (keeps scale stable).
            alpha = alpha / (alpha.mean(dim=1, keepdim=True) + 1e-6)

            # GSTE-A: static guidance (multiply tokens by weights).
            tokens = tokens * alpha

            # Add positional embedding (depends on ht/wt) + rotation embedding (depends on rot_idx).
            tokens = self.posrot(tokens, ht, wt, rot_idx)

            # Save this rotation's token set.
            token_sets.append(tokens)

        # Average tokens over rotations (rotation-invariant-ish representation).
        Tavg = torch.stack(token_sets, dim=0).mean(dim=0)  # (B,49,D)

        # -------------------------
        # Transformer encoder
        # -------------------------
        Tenc, attn_list = self.encoder(Tavg, return_attn=return_xai)

        # Pool tokens by mean (no CLS token).
        z_vit = Tenc.mean(dim=1)  # (B,D)

        # Project to fusion_dim.
        z_vit = F.relu(self.vit_proj(z_vit), inplace=True)  # (B,fusion_dim)

        # -------------------------
        # Fusion + classifier
        # -------------------------
        z = torch.cat([z_cnn, z_vit], dim=1)       # (B,2*fusion_dim)
        h = F.relu(self.fuse_fc(z), inplace=True)  # (B,fusion_dim)
        h = self.fuse_drop(h)
        logits = self.out(h)                      # (B,num_classes)

        if return_xai:
            # Return explanation payload: upsampled mask + attention maps.
            return logits, {"mask": mask_img, "attn": attn_list}

        return logits, None

    @torch.no_grad()
    def mc_dropout_predict(self, x, mc_samples=20):
        # Monte Carlo Dropout:
        # - Put model in eval mode (so BN is stable),
        # - Turn dropout back ON,
        # - Run multiple forward passes to sample predictive distribution.

        self.eval()  # BN uses running stats; dropout would normally be off

        # Re-enable dropout layers only.
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()

        probs = []  # store probabilities from each stochastic forward pass

        for _ in range(mc_samples):
            logits, _ = self.forward(x, return_xai=False)
            probs.append(torch.softmax(logits, dim=1))

        probs = torch.stack(probs, dim=0)  # (S,B,C)

        # Predictive mean probability.
        mu = probs.mean(dim=0)  # (B,C)

        # Predictive variance (uncertainty proxy).
        var = probs.var(dim=0, unbiased=False)  # (B,C)

        return mu, var