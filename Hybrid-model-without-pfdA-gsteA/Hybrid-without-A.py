# THIS IS THE FILE FOR THE HYBRID MODEL WITHOUT PFD-GSTE VARIANT A (ABLATION).

# Libraries I needed

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# CNN: pretrained ResNet50V2 (timm)

class ResNet50V2TimmBackbone(nn.Module):
    
    """
    Uses timm pretrained ResNetV2 feature extractor.
    Returns last feature map (B, C, 7, 7) for 224x224 input.
    """
    def __init__(self, model_name="resnetv2_50x1_bitm", pretrained=True):
        
        # Initialize nn.Module base class so parameters/submodules register correctly
        super().__init__()
        # Import timm only when this backbone is constructed (keeps dependency localized)
        try:
            import timm
        except Exception as e:
            # If timm isn't installed, raise a clear message telling how to fix it
            raise ImportError(
                "timm is required for pretrained ResNet50V2. "
                "Install timm or use an environment (e.g., Kaggle) that includes it."
            ) from e

        # Storing model name for reference/debugging
        self.model_name = model_name
        # Creating timm model in "features_only" mode so it outputs intermediate feature maps
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(4,),  # last stage
        )

    def forward(self, x):
        
        # timm features_only returns a list of feature maps (one per out_indices entry)
        feats = self.backbone(x)  # list length 1 (out_indices=(4,))
        # Returning the last (and only) feature map: (B,C,7,7) for 224x224 inputs
        return feats[-1]


# Tokenisation from FEATURE MAPS

class FeatureTokenEmbed(nn.Module):
    
    """
    Tokenise CNN feature maps:
    (B, C, H, W) -> (B, N, D)
    """
    def __init__(self, in_ch, embed_dim):
        # Initializing base module state
        super().__init__()
        # 1x1 conv projects feature channels C -> embedding dimension D at each spatial location
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, feat):
        # Project feature map channels: (B,C,H,W) -> (B,D,H,W)
        x = self.proj(feat)                    # (B, D, H, W)
        # Extracting shapes for tokenization
        B, D, H, W = x.shape
        # Flattening spatial grid (H*W) into token sequence length N, and transpose to (B,N,D)
        tokens = x.flatten(2).transpose(1, 2)  # (B, N, D), N=H*W
        # Returning tokens plus token grid height/width (for positional embedding interpolation)
        return tokens, H, W


class PositionalAndRotationEmbedding(nn.Module):
    
    """
    Learnable positional embeddings and rotation embeddings.
    """
    def __init__(self, base_h=7, base_w=7, embed_dim=142, n_rot=4):
        
        # Register parameters and submodules
        super().__init__()
        # Base grid shape assumed by the learned positional embedding table
        self.base_h = base_h
        self.base_w = base_w
        # Embedding dimension for each token
        self.embed_dim = embed_dim

        # Learnable positional embeddings for base_h*base_w tokens
        self.pos = nn.Parameter(torch.zeros(1, base_h * base_w, embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

        # Learnable rotation embedding for each rotation index (n_rot = 4 for 0/90/180/270)
        self.rot = nn.Parameter(torch.zeros(n_rot, embed_dim))
        nn.init.trunc_normal_(self.rot, std=0.02)

    def forward(self, tokens, ht, wt, rot_idx):
        
        # If not 7x7, interpolate positional embedding
        if ht == self.base_h and wt == self.base_w:
            # Use stored positional embedding directly when token grid matches base grid
            pos = self.pos
        else:
            # Reshape (1,base_h*base_w,D) -> (1,base_h,base_w,D) -> (1,D,base_h,base_w)
            pos = self.pos.reshape(1, self.base_h, self.base_w, self.embed_dim).permute(0, 3, 1, 2)
            # Interpolate to (ht,wt) so positional encoding matches current token grid
            pos = F.interpolate(pos, size=(ht, wt), mode="bilinear", align_corners=False)
            # Convert back to token table shape: (1,D,ht,wt) -> (1,ht,wt,D) -> (1,ht*wt,D)
            pos = pos.permute(0, 2, 3, 1).reshape(1, ht * wt, self.embed_dim)

        # Add position and rotation embedding to tokens (broadcast over batch)
        tokens = tokens + pos + self.rot[rot_idx].view(1, 1, -1)
        return tokens


# Flexible MHSA (supports dim not divisible by heads)

class FlexibleMHSA(nn.Module):
    
    """
    If dim not divisible by heads, use inner_dim = heads * floor(dim/heads),
    project qkv into inner_dim, then project output back to dim.
    """
    def __init__(self, dim, num_heads, attn_dropout=0.1, proj_dropout=0.1):
        # Initialize module
        super().__init__()
        # Store original embedding dimension and number of attention heads
        self.dim = dim
        self.num_heads = num_heads

        # Integer head dimension using floor division
        head_dim = dim // num_heads
        # Inner dimension used inside attention = heads * head_dim
        inner_dim = num_heads * head_dim
        # Safety check: if embed_dim < num_heads, head_dim becomes 0 and attention is invalid
        if inner_dim <= 0:
            raise ValueError("embed_dim too small for given number of heads.")

        # Saving inner attention dimension and per-head dimension
        self.inner_dim = inner_dim
        self.head_dim = head_dim
        # Scaling factor used in dot-product attention
        self.scale = head_dim ** -0.5

        # Linear projection produces Q, K, V concatenated: dim -> 3*inner_dim
        self.qkv = nn.Linear(dim, inner_dim * 3, bias=True)
        # Dropout on attention probabilities
        self.attn_drop = nn.Dropout(attn_dropout)
        # Project attention output back to original dim
        self.proj = nn.Linear(inner_dim, dim, bias=True)
        # Dropout after output projection
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x, return_attn=False):
        
        # x is token sequence: (B,N,D)
        B, N, D = x.shape
        # Compute QKV in one matmul: (B,N,3*inner_dim)
        qkv = self.qkv(x)  # (B,N,3*inner_dim)
        # Reshape into (3,B,H,N,head_dim) so we can split Q, K, V cleanly
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # Split Q, K, V: each is (B,H,N,head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B,H,N,hd)

        # Scaled dot-product attention: (B,H,N,N)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B,H,N,N)
        # Softmax over keys dimension
        attn = attn.softmax(dim=-1)
        # Apply dropout to attention weights
        attn = self.attn_drop(attn)

        # Weighted sum over values: (B,H,N,hd)
        out = attn @ v  # (B,H,N,hd)
        # Merge heads back: (B,N,inner_dim)
        out = out.transpose(1, 2).reshape(B, N, self.inner_dim)  # (B,N,inner_dim)
        # Project back to original token dimension D
        out = self.proj(out)  # (B,N,dim)
        # Dropout after projection
        out = self.proj_drop(out)

        # Optionally return attention for XAI/rollout
        if return_attn:
            return out, attn
        return out, None


class RViTBlock(nn.Module):
    
    """
    MHSA -> DWConv -> MLP, each with pre-LN and residual.
    """
    def __init__(self, dim, heads, mlp_dim, attn_dropout=0.1, dropout=0.1, ht=7, wt=7):
        
        # Register module components
        super().__init__()
        # Default token grid dimensions used for reshaping into (ht,wt) during DWConv
        self.ht = ht
        self.wt = wt

        # Pre-LN then attention
        self.ln1 = nn.LayerNorm(dim)
        self.attn = FlexibleMHSA(dim, heads, attn_dropout=attn_dropout, proj_dropout=dropout)

        # Pre-LN then depthwise conv branch (token -> image-like grid -> conv -> tokens)
        self.ln2 = nn.LayerNorm(dim)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True)
        self.dw_drop = nn.Dropout(dropout)

        # Pre-LN then MLP feed-forward network
        self.ln3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, return_attn=False):
        
        # Attention sub-layer with residual
        attn_out, attn = self.attn(self.ln1(x), return_attn=return_attn)
        x = x + attn_out

        # Prepare tokens for depthwise convolution:
        # interpret the token sequence as a 2D grid, then apply DWConv spatial mixing
        y = self.ln2(x)
        B, N, D = y.shape
        ht, wt = self.ht, self.wt
        # If stored ht/wt doesn't match token count, infer a grid approximation from sqrt(N)
        if ht * wt != N:
            side = int(math.sqrt(N))
            ht = max(side, 1)
            wt = max(N // ht, 1)

        # Token -> grid: (B,N,D) -> (B,D,ht,wt)
        y2 = y.transpose(1, 2).reshape(B, D, ht, wt)
        # Depthwise convolution for local spatial context
        y2 = self.dwconv(y2)
        # Grid -> tokens: (B,D,ht,wt) -> (B,N,D)
        y2 = y2.reshape(B, D, ht * wt).transpose(1, 2)
        # Dropout on conv output
        y2 = self.dw_drop(y2)
        # Residual connection
        x = x + y2

        # MLP sub-layer with residual
        x = x + self.mlp(self.ln3(x))
        return x, attn


class RViTEncoder(nn.Module):
    
    def __init__(self, dim=142, depth=10, heads=10, mlp_dim=480, attn_dropout=0.1, dropout=0.1, ht=7, wt=7):
        # Register encoder blocks and final layer norm
        super().__init__()
        # Stack depth identical RViTBlocks
        self.blocks = nn.ModuleList([
            RViTBlock(dim, heads, mlp_dim, attn_dropout=attn_dropout, dropout=dropout, ht=ht, wt=wt)
            for _ in range(depth)
        ])
        # Final normalization after the transformer stack
        self.ln = nn.LayerNorm(dim)

    def forward(self, x, return_attn=False):
        # Collect attention matrices if XAI requested
        attn_list = []
        for blk in self.blocks:
            x, attn = blk(x, return_attn=return_attn)
            if return_attn and attn is not None:
                attn_list.append(attn)
        # Final LN then return tokens and optional attention list
        x = self.ln(x)
        return x, attn_list


# Hybrid model (No PFD / No GSTE)

class HybridResNet50V2_RViT(nn.Module):
    def __init__(
        self,
        num_classes=4,
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
    ):
        # Initializing module
        super().__init__()
        # Storing basic model settings
        self.num_classes = num_classes
        self.rotations = rotations
        self.patch_size = patch_size

        # CNN branch
        # Build timm ResNet50V2 backbone that outputs final feature map
        self.cnn = ResNet50V2TimmBackbone(model_name=cnn_name, pretrained=cnn_pretrained)

        # Keeping explicit output channels
        self.cnn_out_ch = 2048

        # CNN pooled vector projection to fusion_dim
        self.cnn_proj = nn.Linear(self.cnn_out_ch, fusion_dim)
        # Global average pooling over (7,7) -> (1,1)
        self.cnn_pool = nn.AdaptiveAvgPool2d(1)

        # Transformer branch
        # Converting CNN feature map into tokens using 1x1 conv projection
        self.patch = FeatureTokenEmbed(in_ch=self.cnn_out_ch, embed_dim=embed_dim)
        # Adding learnable position and rotation embeddings on token grid (base 7x7)
        self.posrot = PositionalAndRotationEmbedding(base_h=7, base_w=7, embed_dim=embed_dim, n_rot=4)
        # RViT encoder stack (attention and depthwise conv and MLP)
        self.encoder = RViTEncoder(
            dim=embed_dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            attn_dropout=attn_dropout,
            dropout=vit_dropout,
            ht=7 // patch_size,  # kept same style as my original (block fixes ht/wt if needed)
            wt=7 // patch_size
        )

        # Projection from transformer embedding -> fusion_dim
        self.vit_proj = nn.Linear(embed_dim, fusion_dim)

        # Fusion head
        # Concatenate [cnn_feat, vit_feat] then reduce -> fusion_dim -> output logits
        self.fuse_fc = nn.Linear(fusion_dim * 2, fusion_dim)
        self.fuse_drop = nn.Dropout(p=fusion_dropout)
        self.out = nn.Linear(fusion_dim, num_classes)

    def freeze_cnn_bn(self):
        
        # Set BatchNorm2d modules inside CNN backbone to eval mode (freeze running stats updates)
        for m in self.cnn.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x, return_xai=False):
        
        # ---- CNN backbone ----
        # Extract final feature map from ResNet50V2: (B,C,7,7)
        feat = self.cnn(x)                       # (B,C,7,7)
        # Pool spatially to a vector: (B,C,1,1) -> (B,C)
        z_cnn = self.cnn_pool(feat).flatten(1)   # (B,C)
        # Project to fusion_dim and apply ReLU
        z_cnn = F.relu(self.cnn_proj(z_cnn), inplace=True)

        # ---- Rotations (NO PFD, NO GSTE) ----
        # Build token sequences for each rotation of the FEATURE MAP (not the image)
        token_sets = []
        for rot_idx, k in enumerate(self.rotations):
            # Rotate feature map by k*90 degrees in spatial dims (H,W) = (2,3)
            f_r = torch.rot90(feat, k=k, dims=(2, 3))
            # Tokenize rotated feature map into (B,N,D)
            tokens, ht, wt = self.patch(f_r)            # (B,N,D)
            # Add position + rotation embedding using rot_idx (0..3)
            tokens = self.posrot(tokens, ht, wt, rot_idx)
            # Collect tokens for averaging across rotations
            token_sets.append(tokens)

        # Rotation-averaged tokens
        # Stack across rotations then mean: produces one token sequence per sample
        Tavg = torch.stack(token_sets, dim=0).mean(dim=0)

        # ---- Transformer ----
        # Run through encoder; optionally collect attention matrices
        Tenc, attn_list = self.encoder(Tavg, return_attn=return_xai)
        # Token global average pooling -> (B,D)
        z_vit = Tenc.mean(dim=1)
        # Project transformer vector to fusion_dim and apply ReLU
        z_vit = F.relu(self.vit_proj(z_vit), inplace=True)

        # ---- Fusion and classifier ----
        # Concatenate CNN and transformer vectors
        z = torch.cat([z_cnn, z_vit], dim=1)
        # Fusion MLP layer + activation
        h = F.relu(self.fuse_fc(z), inplace=True)
        # Dropout for regularization
        h = self.fuse_drop(h)
        # Final classifier logits
        logits = self.out(h)

        # If XAI requested, return logits and attention list (mask is None in this ablation)
        if return_xai:
            return logits, {
                "attn": attn_list,  # kept same key style my predict_xai expects
                "mask": None,       # no PFD mask in ablation
            }

        # Default: no XAI payload
        return logits, None

    @torch.no_grad()
    def mc_dropout_predict(self, x, mc_samples=20):
        
        # Starting in eval mode, then selectively enable dropout layers for MC sampling
        self.eval()
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()

        # Collecting probability predictions across multiple stochastic passes
        probs = []
        for _ in range(mc_samples):
            logits, _ = self.forward(x, return_xai=False)
            probs.append(torch.softmax(logits, dim=1))

        # Stacking samples: (S,B,C) then compute mean and variance over S
        probs = torch.stack(probs, dim=0)
        mu = probs.mean(dim=0)
        var = probs.var(dim=0, unbiased=False)
        return mu, var
