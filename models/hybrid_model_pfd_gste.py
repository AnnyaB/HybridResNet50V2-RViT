
# models/hybrid_model.py
#
# Paper-faithful hybrid (where it matters) + Krsna extensions:
# - CNN: pretrained ResNet50V2 (Sarada: start from pretrained, then modify/fine-tune)
# - PFD: 1x1 conv + sigmoid mask gating on CNN feature maps
# - RViT (Krishnan-style input path): rotate IMAGE, patchify with P=16
# - GSTE: use rotated PFD mask to weight IMAGE patch tokens
# - Rotation integration: average token embeddings across rotations before encoder
# - XAI hooks: retain CNN gated feature grads; collect transformer attn for rollout
#
# NOTE: embed_dim=142, heads=10 -> use flexible attention (inner_dim=140) then project back.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# CNN: pretrained ResNet50V2 (timm)
# -------------------------

class ResNet50V2TimmBackbone(nn.Module):
    """
    Uses timm pretrained ResNetV2 feature extractor.
    Returns last feature map (B, C, 7, 7) for 224x224 input.
    """
    def __init__(self, model_name="resnetv2_50x1_bitm", pretrained=True):
        super().__init__()
        try:
            import timm
        except Exception as e:
            raise ImportError(
                "timm is required for pretrained ResNet50V2. "
                "Install timm or use an environment (e.g., Kaggle) that includes it."
            ) from e

        self.model_name = model_name
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(4,),  # last stage
        )

    def forward(self, x):
        feats = self.backbone(x)  # list length 1 (out_indices=(4,))
        return feats[-1]


# -------------------------
# PFD: pathology mask gating
# -------------------------

class PFD(nn.Module):
    """
    PFD: M = sigmoid(conv1x1(F)), F_path = M * F
    """
    def __init__(self, in_ch):
        super().__init__()
        self.mask_conv = nn.Conv2d(in_ch, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, feat):
        mask = torch.sigmoid(self.mask_conv(feat))  # (B,1,h,w)
        gated = feat * mask
        return gated, mask


# -------------------------
# RViT: patchify IMAGE (Krishnan-style)
# -------------------------

class FeatureTokenEmbed(nn.Module):
    """
    Tokenise semantic feature maps (after PFD):
    (B, C, H, W) -> (B, N, D)
    """
    def __init__(self, in_ch, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, feat):
        x = self.proj(feat)                 # (B, D, H, W)
        B, D, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, N, D), N=H*W
        return tokens, H, W


class PositionalAndRotationEmbedding(nn.Module):
    """
    Learnable positional embeddings + rotation embeddings.
    """
    def __init__(self, base_h=14, base_w=14, embed_dim=142, n_rot=4):
        super().__init__()
        self.base_h = base_h
        self.base_w = base_w
        self.embed_dim = embed_dim

        self.pos = nn.Parameter(torch.zeros(1, base_h * base_w, embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

        self.rot = nn.Parameter(torch.zeros(n_rot, embed_dim))
        nn.init.trunc_normal_(self.rot, std=0.02)

    def forward(self, tokens, ht, wt, rot_idx):
        # fixed 14x14 grid expected; if not, interpolate pos
        if ht == self.base_h and wt == self.base_w:
            pos = self.pos
        else:
            pos = self.pos.reshape(1, self.base_h, self.base_w, self.embed_dim).permute(0, 3, 1, 2)
            pos = F.interpolate(pos, size=(ht, wt), mode="bilinear", align_corners=False)
            pos = pos.permute(0, 2, 3, 1).reshape(1, ht * wt, self.embed_dim)

        tokens = tokens + pos + self.rot[rot_idx].view(1, 1, -1)
        return tokens


# -------------------------
# Flexible MHSA (supports dim not divisible by heads)
# -------------------------

class FlexibleMHSA(nn.Module):
    """
    If dim not divisible by heads, use inner_dim = heads * floor(dim/heads),
    project qkv into inner_dim, then project output back to dim.
    """
    def __init__(self, dim, num_heads, attn_dropout=0.1, proj_dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        head_dim = dim // num_heads
        inner_dim = num_heads * head_dim
        if inner_dim <= 0:
            raise ValueError("embed_dim too small for given number of heads.")

        self.inner_dim = inner_dim
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, inner_dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(inner_dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x, return_attn=False):
        B, N, D = x.shape
        qkv = self.qkv(x)  # (B,N,3*inner_dim)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B,H,N,hd)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B,H,N,N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # (B,H,N,hd)
        out = out.transpose(1, 2).reshape(B, N, self.inner_dim)  # (B,N,inner_dim)
        out = self.proj(out)  # (B,N,dim)
        out = self.proj_drop(out)

        if return_attn:
            return out, attn
        return out, None


class RViTBlock(nn.Module):
    """
    MHSA -> DWConv -> MLP, each with pre-LN + residual.
    """
    def __init__(self, dim, heads, mlp_dim, attn_dropout=0.1, dropout=0.1, ht=14, wt=14):
        super().__init__()
        self.ht = ht
        self.wt = wt

        self.ln1 = nn.LayerNorm(dim)
        self.attn = FlexibleMHSA(dim, heads, attn_dropout=attn_dropout, proj_dropout=dropout)

        self.ln2 = nn.LayerNorm(dim)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True)
        self.dw_drop = nn.Dropout(dropout)

        self.ln3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, return_attn=False):
        attn_out, attn = self.attn(self.ln1(x), return_attn=return_attn)
        x = x + attn_out

        y = self.ln2(x)
        B, N, D = y.shape
        ht, wt = self.ht, self.wt
        if ht * wt != N:
            side = int(math.sqrt(N))
            ht = side
            wt = N // max(side, 1)

        y2 = y.transpose(1, 2).reshape(B, D, ht, wt)
        y2 = self.dwconv(y2)
        y2 = y2.reshape(B, D, ht * wt).transpose(1, 2)
        y2 = self.dw_drop(y2)
        x = x + y2

        x = x + self.mlp(self.ln3(x))
        return x, attn


class RViTEncoder(nn.Module):
    def __init__(self, dim=142, depth=10, heads=10, mlp_dim=480, attn_dropout=0.1, dropout=0.1, ht=14, wt=14):
        super().__init__()
        self.blocks = nn.ModuleList([
            RViTBlock(dim, heads, mlp_dim, attn_dropout=attn_dropout, dropout=dropout, ht=ht, wt=wt)
            for _ in range(depth)
        ])
        self.ln = nn.LayerNorm(dim)

    def forward(self, x, return_attn=False):
        attn_list = []
        for blk in self.blocks:
            x, attn = blk(x, return_attn=return_attn)
            if return_attn and attn is not None:
                attn_list.append(attn)
        x = self.ln(x)
        return x, attn_list


# -------------------------
# Hybrid model
# -------------------------

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
        super().__init__()
        self.num_classes = num_classes
        self.rotations = rotations
        self.patch_size = patch_size

        # CNN branch (Sarada starting point: pretrained)
        self.cnn = ResNet50V2TimmBackbone(model_name=cnn_name, pretrained=cnn_pretrained)

        # Infer CNN output channels with a tiny forward-free assumption (common: 2048)
        # Keep explicit for PFD/projection stability.
        self.cnn_out_ch = 2048

        self.pfd = PFD(in_ch=self.cnn_out_ch)
        self.cnn_proj = nn.Linear(self.cnn_out_ch, fusion_dim)
        self.cnn_pool = nn.AdaptiveAvgPool2d(1)

        # Transformer branch (Krishnan-style patchify image)
        self.patch = FeatureTokenEmbed(in_ch=self.cnn_out_ch, embed_dim=embed_dim)
        self.posrot = PositionalAndRotationEmbedding(base_h=7, base_w=7, embed_dim=embed_dim, n_rot=4)
        self.encoder = RViTEncoder(
            dim=embed_dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            attn_dropout=attn_dropout,
            dropout=vit_dropout,
            ht=7 // patch_size,
            wt=7 // patch_size
        )

        self.vit_proj = nn.Linear(embed_dim, fusion_dim)

        # Fusion head (keep your existing fusion concept)
        self.fuse_fc = nn.Linear(fusion_dim * 2, fusion_dim)
        self.fuse_drop = nn.Dropout(p=fusion_dropout)
        self.out = nn.Linear(fusion_dim, num_classes)

    def freeze_cnn_bn(self):
        # Optional: keep pretrained BN statistics stable
        for m in self.cnn.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _gste_alpha_from_mask(self, mask_img):
        """
        mask_img: (B,1,224,224) -> avg pool to patch grid -> (B,N,1) normalized
        """
        P = self.patch_size
        alpha = F.avg_pool2d(mask_img, kernel_size=P, stride=P)  # (B,1,14,14)
        alpha = alpha.flatten(2).transpose(1, 2)                 # (B,N,1)
        alpha = alpha / (alpha.mean(dim=1, keepdim=True) + 1e-6)
        return alpha

    def forward(self, x, return_xai=False):
        """
        Returns:
            logits OR (logits, xai_dict)
        """

        # =========================
        # CNN backbone
        # =========================
        feat = self.cnn(x)                       # (B, C, H, W)
        z_cnn = self.cnn_pool(feat).flatten(1)   # global CNN descriptor
        z_cnn = F.relu(self.cnn_proj(z_cnn), inplace=True)

        # =========================
        # Pathology-Focused Disentanglement (PFD)
        # =========================
        feat_path, mask_feat = self.pfd(feat)    # semantic features + pathology mask

        # mask resized ONLY for XAI visualization
        mask_img = F.interpolate(
            mask_feat,
            size=(x.shape[2], x.shape[3]),
            mode="bilinear",
            align_corners=False
        )

        # =========================
        # SARADA rotations + GSTE
        # =========================
        token_sets = []

        for rot_idx, k in enumerate(self.rotations):
            f_r = torch.rot90(feat_path, k=k, dims=(2, 3))
            m_r = torch.rot90(mask_feat, k=k, dims=(2, 3))

            tokens, ht, wt = self.patch(f_r)

            alpha = m_r.flatten(2).transpose(1, 2)   # (B, N, 1)
            alpha = alpha / (alpha.mean(dim=1, keepdim=True) + 1e-6)

            tokens = tokens * alpha                  # GSTE (soft guidance)
            tokens = self.posrot(tokens, ht, wt, rot_idx)

            token_sets.append(tokens)

        # Rotation-averaged tokens
        Tavg = torch.stack(token_sets, dim=0).mean(dim=0)

        # =========================
        # Transformer encoder
        # =========================
        Tenc, attn_list = self.encoder(Tavg, return_attn=return_xai)

        z_vit = Tenc.mean(dim=1)
        z_vit = F.relu(self.vit_proj(z_vit), inplace=True)

        # =========================
        # Fusion + classifier
        # =========================
        z = torch.cat([z_cnn, z_vit], dim=1)
        h = F.relu(self.fuse_fc(z), inplace=True)
        h = self.fuse_drop(h)
        logits = self.out(h)

        if return_xai:
            return logits, {
                "mask": mask_img,
                "attn": attn_list
            }

        return logits, None

    @torch.no_grad()
    def mc_dropout_predict(self, x, mc_samples=20):
        """
        MC Dropout: keep dropout active, keep BN in eval.
        """
        self.eval()
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()

        probs = []
        for _ in range(mc_samples):
            logits, _ = self.forward(x, return_xai=False)
            probs.append(torch.softmax(logits, dim=1))

        probs = torch.stack(probs, dim=0)
        mu = probs.mean(dim=0)
        var = probs.var(dim=0, unbiased=False)
        return mu, var
