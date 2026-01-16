# models/hybrid.py
# Implements Algorithm 1: Hybrid ResNet50V2–RViT with PFD + GSTE and Fusion Head.

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnetv2 import ResNetV2
from models.rvit import Tokenizer, PosRotEmbedding, RViTEncoder


class PFD(nn.Module):
    # (10)-(12): M = sigmoid(conv1x1(F)); F_path = M ⊙ F
    def __init__(self, in_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)

    def forward(self, Fmap):
        M = torch.sigmoid(self.conv1x1(Fmap))  # (B,1,H,W) in [0,1]
        F_path = Fmap * M
        return F_path, M


def patch_avg_mask(mask, patch_size_feat):
    # mask: (B,1,H,W) -> alpha: (B,Ntok)
    B, _, H, W = mask.shape
    P = patch_size_feat
    if P == 1:
        return mask.flatten(2).squeeze(1)  # (B, H*W)

    # Average pool over each patch region with stride=P
    pooled = F.avg_pool2d(mask, kernel_size=P, stride=P)  # (B,1,Htok,Wtok)
    return pooled.flatten(2).squeeze(1)


def normalize_01(x, eps=1e-8):
    # Per-sample min-max normalize into [0,1] for stable GSTE weighting
    x_min = x.min(dim=1, keepdim=True).values
    x = x - x_min
    x_max = x.max(dim=1, keepdim=True).values
    x = x / (x_max + eps)
    return x


class FusionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, dropout_p=0.5, num_classes=4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.drop = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, z):
        h = F.relu(self.fc1(z), inplace=True)
        h = self.drop(h)
        logits = self.fc2(h)
        return logits


class HybridResNet50V2_RViT_PFD_GSTE(nn.Module):
    def __init__(
        self,
        num_classes=4,
        rotations=(0, 90, 180, 270),
        # Krishnan hyperparams (paper):
        embed_dim=142,
        depth=10,
        heads=10,
        mlp_dim=480,
        attn_dropout=0.1,
        # Tokenization:
        patch_size_px=16,          # paper patch size (image space)
        feat_stride=16,            # our CNN feature map stride
        # Fusion:
        fusion_hidden=512,
        dropout_p=0.5
    ):
        super().__init__()
        self.rotations = rotations

        # CNN backbone (Sarada-style ResNet50V2 pre-activation)
        self.cnn = ResNetV2.resnet50v2(in_channels=3)
        self.cnn_out_ch = 1024  # layer3 output channels

        # PFD
        self.pfd = PFD(self.cnn_out_ch)

        # Tokenizer: patch size on feature map
        patch_size_feat = max(1, patch_size_px // feat_stride)
        self.patch_size_feat = patch_size_feat
        self.tokenizer = Tokenizer(in_channels=self.cnn_out_ch, embed_dim=embed_dim, patch_size_feat=patch_size_feat)

        # Pos + rot embedding
        self.emb = PosRotEmbedding(dim=embed_dim, base_hw=(14, 14), num_rotations=len(rotations))

        # RViT encoder
        self.rvit = RViTEncoder(dim=embed_dim, depth=depth, heads=heads, mlp_dim=mlp_dim,
                                attn_dropout=attn_dropout, dropout=attn_dropout)

        # Pooling dims
        self.z_vit_dim = embed_dim
        self.z_cnn_dim = self.cnn_out_ch

        # Fusion head
        self.head = FusionHead(in_dim=self.z_cnn_dim + self.z_vit_dim, hidden_dim=fusion_hidden,
                               dropout_p=dropout_p, num_classes=num_classes)

        # Cache for XAI
        self._cache = {}

    def rotate_feat(self, feat, rot_deg):
        # feat: (B,C,H,W)
        # rot_deg in {0,90,180,270}
        k = (rot_deg // 90) % 4
        return torch.rot90(feat, k=k, dims=(2, 3))

    def forward(self, x, collect_attn=False):
        # (46) CNN_BACKBONE
        Fmap = self.cnn.forward_features_stride16(x)  # (B,1024,14,14)

        # (47) PFD
        F_path, M = self.pfd(Fmap)

        # Cache for Grad-CAM++ target (cleaned semantic feature maps)
        self._cache["F_path"] = F_path
        self._cache["M"] = M

        # (48)-(52) RViT rotation set -> tokenize -> GSTE -> add pos+rot -> avg integrate
        tokens_sum = None
        Htok = Wtok = None
        attn_all = None

        alpha = normalize_01(patch_avg_mask(M, self.patch_size_feat))  # (B,Ntok) in [0,1]
        self._cache["alpha"] = alpha

        for rot_id, rot_deg in enumerate(self.rotations):
            F_rot = self.rotate_feat(F_path, rot_deg)

            # TOKENIZE
            T, ht, wt = self.tokenizer(F_rot)  # (B,N,D)
            Htok, Wtok = ht, wt

            # GSTE: T' = alpha ⊙ T (no gradients needed, but this is safe as-is)
            T = T * alpha.unsqueeze(-1)

            # ADD_POS_EMB: T'' = T + Epos + Erot(rot)
            T = self.emb(T, Htok, Wtok, rot_id)

            tokens_sum = T if tokens_sum is None else (tokens_sum + T)

        T_avg = tokens_sum / float(len(self.rotations))  # (24)-(25)
        self._cache["T_avg"] = T_avg

        # (53) RViT encoder
        T_enc, attn_stack = self.rvit(T_avg, Htok, Wtok, collect_attn=collect_attn)
        if collect_attn:
            self._cache["attn_stack"] = attn_stack  # list of (B,heads,N,N)
            self._cache["tok_hw"] = (Htok, Wtok)

        # (31)-(32) pool transformer tokens (GAP over tokens)
        z_vit = T_enc.mean(dim=1)  # (B,D)

        # (33)-(34) pool CNN feature map (GAP over spatial)
        z_cnn = F_path.mean(dim=(2, 3))  # (B,C)

        # (35)-(41) fuse + head
        z = torch.cat([z_cnn, z_vit], dim=1)
        logits = self.head(z)
        probs = F.softmax(logits, dim=1)
        conf = probs.max(dim=1).values

        return logits, probs, conf

    @torch.no_grad()
    def enable_mc_dropout(self):
        # Enable dropout modules for MC sampling while keeping the rest in eval mode
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    @torch.no_grad()
    def mc_predict(self, x, K=20):
        # (42)-(44): MC Dropout mean/variance over probabilities
        self.eval()
        self.enable_mc_dropout()
        preds = []
        for _ in range(K):
            _, probs, _ = self.forward(x, collect_attn=False)
            preds.append(probs)
        P = torch.stack(preds, dim=0)  # (K,B,4)
        mu = P.mean(dim=0)
        var = P.var(dim=0, unbiased=False)
        return mu, var
