# models/rvit.py
# Implements Krishnan-style RViT encoder with:
# - Patch embedding dim D=142, depth=10, heads=10, MLP dim=480, attn dropout=0.1
# - Rotation embeddings + positional embeddings
# - Average integration across rotations {0,90,180,270}
#
# NOTE: In your Algorithm 1, tokens come from CNN feature maps (stride-16).
# That makes "patch size 16 on 224x224" correspond to "patch size 1 on the 14x14 feature map".

import torch
import torch.nn as nn
import torch.nn.functional as F


class Tokenizer(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size_feat=1):
        super().__init__()
        self.patch_size_feat = patch_size_feat

        # If patch_size_feat==1: each spatial location becomes one patch with dim=in_channels.
        # If >1: we unfold and flatten C*(P*P).
        if patch_size_feat == 1:
            patch_dim = in_channels
        else:
            patch_dim = in_channels * patch_size_feat * patch_size_feat

        self.proj = nn.Linear(patch_dim, embed_dim)

    def forward(self, feat):
        # feat: (B, C, H, W)
        B, C, H, W = feat.shape
        P = self.patch_size_feat

        if P == 1:
            # tokens: (B, N, C) where N=H*W
            tokens = feat.flatten(2).transpose(1, 2)
            tokens = self.proj(tokens)  # (B, N, D)
            Htok, Wtok = H, W
            return tokens, Htok, Wtok

        # Unfold: (B, C*P*P, N)
        patches = F.unfold(feat, kernel_size=P, stride=P)
        patches = patches.transpose(1, 2)  # (B, N, C*P*P)
        tokens = self.proj(patches)        # (B, N, D)
        Htok = H // P
        Wtok = W // P
        return tokens, Htok, Wtok


class MLP(nn.Module):
    def __init__(self, dim, mlp_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RViTEncoderLayer(nn.Module):
    def __init__(self, dim, heads, mlp_dim, attn_dropout=0.1, dropout=0.1, dwconv_kernel=3):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=attn_dropout, batch_first=True)
        self.drop_attn = nn.Dropout(dropout)

        self.ln_dw = nn.LayerNorm(dim)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=dwconv_kernel, padding=dwconv_kernel // 2, groups=dim, bias=False)

        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_dim, dropout)

    def forward(self, x, Htok, Wtok, collect_attn=False):
        # (27) MHSA with residual
        q = self.ln1(x)
        attn_out, attn_w = self.attn(q, q, q, need_weights=collect_attn, average_attn_weights=False)
        x = x + self.drop_attn(attn_out)

        # (28) Depth-wise conv with residual (local inductive bias)
        z = self.ln_dw(x)
        B, N, D = z.shape
        z2d = z.transpose(1, 2).reshape(B, D, Htok, Wtok)
        z2d = self.dwconv(z2d)
        z = z2d.reshape(B, D, N).transpose(1, 2)
        x = x + z

        # (29) MLP with residual
        x = x + self.mlp(self.ln2(x))

        return x, attn_w


class RViTEncoder(nn.Module):
    def __init__(self, dim=142, depth=10, heads=10, mlp_dim=480, attn_dropout=0.1, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            RViTEncoderLayer(dim, heads, mlp_dim, attn_dropout=attn_dropout, dropout=dropout)
            for _ in range(depth)
        ])

    def forward(self, x, Htok, Wtok, collect_attn=False):
        attn_stack = []
        for layer in self.layers:
            x, attn_w = layer(x, Htok, Wtok, collect_attn=collect_attn)
            if collect_attn:
                attn_stack.append(attn_w)  # (B, heads, N, N)
        return x, attn_stack


class PosRotEmbedding(nn.Module):
    def __init__(self, dim, base_hw=(14, 14), num_rotations=4):
        super().__init__()
        Hb, Wb = base_hw
        self.base_hw = base_hw
        self.pos_emb = nn.Parameter(torch.zeros(1, Hb * Wb, dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        self.rot_emb = nn.Embedding(num_rotations, dim)
        nn.init.trunc_normal_(self.rot_emb.weight, std=0.02)

    def interpolate_pos(self, Htok, Wtok):
        Hb, Wb = self.base_hw
        if (Htok, Wtok) == (Hb, Wb):
            return self.pos_emb

        # Interpolate in 2D then flatten back.
        D = self.pos_emb.shape[-1]
        pos2d = self.pos_emb.reshape(1, Hb, Wb, D).permute(0, 3, 1, 2)  # (1,D,Hb,Wb)
        pos2d = F.interpolate(pos2d, size=(Htok, Wtok), mode="bilinear", align_corners=False)
        pos = pos2d.permute(0, 2, 3, 1).reshape(1, Htok * Wtok, D)
        return pos

    def forward(self, tokens, Htok, Wtok, rot_id):
        # tokens: (B, N, D)
        pos = self.interpolate_pos(Htok, Wtok)         # (1, N, D)
        rot = self.rot_emb(torch.tensor(rot_id, device=tokens.device)).view(1, 1, -1)  # (1,1,D)
        return tokens + pos + rot
