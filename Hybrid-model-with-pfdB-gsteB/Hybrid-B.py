
# models/hybrid_model.py  # file target (Jupyter magic writes this file)
#
# Hybrid ResNet50V2–RViT (PFDB-GSTEB): paper-faithful where it matters + my extensions.  # high-level summary
# - Sarada baseline: pretrained ResNet50V2 backbone, then add head regularisation (BN/Dropout) without breaking backbone.  # Sarada-style framing (Sarada et al., 2024)
# - Krishnan RViT: rotate IMAGE {0,90,180,270} -> patchify P=16 -> (pos + rot) embeddings -> average -> encoder.  # RViT pipeline (Krishnan et al., 2024)
# - Encoder block: MHSA + depth-wise conv + MLP; token global average pooling (no CLS).  # Krishnan block style
# - PFD-B (my extension): 1×1 conv + sigmoid mask on CNN feature map; gating affects CNN pooled descriptor and guides transformer tokens.  # dual-branch gating idea
# - GSTE-B (my extension): mask-guided dynamic token evolution (optional grid shrink) + interpolated positional embeddings.  # dynamic token grid
# - XAI: attention list for rollout; image-scale mask for overlay; CNN feature map saved for Grad-CAM++ hooks.  # interpretability outputs
#
# Libraries: PyTorch is my core DL stack (Paszke et al., 2019).  # torch citation pointer

import math  # sqrt/ceil utilities (Python stdlib)
import torch  # tensors + autograd (Paszke et al., 2019)
import torch.nn as nn  # layers (Paszke et al., 2019)
import torch.nn.functional as F  # functional ops (Paszke et al., 2019)

# -------------------------  # section divider
# CNN: pretrained ResNet50V2 (timm)  # backbone source
# -------------------------  # section divider
class ResNet50V2TimmBackbone(nn.Module):  # wraps timm backbone
    """  # docstring start
    Uses timm pretrained ResNetV2 feature extractor. Returns last feature map (B, C, 7, 7) for 224x224 input.  # output description
    """  # docstring end
    def __init__(self, model_name="resnetv2_50x1_bitm", pretrained=True):  # config
        super().__init__()  # init base
        try:  # guarded import
            import timm  # timm model zoo; BiT-style weights common (Kolesnikov et al., 2020)
        except Exception as e:  # missing timm
            raise ImportError(  # raise helpful error
                "timm is required for pretrained ResNet50V2. "
                "Install timm or use an environment (e.g., Kaggle) that includes it."
            ) from e  # keep traceback
        self.model_name = model_name  # store name
        self.backbone = timm.create_model(  # create model
            model_name,  # id
            pretrained=pretrained,  # weights
            features_only=True,  # feature maps
            out_indices=(4,),  # last stage
        )  # end create
        try:  # safe channel query
            self.out_ch = int(self.backbone.feature_info.channels()[-1])  # channel count for last stage
        except Exception:  # fallback
            self.out_ch = 2048  # common ResNet last stage channels
    def forward(self, x):  # forward
        feats = self.backbone(x)  # list of features
        return feats[-1]  # last stage feature map

# -------------------------  # section divider
# PFD: pathology mask gating (Krsna extension)  # learned spatial gate
# -------------------------  # section divider
class PFD(nn.Module):  # pathology-focused disentanglement block
    """  # docstring start
    PFD (Krsna): learned pathology gate on CNN feature maps. M = sigmoid(conv1x1(F)), F_path = M * F  # gating equation
    """  # docstring end
    def __init__(self, in_ch):  # in channels
        super().__init__()  # init base
        self.mask_conv = nn.Conv2d(in_ch, 1, kernel_size=1, stride=1, padding=0, bias=True)  # 1×1 mask predictor
    def forward(self, feat):  # feat: (B,C,7,7)
        mask = torch.sigmoid(self.mask_conv(feat))  # (B,1,7,7) gate in [0,1]
        gated = feat * mask  # apply gate (broadcast)
        return gated, mask  # return gated feat + mask

# -------------------------  # section divider
# Krishnan RViT: patchify IMAGE (P=16) -> tokens (B,N,D)  # raw-image patchify path
# -------------------------  # section divider
class ImagePatchEmbed(nn.Module):  # patch embedding module
    """  # docstring start
    Patch embedding as linear projection of flattened patches. Conv2d(kernel=P, stride=P) is equivalent to linear patch projection in ViT.  # ViT equivalence
    """  # docstring end
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=142):  # config
        super().__init__()  # init base
        self.img_size = img_size  # store image size
        self.patch_size = patch_size  # store patch size
        self.grid_h = img_size // patch_size  # grid height (14 for 224/16)
        self.grid_w = img_size // patch_size  # grid width (14 for 224/16)
        self.num_patches = self.grid_h * self.grid_w  # number of tokens (196)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)  # conv patch projection
    def forward(self, x):  # x: (B,3,H,W)
        x = self.proj(x)  # (B,D,gh,gw)
        B, D, H, W = x.shape  # unpack projected map
        tokens = x.flatten(2).transpose(1, 2)  # (B,N,D)
        return tokens, H, W  # return tokens + grid

class PositionalAndRotationEmbedding(nn.Module):  # adds pos+rot embeddings
    """  # docstring start
    Learnable positional embeddings + rotation embeddings (Krishnan-style). Default base grid for 224, P=16 is 14x14.  # expected grid
    """  # docstring end
    def __init__(self, base_h=14, base_w=14, embed_dim=142, n_rot=4):  # config
        super().__init__()  # init base
        self.base_h = base_h  # base height
        self.base_w = base_w  # base width
        self.embed_dim = embed_dim  # token dim
        self.n_rot = int(n_rot)  # number of rotation embeddings
        self.pos = nn.Parameter(torch.zeros(1, base_h * base_w, embed_dim))  # learnable pos table
        nn.init.trunc_normal_(self.pos, std=0.02)  # init
        self.rot = nn.Parameter(torch.zeros(self.n_rot, embed_dim))  # learnable rot table
        nn.init.trunc_normal_(self.rot, std=0.02)  # init
    def forward(self, tokens, ht, wt, rot_idx):  # tokens (B,N,D), grid ht/wt, rotation id
        if ht == self.base_h and wt == self.base_w:  # if grid matches base
            pos = self.pos  # use directly
        else:  # otherwise interpolate pos
            pos = self.pos.reshape(1, self.base_h, self.base_w, self.embed_dim).permute(0, 3, 1, 2)  # (1,D,H,W)
            pos = F.interpolate(pos, size=(ht, wt), mode="bilinear", align_corners=False)  # resize
            pos = pos.permute(0, 2, 3, 1).reshape(1, ht * wt, self.embed_dim)  # back to (1,N,D)
        r = int(rot_idx) % self.n_rot  # keep rotation index in range
        tokens = tokens + pos + self.rot[r].view(1, 1, -1)  # add pos + rot
        return tokens  # return embedded tokens

# -------------------------  # section divider
# Flexible MHSA (supports dim not divisible by heads)  # attention core
# -------------------------  # section divider
class FlexibleMHSA(nn.Module):  # attention module
    """  # docstring start
    Flexible MHSA for dim not divisible by heads. Uses inner_dim = heads * ceil(dim/heads) so we do NOT drop dimensions.  # PFDB uses ceil
    Attention runs in inner_dim, then projected back to dim.  # projection back
    """  # docstring end
    def __init__(self, dim, num_heads, attn_dropout=0.1, proj_dropout=0.1):  # config
        super().__init__()  # init base
        self.dim = dim  # store dim
        self.num_heads = num_heads  # store heads
        head_dim = int(math.ceil(dim / num_heads))  # ceil division for head_dim
        inner_dim = num_heads * head_dim  # internal dim
        self.inner_dim = inner_dim  # store
        self.head_dim = head_dim  # store
        self.scale = head_dim ** -0.5  # scale factor
        self.qkv = nn.Linear(dim, inner_dim * 3, bias=True)  # qkv projection
        self.attn_drop = nn.Dropout(attn_dropout)  # attn dropout
        self.proj = nn.Linear(inner_dim, dim, bias=True)  # output projection
        self.proj_drop = nn.Dropout(proj_dropout)  # proj dropout
    def forward(self, x, return_attn=False):  # x: (B,N,D)
        B, N, D = x.shape  # unpack
        qkv = self.qkv(x)  # (B,N,3*inner_dim)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # (3,B,H,N,hd)
        q, k, v = qkv[0], qkv[1], qkv[2]  # split q,k,v
        attn = (q @ k.transpose(-2, -1)) * self.scale  # logits
        attn = attn.softmax(dim=-1)  # normalize
        attn = self.attn_drop(attn)  # dropout
        out = attn @ v  # apply attention
        out = out.transpose(1, 2).reshape(B, N, self.inner_dim)  # merge heads
        out = self.proj(out)  # project back to dim
        out = self.proj_drop(out)  # dropout
        if return_attn:  # if requested
            return out, attn  # return attention too
        return out, None  # otherwise no attn

class RViTBlock(nn.Module):  # RViT block (MHSA + DWConv + MLP)
    """  # docstring start
    MHSA -> DWConv -> MLP, each with pre-LN + residual.  # structure
    """  # docstring end
    def __init__(self, dim, heads, mlp_dim, attn_dropout=0.1, dropout=0.1, ht=14, wt=14):  # config
        super().__init__()  # init base
        self.ht = ht  # expected grid height
        self.wt = wt  # expected grid width
        self.ln1 = nn.LayerNorm(dim)  # LN before attention
        self.attn = FlexibleMHSA(dim, heads, attn_dropout=attn_dropout, proj_dropout=dropout)  # MHSA
        self.ln2 = nn.LayerNorm(dim)  # LN before DWConv
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True)  # depth-wise conv
        self.dw_drop = nn.Dropout(dropout)  # dropout
        self.ln3 = nn.LayerNorm(dim)  # LN before MLP
        self.mlp = nn.Sequential(  # MLP
            nn.Linear(dim, mlp_dim),  # expand
            nn.GELU(),  # activate
            nn.Dropout(dropout),  # drop
            nn.Linear(mlp_dim, dim),  # project back
            nn.Dropout(dropout),  # drop
        )  # end MLP
    def forward(self, x, return_attn=False):  # x: (B,N,D)
        attn_out, attn = self.attn(self.ln1(x), return_attn=return_attn)  # attention
        x = x + attn_out  # residual add
        y = self.ln2(x)  # normalize
        B, N, D = y.shape  # unpack
        ht, wt = self.ht, self.wt  # configured grid
        if ht * wt != N:  # if mismatch
            side = int(math.sqrt(N))  # infer side
            ht = max(side, 1)  # clamp
            wt = max(N // ht, 1)  # clamp
        y2 = y.transpose(1, 2).reshape(B, D, ht, wt)  # tokens -> grid
        y2 = self.dwconv(y2)  # local mixing
        y2 = y2.reshape(B, D, ht * wt).transpose(1, 2)  # grid -> tokens
        y2 = self.dw_drop(y2)  # dropout
        x = x + y2  # residual add
        x = x + self.mlp(self.ln3(x))  # MLP residual add
        return x, attn  # return tokens + attention

class RViTEncoder(nn.Module):  # stacked encoder
    def __init__(self, dim=142, depth=10, heads=10, mlp_dim=480, attn_dropout=0.1, dropout=0.1, ht=14, wt=14):  # config
        super().__init__()  # init
        self.blocks = nn.ModuleList([  # blocks list
            RViTBlock(dim, heads, mlp_dim, attn_dropout=attn_dropout, dropout=dropout, ht=ht, wt=wt)  # block init
            for _ in range(depth)  # repeat
        ])  # end list
        self.ln = nn.LayerNorm(dim)  # final LN
    def forward(self, x, return_attn=False):  # x: (B,N,D)
        attn_list = []  # collect attention maps
        for blk in self.blocks:  # loop blocks
            x, attn = blk(x, return_attn=return_attn)  # forward
            if return_attn and attn is not None:  # collect if requested
                attn_list.append(attn)  # store
        x = self.ln(x)  # final LN
        return x, attn_list  # output tokens + attention list

# -------------------------  # section divider
# Hybrid model (FULL or ABLATION via flag)  # main classifier class
# -------------------------  # section divider
class HybridResNet50V2_RViT(nn.Module):  # PFDB-GSTEB model
    def __init__(  # constructor
        self,  # self
        num_classes=4,  # multiclass output
        img_size=224,  # input size
        patch_size=16,  # ViT patch size
        embed_dim=142,  # token dim
        depth=10,  # encoder depth
        heads=10,  # heads
        mlp_dim=480,  # MLP dim
        attn_dropout=0.1,  # attn dropout
        vit_dropout=0.1,  # block dropout
        fusion_dim=256,  # fusion dim
        fusion_dropout=0.5,  # fusion dropout
        rotations=(0, 1, 2, 3),  # rot set
        cnn_name="resnetv2_50x1_bitm",  # backbone id
        cnn_pretrained=True,  # pretrained weights
        use_pfd_gste=True,  # toggle ablation
        gste_min_side=7,  # minimum dynamic side
        gste_std_flat=0.05,  # flat-mask threshold
        gste_std_full=0.30,  # full-shrink threshold
        gste_max_shrink=0.50,  # max shrink fraction
    ):  # end signature
        super().__init__()  # init base
        self.num_classes = num_classes  # store
        self.rotations = rotations  # store
        self.patch_size = patch_size  # store
        self.img_size = img_size  # store
        self.use_pfd_gste = bool(use_pfd_gste)  # store as bool

        # CNN branch (Sarada starting point: pretrained backbone)  # backbone path
        self.cnn = ResNet50V2TimmBackbone(model_name=cnn_name, pretrained=cnn_pretrained)  # init CNN
        self.cnn_out_ch = int(self.cnn.out_ch)  # read output channels

        # Krsna PFD  # my gating module
        self.pfd = PFD(in_ch=self.cnn_out_ch)  # init PFD

        # Sarada-style head regularisation (minimal, backbone intact)  # regularize pooled descriptor
        self.cnn_drop = nn.Dropout(p=float(fusion_dropout))  # dropout on pooled CNN vector
        self.cnn_proj = nn.Linear(self.cnn_out_ch, fusion_dim)  # project to fusion dim
        self.cnn_bn = nn.BatchNorm1d(fusion_dim)  # BN on projected CNN vector
        self.cnn_pool = nn.AdaptiveAvgPool2d(1)  # global average pool

        # Transformer branch: Krishnan patchify IMAGE  # raw image -> tokens path
        grid = img_size // patch_size  # 14 for 224/16
        self.base_grid = int(grid)  # store base grid
        self.gste_min_side = int(max(1, min(gste_min_side, self.base_grid)))  # clamp min side

        # GSTE selection knobs (robust dynamic side selection)  # hyperparameters
        self.gste_std_flat = float(gste_std_flat)  # below => keep full grid
        self.gste_std_full = float(gste_std_full)  # above => allow max shrink
        self.gste_max_shrink = float(gste_max_shrink)  # shrink fraction cap

        self.patch = ImagePatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)  # patch embed
        self.posrot = PositionalAndRotationEmbedding(base_h=grid, base_w=grid, embed_dim=embed_dim, n_rot=4)  # pos+rot embed
        self.encoder = RViTEncoder(  # encoder
            dim=embed_dim,  # dim
            depth=depth,  # depth
            heads=heads,  # heads
            mlp_dim=mlp_dim,  # mlp dim
            attn_dropout=attn_dropout,  # attn dropout
            dropout=vit_dropout,  # dropout
            ht=grid,  # expected ht
            wt=grid,  # expected wt
        )  # end encoder
        self.vit_proj = nn.Linear(embed_dim, fusion_dim)  # transformer pooled -> fusion dim

        # Fusion head  # combine CNN and transformer
        self.fuse_fc = nn.Linear(fusion_dim * 2, fusion_dim)  # concat -> fusion
        self.fuse_drop = nn.Dropout(p=float(fusion_dropout))  # dropout
        self.out = nn.Linear(fusion_dim, num_classes)  # logits

        # For external XAI hooks (optional)  # saved tensors for Grad-CAM++ etc.
        self._last_cnn_feat = None  # stores last CNN feature map used
        self._last_cnn_mask_img = None  # stores last upsampled mask

    def freeze_cnn_bn(self):  # freeze BN layers in CNN
        for m in self.cnn.modules():  # iterate CNN modules
            if isinstance(m, nn.BatchNorm2d):  # BN2d layers
                m.eval()  # set eval

    def _mask_to_alpha_grid(self, mask_img, out_side):  # convert image-scale mask to alpha grid
        """ mask_img: (B,1,H,W) -> adaptive pool to (out_side,out_side) Return alpha_grid with mean≈1 per-sample to keep scale stable. """  # doc
        alpha = F.adaptive_avg_pool2d(mask_img, output_size=(out_side, out_side))  # (B,1,s,s)
        alpha = alpha / (alpha.mean(dim=(2, 3), keepdim=True) + 1e-6)  # normalize mean≈1
        return alpha  # return grid weights

    def _choose_dynamic_side(self, alpha_base):  # decide dynamic grid size
        """ GSTE dynamic grid side in [gste_min_side .. base_grid], guided by mask concentration.
        Fixes the "collapse on flat mask" issue:
        - If mask is flat/uncertain (std very low), KEEP full grid.
        - Shrink grid only when mask is truly concentrated/peaked.
        """  # policy docstring
        with torch.no_grad():  # no gradients for side decision
            std = alpha_base.std(dim=(2, 3), keepdim=False)  # (B,1) std across grid
            s = float(std.mean().item())  # scalar proxy for concentration
            if s <= self.gste_std_flat:  # flat mask => no shrink
                return self.base_grid  # keep 14
            denom = max(self.gste_std_full - self.gste_std_flat, 1e-6)  # avoid divide by zero
            t = (s - self.gste_std_flat) / denom  # normalize into [0,1] range
            t = max(0.0, min(1.0, t))  # clamp
            shrink = self.gste_max_shrink * t  # shrink fraction
            target = int(round(self.base_grid * (1.0 - shrink)))  # target side
            side = max(self.gste_min_side, min(self.base_grid, target))  # clamp to allowed
            return side  # return side

    def _gste_dynamic_tokens(self, tokens_base, alpha_base, side):  # dynamic token evolution
        """ Mask-guided dynamic token evolution:
        - tokens_base: (B,N,D) with N=base_grid^2
        - alpha_base : (B,1,base_grid,base_grid), mean≈1
        - side: dynamic grid side (<= base_grid)
        Returns tokens_dyn: (B, side*side, D) and (side,side).
        """  # doc
        B, N, D = tokens_base.shape  # unpack
        g = self.base_grid  # base side
        x = tokens_base.transpose(1, 2).reshape(B, D, g, g)  # tokens -> (B,D,g,g)
        w = alpha_base  # (B,1,g,g) weights
        x = x * w  # apply ROI weighting (broadcast)
        if side < g:  # if downsample needed
            num = F.adaptive_avg_pool2d(x, output_size=(side, side))  # weighted numerator
            den = F.adaptive_avg_pool2d(w, output_size=(side, side)) + 1e-6  # weight sum
            x = num / den  # weighted average pooling
        tokens = x.flatten(2).transpose(1, 2)  # back to (B,side*side,D)
        return tokens, side, side  # return tokens + grid

    def forward(self, x, return_xai=False):  # forward
        # =========================  # section divider
        # CNN backbone  # compute CNN features
        # =========================  # section divider
        feat = self.cnn(x)  # (B,C,7,7)

        # =========================  # section divider
        # PFD (optional)  # compute mask + gated features
        # =========================  # section divider
        if self.use_pfd_gste:  # full model
            feat_path, mask_feat = self.pfd(feat)  # gated feat + mask
            feat_for_cnn = feat_path  # PFDB: also gate the CNN descriptor path
            mask_img = F.interpolate(mask_feat, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)  # mask at image scale
        else:  # ablation path
            feat_for_cnn = feat  # no gating
            mask_img = None  # no mask

        # Save for possible Grad-CAM++ hooking / debugging  # XAI support
        self._last_cnn_feat = feat_for_cnn  # stash feature map
        self._last_cnn_mask_img = mask_img  # stash mask

        # =========================  # section divider
        # CNN pooled vector (Sarada-style head regularisation)  # pooled descriptor branch
        # =========================  # section divider
        z_cnn = self.cnn_pool(feat_for_cnn).flatten(1)  # (B,C)
        z_cnn = self.cnn_drop(z_cnn)  # dropout
        z_cnn = self.cnn_proj(z_cnn)  # (B,fusion_dim)
        z_cnn = self.cnn_bn(z_cnn)  # BN
        z_cnn = F.relu(z_cnn, inplace=True)  # activation

        # =========================  # section divider
        # GSTE: choose dynamic token grid once (batch-consistent) from mask  # dynamic grid decision
        # =========================  # section divider
        if self.use_pfd_gste:  # if mask exists
            alpha0 = self._mask_to_alpha_grid(mask_img, self.base_grid)  # (B,1,14,14)
            dyn_side = self._choose_dynamic_side(alpha0)  # choose side
        else:  # no GSTE
            dyn_side = self.base_grid  # keep 14

        # =========================  # section divider
        # Krishnan RViT rotations: rotate IMAGE, patchify P=16, average embeddings  # core RViT path
        # =========================  # section divider
        token_sets = []  # collect per-rotation token sequences
        for k in self.rotations:  # loop rotations
            rot_id = int(k) % 4  # rotation id in 0..3
            x_r = torch.rot90(x, k=rot_id, dims=(2, 3))  # rotate the raw image
            tokens_base, ht, wt = self.patch(x_r)  # patchify -> (B,196,D), ht=wt=14
            if self.use_pfd_gste:  # apply mask guidance
                m_r = torch.rot90(mask_img, k=rot_id, dims=(2, 3))  # rotate mask with image
                alpha_base = self._mask_to_alpha_grid(m_r, self.base_grid)  # (B,1,14,14)
                tokens, ht, wt = self._gste_dynamic_tokens(tokens_base, alpha_base, dyn_side)  # dynamic grid + weighting
            else:  # ablation
                tokens, ht, wt = tokens_base, ht, wt  # keep base tokens
            tokens = self.posrot(tokens, ht, wt, rot_id)  # add pos + rot embedding (interpolates if needed)
            token_sets.append(tokens)  # store tokens

        Tavg = torch.stack(token_sets, dim=0).mean(dim=0)  # average over rotations before encoder

        # =========================  # section divider
        # Transformer encoder (MHSA + DWConv + MLP)  # global reasoning
        # =========================  # section divider
        Tenc, attn_list = self.encoder(Tavg, return_attn=return_xai)  # encode tokens
        z_vit = Tenc.mean(dim=1)  # global average pool tokens
        z_vit = F.relu(self.vit_proj(z_vit), inplace=True)  # project -> fusion_dim

        # =========================  # section divider
        # Fusion + classifier  # combine branches
        # =========================  # section divider
        z = torch.cat([z_cnn, z_vit], dim=1)  # concat
        h = F.relu(self.fuse_fc(z), inplace=True)  # fuse
        h = self.fuse_drop(h)  # dropout
        logits = self.out(h)  # logits

        if return_xai:  # return explanation payload
            return logits, {  # package dict
                "attn": attn_list,  # transformer attentions
                "mask": mask_img,  # image-scale mask
                "gste_side": dyn_side,  # dynamic grid size used
            }  # end dict
        return logits, None  # default path

    @torch.no_grad()  # inference-only
    def mc_dropout_predict(self, x, mc_samples=20):  # MC dropout
        self.eval()  # eval mode
        for m in self.modules():  # iterate modules
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):  # dropout layers
                m.train()  # enable dropout
        probs = []  # collect probs
        for _ in range(int(mc_samples)):  # repeat
            logits, _ = self.forward(x, return_xai=False)  # forward
            probs.append(torch.softmax(logits, dim=1))  # softmax
        probs = torch.stack(probs, dim=0)  # stack samples
        mu = probs.mean(dim=0)  # mean
        var = probs.var(dim=0, unbiased=False)  # variance
        return mu, var  # return

class HybridResNet50V2_RViT_Ablation(HybridResNet50V2_RViT):  # ablation wrapper
    """ Ablation: Hybrid WITHOUT Krsna extensions (PFD + GSTE). Still paper-faithful to Krishnan RViT (rotate IMAGE, patchify P=16, avg embeddings). """  # what this ablation does
    def __init__(self, *args, **kwargs):  # pass-through init
        kwargs["use_pfd_gste"] = False  # force ablation
        super().__init__(*args, **kwargs)  # call parent init
