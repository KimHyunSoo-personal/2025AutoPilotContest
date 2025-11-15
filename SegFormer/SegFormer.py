import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Patch Embedding
# -------------------------
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels=3, embed_dim=64, patch_size=7, stride=4, padding=3):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)  # [B, C, H', W']
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        x = self.norm(x)
        return x, H, W


# -------------------------
# Attention + Transformer Block
# -------------------------
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, sr_ratio=1, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # spatial reduction
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x, H, W):
        B, N, C = x.shape

        # q
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads)
        q = q.permute(0, 2, 1, 3)   # [B, heads, N, dim_head]

        # spatial reduction for kv
        if self.sr is not None:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
        else:
            x_ = x

        kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Module()
        self.dwconv.dwconv = nn.Conv2d(
            hidden_features, hidden_features,
            kernel_size=3, stride=1, padding=1,
            groups=hidden_features
        )
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.fc1(x)
        x = x.view(B, H, W, -1).permute(0, 3, 1, 2)
        x = self.dwconv.dwconv(x)
        x = x.permute(0, 2, 3, 1).reshape(B, N, -1)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MiTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., sr_ratio=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, sr_ratio=sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=hidden_dim, drop=drop)

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x), H, W)
        return x
    

# -------------------------
# Decoder Head
# -------------------------
class SegFormerHead(nn.Module):
    def __init__(self, dims, num_classes=19):
        super().__init__()
        self.linear_c = nn.ModuleList([nn.Conv2d(dim, 256, 1) for dim in dims])
        self.linear_fuse = nn.Conv2d(256 * len(dims), 256, 1)
        self.pred = nn.Conv2d(256, num_classes, 1)

    def forward(self, features):
        outs = []
        size = features[0].shape[2:]
        for i, f in enumerate(features):
            out = self.linear_c[i](f)
            out = F.interpolate(out, size=size, mode="bilinear", align_corners=False)
            outs.append(out)
        x = self.linear_fuse(torch.cat(outs, dim=1))
        return self.pred(x)


# -------------------------
# SegFormer B0 (checkpoint 구조 동일)
# -------------------------
class SegFormer_B0(nn.Module):
    def __init__(self, num_classes=19):
        super().__init__()
        depths = [2, 2, 2, 2]
        embed_dims = [32, 64, 160, 256]
        num_heads = [1, 2, 5, 8]
        sr_ratios = [8, 4, 2, 1]

        # Patch embeddings
        self.patch_embed1 = OverlapPatchEmbed(3, embed_dims[0], 7, 4, 3)
        self.patch_embed2 = OverlapPatchEmbed(embed_dims[0], embed_dims[1], 3, 2, 1)
        self.patch_embed3 = OverlapPatchEmbed(embed_dims[1], embed_dims[2], 3, 2, 1)
        self.patch_embed4 = OverlapPatchEmbed(embed_dims[2], embed_dims[3], 3, 2, 1)

        # Transformer encoder stages
        self.block1 = nn.ModuleList([MiTBlock(embed_dims[0], num_heads[0], sr_ratio=sr_ratios[0]) for _ in range(depths[0])])
        self.block2 = nn.ModuleList([MiTBlock(embed_dims[1], num_heads[1], sr_ratio=sr_ratios[1]) for _ in range(depths[1])])
        self.block3 = nn.ModuleList([MiTBlock(embed_dims[2], num_heads[2], sr_ratio=sr_ratios[2]) for _ in range(depths[2])])
        self.block4 = nn.ModuleList([MiTBlock(embed_dims[3], num_heads[3], sr_ratio=sr_ratios[3]) for _ in range(depths[3])])

        # Norm layers
        self.norm1 = nn.LayerNorm(embed_dims[0])
        self.norm2 = nn.LayerNorm(embed_dims[1])
        self.norm3 = nn.LayerNorm(embed_dims[2])
        self.norm4 = nn.LayerNorm(embed_dims[3])

        # Classification head (checkpoint에 있음)
        self.head = SegFormerHead(embed_dims, num_classes)

    def forward(self, x):
        B = x.shape[0]
        H0, W0 = x.shape[2:]   # 원본 입력 크기 저장

        # Stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x1 = self.norm1(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # Stage 2
        x, H, W = self.patch_embed2(x1)
        for blk in self.block2:
            x = blk(x, H, W)
        x2 = self.norm2(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # Stage 3
        x, H, W = self.patch_embed3(x2)
        for blk in self.block3:
            x = blk(x, H, W)
        x3 = self.norm3(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # Stage 4
        x, H, W = self.patch_embed4(x3)
        for blk in self.block4:
            x = blk(x, H, W)
        x4 = self.norm4(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        features = [x1, x2, x3, x4]

        # segmentation head
        logits = self.head(features)
        logits = F.interpolate(logits, size=(H0, W0), mode="bilinear", align_corners=False)
        return logits
