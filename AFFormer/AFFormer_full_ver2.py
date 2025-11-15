import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict

# 1) 모델 조립 (키 prefix 맞춤)
class PureAFFormerSmall(torch.nn.Module):
    def __init__(self, num_classes=19):
        super().__init__()
        self.backbone   = AfformerSmallBackbone()
        self.decode_head= AFFHead(in_channels=216, channels=256, num_classes=num_classes)
        self.aux_head = AFFHead(in_channels=176, channels=128, num_classes=num_classes)

    def forward(self, x):
    
        feats = self.backbone(x)  # [stage1(32), stage2(64), stage3(176), stage4(216_refined)]
        main  = self.decode_head(feats[3])           # decode_head는 feats[-1] = 216ch 사용
        main  = F.interpolate(main, size=x.shape[-2:], mode='bilinear', align_corners=False)

        if self.training and hasattr(self, 'aux_head'):
            # ★ 보조헤드는 stage3(176ch)만 사용: AFFHead는 feats[-1]을 쓰므로 리스트로 감싸 전달
            aux = self.aux_head(feats[1])
            aux = F.interpolate(aux, size=x.shape[-2:], mode='bilinear', align_corners=False)
            return main, aux

        return main
    
def dwconv(ch, k=3, s=1, p=1, bias=True):
    return nn.Conv2d(ch, ch, k, s, p, groups=ch, bias=bias)

class Restore(nn.Module):
    def __init__(self, c_in, c_mid, c_out):
        super().__init__()
        self.conv1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(c_in, c_mid, 1, 1, 0, bias=False)),
            ('bn',   nn.BatchNorm2d(c_mid)),
        ]))
        self.dwconv = dwconv(c_mid, 3, 1, 1, bias=True)
        self.norm   = nn.BatchNorm2d(c_mid)  # 키: Restore.norm.*
        self.conv2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(c_mid, c_out, 1, 1, 0, bias=False)),
            ('bn',   nn.BatchNorm2d(c_out)),
        ]))
    def forward(self, x):
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.conv2(x)
        return x
    
def make_crpe_conv_list(C, splits):
    # splits의 합이 C이어야 함 (예: 64 -> 16/24/24, 176 -> 44/66/66, 216 -> 54/81/81)
    ks = [3,5,7]
    pads = [1,2,3]
    convs = []
    for ch, k, p in zip(splits, ks, pads):
        convs.append(nn.Conv2d(ch, ch, k, 1, p, groups=ch, bias=True))
    return nn.ModuleList(convs)

class FactorAttCRPE(nn.Module):
    """
    Factorized axial attention (row + col) + CRPE
    입력: x (B, C, H, W)  →  출력: (B, C, H, W)
    """
    def __init__(self, C, splits):
        super().__init__()
        self.C = C
        self.qkv  = nn.Linear(C, 3*C, bias=True)   # (B, HW, C) -> (B, HW, 3C)
        self.proj = nn.Linear(C, C,  bias=True)    # (B, HW, C) -> (B, HW, C)

        # CRPE: 채널 분할별 depthwise conv (3,5,7)
        self.crpe = nn.Module()
        self.crpe.conv_list = make_crpe_conv_list(C, splits)

    @staticmethod
    def _axial_attn_width(q, k, v, H, W):
        """
        W축(열) 방향 1D attention
        q,k,v: (B, HW, C)  →  반환: (B, HW, C)
        """
        B, HW, C = q.shape
        q = q.view(B, H, W, C)
        k = k.view(B, H, W, C)
        v = v.view(B, H, W, C)

        # 각 row(H 고정)에서 길이 W 시퀀스 self-attn
        q_t = q.permute(0, 1, 3, 2).contiguous()   # (B, H, C, W)
        k_t = k.permute(0, 1, 3, 2).contiguous()   # (B, H, C, W)
        v_t = v.permute(0, 1, 3, 2).contiguous()   # (B, H, C, W)

        attn = torch.matmul(q_t.transpose(-2, -1), k_t) / (C ** 0.5)  # (B,H,W,W)
        attn = attn.softmax(dim=-1)
        out  = torch.matmul(attn, v_t.transpose(-2, -1))  # (B,H,W,C)

        return out.reshape(B, H*W, C).contiguous()

    @staticmethod
    def _axial_attn_height(q, k, v, H, W):
        """
        H축(행) 방향 1D attention
        q,k,v: (B, HW, C)  →  반환: (B, HW, C)
        """
        B, HW, C = q.shape
        q = q.view(B, H, W, C)
        k = k.view(B, H, W, C)
        v = v.view(B, H, W, C)

        # 각 col(W 고정)에서 길이 H 시퀀스 self-attn
        q_t = q.permute(0, 2, 3, 1).contiguous()   # (B, W, C, H)
        k_t = k.permute(0, 2, 3, 1).contiguous()   # (B, W, C, H)
        v_t = v.permute(0, 2, 3, 1).contiguous()   # (B, W, C, H)

        attn = torch.matmul(q_t.transpose(-2, -1), k_t) / (C ** 0.5)  # (B,W,H,H)
        attn = attn.softmax(dim=-1)
        out  = torch.matmul(attn, v_t.transpose(-2, -1))  # (B,W,H,C)
        out  = out.permute(0, 2, 1, 3).contiguous()       # (B,H,W,C)

        return out.reshape(B, H*W, C).contiguous()

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        HW = H * W

        # 1) QKV
        x_flat = x.permute(0, 2, 3, 1).reshape(B, HW, C).contiguous()  # (B, HW, C)
        qkv = self.qkv(x_flat)                                         # (B, HW, 3C)
        q, k, v = torch.chunk(qkv, 3, dim=-1)                          # (B, HW, C)

        # 2) Factorized axial attention (row + col)
        out_w = self._axial_attn_width(q, k, v, H, W)                  # (B, HW, C)
        out_h = self._axial_attn_height(q, k, v, H, W)                 # (B, HW, C)
        x_attn = 0.5 * (out_w + out_h)                                 # 평균 결합

        # 3) CRPE (depthwise conv 분할 적용)
        v_map = v.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()    # (B,C,H,W)
        split_sizes = [conv.in_channels for conv in self.crpe.conv_list]
        v_splits = torch.split(v_map, split_sizes, dim=1)
        crpe_parts = [conv(feat) for conv, feat in zip(self.crpe.conv_list, v_splits)]
        crpe_out = torch.cat(crpe_parts, dim=1).contiguous()           # (B,C,H,W)
        crpe_flat = crpe_out.permute(0, 2, 3, 1).reshape(B, HW, C).contiguous()

        # 4) 최종 projection
        out = self.proj(x_attn + crpe_flat)                            # (B,HW,C)
        out = out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()    # (B,C,H,W)
        return out

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=0, act=True, bn=True, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch) if bn else nn.Identity()
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()
    def forward(self, x): return self.act(self.bn(self.conv(x)))

# --- Stem: backbone.stem.{0,1}.(conv/bn) ---
class StemUnit(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True)
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class Stem(nn.Module):
    # matches backbone.stem.0.*, backbone.stem.1.*
    def __init__(self, c1=3, c2=16, c3=32):
        super().__init__()
        self._modules["0"] = StemUnit(c1, c2)  # stem.0
        self._modules["1"] = StemUnit(c2, c3)  # stem.1
    def forward(self, x):
        x = self._modules["0"](x)
        x = self._modules["1"](x)
        return x

# --- PatchConv: backbone.patch_embed_stages.*.patch_embeds.*.patch_conv.(dwconv/pwconv/bn) ---
class PatchConv(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.dwconv = dwconv(ch, 3, 1, 1, bias=False)
        self.pwconv = nn.Conv2d(ch, ch, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(ch)
    def forward(self, x):
        x = self.dwconv(x); x = self.pwconv(x); x = self.bn(x); return x

class PatchEmbedUnit(nn.Module):
    # matches ...patch_embeds.<idx>.patch_conv.*
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True)
        self.patch_conv = PatchConv(out_ch)
    def forward(self, x):
        x = self.act(self.bn(self.proj(x)))
        x = self.patch_conv(x)
        return x

class PatchEmbedStage(nn.Module):
    # in_ch -> out_ch로 줄이면서 두 번의 패치 컨브를 수행 (ckpt의 .0, .1에 대응)
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.patch_embeds = nn.ModuleList([
            PatchEmbedUnit(in_ch,  out_ch, stride=stride),  # .0
            PatchEmbedUnit(out_ch, out_ch, stride=1),       # .1
        ])

    def forward(self, x):
        for m in self.patch_embeds:
            x = m(x)
        return x


class MLP(nn.Module):
    def __init__(self, C, ratio=2):  # 체크포인트는 2배수(64->128, 176->352, 216->432)
        super().__init__()
        hid = C * ratio
        self.fc1 = nn.Linear(C, hid, bias=True)
        self.dwconv = nn.Module()
        self.dwconv.dwconv = dwconv(hid, 3, 1, 1, bias=True)
        self.fc2 = nn.Linear(hid, C, bias=True)
    def forward(self, x):  # x: (B,C,H,W)
        B, C, H, W = x.shape
        y = x.permute(0, 2, 3, 1).reshape(B, H*W, C)   # (B, HW, C)
        y = self.fc1(y)
        y = y.view(B, H, W, -1).permute(0, 3, 1, 2)    # (B,hid,H,W)
        y = self.dwconv.dwconv(y)
        y = y.permute(0, 2, 3, 1).reshape(B, H*W, -1)  # (B,HW,hid)
        y = self.fc2(y)
        y = y.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return y
    
class MHCALayer(nn.Module):
    def __init__(self, C, splits):
        super().__init__()
        self.splits = splits
        self.cpe  = nn.Module()
        self.cpe.proj = nn.Conv2d(C, C, 3, 1, 1, groups=C, bias=True)
        self.crpe = nn.Module()
        self.crpe.conv_list = nn.ModuleList([
            nn.Conv2d(s, s, k, 1, p, groups=s, bias=True)
            for s, (k,p) in zip(self.splits, [(3,1),(5,2),(7,3)])
        ])
        self.factoratt_crpe = FactorAttCRPE(C, splits)  # 너가 쓰는 구현
        self.mlp  = MLP(C, ratio=2)
        self.norm1 = nn.BatchNorm2d(C)
        self.norm2 = nn.BatchNorm2d(C)

    def _split3(self, x):
        s0, s1, s2 = self.splits
        return x[:, :s0], x[:, s0:s0+s1], x[:, s0+s1:s0+s1+s2]

    def forward(self, x):
        h = self.norm1(x)
        x1, x2, x3 = self._split3(h.contiguous())
        o1 = self.crpe.conv_list[0](x1)
        o2 = self.crpe.conv_list[1](x2)
        o3 = self.crpe.conv_list[2](x3)
        crpe_out = torch.cat([o1, o2, o3], dim=1).contiguous()
        h = self.cpe.proj(h) + crpe_out + self.factoratt_crpe(h)
        x = x + h

        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h
        return x

class MHCABlock(nn.Module):
    def __init__(self, C, n_layers, splits):
        super().__init__()
        self.splits = splits  # 예: 64 -> [16,24,24], 176 -> [44,66,66], 216 -> [54,81,81]
        self.cpe  = nn.Module()
        self.cpe.proj = nn.Conv2d(C, C, 3, 1, 1, groups=C, bias=True)
        self.crpe = nn.Module()
        # conv_list는 이미 로드된 상태(각 분할 채널 수 만큼 groups depthwise)
        self.crpe.conv_list = nn.ModuleList([
            nn.Conv2d(s, s, k, 1, p, groups=s, bias=True)
            for s, (k,p) in zip(self.splits, [(3,1),(5,2),(7,3)])
        ])
        self.MHCA_layers = nn.ModuleList([MHCALayer(C, splits) for _ in range(n_layers)])

    def _split3(self, x):
        s0, s1, s2 = self.splits
        return x[:, :s0], x[:, s0:s0+s1], x[:, s0+s1:s0+s1+s2]

    def forward(self, x):
        # ✅ 블록 바깥 cpe + crpe
        x1, x2, x3 = self._split3(x)
        o1 = self.crpe.conv_list[0](x1)
        o2 = self.crpe.conv_list[1](x2)
        o3 = self.crpe.conv_list[2](x3)
        crpe_out = torch.cat([o1, o2, o3], dim=1)   # 분할별 결과를 concat하여 원 채널로 복원
        x = x + self.cpe.proj(x) + crpe_out

        # 레이어들
        for layer in self.MHCA_layers:
            x = layer(x)
        return x

class MHCAStage(nn.Module):
    def __init__(self, c_in, c_out, n_layers, splits, with_block=True):
        super().__init__()
        self.aggregate = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(c_in, c_out, 1, 1, 0, bias=False)),
            ('bn',   nn.BatchNorm2d(c_out)),
        ]))
        # Restore: c_in -> c_mid -> c_out, c_mid은 절반
        self.Restore = Restore(c_in, c_in//2, c_in)
        if with_block:
            self.mhca_blks = nn.ModuleList([MHCABlock(c_in, n_layers, splits)])
        else:
            self.mhca_blks = nn.ModuleList()  # stage0은 블록 키가 안 보임
    def forward(self, x):
        x = self.Restore(x)
        for b in self.mhca_blks: x = b(x)
        x = self.aggregate(x)
        return x

# --- 전체 Backbone: 키 prefix = backbone.* ---
class AfformerSmallBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = Stem(3, 16, 32)

        # PatchEmbed: 32->96->176->216->216
        self.patch_embed_stages = nn.ModuleList([
            PatchEmbedStage(32,  32,  stride=4),
            PatchEmbedStage(32,  96, stride=2),
            PatchEmbedStage(96, 176, stride=2),
            PatchEmbedStage(176, 216, stride=2),
        ])

        # CRPE 채널 분할 (64=16/24/24, 176=44/66/66, 216=54/81/81)
        sp96  = [24, 36, 36]
        sp176 = [44, 66, 66]
        sp216 = [54, 81, 81]

        self.mhca_stages = nn.ModuleList([
            MHCAStage(32,   96,  n_layers=0, splits=sp96,  with_block=False),  # stage0: 블록 키 없음
            MHCAStage(96,   176, n_layers=2, splits=sp96,  with_block=True),   # MHCA_layers.0,1
            MHCAStage(176,  216, n_layers=6, splits=sp176, with_block=True),   # MHCA_layers.0,1,2,3
            MHCAStage(216,  216, n_layers=2, splits=sp216, with_block=True),   # MHCA_layers.0,1
        ])

    def forward(self, x):
        stem = self.stem(x)
        f1 = self.patch_embed_stages[0](stem)   # 32
        f2 = self.patch_embed_stages[1](f1)  # 64
        f3 = self.patch_embed_stages[2](f2)  # 176
        f4 = self.patch_embed_stages[3](f3)  # 216

        y0 = self.mhca_stages[0](f1)  # 32->64 (Restore 32 mid 16)
        y1 = self.mhca_stages[1](f2)  # 64->176
        y2 = self.mhca_stages[2](f3)  # 176->216
        y3 = self.mhca_stages[3](f4)  # 216->216

        # 보통 모델은 y1~y3(or y0~y3) 중 필요한 것만 반환
        return [y0, y1, y2, y3]
    
class Squeeze(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.gn   = nn.GroupNorm(num_groups=32, num_channels=out_ch)   # ← GroupNorm → BatchNorm2d
        self.act  = nn.ReLU(inplace=True)
    def forward(self, x): return self.act(self.gn(self.conv(x)))

class Align(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 1, 1, 0, bias=False)  # padding=0 (weight shape엔 영향 없음)
        self.gn   = nn.GroupNorm(num_groups=32, num_channels=ch)       # ← GroupNorm → BatchNorm2d
    def forward(self, x): return self.gn(self.conv(x))

class AFFHead(nn.Module):
    """
    - 키 이름: decode_head.squeeze.*, decode_head.align.*, decode_head.conv_seg.*
    - in_channels=[216], num_classes=19 가 pth와 맞음
    """
    def __init__(self, in_channels=176, channels=256, num_classes=19):
        super().__init__()
        self.squeeze = Squeeze(in_channels, channels)
        self.align   = Align(channels)
        self.conv_seg = nn.Conv2d(channels, num_classes, 1, 1, 0, bias=True)

    def forward(self, feat):
        # feats는 backbone에서 나온 리스트. 마지막 인덱스(=216ch) 사용
        x = self.squeeze(feat)
        x = F.relu(self.align(x), inplace=True)
        x = self.conv_seg(x)
        return x
