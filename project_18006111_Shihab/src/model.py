"""Multi-task hand network: box + mask + gesture + confidence from one RGB image.

Implements `docs/02_DESIGN.md` s3 to the frozen contract in `docs/INTERFACES.md`.

Everything here is built from `torch.nn` primitives. Nothing is imported from
`torchvision.models`, `torchvision.ops`, `timm`, or any detection/segmentation library, and no
pretrained weights are loaded (COMP0248 LSA p10/p11).

Layout
------
    HandNetEncoder  stem + 4 residual stages, strides 2/4/8/16/32, channels 24/48/96/192/320
    HandNet         encoder + U-Net segmentation decoder + centre-point detector + classifier
    decode_detection(out, k)  heatmap -> boxes in input-frame pixels
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["HandNet", "HandNetEncoder", "decode_detection", "DET_STRIDE", "HEAT_BIAS_INIT"]

#: Output stride of the detection head. The training target encoder must use the same value:
#: a ground-truth centre at input pixel (cx, cy) lands in cell (cx / 4, cy / 4).
DET_STRIDE: int = 4

#: Bias of the final `heat` convolution. Focal-loss prior trick (RetinaNet / CenterNet): start
#: with sigmoid(bias) = p so the ~99.99% of background cells contribute almost no loss in the
#: first epochs, instead of drowning the handful of positives. NOTE: -2.19 is the CenterNet
#: constant and corresponds to p = 0.1, i.e. -log((1 - 0.1) / 0.1). 02_DESIGN.md quotes it
#: alongside "p = 0.01", which would be -4.595 -- the number, not the probability, is what is
#: specified here and what the CenterNet reference implementation uses.
HEAT_BIAS_INIT: float = -2.19

#: Segmentation logit bias = log(p/(1-p)) at the measured hand-pixel prior of 2.75%
#: (01_DATA.md s2.5 median hand-pixel fraction). Same reasoning as HEAT_BIAS_INIT.
SEG_BIAS_INIT: float = -3.56

_EPS: float = 1e-6


# --------------------------------------------------------------------------------------------
# channel / normalisation helpers
# --------------------------------------------------------------------------------------------
def _ch(c: int, width: float) -> int:
    """Scale a channel count by `width`, rounded to a multiple of 8 (minimum 8).

    Multiples of 8 keep every count even, which IBN-a needs (it halves the channels), and keep
    the tensor cores happy. All nominal counts are already multiples of 8, so `width=1.0`
    reproduces the table in 02_DESIGN.md s3.1 exactly.
    """
    return max(8, int(round(c * width / 8.0)) * 8)


def _base_norm(norm: str) -> str:
    """The plain (non-IBN) norm used wherever IBN-a does not apply.

    IBN-a is defined for the shallow stages of a residual *backbone* only, so the stem, stage 4,
    the decoder and the heads fall back to BatchNorm when `norm="ibn"`.
    """
    return "bn" if norm == "ibn" else norm


def _gn_groups(c: int, target: int = 32) -> int:
    """Group count for GroupNorm: `gcd(32, C)`.

    32 groups is the GN paper's default, but 24 and 48 channels (stem, stage 1) are not divisible
    by 32. Rule used: take the largest divisor of C that also divides 32 -- equivalently the
    largest power of two <= 32 that divides C. Gives 32/32/32/16/8 for 320/192/96/48/24 and
    degrades gracefully for any `width`.
    """
    return math.gcd(target, c)


class IBNorm(nn.Module):
    """IBN-a (Pan et al., ECCV 2018): InstanceNorm on half the channels, BatchNorm on the rest.

    IN removes per-image appearance statistics (illumination, colour cast, sensor response) --
    exactly the RealSense-to-smartphone nuisance we are trying to survive -- while the BN half
    keeps the discriminative content IN would otherwise wash out.
    """

    def __init__(self, c: int) -> None:
        super().__init__()
        self.c_in = c // 2
        self.c_bn = c - self.c_in
        self.inst = nn.InstanceNorm2d(self.c_in, affine=True)
        self.bnorm = nn.BatchNorm2d(self.c_bn)

    def forward(self, x: Tensor) -> Tensor:
        a, b = torch.split(x, [self.c_in, self.c_bn], dim=1)
        return torch.cat([self.inst(a), self.bnorm(b)], dim=1)


def make_norm(kind: str, c: int) -> nn.Module:
    """2-D normalisation layer factory. `kind` in {"bn", "gn", "ibn"}."""
    if kind == "bn":
        return nn.BatchNorm2d(c)
    if kind == "gn":
        return nn.GroupNorm(_gn_groups(c), c)
    if kind == "ibn":
        return IBNorm(c)
    raise ValueError(f"unknown norm {kind!r}; expected one of 'bn', 'gn', 'ibn'")


def _norm1d(kind: str, c: int) -> nn.Module:
    """1-D normalisation for the classifier's (B, C) hidden vector.

    For a tensor with no spatial axis, GroupNorm(g, C) and LayerNorm(C) differ only in the group
    count, and LayerNorm is exactly GroupNorm(1, C). LayerNorm is therefore used for "gn": it is
    the standard batch-independent 1-D choice and avoids inventing a second divisibility rule for
    the head width. "bn" and "ibn" both use BatchNorm1d -- IBN-a is a convolutional construct
    (per-image *spatial* statistics) and has no meaningful 1-D analogue.
    """
    if kind == "gn":
        return nn.LayerNorm(c)
    return nn.BatchNorm1d(c)


# --------------------------------------------------------------------------------------------
# encoder
# --------------------------------------------------------------------------------------------
class BasicBlock(nn.Module):
    """conv3x3 -> Norm -> SiLU -> conv3x3 -> Norm, plus identity, then SiLU.

    The shortcut is projected with a 1x1 conv (+ Norm) whenever the stride or channel count
    changes. When `use_ibn` is set (and `norm == "ibn"`) only the *first* norm is IBN: the IBN-a
    paper places IN on the residual branch's first normalisation and leaves the branch output --
    the tensor that is added to the identity -- as pure BN, because normalising the merged
    identity path is what their IBN-b variant does and it costs accuracy.
    """

    def __init__(
        self, c_in: int, c_out: int, stride: int = 1, norm: str = "bn", use_ibn: bool = False
    ) -> None:
        super().__init__()
        base = _base_norm(norm)
        n1 = "ibn" if (norm == "ibn" and use_ibn) else base
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.norm1 = make_norm(n1, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.norm2 = make_norm(base, c_out)
        self.act = nn.SiLU()
        self.proj: nn.Module | None = None
        if stride != 1 or c_in != c_out:
            self.proj = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride, bias=False), make_norm(base, c_out)
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = x if self.proj is None else self.proj(x)
        y = self.act(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return self.act(y + identity)


class HandNetEncoder(nn.Module):
    """Residual backbone, 02_DESIGN.md s3.1.

    `forward` returns the feature map of every level, coarsest last:
    `[stem(s2), stage1(s4), stage2(s8), stage3(s16), stage4(s32)]`. The U-Net decoder consumes
    all five, so nothing is thrown away.
    """

    #: nominal channels at width=1.0, in stride order (2, 4, 8, 16, 32)
    NOMINAL: tuple[int, ...] = (24, 48, 96, 192, 320)
    #: residual blocks per stage, for stages 1..4
    BLOCKS: tuple[int, ...] = (2, 3, 4, 3)
    out_strides: tuple[int, ...] = (2, 4, 8, 16, 32)

    def __init__(self, norm: str = "bn", width: float = 1.0) -> None:
        super().__init__()
        if norm not in ("bn", "gn", "ibn"):
            raise ValueError(f"unknown norm {norm!r}; expected one of 'bn', 'gn', 'ibn'")
        chs = tuple(_ch(c, width) for c in self.NOMINAL)
        self.out_channels: tuple[int, ...] = chs
        base = _base_norm(norm)

        c0 = chs[0]
        # The stem stays pure BatchNorm under "ibn": IBN-a keeps IN out of conv1 as well as the
        # deep stages, and instance-normalising a 24-channel stride-2 map throws away most of the
        # low-level contrast the rest of the net is built on.
        self.stem = nn.Sequential(
            nn.Conv2d(3, c0, 3, 2, 1, bias=False),
            make_norm(base, c0),
            nn.SiLU(),
            nn.Conv2d(c0, c0, 3, 1, 1, bias=False),
            make_norm(base, c0),
            nn.SiLU(),
        )

        stages: list[nn.Module] = []
        c_prev = c0
        for i, (c_out, n_blocks) in enumerate(zip(chs[1:], self.BLOCKS)):
            # IBN-a in stages 1-3 only (i.e. conv2_x..conv4_x of the paper); stage 4 is pure BN.
            use_ibn = i < 3
            blocks = [BasicBlock(c_prev, c_out, 2, norm, use_ibn)]
            blocks += [BasicBlock(c_out, c_out, 1, norm, use_ibn) for _ in range(n_blocks - 1)]
            stages.append(nn.Sequential(*blocks))
            c_prev = c_out
        #: `stages[i]` is stage i+1, i.e. strides 4, 8, 16, 32.
        self.stages = nn.ModuleList(stages)

    def forward(self, x: Tensor) -> list[Tensor]:
        feats = [self.stem(x)]
        for stage in self.stages:
            feats.append(stage(feats[-1]))
        return feats


# --------------------------------------------------------------------------------------------
# decoder / heads
# --------------------------------------------------------------------------------------------
class UpBlock(nn.Module):
    """One U-Net step: bilinear upsample -> concat lateral -> two (conv3x3 - Norm - SiLU)."""

    def __init__(self, c_in: int, c_lat: int, c_out: int, norm: str) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in + c_lat, c_out, 3, 1, 1, bias=False),
            make_norm(norm, c_out),
            nn.SiLU(),
            nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False),
            make_norm(norm, c_out),
            nn.SiLU(),
        )

    def forward(self, x: Tensor, lateral: Tensor) -> Tensor:
        # Resize to the lateral map's size rather than scale_factor=2, so odd spatial extents
        # (input not a multiple of 32) still concatenate.
        x = F.interpolate(x, size=lateral.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, lateral], dim=1))


def _det_branch(c_in: int, c_out: int) -> nn.Sequential:
    """One detection branch: conv3x3 -> SiLU -> conv1x1 (no norm, as in CenterNet)."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_in, 3, 1, 1), nn.SiLU(), nn.Conv2d(c_in, c_out, 1)
    )


class HandNet(nn.Module):
    """Multi-task network: hand box, hand mask, gesture class (+ confidence from the softmax).

    Args:
        n_classes: gesture classes (10 for this coursework).
        norm: "bn" | "gn" | "ibn" -- the normalisation ablation (Block C in 02_DESIGN.md s8).
        width: multiplier on every channel count.
        use_mask_attn_pool: mask-attended pooling in the classifier (s3.4). `False` is ablation
            A-MAP: global pooling alone, with a correspondingly narrower first Linear.
        heads: subset of ("det", "seg", "cls"). A head not listed builds no parameters and
            contributes no key to the output dict (ablation A-MT). Must be non-empty -- a model
            with no outputs is a caller bug, not a configuration.

    Output dict, with (H, W) the input size and (Hs, Ws) = (H // 4, W // 4):
        "heat" (B,1,Hs,Ws) raw logits | "size" (B,2,Hs,Ws) log(w), log(h) in input pixels |
        "off" (B,2,Hs,Ws) | "seg" (B,1,H,W) raw logits | "cls" (B,n_classes) raw logits
    """

    VALID_HEADS: tuple[str, ...] = ("det", "seg", "cls")
    #: nominal decoder widths at strides 16, 8, 4, 2
    DEC_NOMINAL: tuple[int, ...] = (160, 96, 64, 48)
    CLS_HIDDEN: int = 256

    def __init__(
        self,
        n_classes: int = 10,
        norm: str = "bn",
        width: float = 1.0,
        use_mask_attn_pool: bool = True,
        heads: tuple[str, ...] = ("det", "seg", "cls"),
    ) -> None:
        super().__init__()
        heads = tuple(heads)
        unknown = [h for h in heads if h not in self.VALID_HEADS]
        if unknown:
            raise ValueError(f"unknown head(s) {unknown}; expected a subset of {self.VALID_HEADS}")
        if not heads:
            raise ValueError("`heads` must be non-empty")
        self.heads = heads
        self.n_classes = n_classes
        self.norm_kind = norm

        self.encoder = HandNetEncoder(norm=norm, width=width)
        c_stem, c1, c2, c3, c4 = self.encoder.out_channels
        base = _base_norm(norm)  # decoder + heads are BN under "ibn" (see _base_norm)
        d16, d8, d4, d2 = (_ch(c, width) for c in self.DEC_NOMINAL)

        # The detector hangs off the stride-4 *decoder* feature, so "det" needs the up-path down
        # to stride 4 even when "seg" is absent.
        self._need_dec4 = ("seg" in heads) or ("det" in heads)
        if self._need_dec4:
            self.up3 = UpBlock(c4, c3, d16, base)  # s32 -> s16
            self.up2 = UpBlock(d16, c2, d8, base)  # s16 -> s8
            self.up1 = UpBlock(d8, c1, d4, base)  # s8  -> s4

        if "seg" in heads:
            self.up0 = UpBlock(d4, c_stem, d2, base)  # s4 -> s2
            self.seg_out = nn.Conv2d(d2, 1, 1)

        if "det" in heads:
            # Branch width = the stride-4 decoder width (64 at width=1.0); 02_DESIGN.md fixes the
            # shape of each branch but not its hidden size.
            self.heat_head = _det_branch(d4, 1)
            self.size_head = _det_branch(d4, 2)
            self.off_head = _det_branch(d4, 2)

        if "cls" in heads:
            # Mask-attended pooling needs a predicted mask. Without the "seg" head there is none,
            # so the head silently degrades to plain global pooling and the first Linear is sized
            # for one pooled vector instead of two.
            self.mask_attn = bool(use_mask_attn_pool and "seg" in heads)
            hidden = _ch(self.CLS_HIDDEN, width)
            self.cls_head = nn.Sequential(
                nn.Linear(c4 * (2 if self.mask_attn else 1), hidden),
                _norm1d(base, hidden),
                nn.SiLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden, n_classes),
            )
        else:
            self.mask_attn = False

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # PyTorch's Conv2d default is kaiming_uniform(a=sqrt(5)), which under-scales;
                # fan_out Kaiming is the standard from-scratch residual-net initialisation.
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm2d) and m.affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)
        # Final prediction convs are initialised to near-zero weights so that at step 0 every
        # head emits (almost) exactly its bias: p=0.01 for the heatmap, 0 for size/offset, 0 for the seg logit.
        #
        # This is not cosmetic. A freshly-initialised BatchNorm network in *train* mode whitens
        # every stage, and the accumulated gain drives the stride-4 head logits to +/-27 --
        # measured, not assumed. The focal loss then charges ~9.2 nats for every cell the
        # network confidently and wrongly calls foreground, and the total lands near 800 at
        # step 0 (vs ~4 with this init). Warmup would eventually pull it back, but the first
        # gradients are dominated by noise the head invented. CenterNet zero-inits its heads
        # for exactly this reason.
        for head in ("heat_head", "size_head", "off_head", "seg_out"):
            mod = getattr(self, head, None)
            if mod is None:
                continue
            last = mod[-1] if isinstance(mod, nn.Sequential) else mod
            if isinstance(last, nn.Conv2d):
                # normal(0, 1e-3) rather than exact zeros: with W == 0 the gradient reaching
                # the backbone through this head is identically zero for the first step, so the
                # encoder would sit still while only the head biases move. 1e-3 is small enough
                # that the bias still dominates the output and large enough that gradients flow
                # from step 0.
                nn.init.normal_(last.weight, std=1e-3)
                if last.bias is not None:
                    nn.init.zeros_(last.bias)
        if "det" in self.heads:
            nn.init.constant_(self.heat_head[-1].bias, HEAT_BIAS_INIT)
        if "seg" in self.heads:
            # Same prior trick for the mask: the hand covers a median 2.75% of the image
            # (01_DATA.md), so start the logit at log(0.0275 / 0.9725) rather than at 0.5
            # probability. Saves the first epoch from unlearning a 50% foreground prior.
            nn.init.constant_(self.seg_out.bias, SEG_BIAS_INIT)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        f_stem, f1, f2, f3, f4 = self.encoder(x)
        out: dict[str, Tensor] = {}

        p4: Tensor | None = None
        if self._need_dec4:
            p4 = self.up1(self.up2(self.up3(f4, f3), f2), f1)  # stride 4

        if "seg" in self.heads:
            assert p4 is not None
            seg = self.seg_out(self.up0(p4, f_stem))  # stride 2
            out["seg"] = F.interpolate(
                seg, size=x.shape[-2:], mode="bilinear", align_corners=False
            )

        if "det" in self.heads:
            assert p4 is not None
            out["heat"] = self.heat_head(p4)  # RAW logits; the focal loss applies the sigmoid
            out["size"] = self.size_head(p4)
            out["off"] = self.off_head(p4)

        if "cls" in self.heads:
            g = f4.mean(dim=(2, 3))  # (B, C4) whole-image context
            if self.mask_attn:
                # Detached: a bad mask early in training must not be able to destabilise the
                # classifier, and the classification loss must not be able to reshape the mask.
                # The *predicted* mask is used at train time so there is no train/test mismatch.
                m = torch.sigmoid(out["seg"]).detach()
                m4 = F.adaptive_avg_pool2d(m, f4.shape[-2:])  # (B,1,h32,w32)
                a = (f4 * m4).sum(dim=(2, 3)) / (m4.sum(dim=(2, 3)) + _EPS)
                feat = torch.cat([g, a], dim=1)
            else:
                feat = g
            out["cls"] = self.cls_head(feat)

        return out


# --------------------------------------------------------------------------------------------
# detection decoding
# --------------------------------------------------------------------------------------------
@torch.no_grad()
def decode_detection(out: dict[str, Tensor], k: int = 1) -> tuple[Tensor, Tensor]:
    """Turn the centre-point head's outputs into boxes in **input-frame pixels**.

    sigmoid -> 3x3 max-pool peak suppression (a cell survives iff it equals its 3x3 max, the
    CenterNet stand-in for NMS) -> top-k peaks per image -> gather `off` and `size` there ->
    `(x1, y1, x2, y2)`, clamped to the image.

    Shapes:
        k == 1 -> boxes (B, 4), scores (B,)      [the frozen INTERFACES.md contract]
        k >  1 -> boxes (B, k, 4), scores (B, k), peaks ordered by descending score.

    Image bounds are taken from `out["seg"]` when present (that map is at input resolution) and
    otherwise from `Hs * DET_STRIDE, Ws * DET_STRIDE`. These agree whenever the input size is a
    multiple of 32, which 384x288 and 512x384 are.

    All arithmetic is forced to float32: under bf16 autocast `exp(log w)` would otherwise carry
    ~3 significant bits, which is visible in the IoU.
    """
    heat = out["heat"].float().sigmoid()
    b, _, hs, ws = heat.shape
    keep = F.max_pool2d(heat, kernel_size=3, stride=1, padding=1) == heat
    heat = heat * keep

    k = max(1, min(int(k), hs * ws))
    scores, idx = heat.reshape(b, -1).topk(k, dim=1)  # (B,k)
    ys = torch.div(idx, ws, rounding_mode="floor").float()
    xs = (idx % ws).float()

    gather_idx = idx.unsqueeze(1).expand(b, 2, k)
    off = out["off"].float().reshape(b, 2, -1).gather(2, gather_idx)  # (B,2,k)
    size = out["size"].float().reshape(b, 2, -1).gather(2, gather_idx)

    cx = (xs + off[:, 0]) * DET_STRIDE
    cy = (ys + off[:, 1]) * DET_STRIDE
    w = size[:, 0].exp()
    h = size[:, 1].exp()

    if "seg" in out:
        img_h, img_w = out["seg"].shape[-2:]
    else:
        img_h, img_w = hs * DET_STRIDE, ws * DET_STRIDE

    boxes = torch.stack(
        [
            (cx - w * 0.5).clamp(0, img_w),
            (cy - h * 0.5).clamp(0, img_h),
            (cx + w * 0.5).clamp(0, img_w),
            (cy + h * 0.5).clamp(0, img_h),
        ],
        dim=-1,
    )  # (B,k,4)

    if k == 1:
        return boxes[:, 0], scores[:, 0]
    return boxes, scores
