"""Tests for `src/model.py`. Plain asserts, no pytest.

Run from the project root:
    python tests/test_model.py
"""

from __future__ import annotations

import itertools
import math
import os
import sys

import torch
import torch.nn as nn

# INTERFACES.md: the package root is `project_18006111_Shihab/` and modules import as `from src...`
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."),
)

from src.model import (  # noqa: E402
    DET_STRIDE,
    HEAT_BIAS_INIT,
    HandNet,
    HandNetEncoder,
    decode_detection,
)

SMALL = (96, 128)  # (H, W), divisible by 32 -- keeps the CPU test run quick
ALL_KEYS = {"heat", "size", "off", "seg", "cls"}
HEAD_KEYS = {"det": {"heat", "size", "off"}, "seg": {"seg"}, "cls": {"cls"}}


def _has(module: nn.Module, cls: type) -> bool:
    return any(isinstance(m, cls) for m in module.modules())


# --------------------------------------------------------------------------------------------
def test_forward_shapes() -> None:
    """Every output key has the exact shape the interface contract promises, at two sizes."""
    model = HandNet().eval()
    for h, w in [(288, 384), (384, 512)]:
        x = torch.randn(2, 3, h, w)
        with torch.no_grad():
            out = model(x)
        assert set(out) == ALL_KEYS, set(out)
        hs, ws = h // DET_STRIDE, w // DET_STRIDE
        assert out["heat"].shape == (2, 1, hs, ws), out["heat"].shape
        assert out["size"].shape == (2, 2, hs, ws), out["size"].shape
        assert out["off"].shape == (2, 2, hs, ws), out["off"].shape
        assert out["seg"].shape == (2, 1, h, w), out["seg"].shape  # INPUT resolution
        assert out["cls"].shape == (2, 10), out["cls"].shape
        for k, v in out.items():
            assert torch.isfinite(v).all(), f"non-finite {k}"

    # encoder returns all five levels at the documented strides / channels
    enc = HandNetEncoder().eval()
    with torch.no_grad():
        feats = enc(torch.randn(1, 3, 288, 384))
    assert len(feats) == 5
    assert enc.out_channels == (24, 48, 96, 192, 320), enc.out_channels
    for f, s, c in zip(feats, enc.out_strides, enc.out_channels):
        assert f.shape == (1, c, 288 // s, 384 // s), (f.shape, s, c)
    print("ok  forward shapes @ 384x288 and 512x384")


def test_n_classes_and_width() -> None:
    """`n_classes` and `width` propagate; the GroupNorm fallback survives odd channel counts."""
    m = HandNet(n_classes=7, width=0.5, norm="gn").eval()
    assert m.encoder.out_channels == (16, 24, 48, 96, 160), m.encoder.out_channels
    with torch.no_grad():
        out = m(torch.randn(2, 3, *SMALL))
    assert out["cls"].shape == (2, 7)
    small = sum(p.numel() for p in m.parameters())
    full = sum(p.numel() for p in HandNet(n_classes=7, norm="gn").parameters())
    assert small < full, (small, full)
    print(f"ok  width scaling (0.5 -> {small/1e6:.2f}M vs 1.0 -> {full/1e6:.2f}M params)")


def test_norm_variants() -> None:
    """All three norms build and run; IBN-a lands in stages 1-3 and nowhere else."""
    x = torch.randn(2, 3, *SMALL)
    for norm in ("bn", "gn", "ibn"):
        m = HandNet(norm=norm).eval()
        with torch.no_grad():
            out = m(x)
        assert set(out) == ALL_KEYS
        assert all(torch.isfinite(v).all() for v in out.values()), norm

    bn = HandNet(norm="bn")
    assert not _has(bn, nn.InstanceNorm2d)
    assert _has(bn, nn.BatchNorm2d) and _has(bn, nn.BatchNorm1d)

    gn = HandNet(norm="gn")
    assert _has(gn, nn.GroupNorm)
    # A GN model must be fully batch-independent, head included (hence LayerNorm, not BatchNorm1d)
    assert not _has(gn, nn.BatchNorm2d) and not _has(gn, nn.BatchNorm1d)
    assert not _has(gn, nn.InstanceNorm2d)
    groups = {n: mod.num_groups for n, mod in gn.named_modules() if isinstance(mod, nn.GroupNorm)}
    assert set(groups.values()) <= {8, 16, 32}, groups  # gcd(32, C) for C in 24/48/96/192/320/...

    ibn = HandNet(norm="ibn")
    for i in range(3):  # stages 1, 2, 3
        assert _has(ibn.encoder.stages[i], nn.InstanceNorm2d), f"no IN in stage {i+1}"
    assert not _has(ibn.encoder.stages[3], nn.InstanceNorm2d), "IN leaked into stage 4"
    assert not _has(ibn.encoder.stem, nn.InstanceNorm2d), "IN leaked into the stem"
    # IBN-a replaces only the first norm of a block; norm2 and the projection stay BatchNorm.
    blk = ibn.encoder.stages[0][0]
    assert _has(blk.norm1, nn.InstanceNorm2d) and _has(blk.norm1, nn.BatchNorm2d)
    assert isinstance(blk.norm2, nn.BatchNorm2d)
    # and half the channels each way
    inorm = next(m for m in blk.norm1.modules() if isinstance(m, nn.InstanceNorm2d))
    assert inorm.num_features == 48 // 2, inorm.num_features
    print("ok  norm variants bn/gn/ibn; IBN-a in stages 1-3 only, first norm only")


def test_head_subsets() -> None:
    """Every non-empty subset of `heads` builds, runs, and emits exactly its own keys."""
    x = torch.randn(2, 3, *SMALL)
    names = ("det", "seg", "cls")
    subsets = [
        s
        for r in range(1, 4)
        for s in itertools.combinations(names, r)
    ]
    assert len(subsets) == 7
    for heads in subsets:
        m = HandNet(heads=heads).eval()
        with torch.no_grad():
            out = m(x)
        expected: set[str] = set()
        for h in heads:
            expected |= HEAD_KEYS[h]
        assert set(out) == expected, (heads, set(out))
        assert all(torch.isfinite(v).all() for v in out.values()), heads
        # absent heads own no parameters
        if "seg" not in heads:
            assert not hasattr(m, "seg_out")
        if "det" not in heads:
            assert not hasattr(m, "heat_head")
        if "cls" not in heads:
            assert not hasattr(m, "cls_head")
        if heads == ("det",):  # up-path only as far as stride 4
            assert not hasattr(m, "up0")

    # cls alone: no seg map exists, so mask-attended pooling must fall back to global pooling
    m = HandNet(heads=("cls",), use_mask_attn_pool=True)
    assert m.mask_attn is False, "mask attention must be disabled without a seg head"
    assert m.cls_head[0].in_features == 320, m.cls_head[0].in_features
    assert not hasattr(m, "up3")  # no decoder at all
    m_full = HandNet(heads=("seg", "cls"), use_mask_attn_pool=True)
    assert m_full.mask_attn is True and m_full.cls_head[0].in_features == 640

    for bad in [(), ("det", "pose"), ("depth",)]:
        try:
            HandNet(heads=bad)  # type: ignore[arg-type]
        except ValueError:
            pass
        else:
            raise AssertionError(f"heads={bad} should have raised ValueError")
    for bad_norm in ("in", "ln", ""):
        try:
            HandNet(norm=bad_norm)
        except ValueError:
            pass
        else:
            raise AssertionError(f"norm={bad_norm!r} should have raised ValueError")
    print("ok  all 7 head subsets; ('cls',) falls back to global pooling; bad configs raise")


def test_mask_attn_pool_ablation() -> None:
    """A-MAP: use_mask_attn_pool=False runs and is strictly smaller."""
    on = HandNet(use_mask_attn_pool=True)
    off = HandNet(use_mask_attn_pool=False)
    n_on = sum(p.numel() for p in on.parameters())
    n_off = sum(p.numel() for p in off.parameters())
    assert n_off < n_on, (n_off, n_on)
    # the only difference is the first Linear's input width: 640 -> 320 over 256 hidden units
    assert on.cls_head[0].in_features == 640 and off.cls_head[0].in_features == 320
    assert n_on - n_off == 320 * 256, n_on - n_off
    with torch.no_grad():
        out = off.eval()(torch.randn(2, 3, *SMALL))
    assert out["cls"].shape == (2, 10) and torch.isfinite(out["cls"]).all()
    print(f"ok  A-MAP ablation: {n_on:,} -> {n_off:,} params (-{n_on - n_off:,})")


def test_heat_bias_init() -> None:
    """The focal prior bias is actually on the final heat conv, and only there."""
    m = HandNet()
    assert torch.allclose(
        m.heat_head[-1].bias, torch.full_like(m.heat_head[-1].bias, HEAT_BIAS_INIT)
    )
    assert abs(HEAT_BIAS_INIT + math.log((1 - 0.1) / 0.1)) < 1e-2  # p = 0.1, the CenterNet value
    assert torch.allclose(m.size_head[-1].bias, torch.zeros_like(m.size_head[-1].bias))
    assert torch.allclose(m.off_head[-1].bias, torch.zeros_like(m.off_head[-1].bias))
    with torch.no_grad():
        p = torch.sigmoid(m(torch.randn(2, 3, *SMALL))["heat"]).mean().item()
    assert 0.02 < p < 0.30, p  # background-dominated at init, as intended
    print(f"ok  heat bias init {HEAT_BIAS_INIT} -> mean initial heat prob {p:.3f}")


def test_gradients() -> None:
    """Backward from the sum of all outputs gives finite, non-zero grads and no NaNs."""
    torch.manual_seed(0)
    for norm in ("bn", "gn", "ibn"):
        m = HandNet(norm=norm).train()
        x = torch.randn(2, 3, *SMALL, requires_grad=True)
        out = m(x)
        loss = sum(v.float().sum() for v in out.values())
        loss.backward()

        stem_conv = m.encoder.stem[0]
        assert isinstance(stem_conv, nn.Conv2d)
        g = stem_conv.weight.grad
        assert g is not None, f"{norm}: no grad on the stem conv"
        assert torch.isfinite(g).all(), f"{norm}: non-finite stem grad"
        assert g.abs().sum().item() > 0, f"{norm}: zero stem grad"

        n_none = 0
        for name, p in m.named_parameters():
            if p.grad is None:
                n_none += 1
                continue
            assert torch.isfinite(p.grad).all(), f"{norm}: non-finite grad on {name}"
        assert n_none == 0, f"{norm}: {n_none} parameters got no gradient"
        assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0

    # the classification path must NOT push gradient back through the mask
    m = HandNet().train()
    out = m(torch.randn(2, 3, *SMALL))
    out["cls"].sum().backward()
    assert m.seg_out.weight.grad is None or m.seg_out.weight.grad.abs().sum().item() == 0.0, (
        "sigmoid(seg) into the classifier is not detached"
    )
    print("ok  gradients finite and non-zero for bn/gn/ibn; mask attention is detached")


def test_decode_detection() -> None:
    """Hand-built heatmap -> a box computed by hand."""
    hs, ws = 24, 32
    h, w = hs * DET_STRIDE, ws * DET_STRIDE  # 96 x 128

    heat = torch.full((1, 1, hs, ws), -10.0)
    # a distinctive background so a mis-indexed gather is obvious rather than plausible
    size = torch.full((1, 2, hs, ws), math.log(500.0))
    off = torch.full((1, 2, hs, ws), 0.9)

    heat[0, 0, 5, 7] = 2.0  # peak A (higher)
    size[0, 0, 5, 7] = math.log(40.0)
    size[0, 1, 5, 7] = math.log(30.0)
    off[0, 0, 5, 7] = 0.25
    off[0, 1, 5, 7] = 0.5

    heat[0, 0, 15, 20] = 1.0  # peak B (lower)
    size[0, 0, 15, 20] = math.log(10.0)
    size[0, 1, 15, 20] = math.log(10.0)
    off[0, 0, 15, 20] = 0.0
    off[0, 1, 15, 20] = 0.0

    out = {"heat": heat.requires_grad_(True), "size": size, "off": off}

    # by hand: cx = (7 + 0.25) * 4 = 29, cy = (5 + 0.5) * 4 = 22, w = 40, h = 30
    box_a = torch.tensor([[29.0 - 20, 22.0 - 15, 29.0 + 20, 22.0 + 15]])
    # by hand: cx = 20 * 4 = 80, cy = 15 * 4 = 60, w = h = 10
    box_b = torch.tensor([[75.0, 55.0, 85.0, 65.0]])

    boxes, scores = decode_detection(out, k=1)
    assert boxes.shape == (1, 4) and scores.shape == (1,), (boxes.shape, scores.shape)
    assert torch.allclose(boxes, box_a, atol=1e-4), boxes
    assert abs(scores.item() - torch.sigmoid(torch.tensor(2.0)).item()) < 1e-6, scores
    assert not boxes.requires_grad and not scores.requires_grad  # @torch.no_grad()

    # k > 1: (B,k,4) / (B,k), descending, and the lower peak is B -- so at k=1 it was skipped
    boxes2, scores2 = decode_detection(out, k=2)
    assert boxes2.shape == (1, 2, 4) and scores2.shape == (1, 2), boxes2.shape
    assert scores2[0, 0] > scores2[0, 1]
    assert torch.allclose(boxes2[:, 0], box_a, atol=1e-4)
    assert torch.allclose(boxes2[:, 1], box_b, atol=1e-4), boxes2[:, 1]
    assert abs(scores2[0, 1].item() - torch.sigmoid(torch.tensor(1.0)).item()) < 1e-6

    # 3x3 peak suppression: a neighbour of A, even a strong one, is not returned as a 2nd peak
    heat_n = heat.detach().clone()
    heat_n[0, 0, 5, 8] = 1.5  # adjacent to A, higher than B
    _, s_n = decode_detection({"heat": heat_n, "size": size, "off": off}, k=2)
    assert abs(s_n[0, 1].item() - torch.sigmoid(torch.tensor(1.0)).item()) < 1e-6, s_n

    # clamping at both borders, batch of 2
    heat_c = torch.full((2, 1, hs, ws), -10.0)
    size_c = torch.full((2, 2, hs, ws), math.log(100.0))
    size_c[:, 1] = math.log(80.0)
    off_c = torch.zeros((2, 2, hs, ws))
    heat_c[0, 0, 0, 0] = 3.0  # centre at (0, 0)      -> x1, y1 clamp to 0
    heat_c[1, 0, hs - 1, ws - 1] = 3.0
    off_c[1, :, hs - 1, ws - 1] = 0.75  # centre at (127, 95) -> x2, y2 clamp to (128, 96)
    boxes_c, _ = decode_detection({"heat": heat_c, "size": size_c, "off": off_c}, k=1)
    assert torch.allclose(boxes_c[0], torch.tensor([0.0, 0.0, 50.0, 40.0]), atol=1e-4), boxes_c[0]
    assert torch.allclose(
        boxes_c[1], torch.tensor([77.0, 55.0, float(w), float(h)]), atol=1e-4
    ), boxes_c[1]
    assert (boxes_c[:, 0] >= 0).all() and (boxes_c[:, 2] <= w).all()
    assert (boxes_c[:, 1] >= 0).all() and (boxes_c[:, 3] <= h).all()

    # k is clipped to the number of cells; "seg" (when present) sets the image bounds
    b_big, s_big = decode_detection({"heat": heat_c, "size": size_c, "off": off_c}, k=10**6)
    assert b_big.shape == (2, hs * ws, 4) and s_big.shape == (2, hs * ws)

    # end to end off a real model: boxes stay inside the image
    model = HandNet().eval()
    with torch.no_grad():
        o = model(torch.randn(2, 3, h, w))
    bx, sc = decode_detection(o, k=1)
    assert bx.shape == (2, 4) and sc.shape == (2,)
    assert (bx[:, 0] >= 0).all() and (bx[:, 2] <= w).all() and (bx[:, 3] <= h).all()
    assert ((sc >= 0) & (sc <= 1)).all()
    print("ok  decode_detection: hand-checked box, peak suppression, top-k, clamping")


def test_autocast_bf16() -> None:
    """The whole model runs under CPU bf16 autocast without dtype errors."""
    x = torch.randn(2, 3, *SMALL)
    for norm in ("bn", "gn", "ibn"):
        m = HandNet(norm=norm).train()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = m(x)
            loss = sum(v.float().sum() for v in out.values())
        loss.backward()
        assert out["seg"].shape == (2, 1, *SMALL)
        assert all(torch.isfinite(v.float()).all() for v in out.values()), norm
        for name, p in m.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), f"{norm}/{name}"
        with torch.autocast("cpu", dtype=torch.bfloat16), torch.no_grad():
            o = m.eval()(x)
        bx, sc = decode_detection(o, k=1)
        assert bx.dtype == torch.float32 and torch.isfinite(bx).all(), bx.dtype
    print("ok  autocast('cpu', bfloat16) forward + backward + decode")


# --------------------------------------------------------------------------------------------
def main() -> int:
    torch.manual_seed(0)
    tests = [
        test_forward_shapes,
        test_n_classes_and_width,
        test_norm_variants,
        test_head_subsets,
        test_mask_attn_pool_ablation,
        test_heat_bias_init,
        test_gradients,
        test_decode_detection,
        test_autocast_bf16,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {type(exc).__name__}: {exc}")
            import traceback

            traceback.print_exc()

    m = HandNet()
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    enc = sum(p.numel() for p in m.encoder.parameters())
    dec = sum(
        p.numel()
        for n in ("up3", "up2", "up1", "up0", "seg_out")
        for p in getattr(m, n).parameters()
    )
    det = sum(p.numel() for n in ("heat_head", "size_head", "off_head")
              for p in getattr(m, n).parameters())
    cls = sum(p.numel() for p in m.cls_head.parameters())
    print("-" * 78)
    print(f"HandNet(width=1.0, norm='bn')  parameters: {total:,} ({total/1e6:.2f} M)")
    print(f"  encoder {enc:,} | seg decoder {dec:,} | det heads {det:,} | cls head {cls:,}")
    print(f"  trainable: {trainable:,}")
    print("-" * 78)
    print("ALL TESTS PASSED" if failed == 0 else f"{failed} TEST(S) FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
