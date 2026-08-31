"""Tests for src/utils.py. Plain asserts, no pytest: `python tests/test_utils.py`.

Every expected value below is hand-computed in the comment beside it. Where an aggregate is
awkward to write out (the segmentation means over a 30-record stream) an independent reference
is recomputed from the raw records rather than from the implementation under test.
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import utils  # noqa: E402
from src.utils import MetricAccumulator  # noqa: E402

TOL = 1e-6


def close(a: float, b: float, tol: float = TOL) -> bool:
    return abs(float(a) - float(b)) <= tol


# ----------------------------------------------------------------------------- geometry
def test_mask_to_box() -> None:
    m = np.zeros((10, 12), np.uint8)
    m[2:5, 3:8] = 255                       # rows 2..4 (3), cols 3..7 (5)
    box = utils.mask_to_box(m)
    assert box is not None and box.dtype == np.float32 and box.shape == (4,)
    assert box.tolist() == [3.0, 2.0, 8.0, 5.0], box
    assert box[2] - box[0] == 5.0 and box[3] - box[1] == 3.0     # exclusive-extent convention
    # the box area must equal the pixel count of a filled rectangle
    assert (box[2] - box[0]) * (box[3] - box[1]) == float((m >= 128).sum())

    # the point of the exclusive convention: box IoU == pixel IoU for two rectangle masks
    m2 = np.zeros((10, 12), np.uint8)
    m2[3:9, 5:11] = 255
    b1, b2 = utils.mask_to_box(m), utils.mask_to_box(m2)
    f1, f2 = m >= 128, m2 >= 128
    pixel_iou = (f1 & f2).sum() / (f1 | f2).sum()
    assert close(utils.box_iou(torch.from_numpy(b1), torch.from_numpy(b2)).item(), pixel_iou)

    assert utils.mask_to_box(np.zeros((4, 4), np.uint8)) is None            # no foreground
    grey = np.zeros((6, 6), np.uint8)
    grey[1:3, 1:3] = 100                                                    # below threshold
    assert utils.mask_to_box(grey, thresh=128) is None
    assert utils.mask_to_box(grey, thresh=50).tolist() == [1.0, 1.0, 3.0, 3.0]
    single = np.zeros((5, 5), np.uint8)
    single[4, 0] = 255
    assert utils.mask_to_box(single).tolist() == [0.0, 4.0, 1.0, 5.0]        # 1x1 box
    try:
        utils.mask_to_box(np.zeros((2, 2, 3), np.uint8))
        raise AssertionError("expected ValueError on a 3-D mask")
    except ValueError:
        pass


def test_box_iou() -> None:
    a = torch.tensor([[0., 0., 2., 2.],     # identical -> 1
                      [0., 0., 2., 2.],     # half overlap: inter 1*2=2, union 4+4-2=6 -> 1/3
                      [0., 0., 4., 4.],     # inter 2*2=4, union 16+16-4=28 -> 1/7
                      [0., 0., 1., 1.]])    # disjoint -> 0
    b = torch.tensor([[0., 0., 2., 2.],
                      [1., 0., 3., 2.],
                      [2., 2., 6., 6.],
                      [5., 5., 6., 6.]])
    got = utils.box_iou(a, b)
    assert got.shape == (4,)
    for g, e in zip(got.tolist(), [1.0, 1.0 / 3.0, 1.0 / 7.0, 0.0]):
        assert close(g, e), (got, e)

    # degenerate boxes must give 0, never NaN
    deg = utils.box_iou(torch.tensor([[0., 0., 0., 0.], [3., 3., 1., 1.]]),
                        torch.tensor([[0., 0., 2., 2.], [0., 0., 2., 2.]]))
    assert torch.isfinite(deg).all() and close(deg[0], 0.0) and close(deg[1], 0.0)
    assert close(utils.box_iou(torch.zeros(4), torch.zeros(4)), 0.0)   # (4,) -> scalar
    assert torch.isfinite(utils.box_iou(torch.zeros(2, 4), torch.zeros(2, 4))).all()


def test_box_giou_loss() -> None:
    same = torch.tensor([[1., 2., 5., 9.]])
    assert close(utils.box_giou_loss(same, same).item(), 0.0)                # identical -> 0

    # nested: inter 36, union 100, iou .36; enclosing = pred so the penalty term is 0
    nested = utils.box_giou_loss(torch.tensor([[0., 0., 10., 10.]]),
                                 torch.tensor([[2., 2., 8., 8.]])).item()
    assert close(nested, 1.0 - 0.36)

    # disjoint: iou 0, union 2, C = 3*3 = 9 -> giou = -7/9 -> loss = 16/9
    disj = utils.box_giou_loss(torch.tensor([[0., 0., 1., 1.]]),
                               torch.tensor([[2., 2., 3., 3.]])).item()
    assert close(disj, 16.0 / 9.0) and 1.0 < disj <= 2.0

    loss = utils.box_giou_loss(torch.tensor([[0., 0., 2., 2.], [0., 0., 1., 1.]]),
                               torch.tensor([[1., 0., 3., 2.], [9., 9., 10., 10.]]))
    assert loss.shape == (2,) and (loss >= 0).all() and (loss <= 2).all()

    # gradients flow, are finite, and match finite differences
    pred = torch.tensor([[0.3, 0.6, 4.1, 5.2]], dtype=torch.double, requires_grad=True)
    tgt = torch.tensor([[1.0, 1.0, 5.0, 6.0]], dtype=torch.double)
    assert torch.autograd.gradcheck(lambda p: utils.box_giou_loss(p, tgt), (pred,),
                                    eps=1e-6, atol=1e-6)
    out = utils.box_giou_loss(pred, tgt).sum()
    out.backward()
    assert torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0
    h = 1e-6
    with torch.no_grad():
        bumped = pred.detach().clone()
        bumped[0, 0] += h
        fd = (utils.box_giou_loss(bumped, tgt).sum() - out).item() / h
    assert close(fd, pred.grad[0, 0].item(), 1e-4), (fd, pred.grad[0, 0].item())

    # degenerate (point) boxes stay finite thanks to the clamp on the enclosing area
    assert torch.isfinite(utils.box_giou_loss(torch.zeros(1, 4), torch.zeros(1, 4))).all()

    # bf16 in (autocast) must not poison the ratio: promoted to fp32 internally
    big = torch.tensor([[100., 100., 260., 300.]])
    bf = utils.box_giou_loss(big.to(torch.bfloat16), big.to(torch.bfloat16))
    assert bf.dtype == torch.float32 and close(bf.item(), 0.0)
    assert close(utils.box_iou(big.to(torch.bfloat16), big.to(torch.bfloat16)).item(), 1.0)
    assert close(utils.box_iou(big.to(torch.int32), big.to(torch.int32)).item(), 1.0)


def test_gaussian_radius() -> None:
    # 100x100 box at overlap 0.7: case1 9.2515, case2 8.1670, case3 9.7619 -> min 8.1670
    assert close(utils.gaussian_radius(100, 100, 0.7), (100 - math.sqrt(7000)) / 2.0, 1e-9)
    assert close(utils.gaussian_radius(100, 100, 1.0), 0.0)          # perfect overlap -> r = 0
    assert utils.gaussian_radius(200, 200, 0.7) > utils.gaussian_radius(100, 100, 0.7)
    assert utils.gaussian_radius(100, 100, 0.5) > utils.gaussian_radius(100, 100, 0.9)
    for h in (0.0, 1.0, 5.0, 91.0, 480.0):                            # never NaN / negative
        for w in (0.0, 1.0, 7.0, 69.0, 640.0):
            for o in (0.1, 0.3, 0.7, 0.95, 0.999, 1.0):
                r = utils.gaussian_radius(h, w, o)
                assert math.isfinite(r) and r >= 0.0, (h, w, o, r)
                assert r <= max(h, w) + 1.0, (h, w, o, r)


def test_draw_gaussian() -> None:
    heat = np.zeros((21, 21), np.float32)
    utils.draw_gaussian(heat, cx=10, cy=10, radius=3)
    assert close(heat[10, 10], 1.0)                                   # peak is exactly 1
    assert heat.max() == 1.0
    for d in (1, 2, 3):                                               # 4-fold symmetry
        v = heat[10, 10 + d]
        assert close(heat[10, 10 - d], v) and close(heat[10 + d, 10], v) and close(heat[10 - d, 10], v)
    assert close(heat[8, 9], heat[12, 11]) and heat[10, 11] > heat[10, 13]
    assert heat[10, 10 + 4] == 0.0                                    # support is the radius

    corner = np.zeros((10, 10), np.float32)                           # clipped at a corner
    utils.draw_gaussian(corner, cx=0, cy=0, radius=3)
    assert close(corner[0, 0], 1.0)
    assert (corner[4:, :] == 0).all() and (corner[:, 4:] == 0).all()
    assert close(corner[0, 1], heat[10, 11]) and close(corner[1, 0], heat[11, 10])

    ref_a, ref_b = np.zeros((21, 21), np.float32), np.zeros((21, 21), np.float32)
    utils.draw_gaussian(ref_a, 5, 5, 2)
    utils.draw_gaussian(ref_b, 8, 5, 4)
    merged = np.zeros((21, 21), np.float32)
    utils.draw_gaussian(merged, 5, 5, 2)
    utils.draw_gaussian(merged, 8, 5, 4)
    assert np.allclose(merged, np.maximum(ref_a, ref_b))              # max-merge, not overwrite

    floor = np.full((9, 9), 0.9, np.float32)
    utils.draw_gaussian(floor, 4, 4, 2)
    assert close(floor[4, 4], 1.0) and close(floor[0, 0], 0.9) and floor.min() == 0.9

    dot = np.zeros((5, 5), np.float32)
    utils.draw_gaussian(dot, 2, 2, 0)                                 # radius 0 -> one pixel
    assert close(dot[2, 2], 1.0) and close(dot.sum(), 1.0)

    outside = np.zeros((5, 5), np.float32)
    utils.draw_gaussian(outside, 9, 9, 2)
    utils.draw_gaussian(outside, -1, 2, 2)
    assert (outside == 0).all()                                       # centre off-grid -> no-op


# --------------------------------------------------------------------------- segmentation
def test_seg_scores() -> None:
    n, hw = 5, 4
    pred = np.zeros((n, 1, hw, hw), np.float32)
    tgt = np.zeros((n, 1, hw, hw), np.float32)
    # 0: gt 2x2 at (0,0), pred 2x2 at (1,1) -> inter 1, union 7, |p|=|g|=4
    tgt[0, 0, 0:2, 0:2] = 1.0
    pred[0, 0, 1:3, 1:3] = 0.9
    # 1: both empty   2: gt empty, pred 4 px   3: gt 4 px, pred empty   4: both full
    pred[2, 0, 0:2, 0:2] = 1.0
    tgt[3, 0, 0:2, 0:2] = 1.0
    pred[4] = 1.0
    tgt[4] = 1.0
    s = utils.seg_scores(torch.from_numpy(pred), torch.from_numpy(tgt))
    assert set(s) == {"iou_hand", "iou_bg", "miou", "dice"}
    assert all(v.shape == (n,) for v in s.values())

    assert close(s["iou_hand"][0], 1.0 / 7.0)                 # 1 / 7
    assert close(s["dice"][0], 0.25)                          # 2*1 / (4+4)
    assert close(s["iou_bg"][0], 9.0 / 15.0)                  # (16-7) / (16-1)
    assert close(s["miou"][0], 0.5 * (1.0 / 7.0 + 0.6))

    for k in ("iou_hand", "iou_bg", "miou", "dice"):          # empty & empty -> perfect
        assert close(s[k][1], 1.0), (k, s[k][1])
        assert close(s[k][4], 1.0), (k, s[k][4])              # all-hand image -> iou_bg = 1
    for i in (2, 3):                                          # exactly one side empty -> 0
        assert close(s["iou_hand"][i], 0.0) and close(s["dice"][i], 0.0)
        assert close(s["iou_bg"][i], 12.0 / 16.0) and close(s["miou"][i], 0.375)

    thr = utils.seg_scores(torch.tensor([[[[0.49, 0.51]]]]), torch.tensor([[[[0.0, 1.0]]]]))
    assert close(thr["iou_hand"][0], 1.0) and close(thr["dice"][0], 1.0)   # threshold at 0.5
    try:
        utils.seg_scores(torch.zeros(2, 1, 4, 4), torch.zeros(2, 1, 5, 5))
        raise AssertionError("expected ValueError on mismatched shapes")
    except ValueError:
        pass


# ------------------------------------------------------------------------------- metrics
def _records() -> list[dict]:
    """30 frames / 3 clips with a hand-designed confusion pattern.

    clipA: 10 frames of class 0, 8 correct + 2 predicted as 1
    clipB: 10 frames of class 1, all correct
    clipC: 10 frames of class 2, 5 correct + 5 predicted as 0
    box_iou cycles 0.6 / 0.4 / None; every 5th frame carries no seg dict.
    """
    plan = [("clipA", 0, [0] * 8 + [1] * 2), ("clipB", 1, [1] * 10), ("clipC", 2, [2] * 5 + [0] * 5)]
    out, i = [], 0
    for clip, gt, preds in plan:
        for p in preds:
            seg = None if i % 5 == 0 else {"iou_hand": 0.30 + 0.01 * i,
                                           "iou_bg": 0.90, "dice": 0.50 + 0.01 * i}
            out.append(dict(clip_key=clip, box_iou={0: 0.6, 1: 0.4, 2: None}[i % 3], seg=seg,
                            gt_class=gt, pred_class=p,
                            cls_conf=0.9 if p == gt else 0.4, det_conf=0.5))
            i += 1
    return out


def test_metric_accumulator() -> None:
    acc = MetricAccumulator()
    recs = _records()
    for r in recs:
        acc.add(**r)
    s = acc.summary()
    f, c = s["frame"], s["clip"]

    assert s["n_frames"] == 30 and s["n_clips"] == 3 and len(acc) == 30

    # detection: 20 scored (10 x 0.6, 10 x 0.4), 10 excluded -- NOT counted as zero
    assert f["n_boxes_scored"] == 20 and f["n_boxes_skipped"] == 10
    assert close(f["det_acc@0.5"], 0.5) and not close(f["det_acc@0.5"], 10 / 30)
    assert close(f["mean_box_iou"], 0.5) and not close(f["mean_box_iou"], 10.0 / 30.0)
    # per clip: acc 4/7, 3/7, 3/6 and iou 3.6/7, 3.4/7, 3.0/6 -> both average to exactly 0.5
    assert close(c["det_acc@0.5"], 0.5) and close(c["mean_box_iou"], 0.5)
    assert c["n_boxes_scored"] == 3 and c["n_boxes_skipped"] == 0

    # segmentation: 6 of 30 frames carry no seg dict and are excluded
    kept = [r["seg"] for r in recs if r["seg"] is not None]
    assert f["n_seg_scored"] == 24 and f["n_seg_skipped"] == 6 and len(kept) == 24
    ref_hand = sum(k["iou_hand"] for k in kept) / len(kept)
    assert close(f["seg_iou_hand"], ref_hand)
    assert not close(f["seg_iou_hand"], sum(k["iou_hand"] for k in kept) / 30.0)
    assert close(f["seg_dice"], sum(k["dice"] for k in kept) / len(kept))
    assert close(f["seg_iou_bg"], 0.9)
    assert close(f["seg_miou"], 0.5 * (ref_hand + 0.9))          # miou derived from the pair
    assert c["n_seg_scored"] == 3 and c["n_seg_skipped"] == 0

    # classification: 23/30 correct; F1 = 16/23, 20/22, 10/15 for classes 0,1,2 and 0 elsewhere
    assert close(f["cls_top1"], 23.0 / 30.0)
    expect_f1 = [16.0 / 23.0, 20.0 / 22.0, 10.0 / 15.0] + [0.0] * 7
    got_f1 = [f["per_class_f1"][g] for g in utils.GESTURES]
    for g, e in zip(got_f1, expect_f1):
        assert close(g, e), (got_f1, expect_f1)
    assert close(f["cls_macro_f1"], sum(expect_f1) / 10.0)
    assert close(f["cls_macro_f1"], 0.22714097497)
    assert f["per_class_support"][utils.GESTURES[0]] == 10
    assert sum(f["per_class_support"].values()) == 30

    cm = np.array(s["confusion_matrix"])                         # rows = gt, cols = pred
    assert cm.shape == (10, 10) and cm.sum() == 30
    assert cm[0].tolist() == [8, 2] + [0] * 8
    assert cm[1].tolist() == [0, 10] + [0] * 8
    assert cm[2].tolist() == [5, 0, 5] + [0] * 7
    assert cm[3:].sum() == 0

    # clips are all 10 frames long here, so the clip block reproduces the frame block
    assert close(c["cls_top1"], (0.8 + 1.0 + 0.5) / 3.0)
    assert close(c["cls_macro_f1"], f["cls_macro_f1"])
    assert close(sum(sum(r) for r in s["confusion_matrix_clip"]), 3.0)

    assert close(s["cls_conf_correct"], 0.9) and close(s["cls_conf_incorrect"], 0.4)
    assert close(s["mean_cls_conf"], (23 * 0.9 + 7 * 0.4) / 30.0)
    assert close(s["mean_det_conf"], 0.5)

    json.dumps(s, allow_nan=False)                               # strict-JSON serialisable

    empty = MetricAccumulator().summary()                        # no records -> zeros, no NaN
    assert empty["n_frames"] == 0 and empty["n_clips"] == 0
    assert close(empty["frame"]["cls_macro_f1"], 0.0) and close(empty["frame"]["mean_box_iou"], 0.0)
    json.dumps(empty, allow_nan=False)

    try:
        MetricAccumulator().add(clip_key="x", box_iou=None, seg=None, gt_class=10,
                                pred_class=0, cls_conf=1.0, det_conf=1.0)
        raise AssertionError("expected ValueError on an out-of-range class")
    except ValueError:
        pass


def test_clip_weighting_differs_from_frame() -> None:
    """Unequal clip lengths: the clip block must not collapse to the frame block."""
    acc = MetricAccumulator()
    for _ in range(2):                                   # clipX: 2 frames of class 0, both right
        acc.add(clip_key="X", box_iou=1.0, seg=None, gt_class=0, pred_class=0,
                cls_conf=1.0, det_conf=1.0)
    for i in range(6):                                   # clipY: 6 frames of class 1, 3 right
        acc.add(clip_key="Y", box_iou=0.0, seg=None, gt_class=1, pred_class=1 if i < 3 else 0,
                cls_conf=1.0, det_conf=1.0)
    s = acc.summary()
    assert close(s["frame"]["cls_top1"], 5.0 / 8.0)      # 5 of 8 frames
    assert close(s["clip"]["cls_top1"], (1.0 + 0.5) / 2.0)
    assert close(s["frame"]["mean_box_iou"], 2.0 / 8.0)
    assert close(s["clip"]["mean_box_iou"], (1.0 + 0.0) / 2.0)
    # clip support = one unit per clip, spread over that clip's frames
    assert close(s["clip"]["per_class_support"][utils.GESTURES[0]], 1.0)
    assert close(s["clip"]["per_class_support"][utils.GESTURES[1]], 1.0)
    assert s["frame"]["n_seg_scored"] == 0 and close(s["frame"]["seg_miou"], 0.0)


def test_against_sklearn() -> None:
    """Cross-check our confusion matrix and macro-F1 against sklearn on the same records.

    scikit-learn is a development-only dependency (requirements.txt keeps it commented out), so
    this suite has to pass without it: an optional cross-check that fails when the optional
    package is absent is a broken test, not a strict one.
    """
    try:
        from sklearn.metrics import confusion_matrix, f1_score
    except ImportError:
        print("  skip test_against_sklearn (scikit-learn not installed)")
        return

    acc = MetricAccumulator()
    recs = _records()
    rng = np.random.default_rng(0)
    for r in recs:                                       # add noisier predictions for coverage
        acc.add(**r)
    extra = [(int(g), int(p)) for g, p in rng.integers(0, 10, size=(40, 2))]
    for i, (g, p) in enumerate(extra):
        acc.add(clip_key=f"clipR{i % 4}", box_iou=None, seg=None, gt_class=g, pred_class=p,
                cls_conf=0.5, det_conf=0.5)
    s = acc.summary()
    y_true = [r["gt_class"] for r in recs] + [g for g, _ in extra]
    y_pred = [r["pred_class"] for r in recs] + [p for _, p in extra]

    ref_cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    assert np.array_equal(np.array(s["confusion_matrix"]), ref_cm)
    ref_f1 = f1_score(y_true, y_pred, labels=list(range(10)), average="macro", zero_division=0)
    assert close(s["frame"]["cls_macro_f1"], float(ref_f1), 1e-9)
    ref_per = f1_score(y_true, y_pred, labels=list(range(10)), average=None, zero_division=0)
    for name, e in zip(utils.GESTURES, ref_per):
        assert close(s["frame"]["per_class_f1"][name], float(e), 1e-9)
    assert close(s["frame"]["cls_top1"],
                 sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true))


# -------------------------------------------------------------------------------- sundry
def test_seed_meter_json_params() -> None:
    utils.set_seed(1234)
    a = (torch.randn(3).tolist(), np.random.rand(3).tolist())
    utils.set_seed(1234)
    assert a == (torch.randn(3).tolist(), np.random.rand(3).tolist())
    utils.set_seed(7, deterministic=True)                # must not raise without a GPU

    m = utils.AverageMeter("loss")
    assert close(m.avg, 0.0)                             # no updates -> 0, not a division error
    m.update(1.0, n=2)
    m.update(torch.tensor(4.0))
    assert m.count == 3 and close(m.sum, 6.0) and close(m.avg, 2.0) and close(m.val, 4.0)
    m.reset()
    assert m.count == 0 and close(m.avg, 0.0)

    payload = {"a": np.float32(0.5), "b": np.int64(3), "c": np.arange(3),
               "d": torch.tensor([1.0, 2.0]), "e": np.bool_(True), "f": {"g": np.float64(1.5)}}
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "nested" / "out.json"
        utils.save_json(payload, p)
        back = utils.load_json(p)
    assert back == {"a": 0.5, "b": 3, "c": [0, 1, 2], "d": [1.0, 2.0], "e": True, "f": {"g": 1.5}}

    net = nn.Sequential(nn.Linear(4, 5), nn.Linear(5, 2))          # 4*5+5 + 5*2+2 = 37
    assert utils.count_parameters(net) == 37
    net[0].weight.requires_grad_(False)
    assert utils.count_parameters(net) == 17
    assert utils.count_parameters(net, trainable_only=False) == 37


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:                                   # noqa: BLE001
            failed += 1
            print(f"FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
