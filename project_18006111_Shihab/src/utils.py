"""Shared utilities: geometry, heatmap targets, metrics, bookkeeping.

Everything here is implemented from scratch - no ``torchvision.ops``, no detection or
segmentation library. The conventions fixed in this module are relied on by ``dataloader``,
``model``, ``train`` and ``evaluate``, so they are stated explicitly in the docstrings.

Box convention (used *everywhere*, including IoU and GIoU)
---------------------------------------------------------
A box is ``(x1, y1, x2, y2)`` in absolute pixels with ``x2``/``y2`` **exclusive**, i.e.
``width = x2 - x1`` is exactly the number of foreground pixel columns. With this convention the
rectangle area equals its pixel count, so box IoU and pixel-mask IoU of two rectangles agree
(the ``+1`` of the inclusive convention is never needed and never double counted).
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

__all__ = [
    "GESTURES", "IMAGENET_MEAN", "IMAGENET_STD",
    "set_seed", "AverageMeter", "save_json", "load_json", "count_parameters",
    "mask_to_box", "box_iou", "box_giou_loss", "gaussian_radius", "draw_gaussian",
    "seg_scores", "MetricAccumulator",
]

GESTURES: list[str] = [
    "G01_call", "G02_dislike", "G03_like", "G04_ok", "G05_one",
    "G06_palm", "G07_peace", "G08_rock", "G09_stop", "G10_three",
]
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

_EPS = 1e-7


# reproducibility / bookkeeping --------------------------------------------------------
def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed ``random``, ``numpy`` and ``torch`` (CPU + all CUDA devices if present).

    ``deterministic=True`` additionally disables cuDNN autotuning and requests deterministic
    kernels with ``warn_only=True``: several ops we need (bilinear upsample backward in the
    U-Net decoder) have no deterministic CUDA kernel, and a hard error there would kill a run
    for no benefit. Full bitwise reproducibility therefore holds on CPU and is best-effort on
    GPU; this is stated in the report rather than silently assumed.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)


class AverageMeter:
    """Running mean of a scalar, weighted by the number of items each update covers."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        if isinstance(val, Tensor):
            val = float(val.detach().item())
        self.val = float(val)
        self.sum += self.val * int(n)
        self.count += int(n)

    @property
    def avg(self) -> float:
        """Mean so far; ``0.0`` before the first update."""
        return self.sum / self.count if self.count else 0.0

    def __repr__(self) -> str:
        return f"AverageMeter({self.name!r}, avg={self.avg:.4f}, n={self.count})"


class _NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that understands numpy scalars/arrays, torch tensors, ``Path`` and sets."""

    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Tensor):
            return o.detach().cpu().tolist()
        if isinstance(o, (set, frozenset)):
            return sorted(o)
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def save_json(obj: Any, path: str | Path) -> None:
    """Write ``obj`` as indented JSON, creating parent directories.

    Numpy/torch scalars are converted (every module dumps numpy floats sooner or later). The
    write goes to a temporary file and is then renamed, so an interrupted run cannot leave a
    half-written results file behind.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, cls=_NumpyJSONEncoder)
        fh.write("\n")
    os.replace(tmp, path)


def load_json(path: str | Path) -> Any:
    """Read a JSON file written by :func:`save_json` (or any other JSON file)."""
    with open(Path(path), "r", encoding="utf-8") as fh:
        return json.load(fh)


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    """Number of (by default, trainable) scalar parameters in ``module``."""
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only))


# geometry -----------------------------------------------------------------------------
def mask_to_box(mask: np.ndarray, thresh: int = 128) -> np.ndarray | None:
    """Tight box of ``mask >= thresh``.

    Args:
        mask: ``(H, W)`` array. ``thresh`` is in the *units of the array*: the released masks
            are ``uint8`` and are binarised at 128; a float ``{0,1}`` mask
            must be passed with ``thresh=0.5``.

    Returns:
        ``(4,) float32`` ``(x1, y1, x2, y2)`` with exclusive ``x2``/``y2`` - ``x2 - x1`` is the
        number of foreground columns and ``y2 - y1`` the number of foreground rows - or
        ``None`` when the mask has no foreground pixel at all.
    """
    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError(f"mask_to_box expects a 2-D (H,W) mask, got shape {m.shape}")
    fg = m >= thresh
    rows = np.flatnonzero(fg.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(fg.any(axis=0))
    return np.array([cols[0], rows[0], cols[-1] + 1, rows[-1] + 1], dtype=np.float32)


def _iou_parts(a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    """Elementwise ``(iou, union)`` for paired boxes; degenerate boxes give ``0``, never NaN.

    Anything that is not already float64 is promoted to float32: under ``autocast(bf16)`` the
    incoming boxes are bf16, whose ~3 significant digits are not enough for an area ratio.
    """
    if a.shape != b.shape or a.shape[-1] != 4:
        raise ValueError(f"paired boxes must have matching (...,4) shapes, "
                         f"got {tuple(a.shape)} vs {tuple(b.shape)}")
    a = a if a.dtype == torch.float64 else a.float()
    b = b if b.dtype == torch.float64 else b.float()
    area_a = (a[..., 2] - a[..., 0]).clamp(min=0) * (a[..., 3] - a[..., 1]).clamp(min=0)
    area_b = (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)
    lt = torch.maximum(a[..., :2], b[..., :2])
    rb = torch.minimum(a[..., 2:], b[..., 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a + area_b - inter
    return inter / union.clamp(min=_EPS), union


def box_iou(a: Tensor, b: Tensor) -> Tensor:
    """Elementwise IoU of *paired* rows: ``(N,4), (N,4) -> (N,)``. Not a pairwise matrix.

    Any leading shape is accepted as long as both arguments match (``(4,)`` gives a scalar).
    Zero-area or inverted boxes yield ``0.0`` rather than NaN.
    """
    return _iou_parts(a, b)[0]


def box_giou_loss(pred: Tensor, tgt: Tensor) -> Tensor:
    """``1 - GIoU`` per row, ``(N,4), (N,4) -> (N,)``; differentiable w.r.t. ``pred``.

    GIoU = IoU - (area(C) - union) / area(C) with C the smallest enclosing box, so the loss
    lies in ``[0, 2]``: ``0`` for identical boxes and ``> 1`` whenever the boxes are disjoint.
    ``area(C)`` is clamped away from zero so two degenerate (point) boxes give a finite value.
    """
    iou, union = _iou_parts(pred, tgt)
    lt = torch.minimum(pred[..., :2], tgt[..., :2])
    rb = torch.maximum(pred[..., 2:], tgt[..., 2:])
    cwh = (rb - lt).clamp(min=0)
    c_area = (cwh[..., 0] * cwh[..., 1]).clamp(min=_EPS)
    return 1.0 - (iou - (c_area - union) / c_area)


# centre-point heatmap targets ---------------------------------------------------------
def gaussian_radius(h: float, w: float, min_overlap: float = 0.7) -> float:
    """CornerNet Gaussian radius for a ``h x w`` box: the largest corner displacement ``r``
    that still leaves IoU >= ``min_overlap``.

    Three configurations are solved and the minimum taken: (1) both corners displaced the same
    way (box translated), (2) the box shrunk by ``r`` on every side, (3) the box grown by ``r``
    on every side. Each is a quadratic in ``r``; we take the **smallest non-negative root**,
    which is the mathematically correct one - the widely copied CornerNet/CenterNet snippet
    uses ``(b + sqrt(d)) / 2`` for cases 1 and 2, which returns the *larger* root and forgets
    the ``/(2a)``, and so over-estimates the radius (e.g. 9.76 instead of 8.17 px for a
    100x100 box at overlap 0.7). Negative discriminants are clamped to 0.
    """
    h, w = max(float(h), 0.0), max(float(w), 0.0)
    o = min(max(float(min_overlap), 1e-6), 1.0)

    def _root(a: float, b: float, c: float) -> float:
        """Smallest non-negative root of ``a*r^2 + b*r + c = 0`` (a > 0); 0.0 if there is none."""
        disc = max(b * b - 4.0 * a * c, 0.0) ** 0.5
        roots = [r for r in ((-b - disc) / (2.0 * a), (-b + disc) / (2.0 * a)) if r >= 0.0]
        return min(roots) if roots else 0.0

    r1 = _root(1.0, -(h + w), w * h * (1.0 - o) / (1.0 + o))       # translated box
    r2 = _root(4.0, -2.0 * (h + w), (1.0 - o) * w * h)             # box shrunk by r per side
    r3 = _root(4.0 * o, 2.0 * o * (h + w), (o - 1.0) * w * h)      # box grown by r per side
    return float(max(0.0, min(r1, r2, r3)))


def draw_gaussian(heat: np.ndarray, cx: int, cy: int, radius: int) -> None:
    """Splat an unnormalised 2-D Gaussian (peak exactly 1.0) into ``heat`` **in place**.

    ``heat`` is ``(H, W) float32``; ``sigma = (2*radius + 1) / 6``; overlapping splats are
    merged with ``np.maximum`` so an existing stronger peak is never overwritten. The splat is
    clipped at the array edges, and a centre outside the array is a no-op. ``radius`` is
    floored at 0, which writes the single centre pixel.
    """
    if heat.ndim != 2:
        raise ValueError(f"draw_gaussian expects a 2-D (H,W) heatmap, got shape {heat.shape}")
    radius = max(int(radius), 0)
    cx, cy = int(cx), int(cy)
    height, width = heat.shape
    if not (0 <= cx < width and 0 <= cy < height):
        return
    sigma = (2.0 * radius + 1.0) / 6.0
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    g = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma)).astype(np.float32)
    g[g < np.finfo(np.float32).eps * g.max()] = 0.0
    left, right = min(cx, radius), min(width - cx, radius + 1)
    top, bottom = min(cy, radius), min(height - cy, radius + 1)
    sub_heat = heat[cy - top:cy + bottom, cx - left:cx + right]
    sub_g = g[radius - top:radius + bottom, radius - left:radius + right]
    if sub_heat.size and sub_g.size:
        np.maximum(sub_heat, sub_g, out=sub_heat)


# segmentation metrics -----------------------------------------------------------------
@torch.no_grad()
def seg_scores(pred: Tensor, tgt: Tensor) -> dict[str, Tensor]:
    """Per-image binary segmentation scores.

    Args:
        pred: ``(N,1,H,W)`` probability in ``[0,1]``, thresholded at 0.5.
        tgt: ``(N,1,H,W)`` in ``{0,1}`` (ground-truth masks are binarised at 128 upstream).

    Returns:
        ``{"iou_hand", "iou_bg", "miou", "dice"}``, each a ``(N,)`` float32 tensor.
        ``miou`` is the mean of the two class IoUs; ``dice`` is on the hand class only.

    Edge cases:
      * no ground-truth hand **and** no predicted hand -> ``iou_hand = dice = 1.0``
        (a correct prediction of "nothing here" scores as perfect, not as zero);
      * exactly one of the two empty -> ``iou_hand = dice = 0.0``;
      * the same rule applied to the background class gives ``iou_bg = 1.0`` for an image whose
        every pixel is hand in both masks.
    """
    if pred.shape != tgt.shape:
        raise ValueError(f"seg_scores shape mismatch: {tuple(pred.shape)} vs {tuple(tgt.shape)}")
    if pred.dim() < 2:
        raise ValueError("seg_scores expects at least (N, ...) inputs")
    n = pred.shape[0]
    p = (pred.detach().reshape(n, -1) >= 0.5).to(torch.float32)
    g = (tgt.detach().reshape(n, -1) >= 0.5).to(torch.float32)
    npix = float(p.shape[1])

    inter = (p * g).sum(1)
    p_sum, g_sum = p.sum(1), g.sum(1)
    union = p_sum + g_sum - inter
    one = torch.ones_like(inter)

    # counts are integral, so clamp(min=1) is exact wherever the branch is taken
    iou_hand = torch.where(union > 0, inter / union.clamp(min=1.0), one)
    inter_bg, union_bg = npix - union, npix - inter
    iou_bg = torch.where(union_bg > 0, inter_bg / union_bg.clamp(min=1.0), one)
    den = p_sum + g_sum
    dice = torch.where(den > 0, 2.0 * inter / den.clamp(min=1.0), one)
    return {"iou_hand": iou_hand, "iou_bg": iou_bg, "miou": 0.5 * (iou_hand + iou_bg), "dice": dice}


# evaluation accumulator ---------------------------------------------------------------
_SEG_KEYS: tuple[str, ...] = ("miou", "iou_hand", "iou_bg", "dice")


class MetricAccumulator:
    """Streams one record per frame and produces the full metric set.

    Aggregation:
      * ``frame`` - every frame weighted equally (the headline numbers);
      * ``clip``  - each metric averaged *within* a ``clip_key`` first and then across clips,
        so the 15 correlated frames of a clip count once. For the F1-family (which does not
        decompose per frame) the same principle is applied to the confusion matrix: a frame
        contributes ``1 / frames-in-its-clip`` counts, so each clip carries total weight 1 and
        ``cls_top1`` equals the mean of the per-clip accuracies. ``per_class_support`` in the
        clip block is therefore a clip count (fractional only if a clip mixes labels).

    Exclusions - never silently zero:
      * ``box_iou=None`` (no usable ground-truth box: ``has_mask`` false or an empty mask) is
        dropped from ``det_acc@0.5`` and ``mean_box_iou``; ``n_boxes_scored`` reports the
        denominator actually used and ``n_boxes_skipped`` the number dropped.
      * a record whose ``seg`` is ``None``/empty is dropped from the four segmentation metrics,
        with ``n_seg_scored`` / ``n_seg_skipped`` reported the same way.
      * in the clip block these counts are counts of *clips* that did / did not contribute.

    F1 convention: macro-F1 is the unweighted mean of per-class F1 over **all** ``n_classes``
    classes, and a class with zero support and zero predictions contributes ``F1 = 0.0``. This
    is deliberate - it keeps macro-F1 comparable across splits with different class coverage
    and refuses to reward a model for a class it was never asked about. Empty accumulators
    return ``0.0`` everywhere (never NaN) so ``summary()`` is always strict-JSON serialisable.
    """

    def __init__(self, n_classes: int = 10, class_names: Sequence[str] | None = None,
                 iou_thresh: float = 0.5) -> None:
        self.n_classes = int(n_classes)
        if class_names is not None:
            names = list(class_names)
        elif self.n_classes == len(GESTURES):
            names = list(GESTURES)
        else:
            names = [f"class_{i}" for i in range(self.n_classes)]
        if len(names) != self.n_classes:
            raise ValueError("class_names length must equal n_classes")
        self.class_names = names
        self.iou_thresh = float(iou_thresh)
        self.reset()

    def reset(self) -> None:
        self._rows: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._rows)

    def add(self, *, clip_key: str, box_iou: float | None, seg: Mapping[str, Any] | None,
            gt_class: int, pred_class: int, cls_conf: float, det_conf: float) -> None:
        """Record one frame.

        ``seg`` holds scalar per-frame values, i.e. ``{k: v[i].item() for k, v in
        seg_scores(...).items()}``; pass ``None`` when the frame has no ground-truth mask.
        ``box_iou`` is ``None`` for a frame with no usable ground-truth box.
        """
        gt, pred = int(gt_class), int(pred_class)
        for name, v in (("gt_class", gt), ("pred_class", pred)):
            if not 0 <= v < self.n_classes:
                raise ValueError(f"{name}={v} outside [0,{self.n_classes})")
        self._rows.append({
            "clip": str(clip_key),
            "box_iou": None if box_iou is None else float(box_iou),
            "seg": self._coerce_seg(seg),
            "gt": gt, "pred": pred,
            "cls_conf": float(cls_conf), "det_conf": float(det_conf),
        })

    @staticmethod
    def _coerce_seg(seg: Mapping[str, Any] | None) -> dict[str, float] | None:
        if not seg:
            return None
        out = {k: float(seg[k]) for k in ("iou_hand", "iou_bg", "dice") if k in seg}
        missing = {"iou_hand", "iou_bg", "dice"} - set(out)
        if missing:
            raise KeyError(f"seg record missing {sorted(missing)}")
        out["miou"] = float(seg["miou"]) if "miou" in seg else 0.5 * (out["iou_hand"] + out["iou_bg"])
        return out

    @staticmethod
    def _frame_mean(pairs: Sequence[tuple[str, float]]) -> float:
        return float(np.mean([v for _, v in pairs])) if pairs else 0.0

    @staticmethod
    def _clip_mean(pairs: Sequence[tuple[str, float]]) -> float:
        if not pairs:
            return 0.0
        groups: dict[str, list[float]] = defaultdict(list)
        for key, val in pairs:
            groups[key].append(val)
        return float(np.mean([float(np.mean(v)) for v in groups.values()]))

    def _cls_metrics(self, cm: np.ndarray) -> dict[str, Any]:
        total = float(cm.sum())
        tp = np.diag(cm).astype(np.float64)
        fp = cm.sum(axis=0).astype(np.float64) - tp
        fn = cm.sum(axis=1).astype(np.float64) - tp
        den = 2.0 * tp + fp + fn
        f1 = np.where(den > 0, 2.0 * tp / np.where(den > 0, den, 1.0), 0.0)
        cast = int if cm.dtype.kind in "iu" else float
        return {
            "cls_top1": float(tp.sum() / total) if total > 0 else 0.0,
            "cls_macro_f1": float(f1.mean()),
            "per_class_f1": {n: float(v) for n, v in zip(self.class_names, f1)},
            "per_class_support": {n: cast(v) for n, v in zip(self.class_names, cm.sum(axis=1))},
        }

    def summary(self) -> dict[str, Any]:
        """Aggregate everything; JSON-serialisable, safe to call on an empty accumulator."""
        rows = self._rows
        n_frames = len(rows)
        per_clip: dict[str, int] = defaultdict(int)
        for r in rows:
            per_clip[r["clip"]] += 1
        n_clips = len(per_clip)

        det = [(r["clip"], r["box_iou"]) for r in rows if r["box_iou"] is not None]
        hit = [(c, float(v >= self.iou_thresh)) for c, v in det]
        seg = {k: [(r["clip"], r["seg"][k]) for r in rows if r["seg"] is not None] for k in _SEG_KEYS}
        n_seg = sum(1 for r in rows if r["seg"] is not None)

        cm_frame = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)
        cm_clip = np.zeros((self.n_classes, self.n_classes), dtype=np.float64)
        for r in rows:
            cm_frame[r["gt"], r["pred"]] += 1
            cm_clip[r["gt"], r["pred"]] += 1.0 / per_clip[r["clip"]]

        def build(agg, cm: np.ndarray, per_clip_counts: bool) -> dict[str, Any]:
            if per_clip_counts:
                n_box_ok = len({c for c, _ in det})
                n_seg_ok = len({r["clip"] for r in rows if r["seg"] is not None})
                total = n_clips
            else:
                n_box_ok, n_seg_ok, total = len(det), n_seg, n_frames
            block: dict[str, Any] = {
                f"det_acc@{self.iou_thresh:g}": agg(hit),
                "mean_box_iou": agg(det),
                "n_boxes_scored": n_box_ok,
                "n_boxes_skipped": total - n_box_ok,
            }
            block.update({f"seg_{k}": agg(seg[k]) for k in _SEG_KEYS})
            block["n_seg_scored"] = n_seg_ok
            block["n_seg_skipped"] = total - n_seg_ok
            block.update(self._cls_metrics(cm))
            return block

        conf_ok = [(r["clip"], r["cls_conf"]) for r in rows if r["pred"] == r["gt"]]
        conf_bad = [(r["clip"], r["cls_conf"]) for r in rows if r["pred"] != r["gt"]]
        return {
            "n_frames": n_frames,
            "n_clips": n_clips,
            "frame": build(self._frame_mean, cm_frame, per_clip_counts=False),
            "clip": build(self._clip_mean, cm_clip, per_clip_counts=True),
            "confusion_matrix": cm_frame.tolist(),
            "confusion_matrix_clip": np.round(cm_clip, 9).tolist(),
            "mean_cls_conf": self._frame_mean([(r["clip"], r["cls_conf"]) for r in rows]),
            "mean_det_conf": self._frame_mean([(r["clip"], r["det_conf"]) for r in rows]),
            "cls_conf_correct": self._frame_mean(conf_ok),
            "cls_conf_incorrect": self._frame_mean(conf_bad),
            "class_names": list(self.class_names),
            "iou_thresh": self.iou_thresh,
        }
