#!/usr/bin/env python3
"""Non-learned reference point: skin-chroma segmentation + HOG + softmax regression.

The brief asks for the model to be compared against a baseline. The most informative baseline is
not a weaker network — it is the pre-deep-learning solution to exactly this problem, because it
says how much of the task is carried by *learning* rather than by the fact that hands are
skin-coloured blobs near the middle of the frame.

Pipeline (identical at train and test time — no ground truth is used at inference):

1. **Segment**: YCrCb skin-chroma threshold -> morphological open/close -> largest connected
   component. This is the classical hand-segmentation recipe, and it is also a fair stand-in for
   "what you get without learning anything".
2. **Detect**: the box is the tight box of that mask — the same mask -> box rule as everything
   else in this study, so the detection metric compares like with like.
3. **Classify**: HOG over the 64x64 crop of the predicted box, then multinomial logistic
   regression trained with plain gradient descent.

Deliberately implemented without scikit-learn: the classifier is ~30 lines of numpy, and keeping
the dependency out means the baseline runs in the same environment as everything else.

The output JSON has the same shape as `src/evaluate.py` writes, so the baseline drops straight
into the comparison tables and figures with no special-casing.

Usage
-----
    python tools/baseline_classical.py --index data/realsense_trainval/index.json \
        --test-index data/realsense_test/index.json --out results/b0_classical
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(1)

# tools/ lives inside the deliverable, so the package root is one level up. Keeping the
# tools inside project_<studentno>_<surname>/ makes the submitted zip self-contained:
# every command in README.md runs from the tree that is actually handed in.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from src import utils                                        # noqa: E402

CROP = 64
MASK_THRESH = 128


# ======================================================================================
# stage 1-2: segment and detect
# ======================================================================================
def skin_mask(bgr: np.ndarray) -> np.ndarray:
    """Classical YCrCb skin threshold, opened then closed, largest component only.

    Thresholds are the widely-used Cr in [133,173], Cb in [77,127] band. They are *not* tuned on
    this dataset: tuning a baseline on the same data it is measured on would flatter it, and the
    point of the baseline is to be the honest off-the-shelf answer.
    """
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycrcb[..., 1], ycrcb[..., 2]
    m = ((cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return np.zeros(m.shape, np.uint8)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return ((lab == k).astype(np.uint8)) * 255


def mask_box(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(mask >= MASK_THRESH)
    if xs.size == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)


# ======================================================================================
# stage 3: HOG + softmax regression
# ======================================================================================
#: HOG geometry: 8x8-pixel cells, 2x2-cell blocks with 1-cell stride, 9 unsigned orientation
#: bins -- the Dalal & Triggs (CVPR 2005) parameters. On a 64x64 crop that is 8x8 cells,
#: 7x7 blocks, 7*7*4*9 = 1764 dimensions.
HOG_CELL, HOG_BLOCK, HOG_BINS = 8, 2, 9


def hog(gray: np.ndarray) -> np.ndarray:
    """Dalal-Triggs HOG, written out in numpy.

    `cv2.HOGDescriptor` is absent from some OpenCV builds (including the headless wheel on the
    GPU host), and a baseline that cannot run in the same environment as the model is useless
    for comparison. Thirty lines of numpy removes the dependency question entirely -- and since
    the point of this baseline is "the pre-deep-learning answer", implementing its feature
    extractor rather than importing it is the honest version.
    """
    g = gray.astype(np.float32) / 255.0
    gx = np.zeros_like(g); gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.rad2deg(np.arctan2(gy, gx)) % 180.0          # unsigned gradients

    H, W = g.shape
    cy, cx = H // HOG_CELL, W // HOG_CELL
    bin_w = 180.0 / HOG_BINS
    # Linear interpolation between the two neighbouring orientation bins: hard assignment
    # makes the descriptor jump discontinuously as an edge rotates, which is exactly the
    # instability HOG is supposed to avoid.
    b = ang / bin_w - 0.5
    b0 = np.floor(b).astype(np.int32)
    frac = b - b0
    b0 = b0 % HOG_BINS
    b1 = (b0 + 1) % HOG_BINS

    cells = np.zeros((cy, cx, HOG_BINS), np.float32)
    yy = (np.arange(H) // HOG_CELL)[:, None].repeat(W, 1)
    xx = (np.arange(W) // HOG_CELL)[None, :].repeat(H, 0)
    ok = (yy < cy) & (xx < cx)
    np.add.at(cells, (yy[ok], xx[ok], b0[ok]), (mag * (1 - frac))[ok])
    np.add.at(cells, (yy[ok], xx[ok], b1[ok]), (mag * frac)[ok])

    # L2-Hys block normalisation over overlapping 2x2-cell blocks.
    nb_y, nb_x = cy - HOG_BLOCK + 1, cx - HOG_BLOCK + 1
    out = np.empty((nb_y, nb_x, HOG_BLOCK * HOG_BLOCK * HOG_BINS), np.float32)
    for i in range(nb_y):
        for j in range(nb_x):
            v = cells[i:i + HOG_BLOCK, j:j + HOG_BLOCK].ravel()
            v = v / np.sqrt((v * v).sum() + 1e-6)
            v = np.minimum(v, 0.2)                        # the "Hys" clip
            out[i, j] = v / np.sqrt((v * v).sum() + 1e-6)
    return out.ravel()


def hog_features(bgr: np.ndarray, box: np.ndarray | None) -> np.ndarray:
    """HOG over the crop of the *predicted* box, or of the whole frame when detection failed.

    Falling back to the whole frame rather than skipping the sample matters: at test time a
    detection failure must still produce a class prediction, because the metric set scores
    classification on every frame.
    """
    h, w = bgr.shape[:2]
    if box is None:
        crop = bgr
    else:
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, max(x1 + 2, x2)), min(h, max(y1 + 2, y2))
        crop = bgr[y1:y2, x1:x2]
        if crop.size == 0:
            crop = bgr
    g = cv2.cvtColor(cv2.resize(crop, (CROP, CROP)), cv2.COLOR_BGR2GRAY)
    return hog(g).astype(np.float32)


def softmax_fit(X: np.ndarray, y: np.ndarray, n_cls: int = 10, epochs: int = 300,
                lr: float = 0.5, l2: float = 1e-4, seed: int = 0, verbose: bool = True):
    """Multinomial logistic regression, full-batch gradient descent with L2.

    ~30 lines instead of a scikit-learn dependency. Features are standardised first, which is
    what makes a single fixed learning rate work across HOG dimensions of very different scale.
    """
    rng = np.random.default_rng(seed)
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6
    Xs = (X - mu) / sd
    Xs = np.hstack([Xs, np.ones((Xs.shape[0], 1), np.float32)])
    W = rng.normal(0, 0.01, (Xs.shape[1], n_cls)).astype(np.float32)
    Y = np.zeros((len(y), n_cls), np.float32)
    Y[np.arange(len(y)), y] = 1.0
    n = len(y)
    for ep in range(epochs):
        z = Xs @ W
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        grad = Xs.T @ (p - Y) / n + l2 * W
        W -= lr * grad
        if verbose and (ep + 1) % 100 == 0:
            loss = -np.log(np.clip(p[np.arange(n), y], 1e-12, None)).mean()
            acc = float((p.argmax(1) == y).mean())
            print(f"  [b0] epoch {ep + 1}/{epochs} loss={loss:.4f} train-acc={acc:.4f}", flush=True)
    return {"W": W, "mu": mu, "sd": sd}


def softmax_predict(model, X: np.ndarray):
    Xs = (X - model["mu"]) / model["sd"]
    Xs = np.hstack([Xs, np.ones((Xs.shape[0], 1), np.float32)])
    z = Xs @ model["W"]
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    return p.argmax(1), p.max(1)


# ======================================================================================
# dataset walking
# ======================================================================================
def load_split(index_path: str, split: str):
    ix = utils.load_json(index_path)
    root = Path(index_path).parent
    smap = ix["split_map"]
    return root, [r for r in ix["records"] if smap.get(r["subject"], "train") == split], ix


def process(root: Path, recs, want_mask: bool, tag: str):
    """Run segmentation over a split; return features, labels and per-record predictions."""
    feats, labels, preds = [], [], []
    t0 = time.time()
    for i, r in enumerate(recs):
        bgr = cv2.imread(str(root / r["rgb"]), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        m = skin_mask(bgr)
        b = mask_box(m)
        feats.append(hog_features(bgr, b))
        labels.append(int(r["cls"]))
        preds.append({"rec": r, "mask": m if want_mask else None, "box": b})
        if (i + 1) % 2000 == 0:
            print(f"  [b0] {tag} {i + 1}/{len(recs)} ({time.time() - t0:.0f}s)", flush=True)
    return np.stack(feats), np.array(labels, np.int64), preds


def score_split(root: Path, recs, model, ix, tag: str) -> dict:
    X, y, preds = process(root, recs, want_mask=True, tag=tag)
    yhat, conf = softmax_predict(model, X)
    acc = utils.MetricAccumulator(n_classes=10, class_names=utils.GESTURES, iou_thresh=0.5)
    import torch
    for j, p in enumerate(preds):
        r = p["rec"]
        ck = f"{r['subject']}/{r['gesture']}/{r['clip']}"
        has = r["ann"] is not None and r["box"] is not None
        box_iou = None
        seg = None
        if has:
            gt_box = torch.from_numpy(np.asarray([r["box"]], dtype=np.float32))
            pbox = p["box"] if p["box"] is not None else np.zeros(4, np.float32)
            pb = torch.from_numpy(np.asarray([pbox], dtype=np.float32))
            box_iou = float(utils.box_iou(pb, gt_box)[0])
            gm = cv2.imread(str(root / r["ann"]), cv2.IMREAD_GRAYSCALE)
            if gm is not None:
                pm = torch.from_numpy((p["mask"] >= MASK_THRESH).astype(np.float32))[None, None]
                gmm = torch.from_numpy((gm >= MASK_THRESH).astype(np.float32))[None, None]
                s = utils.seg_scores(pm, gmm)
                seg = {k: float(v[0]) for k, v in s.items()}
        acc.add(clip_key=ck, box_iou=box_iou, seg=seg, gt_class=int(r["cls"]),
                pred_class=int(yhat[j]), cls_conf=float(conf[j]),
                det_conf=1.0 if p["box"] is not None else 0.0)
    res = acc.summary()
    res["frame"]["cls_ece"] = 0.0     # not meaningful for this baseline; kept for schema parity
    res["method"] = "B0 classical: YCrCb skin -> largest component -> HOG -> softmax regression"
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True, help="packed train/val index.json")
    ap.add_argument("--test-index", default=None)
    ap.add_argument("--out", required=True, help="output prefix, e.g. results/b0_classical")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--max-train", type=int, default=8000,
                    help="cap on training frames (HOG+GD is CPU-bound; 8k is ample for 10 classes)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    root, tr, ix = load_split(a.index, "train")
    rng = np.random.default_rng(a.seed)
    if a.max_train and len(tr) > a.max_train:
        tr = [tr[i] for i in rng.permutation(len(tr))[:a.max_train]]
    print(f"[b0] fitting on {len(tr)} training frames", flush=True)
    Xtr, ytr, _ = process(root, tr, want_mask=False, tag="train")
    model = softmax_fit(Xtr, ytr, epochs=a.epochs, seed=a.seed)

    out_prefix = Path(a.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    _, va, _ = load_split(a.index, "val")
    print(f"[b0] scoring {len(va)} val frames", flush=True)
    utils.save_json(score_split(root, va, model, ix, "val"), f"{out_prefix}_rs_val.json")

    if a.test_index:
        troot, te, tix = load_split(a.test_index, "test")
        print(f"[b0] scoring {len(te)} test frames", flush=True)
        utils.save_json(score_split(troot, te, model, tix, "test"), f"{out_prefix}_rs_test.json")

    print(f"[b0] wrote {out_prefix}_rs_*.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
