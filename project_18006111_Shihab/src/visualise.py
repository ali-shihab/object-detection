"""Figures for the COMP0248 CW1 LSA report.

Every figure here is drawn to survive the actual publication conditions: a two-column IEEE
paper at ~8.5 cm column width, possibly printed in greyscale. That constrains the design more
than it might look:

* **Never encode meaning in hue alone.** Series are separated by marker, hatch, linestyle or
  position as well as colour, so a greyscale print loses nothing.
* **Font sizes are set for the final size**, not the screen. A 7 pt tick label at 8.5 cm is
  legible; matplotlib's 10 pt default at figure width 6 in scaled down to 8.5 cm is not.
* **Both PNG (200 dpi) and PDF** are written for every figure — PDF for LaTeX inclusion, PNG
  for quick inspection and for pasting into the working documents.

Nothing here computes a metric. Everything is read from the JSONs that `src/evaluate.py` and
`src/train.py` write, so a figure can never disagree with the number in the results table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                                   # no display on the GPU host
import matplotlib.pyplot as plt                          # noqa: E402
import numpy as np                                       # noqa: E402
from matplotlib.patches import Rectangle                 # noqa: E402

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
from src import utils                                    # noqa: E402

COL_W = 3.35            # inches; one IEEE column
FULL_W = 6.9            # inches; both columns

#: Greyscale-safe qualitative set: distinguishable by lightness as well as hue.
PALETTE = ["#1b3a6b", "#c2564a", "#4d8b6f", "#8a6bab", "#b8892a", "#4a4a4a"]
HATCH = ["", "///", "...", "xxx", "\\\\\\", "+++"]


def _rc() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200,
        "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.2, "patch.linewidth": 0.6,
        "figure.constrained_layout.use": False,
    })


def _save(fig, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    for suf in (".png", ".pdf"):
        fig.savefig(out.with_suffix(suf), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out.with_suffix(".png")


# ======================================================================================
# confusion matrix
# ======================================================================================
def plot_confusion_matrix(cm, classes, out_path, normalise: bool = True, title=None):
    """10x10 confusion matrix with per-cell annotation.

    Text colour is chosen per cell against that cell's own luminance rather than by a single
    global threshold — with a nearly diagonal matrix a global rule makes either the diagonal
    or the off-diagonal text vanish.
    """
    _rc()
    cm = np.asarray(cm, dtype=np.float64)
    raw = cm.copy()
    if normalise:
        row = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, np.where(row > 0, row, 1.0))
    n = len(classes)
    fig, ax = plt.subplots(figsize=(COL_W * 1.35, COL_W * 1.35))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalise else cm.max())
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    short = [c.split("_", 1)[-1] for c in classes]
    ax.set_xticklabels(short, rotation=45, ha="right")
    ax.set_yticklabels(short)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    if title:
        ax.set_title(title)
    ax.grid(False)
    thresh_lum = 0.55
    for i in range(n):
        for j in range(n):
            v = cm[i, j]
            if v <= 0 and raw[i, j] <= 0:
                continue
            lum = im.cmap(im.norm(v))[:3]
            lum = 0.2126 * lum[0] + 0.7152 * lum[1] + 0.0722 * lum[2]
            txt = f"{v:.2f}".lstrip("0") if normalise else f"{int(raw[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=5.6,
                    color="white" if lum < thresh_lum else "#111111")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    return _save(fig, out_path)


# ======================================================================================
# qualitative overlays
# ======================================================================================
def plot_qualitative(examples, out_path, n_cols: int = 4, show_gt: bool = True,
                     title=None):
    """The brief's required mask-and-box overlays (LSA p9).

    Predicted mask = translucent fill + solid outline; predicted box solid; ground-truth box
    dashed. A misclassified panel gets a thick border AND a "X" prefix in its caption, so the
    error is visible in greyscale and to a reader who cannot see colour at all.
    """
    _rc()
    ex = list(examples)
    if not ex:
        raise ValueError("no examples to plot")
    n = len(ex)
    n_cols = max(1, min(n_cols, n))
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(FULL_W, FULL_W / n_cols * 0.80 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")

    for ax, e in zip(axes, ex):
        img = np.asarray(e["image"])
        if img.ndim == 3 and img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        H, W = img.shape[:2]

        pm = np.asarray(e["pred_mask"]) >= 0.5
        if pm.any():
            overlay = np.zeros((*pm.shape, 4))
            overlay[pm] = (0.10, 0.85, 0.55, 0.35)
            ax.imshow(overlay)
            ax.contour(pm.astype(float), levels=[0.5], colors=["#067a4e"],
                       linewidths=0.9)
        if show_gt and e.get("has_mask", True):
            gm = np.asarray(e["gt_mask"]) >= 0.5
            if gm.any():
                ax.contour(gm.astype(float), levels=[0.5], colors=["#ffffff"],
                           linewidths=0.7, linestyles="dotted")

        x1, y1, x2, y2 = e["pred_box"]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                               edgecolor="#ffd400", linewidth=1.3))
        if show_gt and e.get("has_mask", True):
            gx1, gy1, gx2, gy2 = e["gt_box"]
            ax.add_patch(Rectangle((gx1, gy1), gx2 - gx1, gy2 - gy1, fill=False,
                                   edgecolor="#ffffff", linewidth=1.0, linestyle="--"))

        gt_name = utils.GESTURES[e["gt_class"]].split("_", 1)[-1]
        pr_name = utils.GESTURES[e["pred_class"]].split("_", 1)[-1]
        wrong = e["pred_class"] != e["gt_class"]
        cap = f"{'X ' if wrong else ''}{pr_name} {e['cls_conf']:.2f}  (gt {gt_name})"
        ax.set_title(cap, fontsize=6.2, pad=2,
                     color="#8c1d1d" if wrong else "#111111")
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(2.0 if wrong else 0.4)
            s.set_color("#8c1d1d" if wrong else "#bbbbbb")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_xlim(0, W); ax.set_ylim(H, 0)
    if title:
        fig.suptitle(title, fontsize=8.5, y=1.005)
    return _save(fig, out_path)


# ======================================================================================
# reliability
# ======================================================================================
def plot_reliability(bins, out_path, title=None):
    """Calibration curve with per-bin counts on a twin axis."""
    _rc()
    b = [x for x in bins if x.get("n")]
    if not b:
        raise ValueError("no populated confidence bins")
    conf = np.array([x["conf"] for x in b])
    acc = np.array([x["acc"] for x in b])
    cnt = np.array([x["n"] for x in b], dtype=float)
    mid = np.array([(x["lo"] + x["hi"]) / 2 for x in b])
    w = float(b[0]["hi"] - b[0]["lo"])

    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.78))
    ax2 = ax.twinx()
    ax2.bar(mid, cnt / cnt.sum(), width=w * 0.92, color="#d9d9d9",
            edgecolor="#9a9a9a", zorder=1)
    ax2.set_ylabel("fraction of frames", color="#6f6f6f")
    ax2.tick_params(axis="y", colors="#6f6f6f")
    ax2.grid(False)
    ax.plot([0, 1], [0, 1], color="#7a7a7a", linestyle=(0, (4, 3)), linewidth=0.9,
            zorder=2, label="perfect calibration")
    ax.plot(conf, acc, marker="o", markersize=3.2, color=PALETTE[0], zorder=3,
            label="observed")
    ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    ax.legend(loc="upper left", frameon=False)
    if title:
        ax.set_title(title)
    return _save(fig, out_path)


# ======================================================================================
# training curves
# ======================================================================================
def plot_training_curves(jsonl_path, out_path, keys=None, title=None):
    """Loss terms and validation metrics from the per-epoch JSONL that train.py appends."""
    _rc()
    rows = [json.loads(l) for l in Path(jsonl_path).read_text().splitlines() if l.strip()]
    if not rows:
        raise ValueError(f"{jsonl_path} is empty")
    ep = [r["epoch"] for r in rows]
    loss_keys = keys or [k for k in rows[0] if k.startswith("train/") and k != "train/n_annotated"]
    val_keys = [k for k in rows[-1] if k.startswith("val/")]

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, FULL_W * 0.34))
    for i, k in enumerate(sorted(loss_keys)):
        y = [r.get(k) for r in rows]
        axes[0].plot(ep, y, color=PALETTE[i % len(PALETTE)],
                     linestyle=["-", "--", ":", "-."][i % 4],
                     label=k.split("/", 1)[1])
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("training loss (log)")
    axes[0].legend(ncol=2, frameon=False, fontsize=6)

    for i, k in enumerate(sorted(val_keys)):
        y = [r.get(k) for r in rows]
        xs = [e for e, v in zip(ep, y) if v is not None]
        ys = [v for v in y if v is not None]
        axes[1].plot(xs, ys, color=PALETTE[i % len(PALETTE)],
                     marker=["o", "s", "^", "D", "v", "P", "*", "X"][i % 8],
                     markersize=2.6, linestyle=["-", "--", ":", "-."][i % 4],
                     label=k.split("/", 1)[1])
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("validation metric")
    axes[1].set_ylim(0, 1)
    axes[1].legend(ncol=2, frameon=False, fontsize=6)
    if title:
        fig.suptitle(title, fontsize=8.5, y=1.02)
    return _save(fig, out_path)


# ======================================================================================
# ablation bars
# ======================================================================================
def plot_ablation_bars(rows, metric, out_path, baseline_key=None, title=None,
                       label_key: str = "run"):
    """Horizontal bars with bootstrap CIs; optionally as a delta against one baseline row.

    Horizontal because ablation labels are long, and error bars because the effects in this
    study are small: a bar chart of point estimates alone would let a 0.4-point difference look
    like a result when the interval says it is noise.
    """
    _rc()
    rows = list(rows)
    if not rows:
        raise ValueError("no rows")
    lo_k, hi_k = f"{metric}_lo", f"{metric}_hi"
    labels = [str(r.get(label_key, "?")) for r in rows]
    vals = np.array([float(r.get(metric, np.nan)) for r in rows])
    lo = np.array([float(r.get(lo_k, np.nan)) for r in rows])
    hi = np.array([float(r.get(hi_k, np.nan)) for r in rows])

    base = None
    if baseline_key is not None:
        for r in rows:
            if str(r.get(label_key)) == str(baseline_key):
                base = float(r.get(metric, np.nan))
                break
        if base is None:
            raise KeyError(f"baseline {baseline_key!r} not among {labels}")
        vals, lo, hi = vals - base, lo - base, hi - base

    order = np.argsort(vals)
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(COL_W * 1.6, 0.26 * len(rows) + 1.0))
    err = np.vstack([np.nan_to_num(vals[order] - lo[order]),
                     np.nan_to_num(hi[order] - vals[order])])
    colors = [PALETTE[0] if v >= 0 else PALETTE[1] for v in vals[order]]
    ax.barh(y, vals[order], xerr=err, color=colors, edgecolor="#333333",
            error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": "#333333"})
    ax.set_yticks(y); ax.set_yticklabels([labels[i] for i in order])
    ax.set_xlabel(f"delta {metric}" if baseline_key else metric)
    if baseline_key:
        ax.axvline(0.0, color="#333333", linewidth=0.8)
        ax.set_title(title or f"{metric} relative to {baseline_key}")
    elif title:
        ax.set_title(title)
    return _save(fig, out_path)


# ======================================================================================
# augmentation grid
# ======================================================================================
def plot_augmentation_grid(img, out_path, modes=None, seed: int = 0, n_draws: int = 5):
    """One frame through each photometric mode, plus a row of independent CPR draws.

    This is the figure that makes the method legible: the reader can see that CPR spans a much
    wider imaging-pipeline range than colour jitter, and that the pseudo-target shift sits
    outside it (which is the whole basis of the target-free model selection in s7.1).
    """
    _rc()
    from src import augment as A
    modes = list(modes or ["none", "jitter", "cpr", "randconv", "aprs", "augmix"])
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

    top = []
    for m in modes:
        if m == "none":
            top.append(("original", img))
            continue
        pol = A.AugmentPolicy(geometric=False, photometric=m, seed=seed)
        top.append((m, pol(img, None, None)[0]))
    top.append(("pseudo-target", A.pseudo_target_transform(img)))

    draws = []
    for i in range(n_draws):
        pol = A.AugmentPolicy(geometric=False, photometric="cpr", seed=seed + 100 + i)
        draws.append((f"CPR draw {i + 1}", pol(img, None, None)[0]))

    ncol = max(len(top), len(draws))
    fig, axes = plt.subplots(2, ncol, figsize=(FULL_W, FULL_W / ncol * 0.82 * 2))
    axes = np.atleast_2d(axes)
    for r, row in enumerate((top, draws)):
        for c in range(ncol):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if c < len(row):
                name, im = row[c]
                ax.imshow(im)
                ax.set_title(name, fontsize=6.2, pad=2)
            else:
                ax.axis("off")
    return _save(fig, out_path)


# ======================================================================================
# domain gap
# ======================================================================================
def plot_domain_gap(result_paths, labels, out_path, metrics=None, title=None):
    """Grouped bars of the same metrics across evaluation domains.

    The point of the figure is the *shape*: how far each metric falls from RealSense to phone,
    and how much of that fall E3 recovers. Grouping by metric (not by domain) puts the two bars
    a reader must compare next to each other.
    """
    _rc()
    metrics = metrics or ["det_acc@0.5", "mean_box_iou", "seg_iou_hand", "seg_dice",
                          "cls_top1", "cls_macro_f1"]
    data, cis = [], []
    for p in result_paths:
        d = utils.load_json(p)
        f, ci = d.get("frame", {}), d.get("ci", {})
        data.append([float(f.get(m, np.nan)) for m in metrics])
        cis.append([(ci.get(m, {}).get("lo", np.nan), ci.get(m, {}).get("hi", np.nan))
                    for m in metrics])
    data = np.array(data)
    n_series, n_metric = data.shape
    x = np.arange(n_metric)
    w = 0.8 / n_series
    fig, ax = plt.subplots(figsize=(FULL_W, FULL_W * 0.36))
    for s in range(n_series):
        lo = np.array([c[0] for c in cis[s]]); hi = np.array([c[1] for c in cis[s]])
        err = np.vstack([np.nan_to_num(data[s] - lo), np.nan_to_num(hi - data[s])])
        ax.bar(x + s * w - 0.4 + w / 2, data[s], width=w * 0.92,
               yerr=err, color=PALETTE[s % len(PALETTE)], hatch=HATCH[s % len(HATCH)],
               edgecolor="#222222", label=labels[s],
               error_kw={"elinewidth": 0.7, "capsize": 1.6, "ecolor": "#222222"})
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ") for m in metrics], rotation=18, ha="right")
    ax.set_ylabel("score"); ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncol=min(4, n_series))
    if title:
        ax.set_title(title)
    return _save(fig, out_path)


# ======================================================================================
# CLI
# ======================================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="*", default=[], help="result JSONs from src.evaluate")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--curves", default=None, help="log.jsonl from src.train")
    ap.add_argument("--examples", default=None, help="npz written by src.evaluate --examples-out")
    ap.add_argument("--aug-source", default=None, help="an image file for the augmentation grid")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pre = a.prefix
    made = []

    for p in a.results:
        d = utils.load_json(p)
        stem = Path(p).stem
        if "confusion_matrix" in d:
            made.append(plot_confusion_matrix(d["confusion_matrix"],
                                              d.get("class_names", utils.GESTURES),
                                              out / f"{pre}cm_{stem}", normalise=True,
                                              title=f"confusion matrix — {stem}"))
        if d.get("reliability"):
            made.append(plot_reliability(d["reliability"], out / f"{pre}rel_{stem}",
                                         title=f"calibration — {stem}"))
    if len(a.results) > 1:
        labels = a.labels or [Path(p).stem for p in a.results]
        made.append(plot_domain_gap(a.results, labels, out / f"{pre}domain_gap"))
        from src.evaluate import compare_runs
        rows = compare_runs(a.results, out_csv=str(out / f"{pre}comparison.csv"))
        made.append(plot_ablation_bars(rows, "cls_macro_f1", out / f"{pre}ablation_macro_f1"))
        made.append(plot_ablation_bars(rows, "seg_iou_hand", out / f"{pre}ablation_seg_iou"))
    if a.curves:
        made.append(plot_training_curves(a.curves, out / f"{pre}curves"))
    if a.examples:
        z = np.load(a.examples, allow_pickle=True)
        made.append(plot_qualitative(list(z["examples"])[:12], out / f"{pre}qualitative"))
    if a.aug_source:
        from PIL import Image
        made.append(plot_augmentation_grid(np.asarray(Image.open(a.aug_source).convert("RGB")),
                                           out / f"{pre}augmentation_grid"))
    for m in made:
        print(f"[viz] {m}")
    if not made:
        print("[viz] nothing to draw — pass --results / --curves / --examples / --aug-source")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
