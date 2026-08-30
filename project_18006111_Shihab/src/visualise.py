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
# DESTINATION WIDTH. Every plotting function takes the width of the slot it will be rendered
# into and sizes itself to it, so \includegraphics never has to scale. A figure authored at
# FULL_W and included at \columnwidth has all of its text shrunk by 0.49; at 6.8 pt titles that
# is 3.3 pt on the page. Which width a figure gets is a report-layout decision, so it is made
# once, in main(), against what the .tex actually does with each file.

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
def _errorbars(vals, lo, hi, where=""):
    """Asymmetric error bars from a point estimate and an interval, clamped at zero.

    matplotlib raises on a negative yerr, which kills the whole figure run. A point estimate can
    legitimately fall outside its own interval here: the interval is a percentile bootstrap over
    clips, and an evaluation run with a small --n-boot produces a degenerate interval that need
    not bracket anything. Clamping keeps the figure honest -- a zero-length bar on that side is
    exactly the right thing to draw for an interval that does not extend there -- and says so
    rather than failing or silently drawing a mirror image.
    """
    lower = np.nan_to_num(np.asarray(vals, float) - np.asarray(lo, float))
    upper = np.nan_to_num(np.asarray(hi, float) - np.asarray(vals, float))
    n_bad = int((lower < 0).sum() + (upper < 0).sum())
    if n_bad:
        print(f"[viz] {n_bad} error-bar arm(s) clamped to zero in {where}: the point estimate "
              f"lies outside its bootstrap interval (a very small --n-boot does this)")
    return np.vstack([np.clip(lower, 0, None), np.clip(upper, 0, None)])


def plot_confusion_matrix(cm, classes, out_path, normalise: bool = True, title=None,
                          width: float = COL_W):
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
    fig, ax = plt.subplots(figsize=(width, width))
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
                     title=None, width: float = FULL_W):
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
                             figsize=(width, width / n_cols * 0.80 * n_rows))
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

        # Ground truth is drawn FIRST and slightly heavier, so that when the two boxes very
        # nearly coincide -- which is the common case at 0.84 mean IoU -- the dashed white line
        # still shows through the solid prediction rather than being completely hidden by it.
        # A figure in which the ground truth is invisible reads as if it were never plotted.
        if show_gt and e.get("has_mask", True):
            gx1, gy1, gx2, gy2 = e["gt_box"]
            ax.add_patch(Rectangle((gx1, gy1), gx2 - gx1, gy2 - gy1, fill=False,
                                   edgecolor="#111111", linewidth=2.2))
            ax.add_patch(Rectangle((gx1, gy1), gx2 - gx1, gy2 - gy1, fill=False,
                                   edgecolor="#ffffff", linewidth=1.6, linestyle=(0, (3, 2))))
        x1, y1, x2, y2 = e["pred_box"]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                               edgecolor="#ffd400", linewidth=1.1))

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
def plot_training_curves(jsonl_path, out_path, keys=None, title=None,
                         width: float = COL_W):
    """Loss terms and validation metrics from the per-epoch JSONL that train.py appends."""
    _rc()
    rows = [json.loads(l) for l in Path(jsonl_path).read_text().splitlines() if l.strip()]
    if not rows:
        raise ValueError(f"{jsonl_path} is empty")
    ep = [r["epoch"] for r in rows]
    loss_keys = keys or [k for k in rows[0] if k.startswith("train/") and k != "train/n_annotated"]
    val_keys = [k for k in rows[-1] if k.startswith("val/")]

    # Side by side, the two panels' legends collide with each other's axis labels below about
    # 5 inches -- which is every column-width use. Below that, stack the two panels instead of
    # letting them overlap, and let tight_layout resolve the spacing rather than bbox_inches,
    # which only trims the outside and cannot fix an interior collision.
    wide = width >= 5.0
    if wide:
        fig, axes = plt.subplots(1, 2, figsize=(width, width * 0.34))
    else:
        fig, axes = plt.subplots(2, 1, figsize=(width, width * 0.88), sharex=True)
    for i, k in enumerate(sorted(loss_keys)):
        y = [r.get(k) for r in rows]
        axes[0].plot(ep, y, color=PALETTE[i % len(PALETTE)],
                     linestyle=["-", "--", ":", "-."][i % 4],
                     label=k.split("/", 1)[1])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("training loss (log)")
    # When stacked, the panels share an x-axis, so the upper panel's own "epoch" label would sit
    # in the gap between them saying nothing.
    if wide:
        axes[0].set_xlabel("epoch")
    axes[0].legend(ncol=2, frameon=False, fontsize=6, loc="lower left")

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
    axes[1].legend(ncol=2, frameon=False, fontsize=6, loc="lower right")
    if title:
        fig.suptitle(title, fontsize=8.5, y=1.02)
    fig.tight_layout(pad=0.4)
    return _save(fig, out_path)


# ======================================================================================
# ablation bars
# ======================================================================================
def plot_ablation_bars(rows, metric, out_path, baseline_key=None, title=None,
                       width: float = COL_W, max_rows: int = 14,
                       always_keep=("e1", "e3"),
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

    # Keep the chart readable. At 0.26 in per bar, every row of a 121-row comparison table makes
    # a figure 32 in tall for a 3.35 in slot. Keep the most extreme rows -- those are the ones a
    # reader is looking for -- plus the reference runs, which have to be present for any of the
    # others to mean anything.
    if len(rows) > max_rows:
        keep = {str(r.get(label_key)) for r in rows if str(r.get(label_key)) in set(always_keep)}
        ranked = sorted(rows, key=lambda r: float(r.get(metric, float("nan"))))
        n_ends = max(1, (max_rows - len(keep)) // 2)
        chosen = ranked[:n_ends] + ranked[-n_ends:]
        chosen += [r for r in rows if str(r.get(label_key)) in keep and r not in chosen]
        # Two statements, not a tuple assignment: the right-hand side of `seen, rows = set(), [...]`
        # is evaluated in full before either name is bound, so a comprehension would reference
        # `seen` before it exists.
        seen: set[str] = set()
        rows = []
        for r in chosen:
            lbl = str(r.get(label_key))
            if lbl not in seen:
                seen.add(lbl)
                rows.append(r)
        print(f"[viz] ablation bars: showing {len(rows)} of {len(ranked)} rows "
              f"(the extremes plus {sorted(keep)}); the full set is in comparison.csv")

    lo_k, hi_k = f"{metric}_lo", f"{metric}_hi"
    labels = [str(r.get(label_key, "?")) for r in rows]
    # The chart is filtered to one split, so a suffix every label shares is noise on every row.
    for suf in ("_rs_test", "_rs_val", "_pseudo"):
        if labels and all(l.endswith(suf) for l in labels):
            labels = [l[: -len(suf)] for l in labels]
            break
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
    fig, ax = plt.subplots(figsize=(width, 0.26 * len(rows) + 1.0))
    err = _errorbars(vals[order], lo[order], hi[order], out_path)
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
def plot_augmentation_grid(img, out_path, modes=None, seed: int = 0, n_draws: int = 5,
                           width: float = COL_W, ncol: int = 3):
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

    # One flat list in a grid rather than two ragged rows. Panel size is what decides whether
    # this figure works: the reader has to be able to see what each policy did to the image. At
    # the default 3 columns across one IEEE column that is ~2.8 cm per panel; at 7 columns it
    # was ~1.2 cm, which showed nothing.
    panels = top + draws
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(width, width / ncol * 0.80 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, item in zip(axes, panels):
        name, im = item
        ax.imshow(im)
        ax.set_title(name, fontsize=6.8, pad=2)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for ax in axes[len(panels):]:
        ax.axis("off")
    return _save(fig, out_path)


# ======================================================================================
# domain gap
# ======================================================================================
def plot_domain_gap(result_paths, labels, out_path, metrics=None, title=None,
                    width: float = COL_W):
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
    fig, ax = plt.subplots(figsize=(width, width * 0.62))
    for s in range(n_series):
        lo = np.array([c[0] for c in cis[s]]); hi = np.array([c[1] for c in cis[s]])
        err = _errorbars(data[s], lo, hi, out_path)
        ax.bar(x + s * w - 0.4 + w / 2, data[s], width=w * 0.92,
               yerr=err, color=PALETTE[s % len(PALETTE)], hatch=HATCH[s % len(HATCH)],
               edgecolor="#222222", label=labels[s],
               error_kw={"elinewidth": 0.7, "capsize": 1.6, "ecolor": "#222222"})
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ") for m in metrics], rotation=18, ha="right")
    ax.set_ylabel("score"); ax.set_ylim(0, 1.02)
    # Above the axes, not inside them. With six metric groups there is no interior region the
    # legend can occupy without covering bars, and a legend over the data is worse than a
    # slightly taller figure.
    # Two legend columns, not four. bbox_inches="tight" grows the saved figure to fit whatever
    # the legend needs, so a single wide row silently makes the figure 1.5x wider than the column
    # it is printed in -- and \includegraphics then scales every label down to compensate.
    ax.legend(frameon=False, ncol=2 if width < 5.0 else min(4, n_series), loc="lower center",
              bbox_to_anchor=(0.5, 1.01), borderaxespad=0.0)
    if title:
        fig.suptitle(title, y=1.12, fontsize=8.5)
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
    ap.add_argument("--per-run", nargs="*", default=None, metavar="STEM",
                    help="result stems that get their own confusion matrix and reliability "
                         "curve (default: all of --results). Every --results file is still read "
                         "for the comparison table and the ablation bars.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pre = a.prefix
    made = []

    per_run = set(a.per_run) if a.per_run else None
    skipped = 0
    for p in a.results:
        stem = Path(p).stem
        if per_run is not None and stem not in per_run:
            skipped += 1
            continue
        d = utils.load_json(p)
        if "confusion_matrix" in d:
            made.append(plot_confusion_matrix(d["confusion_matrix"],
                                              d.get("class_names", utils.GESTURES),
                                              out / f"{pre}cm_{stem}", normalise=True,
                                              title=f"confusion matrix — {stem}",
                                              width=COL_W))
        if d.get("reliability"):
            made.append(plot_reliability(d["reliability"], out / f"{pre}rel_{stem}",
                                         title=f"calibration — {stem}"))
    if skipped:
        print(f"[viz] per-run figures for {len(per_run & {Path(x).stem for x in a.results})} of "
              f"{len(a.results)} results; {skipped} skipped (still counted in the comparisons)")
    if len(a.results) > 1:
        # The domain-gap figure is a *story*, not a dump: RealSense -> synthetic shift -> phone,
        # for the two models being compared. Feeding it every result file produces a wall of
        # bars in alphabetical order that answers no question. Curate when the canonical runs
        # are present; fall back to whatever was passed otherwise.
        # Four series, in the order the brief's question is asked: each model on its own source
        # domain, then the same weights on the shifted one. Both source bars have to be there --
        # without E3's, the figure shows the recovery but not that it was free, and "no in-domain
        # cost" is half the claim. The shifted bar is the smartphone set when it exists and the
        # synthetic proxy until then, and the label says which, because they are not the same
        # measurement and a figure that blurs them is worse than no figure.
        shifted_e1 = ("e1_phone", "E2 phone (zero-shot)") if "e1_phone" in {
            Path(p).stem for p in a.results} else ("e1_pseudo", "E1 proxy shift")
        shifted_e3 = ("e3_phone", "E3 phone (+CPR)") if "e3_phone" in {
            Path(p).stem for p in a.results} else ("e3_pseudo", "E3 proxy shift")
        want = [("e1_rs_test", "E1 RealSense test"), shifted_e1,
                ("e3_rs_test", "E3 RealSense test"), shifted_e3]
        by_stem = {Path(p).stem: p for p in a.results}
        curated = [(by_stem[k], lbl) for k, lbl in want if k in by_stem]
        if len(curated) >= 2:
            made.append(plot_domain_gap([c[0] for c in curated], [c[1] for c in curated],
                                        out / f"{pre}domain_gap"))
        else:
            labels = a.labels or [Path(p).stem for p in a.results]
            made.append(plot_domain_gap(a.results, labels, out / f"{pre}domain_gap"))
        from src.evaluate import compare_runs
        rows = compare_runs(a.results, out_csv=str(out / f"{pre}comparison.csv"))
        # comparison.csv keeps every row; the charts show one split, or the same configuration
        # appears three times with three different bars and the ranking means nothing.
        test_rows = [r for r in rows if str(r.get("split")) == "test"
                     and not r.get("pseudo_target")] or rows
        # compare_runs names a row by its result-file stem, so the references are "e1_rs_test"
        # and "e3_rs_test", not "e1"/"e3".
        REFS = ("e1_rs_test", "e3_rs_test")
        # Drawn as deltas against E3 rather than absolute scores. Every configuration here scores
        # between about 0.85 and 0.92, so a zero-based bar chart of absolutes is thirteen bars of
        # the same length; the question the figure exists to answer is how each differs from the
        # proposed method, and that is what a delta chart shows.
        base = "e3_rs_test" if any(str(r.get("run")) == "e3_rs_test" for r in test_rows) else None
        made.append(plot_ablation_bars(test_rows, "cls_macro_f1",
                                       out / f"{pre}ablation_macro_f1",
                                       baseline_key=base, always_keep=REFS))
        made.append(plot_ablation_bars(test_rows, "seg_iou_hand",
                                       out / f"{pre}ablation_seg_iou",
                                       baseline_key=base, always_keep=REFS))
    if a.curves:
        made.append(plot_training_curves(a.curves, out / f"{pre}curves"))
    if a.examples:
        z = np.load(a.examples, allow_pickle=True)
        made.append(plot_qualitative(list(z["examples"])[:12], out / f"{pre}qualitative"))
    if a.aug_source:
        from PIL import Image
        src_img = np.asarray(Image.open(a.aug_source).convert("RGB"))
        # Two versions, because the choice between them is a page-budget decision the report
        # makes, not one this script should make for it. The column version is what the .tex
        # includes: six panels telling the E1 / E3 / held-out-proxy story at 2.8 cm each. The
        # full-width version additionally shows the competing randomisers and more CPR draws,
        # for use in a \begin{figure*} if the six pages turn out to have room.
        made.append(plot_augmentation_grid(
            src_img, out / f"{pre}augmentation_grid",
            modes=["none", "jitter"], n_draws=3, width=COL_W, ncol=3))
        made.append(plot_augmentation_grid(
            src_img, out / f"{pre}augmentation_grid_full",
            modes=["none", "jitter", "cpr", "randconv", "aprs", "augmix"],
            n_draws=5, width=FULL_W, ncol=4))
    for m in made:
        print(f"[viz] {m}")
    if not made:
        print("[viz] nothing to draw — pass --results / --curves / --examples / --aug-source")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
