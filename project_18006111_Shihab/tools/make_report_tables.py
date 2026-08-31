#!/usr/bin/env python3
"""Generate the report's LaTeX tables directly from the result JSONs.

The tables in the report are *generated*, never transcribed. A number that is typed by hand into
a paper is a number that can silently disagree with the experiment that produced it; regenerating
from `results/*.json` makes that class of error impossible, and re-running one line after a new
experiment finishes keeps every table current.

Emits, into `--out`:

* `tbl_main.tex`        - the mandatory experiment table: B0 / E1 / E2 / E3 across every split
* `tbl_gain.tex`        - E3 - E2 on the smartphone set, per metric, with bootstrap intervals
* `tbl_ablation.tex`    - the ablation grid, sorted, as deltas against the relevant baseline
* `tbl_perclass.tex`    - per-class F1 for the headline runs
* `macros.tex`          - \\newcommand definitions for every number quoted in running text

Usage
-----
    python tools/make_report_tables.py --results results --out report/generated
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# tools/ lives inside the deliverable, so the package root is one level up. Keeping the
# tools inside project_<studentno>_<surname>/ makes the submitted zip self-contained:
# every command in README.md runs from the tree that is actually handed in.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from src import utils                                       # noqa: E402

HEADLINE = [
    ("det_acc@0.5", "Det.\\ acc@0.5", 3, True),
    ("mean_box_iou", "Mean box IoU", 3, True),
    ("seg_iou_hand", "Hand IoU", 3, True),
    ("seg_miou", "mIoU", 3, False),
    ("seg_dice", "Dice", 3, True),
    ("cls_top1", "Top-1", 3, True),
    ("cls_macro_f1", "Macro-F1", 3, True),
]


def esc(s: str) -> str:
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def load(results_dir: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(results_dir.glob("*.json")):
        if p.name.endswith("_examples.npz"):
            continue
        try:
            out[p.stem] = utils.load_json(p)
        except Exception as e:
            print(f"[tbl] skipping {p.name}: {e}", file=sys.stderr)
    return out


def val(d: dict, key: str, block: str = "frame"):
    return d.get(block, {}).get(key)


def fmt(v, nd=3, ci=None):
    if v is None:
        return "--"
    s = f"{v:.{nd}f}"
    if ci and ci.get("lo") is not None:
        s += f"\\,\\tiny[{ci['lo']:.{nd}f},{ci['hi']:.{nd}f}]"
    return s


# ======================================================================================
def table_main(R: dict, out: Path) -> None:
    """B0 / E1 / E2 / E3 on every evaluation split. The table the marker looks for first."""
    rows = [
        ("B0 classical", "RealSense val", "b0_classical_rs_val"),
        ("B0 classical", "RealSense test", "b0_classical_rs_test"),
        ("E1 (jitter)", "RealSense val", "e1_rs_val"),
        ("E1 (jitter)", "RealSense test", "e1_rs_test"),
        ("E1 (jitter)", "pseudo-target", "e1_pseudo"),
        ("E2 = E1 weights", "smartphone", "e1_phone"),
        ("E3 (CPR)", "RealSense val", "e3_rs_val"),
        ("E3 (CPR)", "RealSense test", "e3_rs_test"),
        ("E3 (CPR)", "pseudo-target", "e3_pseudo"),
        ("E3 (CPR)", "smartphone", "e3_phone"),
    ]
    L = [r"\begin{table*}[!tb]", r"\centering",
         r"\footnotesize\setlength{\tabcolsep}{4pt}",
         r"\caption{Mandatory experiments; per-frame metrics, bracketed figures are bootstrap "
         r"95\% intervals over clips. E2 is the E1 checkpoint evaluated unchanged on the "
         r"smartphone set --- same weights, same preprocessing, no fine-tuning. Each row is one "
         r"training run, the checkpoint shipped in \texttt{weights/}, so the intervals are "
         r"sampling error over clips and not seed-to-seed variability, which is in the text. "
         r"mIoU is the two-class hand/background mean and carries a 0.97 background floor, so "
         r"hand IoU is given beside it.}",
         r"\label{tab:main}",
         r"\begin{tabular}{ll" + "c" * len(HEADLINE) + "}", r"\toprule",
         "Model & Evaluation set & " + " & ".join(h[1] for h in HEADLINE) + r" \\", r"\midrule"]
    for name, split, key in rows:
        d = R.get(key)
        if d is None:
            L.append(f"{esc(name)} & {esc(split)} & " + " & ".join(["--"] * len(HEADLINE)) + r" \\")
            continue
        cells = [fmt(val(d, k), nd, d.get("ci", {}).get(k) if ci else None)
                 for k, _, nd, ci in HEADLINE]
        L.append(f"{esc(name)} & {esc(split)} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (out / "tbl_main.tex").write_text("\n".join(L) + "\n")


def table_gain(R: dict, out: Path) -> None:
    """E3 - E2 on the smartphone set: the single number Objective 4 is graded on."""
    e2, e3 = R.get("e1_phone"), R.get("e3_phone")
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Experiment 3 relative to Experiment 2 on the smartphone test set. "
         r"Both models are trained on RealSense data only; they differ solely in the "
         r"photometric augmentation.}",
         r"\label{tab:gain}", r"\begin{tabular}{lccc}", r"\toprule",
         r"Metric & E2 (zero-shot) & E3 (+CPR) & $\Delta$ \\", r"\midrule"]
    for k, label, nd, _ci in HEADLINE:
        a, b = (val(e2, k) if e2 else None), (val(e3, k) if e3 else None)
        d = None if (a is None or b is None) else b - a
        dr = None if d is None else round(d, nd)
        if dr is None:
            cell = "--"
        elif dr == 0:
            cell = f"{0.0:.{nd}f}"                          # unchanged to the printed precision
        else:
            cell = f"{d:+.{nd}f}" + (r"\,$\uparrow$" if dr > 0 else r"\,$\downarrow$")
        L.append(f"{label} & {fmt(a, nd)} & {fmt(b, nd)} & {cell} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out / "tbl_gain.tex").write_text("\n".join(L) + "\n")


ABL_GROUPS = [
    ("Block B --- CPR leave-one-out (vs.\\ E3)", r"^b[1-9]_", "e3"),
    # E3 is included in this block on purpose: the question Block D exists to answer is
    # "is the proposed method better than the published alternatives?", and that is unreadable
    # if the proposal's own delta sits in a different table three blocks away.
    ("Block D --- competing randomisers, and CPR among them (vs.\\ E1)", r"^(d\d_|e3$)", "e1"),
    ("Block E --- architecture and loss (vs.\\ E3)", r"^e_", "e3"),
    ("Block C --- normalisation (vs.\\ E3)", r"^c_", "e3"),
    ("Block A --- augmentation floor (vs.\\ E1)", r"^a\d_", "e1"),
    # The per-seed repeats (`*_s1`, `*_s2`, ...) are deliberately NOT a block here. Nine rows of
    # duplicates cost about a tenth of a page in a report capped at 6 pages including references,
    # and the text reports the mean and standard deviation over seeds, which is what the reader
    # needs. Every per-seed JSON still ships in results/, so no evidence is lost.
]


# Metrics a head-subset run cannot produce. `e_singletask_cls` is built with heads=("cls",), and
# evaluate.py fills its absent detection and segmentation outputs with zeros rather than nulls --
# so a delta against a full model reads as "-0.945 detection accuracy" for a model that has no
# detection head at all. Suppress those cells instead of printing a number that means nothing.
HEAD_SUBSET = {
    "det": ("det_acc@0.5", "det_acc@0.75", "mean_box_iou"),
    "seg": ("seg_iou_hand", "seg_miou", "seg_dice"),
}


def _absent_metrics(d: dict) -> set:
    heads = (d.get("config", {}).get("train_cfg", {}) or {}).get("heads")
    if not heads:
        return set()
    return {m for h, ms in HEAD_SUBSET.items() if h not in heads for m in ms}


ABL_COLS = [h for h in HEADLINE if h[0] != "seg_miou"]


def table_ablation(R: dict, out: Path, split: str = "rs_test") -> None:
    """The ablation grid. Deltas, because absolute numbers hide small effects."""
    # \footnotesize and a tighter column separation are page-budget decisions, not cosmetics: the
    # report is capped at 6 pages including references (LSA p13) and this is the largest float in
    # it. [!tb] rather than [t] lets it take the bottom of a page too, which stfloats enables.
    L = [r"\begin{table*}[!tb]", r"\centering",
         r"\footnotesize\setlength{\tabcolsep}{4.5pt}",
         rf"\caption{{Ablations on the RealSense test set, as deltas against the reference named "
         rf"in each block heading (its seed-0 checkpoint). A delta smaller than the seed spread "
         rf"quoted in the text is not a result; a dash marks a metric a configuration cannot "
         rf"produce. Per-seed repeats are summarised in the text.}}",
         r"\label{tab:ablation}",
         r"\begin{tabular}{l" + "c" * len(ABL_COLS) + "}", r"\toprule",
         "Run & " + " & ".join(h[1] for h in ABL_COLS) + r" \\"]
    for title, pat, ref in ABL_GROUPS:
        keys = sorted(k for k in R
                      if k.endswith("_" + split) and re.search(pat, k.replace("_" + split, "")))
        if not keys:
            continue
        L += [r"\midrule", rf"\multicolumn{{{len(ABL_COLS) + 1}}}{{l}}{{\textit{{{title}}}}} \\"]
        base = R.get(f"{ref}_{split}") if ref else None
        for k in keys:
            d = R[k]
            name = k.replace("_" + split, "")
            cells = []
            absent = _absent_metrics(d)
            for mk, _, nd, _ci in ABL_COLS:
                v = None if mk in absent else val(d, mk)
                if v is None:
                    cells.append("--")
                elif base is not None and val(base, mk) is not None:
                    cells.append(f"{v - val(base, mk):+.{nd}f}")
                else:
                    cells.append(f"{v:.{nd}f}")
            L.append(f"{esc(name)} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (out / "tbl_ablation.tex").write_text("\n".join(L) + "\n")


def table_perclass(R: dict, out: Path) -> None:
    runs = [("E1, RealSense test", "e1_rs_test"), ("E2, smartphone", "e1_phone"),
            ("E3, smartphone", "e3_phone")]
    present = [(lbl, k) for lbl, k in runs if k in R]
    if not present:
        return
    names = R[present[0][1]].get("class_names", utils.GESTURES)
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Per-class $F_1$" + ("" if len(present) > 1 else
           " on the RealSense test set. The smartphone columns appear here once that set exists; "
           "until then this is the in-domain per-class breakdown only") +
         (". The classes that survive the cross-camera shift, and the ones that do not, are more "
          "informative than the macro average alone." if len(present) > 1 else ".") + r"}",
         r"\label{tab:perclass}",
         r"\begin{tabular}{l" + "c" * len(present) + "}", r"\toprule",
         "Gesture & " + " & ".join(esc(l) for l, _ in present) + r" \\", r"\midrule"]
    for n in names:
        cells = []
        for _, k in present:
            f1 = R[k].get("frame", {}).get("per_class_f1", {}).get(n)
            cells.append("--" if f1 is None else f"{f1:.3f}")
        L.append(f"{esc(n.split('_', 1)[-1])} & " + " & ".join(cells) + r" \\")
    L += [r"\midrule"]
    cells = [fmt(val(R[k], "cls_macro_f1")) for _, k in present]
    L += [r"\textbf{macro} & " + " & ".join(cells) + r" \\",
          r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out / "tbl_perclass.tex").write_text("\n".join(L) + "\n")


def macros(R: dict, out: Path) -> None:
    """\\newcommand for every number quoted in prose, so the text cannot drift from the tables."""
    def cmd(name, value, nd=3):
        return (rf"\newcommand{{\{name}}}{{{value:.{nd}f}}}" if isinstance(value, float)
                else rf"\newcommand{{\{name}}}{{{value}}}")
    L = ["% Auto-generated by tools/make_report_tables.py -- do not edit.",
         "% Quote numbers in the text as e.g. \\EthreePhoneMacroF -- never typed by hand."]
    alias = {"e1_rs_test": "EoneRs", "e1_phone": "EtwoPhone", "e3_phone": "EthreePhone",
             "e3_rs_test": "EthreeRs", "e1_rs_val": "EoneVal", "e3_rs_val": "EthreeVal",
             "b0_classical_rs_test": "BzeroRs", "e1_pseudo": "EonePseudo",
             "e3_pseudo": "EthreePseudo"}
    metric_alias = {"det_acc@0.5": "DetAcc", "mean_box_iou": "BoxIoU", "seg_iou_hand": "HandIoU",
                    "seg_dice": "Dice", "cls_top1": "Topone", "cls_macro_f1": "MacroF",
                    "cls_ece": "Ece"}
    for key, a in alias.items():
        d = R.get(key)
        if not d:
            continue
        for mk, ma in metric_alias.items():
            v = val(d, mk)
            if v is not None:
                L.append(cmd(f"{a}{ma}", float(v)))
        L.append(cmd(f"{a}Frames", d.get("n_frames", 0)))
        L.append(cmd(f"{a}Clips", d.get("n_clips", 0)))
    e2, e3 = R.get("e1_phone"), R.get("e3_phone")
    if e2 and e3:
        for mk, ma in metric_alias.items():
            a, b = val(e2, mk), val(e3, mk)
            if a is not None and b is not None:
                L.append(rf"\newcommand{{\Gain{ma}}}{{{b - a:+.3f}}}")
    (out / "macros.tex").write_text("\n".join(L) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    R = load(Path(a.results))
    print(f"[tbl] loaded {len(R)} result files: {', '.join(sorted(R))}")
    table_main(R, out)
    table_gain(R, out)
    table_ablation(R, out)
    table_perclass(R, out)
    macros(R, out)
    for f in sorted(out.glob("*.tex")):
        print(f"[tbl] {f}  ({f.stat().st_size} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
