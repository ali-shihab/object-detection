"""Evaluation for the multi-task RGB hand model (COMP0248 CW1 LSA).

Implements the metric set of s9 (which is the brief's p9 list plus the extras
that make small ablation deltas interpretable):

* detection      - accuracy@0.5 IoU (also 0.75 / 0.9), mean box IoU
* segmentation   - hand IoU, background IoU, mIoU, Dice
* classification - top-1, macro-F1, per-class F1, 10x10 confusion matrix
* calibration    - ECE and a reliability curve for the required gesture confidence
* every metric per-frame (headline) **and** per-clip, plus bootstrap 95% CIs over clips

Two rules are enforced here rather than left to the caller:

* Frames with ``has_mask=False`` are excluded from every detection and segmentation metric
  and counted in ``n_boxes_skipped`` / ``n_seg_skipped``. Scoring a placeholder zero box as a
  miss would make 86% of the RealSense training frames look like detection failures.
* Confidence intervals resample **clips**, not frames. The 15 frames of a clip have a median
  consecutive mask IoU of 0.649, so a frame-level bootstrap would report an
  interval several times narrower than the data supports.

Usage
-----
    python -m src.evaluate --ckpt runs/e1/best.pt --index data/realsense_test/index.json \
        --split test --out results/e1_test.json --examples 24
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from src import utils                                     # noqa: E402
from src.model import HandNet, decode_detection           # noqa: E402
from src.dataloader import HandGestureDataset             # noqa: E402
from src.augment import pseudo_target_transform           # noqa: E402

MEAN = torch.tensor(utils.IMAGENET_MEAN).view(1, 3, 1, 1)
STD = torch.tensor(utils.IMAGENET_STD).view(1, 3, 1, 1)


# ======================================================================================
# calibration
# ======================================================================================
def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15):
    """ECE with equal-width bins, plus the reliability-curve data.

    The brief requires the model to emit a gesture confidence (p5). A confidence that is never
    checked is decoration: ECE is the cheapest statement of whether the number means anything,
    and it is the natural place to see whether cross-camera transfer breaks calibration as well
    as accuracy - a model that is wrong *and* confident on phone images fails differently from
    one that is wrong and knows it.
    """
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    if conf.size == 0:
        return 0.0, []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, curve = 0.0, []
    n = conf.size
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        k = int(m.sum())
        if k == 0:
            curve.append({"lo": float(lo), "hi": float(hi), "n": 0,
                          "conf": None, "acc": None})
            continue
        c, a = float(conf[m].mean()), float(correct[m].mean())
        ece += (k / n) * abs(a - c)
        curve.append({"lo": float(lo), "hi": float(hi), "n": k, "conf": c, "acc": a})
    return float(ece), curve


# ======================================================================================
# bootstrap
# ======================================================================================
def bootstrap_ci(values_by_clip: dict[str, list[float]], n_boot: int = 1000,
                 seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile bootstrap over clips for a per-frame metric.

    Resamples clips with replacement, recomputes the frame-weighted mean each time. The point
    estimate is the frame mean over the real data, so it does not move with the seed; only the
    interval does.
    """
    keys = list(values_by_clip)
    if not keys:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n_clips": 0}
    flat = [v for k in keys for v in values_by_clip[k]]
    point = float(np.mean(flat)) if flat else 0.0
    rng = np.random.default_rng(seed)
    idx = np.arange(len(keys))
    draws = np.empty(n_boot, dtype=np.float64)
    per_clip = [np.asarray(values_by_clip[k], dtype=np.float64) for k in keys]
    for b in range(n_boot):
        pick = rng.choice(idx, size=len(idx), replace=True)
        cat = np.concatenate([per_clip[i] for i in pick]) if len(pick) else np.zeros(0)
        draws[b] = cat.mean() if cat.size else 0.0
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": point, "lo": float(lo), "hi": float(hi), "n_clips": len(keys)}


# ======================================================================================
# core
# ======================================================================================
@torch.no_grad()
def evaluate_model(model, loader, device, *, amp: bool = True,
                   collect_examples: int = 0, pseudo_target: bool = False,
                   n_boot: int = 1000, boot_seed: int = 0,
                   collect_predictions: bool = False) -> dict:
    """Run the model over ``loader`` and return the full metric block.

    ``pseudo_target=True`` applies the held-out phone-style shift to
    each image. It is applied to the **de-normalised uint8 image and then re-normalised**,
    because the shift models an imaging pipeline that operates on 8-bit display-referred pixels;
    applying it to a normalised float tensor would be modelling nothing physical, and the
    quantisation step (which is part of what a phone actually does) would be skipped.
    """
    model.eval()
    acc = utils.MetricAccumulator(n_classes=10, class_names=utils.GESTURES, iou_thresh=0.5)
    extra_thresh = (0.75, 0.9)
    hits: dict[float, dict[str, list[float]]] = {t: {} for t in extra_thresh}
    by_clip: dict[str, dict[str, list[float]]] = {}
    conf_all, correct_all = [], []
    examples: list[dict] = []
    predictions: list[dict] = []
    cls_taken: dict[int, int] = {}
    n_failures = 0
    mean_d, std_d = MEAN.to(device), STD.to(device)
    n_seen = 0
    t0 = time.time()
    # One generator for the whole pass, created outside the batch loop. Constructing it inside
    # restarts the draw sequence every batch, which makes the transform a frame receives depend
    # on its position within its batch - i.e. on --batch-size. A pseudo-target number reported
    # at batch 32 would then not be the number model selection saw at batch 48.
    pt_rng = np.random.default_rng(1234)

    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        if pseudo_target:
            u8 = (img * std_d + mean_d).clamp(0, 1).mul(255).round().to(torch.uint8)
            u8 = u8.permute(0, 2, 3, 1).cpu().numpy()
            u8 = np.stack([pseudo_target_transform(f, pt_rng) for f in u8])
            img = torch.from_numpy(u8).permute(0, 3, 1, 2).to(device).float().div_(255)
            img = (img - mean_d) / std_d

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(amp and device.type == "cuda")):
            out = model(img)
        out = {k: (v.float() if torch.is_tensor(v) else v) for k, v in out.items()}

        b = img.shape[0]
        has = batch["has_mask"].to(device)
        gt_box = batch["box"].to(device)
        gt_mask = batch["mask"].to(device)
        labels = batch["label"].to(device)

        if "heat" in out:
            pred_box, det_conf = decode_detection(out, k=1)
            ious = utils.box_iou(pred_box.float(), gt_box.float())
        else:
            pred_box = torch.zeros((b, 4), device=device)
            det_conf = torch.zeros((b,), device=device)
            ious = torch.zeros((b,), device=device)

        if "seg" in out:
            probs = torch.sigmoid(out["seg"])
            segs = utils.seg_scores(probs, gt_mask)
        else:
            probs = torch.zeros_like(gt_mask)
            segs = {k: torch.zeros((b,), device=device)
                    for k in ("iou_hand", "iou_bg", "miou", "dice")}

        if "cls" in out:
            p = torch.softmax(out["cls"], dim=1)
            cls_conf, pred_cls = p.max(dim=1)
        else:
            cls_conf = torch.zeros((b,), device=device)
            pred_cls = torch.zeros((b,), dtype=torch.long, device=device)

        meta = batch.get("meta")
        for i in range(b):
            ck = _clip_key(meta, i, n_seen)
            hm = bool(has[i].item())
            iou = float(ious[i].item()) if hm else None
            seg_i = {k: float(v[i].item()) for k, v in segs.items()} if hm else None
            acc.add(clip_key=ck, box_iou=iou, seg=seg_i,
                    gt_class=int(labels[i].item()), pred_class=int(pred_cls[i].item()),
                    cls_conf=float(cls_conf[i].item()), det_conf=float(det_conf[i].item()))
            if hm:
                for t in extra_thresh:
                    hits[t].setdefault(ck, []).append(float(iou >= t))
                by_clip.setdefault("mean_box_iou", {}).setdefault(ck, []).append(iou)
                for k, v in seg_i.items():
                    by_clip.setdefault(f"seg_{k}", {}).setdefault(ck, []).append(v)
                by_clip.setdefault("det_acc@0.5", {}).setdefault(ck, []).append(float(iou >= 0.5))
            gt_i = int(labels[i].item())
            correct_i = int(pred_cls[i].item()) == gt_i
            ok = float(correct_i)
            by_clip.setdefault("cls_top1", {}).setdefault(ck, []).append(ok)
            conf_all.append(float(cls_conf[i].item()))
            correct_all.append(ok)

            if collect_predictions:
                # One row per input image: the four required outputs (box, mask, class,
                # confidence) beside the ground truth they are scored against. The aggregate
                # JSON says how well the model did; this says what it actually predicted, which
                # is what makes any number in the report checkable frame by frame.
                gb = gt_box[i].tolist()
                pb = pred_box[i].tolist()
                predictions.append({
                    "id": _meta_field(meta, i, "id", f"_frame{n_seen + i}"),
                    "rgb": _meta_field(meta, i, "rgb", ""),
                    "clip": ck,
                    "subject": _meta_field(meta, i, "subject", ""),
                    "gesture": _meta_field(meta, i, "gesture", ""),
                    "frame": _meta_field(meta, i, "frame", n_seen + i),
                    "gt_class": gt_i,
                    "gt_class_name": utils.GESTURES[gt_i],
                    "pred_class": int(pred_cls[i].item()),
                    "pred_class_name": utils.GESTURES[int(pred_cls[i].item())],
                    "cls_conf": round(float(cls_conf[i].item()), 6),
                    "correct": int(correct_i),
                    "det_conf": round(float(det_conf[i].item()), 6),
                    "pred_xmin": round(pb[0], 2), "pred_ymin": round(pb[1], 2),
                    "pred_xmax": round(pb[2], 2), "pred_ymax": round(pb[3], 2),
                    "pred_mask_area_frac": round(
                        float((probs[i, 0] >= 0.5).float().mean().item()), 6),
                    "has_gt_mask": int(hm),
                    "gt_xmin": round(gb[0], 2) if hm else "",
                    "gt_ymin": round(gb[1], 2) if hm else "",
                    "gt_xmax": round(gb[2], 2) if hm else "",
                    "gt_ymax": round(gb[3], 2) if hm else "",
                    "box_iou": round(iou, 6) if hm else "",
                    "seg_iou_hand": round(seg_i["iou_hand"], 6) if hm else "",
                    "seg_miou": round(seg_i["miou"], 6) if hm else "",
                    "seg_dice": round(seg_i["dice"], 6) if hm else "",
                })

            # Example selection is STRATIFIED, not "the first N the loader hands over".
            # The loader is unshuffled, so taking the first 24 gives 24 consecutive frames of
            # one clip of one class - which is what the first version of this figure did, and
            # it is useless as the brief's "qualitative overlays on a few val/test images".
            # Cap per class, and always keep room for mistakes: a qualitative figure showing
            # only successes tells the reader nothing about how the model fails.
            per_cls_cap = max(1, collect_examples // 10)
            fail_budget = max(1, collect_examples // 4)
            take = False
            if collect_examples:
                if not correct_i and n_failures < fail_budget:
                    take = True
                elif (cls_taken.get(gt_i, 0) < per_cls_cap
                      and len(examples) < collect_examples):
                    take = True
            if take:
                if len(examples) >= collect_examples:
                    # replace a correct example so a newly-found failure still gets in
                    for j, ex in enumerate(examples):
                        if ex["pred_class"] == ex["gt_class"]:
                            examples.pop(j)
                            cls_taken[ex["gt_class"]] = cls_taken.get(ex["gt_class"], 1) - 1
                            break
                    else:
                        take = False
            if take:
                cls_taken[gt_i] = cls_taken.get(gt_i, 0) + 1
                n_failures += 0 if correct_i else 1
                examples.append({
                    "image": (img[i].detach().cpu() * STD[0] + MEAN[0]).clamp(0, 1).numpy(),
                    "gt_mask": gt_mask[i, 0].detach().cpu().numpy(),
                    "pred_mask": probs[i, 0].detach().cpu().numpy(),
                    "gt_box": gt_box[i].detach().cpu().numpy().tolist(),
                    "pred_box": pred_box[i].detach().cpu().numpy().tolist(),
                    "gt_class": int(labels[i].item()),
                    "pred_class": int(pred_cls[i].item()),
                    "cls_conf": float(cls_conf[i].item()),
                    "det_conf": float(det_conf[i].item()),
                    "has_mask": hm,
                    "clip": ck,
                })
        n_seen += b

    res = acc.summary()
    ece, curve = expected_calibration_error(np.array(conf_all), np.array(correct_all))
    res["frame"]["cls_ece"] = ece
    res["reliability"] = curve
    for t in extra_thresh:
        flat = [v for vs in hits[t].values() for v in vs]
        res["frame"][f"det_acc@{t:g}"] = float(np.mean(flat)) if flat else 0.0
        res["clip"][f"det_acc@{t:g}"] = float(
            np.mean([float(np.mean(v)) for v in hits[t].values()])) if hits[t] else 0.0
    res["ci"] = {k: bootstrap_ci(v, n_boot=n_boot, seed=boot_seed) for k, v in by_clip.items()}
    res["eval_seconds"] = round(time.time() - t0, 2)
    res["pseudo_target"] = bool(pseudo_target)
    if collect_examples:
        # Deterministic order: by class, then failures first within a class, so the figure reads
        # left-to-right as a tour of the label set rather than in loader order.
        examples.sort(key=lambda e: (e["gt_class"], e["pred_class"] == e["gt_class"]))
        res["examples"] = examples
        res["n_example_failures"] = sum(1 for e in examples
                                        if e["pred_class"] != e["gt_class"])
    if collect_predictions:
        res["predictions"] = predictions
    return res


def _meta_field(meta, i: int, key: str, default):
    """Pull one field out of a collated meta dict, whichever shape the collate produced.

    ``default_collate`` turns a per-item dict of scalars into a dict of batched sequences, but a
    custom collate (or a single-item batch) can leave it as a list of dicts, and a string field
    can arrive already collapsed. Handle all three rather than assume one.
    """
    if isinstance(meta, dict) and key in meta:
        v = meta[key]
        if isinstance(v, (str, int, float)):
            return v
        try:
            v = v[i]
        except (TypeError, IndexError, KeyError):
            return default
        return v.item() if hasattr(v, "item") else v
    if isinstance(meta, (list, tuple)) and i < len(meta) and isinstance(meta[i], dict):
        return meta[i].get(key, default)
    return default


def _clip_key(meta, i: int, offset: int) -> str:
    """Prefer the loader's clip identity; fall back to a per-frame key.

    Falling back to a unique per-frame key makes the clip block degenerate to the frame block
    rather than silently lumping every frame into one giant pseudo-clip, which would make the
    clip-level CIs meaningless without anything looking wrong.
    """
    if isinstance(meta, dict) and "clip_key" in meta:
        v = meta["clip_key"]
        return str(v[i]) if not isinstance(v, str) else v
    if isinstance(meta, (list, tuple)) and i < len(meta) and isinstance(meta[i], dict):
        return str(meta[i].get("clip_key", f"_frame{offset + i}"))
    return f"_frame{offset + i}"


# ======================================================================================
# checkpoint / CLI
# ======================================================================================
def load_checkpoint(path: str, device) -> tuple[torch.nn.Module, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {})
    model = HandNet(n_classes=10,
                    norm=cfg.get("norm", "bn"),
                    width=cfg.get("width", 1.0),
                    use_mask_attn_pool=cfg.get("use_mask_attn_pool", True),
                    heads=tuple(cfg.get("heads", ("det", "seg", "cls"))))
    state = ck.get("ema") or ck["model"]      # EMA weights are what training selected on
    model.load_state_dict(state)
    return model.to(device).eval(), cfg


def _code_hash() -> str:
    """Hash of the source files, so a result JSON can be tied to the code that made it."""
    h = hashlib.sha256()
    for p in sorted(_HERE.glob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def compare_runs(paths: list[str], out_csv: str | None = None) -> list[dict]:
    """Tidy one row per result JSON for the ablation tables of s8."""
    keys = ["det_acc@0.5", "mean_box_iou", "seg_iou_hand", "seg_miou", "seg_dice",
            "cls_top1", "cls_macro_f1", "cls_ece"]
    rows = []
    for p in paths:
        d = utils.load_json(p)
        f, ci = d.get("frame", {}), d.get("ci", {})
        row = {"run": Path(p).stem,
               "n_frames": d.get("n_frames"), "n_clips": d.get("n_clips"),
               "split": d.get("config", {}).get("split"),
               "pseudo_target": d.get("pseudo_target")}
        for k in keys:
            row[k] = round(float(f.get(k, float("nan"))), 4) if k in f else None
            if k in ci:
                row[f"{k}_lo"] = round(ci[k]["lo"], 4)
                row[f"{k}_hi"] = round(ci[k]["hi"], 4)
        rows.append(row)
    if out_csv:
        cols = sorted({c for r in rows for c in r})
        cols = ["run", "split", "pseudo_target", "n_frames", "n_clips"] + \
               [c for c in cols if c not in ("run", "split", "pseudo_target", "n_frames", "n_clips")]
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--img-size", nargs=2, type=int, default=None, metavar=("W", "H"))
    ap.add_argument("--pseudo-target", action="store_true")
    ap.add_argument("--examples", type=int, default=0)
    ap.add_argument("--examples-out", default=None, help="npz path for the collected examples")
    ap.add_argument("--predictions-out", default=None,
                    help="CSV path for one prediction row per input image (box, mask area, "
                         "class, confidence, and the ground truth it is scored against)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)

    device = torch.device(a.device)
    model, cfg = load_checkpoint(a.ckpt, device)
    img_size = tuple(a.img_size) if a.img_size else tuple(cfg.get("img_size", (384, 288)))

    ds = HandGestureDataset(a.index, a.split, img_size=img_size, aug=None, return_meta=True)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers,
        pin_memory=(device.type == "cuda"), persistent_workers=a.num_workers > 0)

    res = evaluate_model(model, loader, device, amp=True,
                         collect_examples=a.examples, pseudo_target=a.pseudo_target,
                         n_boot=a.n_boot,
                         collect_predictions=bool(a.predictions_out))
    examples = res.pop("examples", None)
    preds = res.pop("predictions", None)
    res["config"] = {
        "ckpt": str(Path(a.ckpt).resolve()), "index": str(Path(a.index).resolve()),
        "split": a.split, "img_size": list(img_size), "batch_size": a.batch_size,
        "pseudo_target": bool(a.pseudo_target), "code_hash": _code_hash(),
        "train_cfg": cfg, "torch": torch.__version__,
        "evaluated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    utils.save_json(res, a.out)

    if preds and a.predictions_out:
        Path(a.predictions_out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.predictions_out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(preds[0].keys()))
            w.writeheader()
            w.writerows(preds)
        print(f"[eval] wrote {len(preds)} prediction rows to {a.predictions_out}")

    if examples is not None and a.examples_out:
        Path(a.examples_out).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.examples_out, examples=np.array(examples, dtype=object),
                            allow_pickle=True)

    f = res["frame"]
    print(f"[eval] {a.split}: n={res['n_frames']} frames / {res['n_clips']} clips "
          f"({res['eval_seconds']}s){' [pseudo-target]' if a.pseudo_target else ''}")
    for k in ("det_acc@0.5", "det_acc@0.75", "mean_box_iou", "seg_iou_hand", "seg_miou",
              "seg_dice", "cls_top1", "cls_macro_f1", "cls_ece"):
        if k in f:
            ci = res["ci"].get(k)
            s = f"  {k:<14} {f[k]:.4f}"
            if ci:
                s += f"   95% CI [{ci['lo']:.4f}, {ci['hi']:.4f}]"
            print(s)
    print(f"[eval] boxes scored {f['n_boxes_scored']} / skipped {f['n_boxes_skipped']}; "
          f"seg scored {f['n_seg_scored']} / skipped {f['n_seg_skipped']}")
    print(f"[eval] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
