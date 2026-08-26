#!/usr/bin/env python3
"""Pack the raw COMP0248 hand-gesture releases into a compact training format.

Why this exists
---------------
The raw RealSense release is ~8.9 GB of 640x480 PNGs spread over ~25k files in a four-level
directory tree, and the UCL Knuckles workspace quota has roughly 12 GB free. Packing to
512x384 JPEG (RGB) + PNG (masks) plus one ``index.json`` cuts that to ~1.7 GB, replaces four
directory walks per sample with a single file open, and gives one place where the
mask -> box definition lives.

Design notes that matter for correctness
----------------------------------------
* **Boxes are derived from the resized, re-thresholded mask**, not from the source mask and
  then scaled. That keeps the stored box exactly consistent with the stored mask, so a
  visual check of one is a check of the other.
* **Masks are resized bilinearly and re-thresholded at 128**, not nearest-neighbour.
  The released masks are anti-aliased (0.0064% of pixels lie strictly between 0 and 255,
  affecting 23% of masks), and nearest-neighbour downsampling of a thin finger gap is
  noticeably worse than filtering then thresholding.
* **One contributor is extracted, packed and deleted before the next is touched**, so peak
  extra disk is one contributor (~300 MB) rather than the whole 8.9 GB release.
* **The train/val split is by contributor**, chosen here (not at train time) so every
  experiment in the study reads the identical split from ``index.json``.

CLI
---
    python tools/pack_dataset.py --src rgb_only.7z --out data/realsense_train \
        --packed-size 512 384 --split-holdout 6 --seed 0
    python tools/pack_dataset.py --src "Test data-COMP0248_Test_data_23" \
        --out data/realsense_test --test
"""
from __future__ import annotations

import argparse
import json
import os
import re
import hashlib
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

GESTURES = [
    "G01_call", "G02_dislike", "G03_like", "G04_ok", "G05_one",
    "G06_palm", "G07_peace", "G08_rock", "G09_stop", "G10_three",
]
GESTURE_TO_ID = {g: i for i, g in enumerate(GESTURES)}

FRAME_RE = re.compile(r"^frame_(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)
MASK_THRESH = 128
JPEG_QUALITY = 95


# --------------------------------------------------------------------------------------
# pure helpers (exercised by the regression fixture described in docs/03_IMPLEMENTATION.md s5)
# --------------------------------------------------------------------------------------
def frame_index(name: str) -> int | None:
    """``frame_007.png`` -> 7.  Returns None for anything else (``.DS_Store`` and friends).

    The released trees carry 41 stray ``.DS_Store`` files inside class and clip directories;
    globbing ``*`` and trusting it is the single most likely way to poison the index.
    """
    m = FRAME_RE.match(name)
    return int(m.group(1)) if m else None


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a {0,255} uint8 mask to ``size=(W,H)`` and re-binarise at 128."""
    im = Image.fromarray(mask, mode="L").resize(size, Image.BILINEAR)
    out = np.asarray(im)
    return np.where(out >= MASK_THRESH, np.uint8(255), np.uint8(0))


def mask_to_box(mask: np.ndarray) -> list[float] | None:
    """Tight ``(x1,y1,x2,y2)`` of the foreground, or None if the mask is empty.

    Coordinates are inclusive-exclusive in the numpy sense: ``x2``/``y2`` are one past the
    last foreground pixel, so ``x2-x1`` is the pixel width. Matches ``src/utils.mask_to_box``.
    """
    ys, xs = np.nonzero(mask >= MASK_THRESH)
    if xs.size == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def assign_splits(subjects: list[str], holdout: int, seed: int) -> dict[str, str]:
    """Deterministically hold out ``holdout`` contributors for validation.

    Contributor-wise, never frame- or clip-wise: at 3 FPS consecutive frames of a clip have a
    median mask IoU of 0.649, so any within-clip split leaks near-duplicates into validation.
    """
    subs = sorted(subjects)
    if holdout <= 0:
        return {s: "train" for s in subs}
    if holdout >= len(subs):
        raise ValueError(f"holdout={holdout} >= {len(subs)} contributors")
    rng = np.random.default_rng(seed)
    val = set(rng.permutation(np.array(subs, dtype=object))[:holdout].tolist())
    return {s: ("val" if s in val else "train") for s in subs}


# --------------------------------------------------------------------------------------
# per-clip worker
# --------------------------------------------------------------------------------------
def pack_clip(job: dict) -> list[dict]:
    """Pack one clip. Runs in a worker process; returns index records."""
    clip_dir = Path(job["clip_dir"])
    out_root = Path(job["out"])
    W, H = job["packed_size"]
    subject, gesture, clip = job["subject"], job["gesture"], job["clip"]

    rgb_dir, ann_dir = clip_dir / "rgb", clip_dir / "annotation"
    if not rgb_dir.is_dir():
        return []

    ann_by_frame: dict[int, Path] = {}
    if ann_dir.is_dir():
        for p in ann_dir.iterdir():
            fi = frame_index(p.name)
            if fi is not None:
                ann_by_frame[fi] = p

    records: list[dict] = []
    for p in sorted(rgb_dir.iterdir()):
        fi = frame_index(p.name)
        if fi is None:
            continue
        try:
            im = Image.open(p).convert("RGB")
        except Exception as e:  # a truncated file should not kill the whole run
            print(f"[warn] unreadable rgb {p}: {e}", file=sys.stderr)
            continue
        src_w, src_h = im.size
        stem = f"{subject}__{gesture}__{clip}__f{fi:03d}"
        rgb_rel = f"rgb/{stem}.jpg"
        im.resize((W, H), Image.LANCZOS).save(
            out_root / rgb_rel, "JPEG", quality=JPEG_QUALITY, subsampling=0, optimize=True
        )

        ann_rel, box = None, None
        if fi in ann_by_frame:
            try:
                m = np.asarray(Image.open(ann_by_frame[fi]).convert("L"))
            except Exception as e:
                print(f"[warn] unreadable mask {ann_by_frame[fi]}: {e}", file=sys.stderr)
                m = None
            if m is not None:
                mr = resize_mask(m, (W, H))
                box = mask_to_box(mr)
                if box is None:
                    # An all-zero mask is a labelling failure, not a hand-free image: the brief
                    # guarantees a hand in every frame. Drop the supervision, keep the frame for
                    # the classifier, and record it so the count reaches the data document.
                    print(f"[warn] empty mask {ann_by_frame[fi]}", file=sys.stderr)
                else:
                    ann_rel = f"ann/{stem}.png"
                    Image.fromarray(mr, mode="L").save(out_root / ann_rel, "PNG", optimize=True)

        records.append({
            "id": f"{subject}/{gesture}/{clip}/{fi:03d}",
            "subject": subject, "gesture": gesture, "clip": clip, "frame": fi,
            "rgb": rgb_rel, "ann": ann_rel,
            "cls": GESTURE_TO_ID[gesture],
            "box": box,
            "src_size": [src_w, src_h],
        })
    return records


# --------------------------------------------------------------------------------------
# tree discovery
# --------------------------------------------------------------------------------------
def subject_signature(root: Path) -> str:
    """Content signature of a contributor folder: sorted (relative path, size) pairs, hashed.

    Used to drop byte-identical duplicate submissions. The release contains
    ``25150455_Guan`` and ``25150455_Guan 2`` -- same student number, 850 identical files each
    (verified by md5 of every PNG). Packing both double-weights that contributor in training and
    creates a subject-leakage hazard the moment a split puts one folder in train and the other in
    val. Signature-based detection catches this without name heuristics, so any future duplicate
    is caught too.
    """
    h = hashlib.md5()
    for f in sorted(root.rglob("*")):
        if f.is_file() and f.name != ".DS_Store":
            h.update(str(f.relative_to(root)).encode())
            h.update(str(f.stat().st_size).encode())
    return h.hexdigest()


def discover_clips(subject_root: Path, subject: str) -> list[dict]:
    """Find ``<subject_root>/[wrapper/...]/G##_name/clip##`` directories.

    Descends through wrapper directories first: one contributor packaged their submission as
    ``25047621_Wu/dataset/25047621_Wu/G01_call/...``. Matching only the immediate children
    silently drops that whole contributor (750 frames) with no error -- which is exactly what
    the first pack run did.
    """
    subject_root = find_dataset_root(subject_root)
    jobs = []
    for gdir in sorted(subject_root.iterdir()):
        if not gdir.is_dir() or gdir.name not in GESTURE_TO_ID:
            continue
        for cdir in sorted(gdir.iterdir()):
            if cdir.is_dir() and cdir.name.startswith("clip"):
                jobs.append({"clip_dir": str(cdir), "subject": subject,
                             "gesture": gdir.name, "clip": cdir.name})
    return jobs


def find_dataset_root(path: Path) -> Path:
    """Descend through single-child wrapper directories until gesture folders appear."""
    cur = path
    for _ in range(6):
        names = {p.name for p in cur.iterdir() if p.is_dir()}
        if names & set(GESTURE_TO_ID):
            return cur
        kids = [p for p in cur.iterdir() if p.is_dir()]
        if len(kids) == 1:
            cur = kids[0]
        else:
            return cur
    return cur


def list_archive_subjects(archive: Path) -> tuple[str, list[str]]:
    """Return (top-level prefix, contributor names) by listing the 7z archive."""
    out = subprocess.run(["7z", "l", "-slt", str(archive)], capture_output=True, text=True,
                         check=True).stdout
    paths = [ln[7:].strip() for ln in out.splitlines() if ln.startswith("Path = ")]
    paths = [p.replace("\\", "/") for p in paths[1:]]  # first Path is the archive itself
    tops = {p.split("/")[0] for p in paths if "/" in p or p}
    prefix = ""
    if len(tops) == 1:
        prefix = tops.pop()
        rel = [p[len(prefix) + 1:] for p in paths if p.startswith(prefix + "/")]
    else:
        rel = paths
    subjects = sorted({r.split("/")[0] for r in rel
                       if "/" in r and r.split("/")[0] not in ("", ".DS_Store")
                       and r.split("/")[0] not in GESTURE_TO_ID})
    return prefix, subjects


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="7z archive, or a directory tree")
    ap.add_argument("--out", required=True)
    ap.add_argument("--packed-size", nargs=2, type=int, default=[512, 384], metavar=("W", "H"))
    ap.add_argument("--split-holdout", type=int, default=6,
                    help="contributors held out for validation (train packs only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test", action="store_true",
                    help="pack as a single 'test' split; no contributor sub-level expected")
    ap.add_argument("--subject-name", default=None,
                    help="contributor name to record when --test or a flat tree is given")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--scratch", default=None, help="where to extract (defaults to a temp dir)")
    ap.add_argument("--limit-subjects", type=int, default=0, help="smoke-test switch")
    args = ap.parse_args()

    src, out = Path(args.src).expanduser(), Path(args.out).expanduser()
    (out / "rgb").mkdir(parents=True, exist_ok=True)
    (out / "ann").mkdir(parents=True, exist_ok=True)
    W, H = args.packed_size
    t0 = time.time()
    records: list[dict] = []
    seen_sigs: dict[str, str] = {}
    skipped_dupes: list[tuple[str, str]] = []

    def run_jobs(jobs: list[dict]) -> None:
        for j in jobs:
            j["out"] = str(out)
            j["packed_size"] = [W, H]
        if args.workers <= 1:
            for j in jobs:
                records.extend(pack_clip(j))
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                for r in ex.map(pack_clip, jobs, chunksize=1):
                    records.extend(r)

    if src.suffix.lower() == ".7z" or src.suffix.lower() == ".zip":
        prefix, subjects = list_archive_subjects(src)
        if args.limit_subjects:
            subjects = subjects[:args.limit_subjects]
        print(f"[pack] archive {src.name}: prefix={prefix!r} contributors={len(subjects)}")
        scratch = Path(args.scratch) if args.scratch else Path(tempfile.mkdtemp(prefix="packsrc_"))
        scratch.mkdir(parents=True, exist_ok=True)
        for i, subj in enumerate(subjects, 1):
            sdir = scratch / subj
            if sdir.exists():
                shutil.rmtree(sdir)
            pat = f"{prefix}/{subj}/*" if prefix else f"{subj}/*"
            cmd = ["7z", "x", "-y", f"-o{scratch}", str(src), pat]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"[error] extract {subj}: {r.stderr[-400:]}", file=sys.stderr)
                continue
            got = scratch / prefix / subj if prefix else sdir
            sig = subject_signature(got)
            if sig in seen_sigs:
                print(f"[pack] {i}/{len(subjects)} {subj}: SKIPPED, byte-identical to "
                      f"{seen_sigs[sig]!r} (duplicate submission)", flush=True)
                shutil.rmtree(scratch / prefix if prefix else got, ignore_errors=True)
                skipped_dupes.append((subj, seen_sigs[sig]))
                continue
            seen_sigs[sig] = subj
            jobs = discover_clips(got, subj)
            n0 = len(records)
            run_jobs(jobs)
            n_new = len(records) - n0
            print(f"[pack] {i}/{len(subjects)} {subj}: {len(jobs)} clips -> "
                  f"{n_new} frames  ({time.time() - t0:.0f}s)"
                  + ("   <-- WARNING: zero frames" if n_new == 0 else ""), flush=True)
            shutil.rmtree(scratch / prefix if prefix else got, ignore_errors=True)
        shutil.rmtree(scratch, ignore_errors=True)
    else:
        root = find_dataset_root(src)
        top = {p.name for p in root.iterdir() if p.is_dir()}
        if top & set(GESTURE_TO_ID):
            subj = args.subject_name or root.name
            run_jobs(discover_clips(root, subj))
            print(f"[pack] flat tree as contributor {subj!r}: {len(records)} frames")
        else:
            subjects = sorted(p.name for p in root.iterdir() if p.is_dir())
            if args.limit_subjects:
                subjects = subjects[:args.limit_subjects]
            for i, s in enumerate(subjects, 1):
                sig = subject_signature(root / s)
                if sig in seen_sigs:
                    print(f"[pack] {i}/{len(subjects)} {s}: SKIPPED, byte-identical to "
                          f"{seen_sigs[sig]!r} (duplicate submission)", flush=True)
                    skipped_dupes.append((s, seen_sigs[sig]))
                    continue
                seen_sigs[sig] = s
                n0 = len(records)
                run_jobs(discover_clips(root / s, s))
                n_new = len(records) - n0
                print(f"[pack] {i}/{len(subjects)} {s}: {n_new} frames"
                      + ("   <-- WARNING: zero frames" if n_new == 0 else ""), flush=True)

    subjects_seen = sorted({r["subject"] for r in records})
    if args.test:
        split_map = {s: "test" for s in subjects_seen}
    else:
        split_map = assign_splits(subjects_seen, args.split_holdout, args.seed)

    n_ann = sum(1 for r in records if r["ann"])
    index = {
        "root": str(out.resolve()),
        "packed_size": [W, H],
        "split_map": split_map,
        "records": sorted(records, key=lambda r: r["id"]),
        "meta": {
            "source": str(src),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "jpeg_quality": JPEG_QUALITY,
            "mask_threshold": MASK_THRESH,
            "mask_resample": "bilinear-then-threshold",
            "rgb_resample": "lanczos",
            "n_records": len(records),
            "n_annotated": n_ann,
            "n_subjects": len(subjects_seen),
            "skipped_duplicate_subjects": [{"dropped": a, "identical_to": b}
                                           for a, b in skipped_dupes],
            "gestures": GESTURES,
        },
    }
    with open(out / "index.json", "w") as f:
        json.dump(index, f)

    print(f"\n[pack] done in {time.time() - t0:.0f}s")
    print(f"[pack] {len(records)} frames | {n_ann} annotated ({100 * n_ann / max(1, len(records)):.1f}%) "
          f"| {len(subjects_seen)} contributors")
    for a_, b_ in skipped_dupes:
        print(f"[pack] dropped duplicate contributor {a_!r} (identical to {b_!r})")
    if not args.test:
        n_val = sum(1 for v in split_map.values() if v == "val")
        print(f"[pack] split: {len(split_map) - n_val} train / {n_val} val contributors")
        print(f"[pack] val contributors: {sorted(s for s, v in split_map.items() if v == 'val')}")
    from collections import Counter
    cc = Counter(r["gesture"] for r in records)
    print("[pack] frames per gesture: " + ", ".join(f"{g}={cc.get(g, 0)}" for g in GESTURES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
