#!/usr/bin/env python3
"""Produce binary hand masks for the smartphone set, following the COMP0248 Week-4 data-curation
tutorial (`Realsense_datacollection_n_annotation.pdf`, Part 3).

The tutorial's pipeline is: pre-annotate with a Segment Anything model, refine every mask by hand
in Label Studio, then decode the export back into `annotation/` PNGs. The RealSense masks this
project trains and evaluates against were produced that way, so the smartphone masks are produced
that way too -- matching the annotation convention matters more for a cross-domain segmentation
metric than any property of the annotator.

WHAT THIS SCRIPT IS, AND IS NOT
    It is annotation tooling. It runs once, offline, to produce ground truth. Nothing here is
    imported by `src/`, runs at training or inference time, or contributes to any reported
    prediction -- the brief's restriction on detection/segmentation frameworks governs the
    *solution*, and the course tutorial explicitly prescribes SAM for *annotation*
    (CW1 p9: "Pre-annotating with SAM3 and refining in Label Studio").

TWO SUBSTITUTIONS, BOTH DELIBERATE
    * SAM 1 (`segment_anything`, ViT-B) instead of SAM3. SAM3's weights are gated behind a manual
      approval form, so they cannot be fetched non-interactively. SAM 1 is the same family and the
      same role.
    * SAM3 takes a *text* prompt ("hand") and returns instances directly. SAM 1 is promptable by
      points and boxes but not by text, so the hand is located first with MediaPipe's hand
      landmarker and its landmarks become the prompt. MediaPipe contributes no pixels: it decides
      only where to click. Every boundary in the output is SAM's.

STAGES
    1. detect   -- MediaPipe hand landmarks -> a padded box and ten positive points
    2. segment  -- SAM, prompted with both, best of three candidate masks by predicted IoU
    3. wrist    -- cut perpendicular to the palm axis through the wrist landmark, so the mask
                   stops at the wrist crease as the RealSense convention does
    4. export   -- write `labelling/{images,labels}` for Label Studio, and a QC report

Steps 3 and 4 of the tutorial (Label Studio refinement, then `process_LS_output.py`) are manual
and are documented in README.md; this script produces their input.

Usage
-----
    python tools/annotate_smartphone_sam.py \
        --src smartphone_dataset/18006111_Shihab \
        --out _scratch/ls_work --sam-checkpoint sam_vit_b_01ec64.pth
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# RealSense hand-area reference band, measured over all 3,450 test masks:
# median 2.75 % of the frame, p05 1.59 %, p95 4.73 %. Used only to FLAG frames for the manual
# pass -- nothing is rejected automatically, because a phone held closer than the RealSense rig
# legitimately produces a larger hand.
RS_P05, RS_P95 = 1.59, 4.73

# MediaPipe hand-landmark indices: 0 wrist, 4/8/12/16/20 fingertips, 5/9/13/17 knuckle bases.
PROMPT_IDS = [0, 4, 8, 12, 16, 20, 5, 9, 13, 17]
MCP_IDS = [5, 9, 13, 17]
BOX_PAD = 0.12          # landmarks sit inside the silhouette; pad the hull outwards


def build_landmarker(model_path: str, min_conf: float = 0.2):
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    return vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=model_path),
            num_hands=2, min_hand_detection_confidence=min_conf))


def build_sam(checkpoint: str, model_type: str = "vit_b"):
    # Imported lazily so that `src/` never pulls this in and the submission runs without it.
    from segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.eval()
    return SamPredictor(sam)


def largest_hand(hands):
    """The brief's protocol is one right hand in frame; if two are found take the bigger."""
    return max(hands, key=lambda L: (max(p.x for p in L) - min(p.x for p in L))
                                     * (max(p.y for p in L) - min(p.y for p in L)))


def cut_at_wrist(mask: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Drop everything behind the wrist crease.

    The palm axis is wrist -> mean(knuckle bases). The cut is the line through the wrist landmark
    perpendicular to that axis; the landmark sits at the crease, so there is no tuning constant.
    A bent wrist or a folded cuff can leave a sliver forward of the line, so the largest connected
    component is kept.
    """
    wrist = pts[0]
    d = pts[MCP_IDS].mean(0) - wrist
    n = np.linalg.norm(d)
    if n < 1e-6:
        return mask
    d = d / n
    H, W = mask.shape
    ys, xs = np.mgrid[0:H, 0:W]
    forward = (xs - wrist[0]) * d[0] + (ys - wrist[1]) * d[1] >= 0
    cut = mask & forward
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(cut.astype(np.uint8), 8)
    if nlab > 1:
        return lab == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))
    return cut


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True,
                    help="smartphone tree: <studentno>_<surname>/G##_x/clip##/rgb/frame_###.png")
    ap.add_argument("--out", required=True, help="working directory for the Label Studio pass")
    ap.add_argument("--sam-checkpoint", required=True)
    ap.add_argument("--sam-model", default="vit_b")
    ap.add_argument("--hand-model", default="hand_landmarker.task",
                    help="MediaPipe hand_landmarker.task")
    ap.add_argument("--no-wrist-cut", action="store_true",
                    help="skip stage 3 (kept for the ablation)")
    a = ap.parse_args(argv)

    import mediapipe as mp
    cv2.setNumThreads(1)                   

    src = Path(a.src)
    images = Path(a.out) / "dataset/labelling/images"
    labels = Path(a.out) / "dataset/labelling/labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)

    landmarker = build_landmarker(a.hand_model)
    predictor = build_sam(a.sam_checkpoint, a.sam_model)

    frames = sorted(src.glob("*/*/rgb/*.png"))
    if not frames:
        print(f"no frames under {src}/*/*/rgb/", file=sys.stderr)
        return 1
    print(f"{len(frames)} frames", flush=True)

    report = []
    for i, f in enumerate(frames, 1):
        rel = f.relative_to(src)                       # G##_x/clip##/rgb/frame_###.png
        bgr = cv2.imread(str(f))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

        rec = {"frame": str(rel), "detected": bool(res.hand_landmarks)}
        if not res.hand_landmarks:
            # An empty mask, not a missing file: the frame must still reach the manual pass, and
            # a missing label would silently drop it from labels.json.
            mask = np.zeros((H, W), bool)
            rec.update(area_pct=0.0, sam_score=0.0, flags=["NO_HAND_DETECTED"])
        else:
            pts = np.array([[p.x * W, p.y * H] for p in largest_hand(res.hand_landmarks)])
            x0, y0 = pts.min(0)
            x1, y1 = pts.max(0)
            px, py = BOX_PAD * (x1 - x0), BOX_PAD * (y1 - y0)
            box = np.array([max(0, x0 - px), max(0, y0 - py), min(W, x1 + px), min(H, y1 + py)])
            t0 = time.time()
            predictor.set_image(rgb)
            masks, scores, _ = predictor.predict(
                point_coords=pts[PROMPT_IDS], point_labels=np.ones(len(PROMPT_IDS)),
                box=box, multimask_output=True)
            mask = masks[int(np.argmax(scores))]
            if not a.no_wrist_cut:
                mask = cut_at_wrist(mask, pts)
            area = float(mask.mean()) * 100
            flags = []
            if not (RS_P05 <= area <= RS_P95):
                flags.append(f"AREA_OUT_OF_BAND({area:.2f}%)")
            ys, xs = np.where(mask)
            if len(xs) and (xs.min() == 0 or ys.min() == 0 or xs.max() == W-1 or ys.max() == H-1):
                flags.append("TOUCHES_BORDER")
            rec.update(area_pct=round(area, 3), sam_score=round(float(scores.max()), 4),
                       seconds=round(time.time() - t0, 1), flags=flags)

        for root, payload in ((images, None), (labels, mask)):
            dst = root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if payload is None:
                shutil.copyfile(f, dst)
            else:
                cv2.imwrite(str(dst), payload.astype(np.uint8) * 255)
        report.append(rec)
        print(f"[{i:3d}/{len(frames)}] {rel}  {rec.get('area_pct',0):5.2f}%  "
              f"s={rec.get('sam_score',0):.3f}  {' '.join(rec.get('flags',[]))}", flush=True)

    qc = Path(a.out) / "qc"
    qc.mkdir(parents=True, exist_ok=True)
    (qc / "sam_preannotation.json").write_text(json.dumps(report, indent=1))
    flagged = sum(1 for r in report if r.get("flags"))
    print(f"\n{len(report)} frames; {flagged} flagged for the manual pass; "
          f"{sum(1 for r in report if not r['detected'])} with no hand detected")
    print(f"next: convert_annotations_for_LS.py -> Label Studio -> process_LS_output.py ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
