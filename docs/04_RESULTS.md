# Results — every measured number

**Scope owner:** this document. All quantitative outcomes, the analysis of them, and the
evidence pack the report is written from. Design rationale lives in `02_DESIGN.md`; dataset
facts in `01_DATA.md`; review findings in `05_REVIEW.md`. **No number appears here that was not
read out of a result JSON produced by `src/evaluate.py`.**

**Last updated:** 2026-08-26 — E1 and B0 complete; E3 and the ablation grid in flight.

**Status:** IN PROGRESS. Rows marked *pending* have no number yet. Experiments 2 and 3 on the
smartphone set are blocked on the smartphone recording (`05_REVIEW.md` F1).

---

## 1. How to read these tables

Every metric is **per-frame** unless a table says otherwise. `src/evaluate.py` also writes
per-clip aggregates and bootstrap 95 % confidence intervals resampled over clips; those are in
the JSONs and are what the report's error bars come from. The per-frame `n` overstates the
effective sample size, because the 15 frames of a clip have a median consecutive mask IoU of
0.649 (`01_DATA.md` s2) — this is why the intervals are clip-level.

`hand IoU` is reported separately from `seg mIoU` throughout. Background IoU is ~0.97 by
construction on a dataset where the hand covers 2.75 % of the image, so the two-class mean
flatters every model, including the ones that are barely working.

Evaluation splits:

| Name | What it is | n frames | n clips |
|---|---|---|---|
| `rs_val` | 6 held-out RealSense contributors | 4,500 | 300 |
| `rs_test` | the official RealSense test release, 5 unidentified holdout subjects | 3,450 | 230 |
| `pseudo` | `rs_test` pushed through the held-out synthetic phone-style shift (`02_DESIGN.md` s7.1) | 3,450 | 230 |
| `phone` | the smartphone test set | 60 | 20 |

Note `rs_val` scores detection and segmentation on only the annotated 14.1 % of its frames
(600 of 4,500 boxes); classification is scored on all of them. The evaluator reports the
denominators it used in `n_boxes_scored` / `n_seg_scored` rather than silently averaging zeros.

## 2. Mandatory experiments

| Model | Split | det acc@0.5 | mean box IoU | hand IoU | Dice | top-1 | macro-F1 | ECE |
|---|---|---|---|---|---|---|---|---|
| B0 classical | rs_val | 0.0800 | 0.1985 | 0.2336 | 0.3369 | 0.2078 | 0.2009 | n/a |
| B0 classical | rs_test | 0.0562 | 0.1589 | 0.1722 | 0.2471 | 0.1510 | 0.1528 | n/a |
| **E1** (jitter) | rs_val | **0.9617** | **0.8791** | **0.8858** | **0.9233** | **0.9047** | **0.9068** | 0.0331 |
| **E1** (jitter) | rs_test | **0.9186** | **0.8405** | **0.8270** | **0.8763** | **0.8722** | **0.8747** | 0.0310 |
| E1 | pseudo | 0.8748 | 0.7948 | 0.7465 | 0.8128 | 0.8226 | 0.8256 | 0.0401 |
| **E2** = E1 weights | phone | *pending* | | | | | | |
| E3 (CPR) | rs_val | *pending* | | | | | | |
| E3 (CPR) | rs_test | *pending* | | | | | | |
| E3 (CPR) | pseudo | *pending* | | | | | | |
| **E3** (CPR) | phone | *pending* | | | | | | |

### 2.1 What E1 establishes

* The multi-task model reaches **0.875 macro-F1 and 0.827 hand IoU on the official RealSense
  test set**, with 92 % of frames localised to better than 0.5 IoU.
* **The classical baseline is not close.** 0.153 macro-F1 and 0.056 detection accuracy on the
  same frames. This matters for the report's argument: it rules out the reading that the task is
  easy because hands are skin-coloured blobs near the centre of the frame. Skin-chroma
  segmentation followed by HOG and a linear classifier — the pre-deep-learning answer, with
  untuned literature thresholds — recovers a usable box in a twentieth of frames. The gap is
  0.86 detection accuracy and 0.72 macro-F1.
* **Val is consistently better than test** by 3-4 points on every metric (macro-F1 0.907 vs
  0.875, hand IoU 0.886 vs 0.827). Both are subject-disjoint from training, so this is not
  leakage; the most likely reading is that the 5 test subjects are a harder or differently
  distributed sample than the 6 held-out contributors, which is what a genuinely held-out test
  release is for. It also means **val is a mildly optimistic proxy**, and any model selected on
  val carries that bias — stated because every checkpoint in this study is selected on val.
* **Calibration is good in-domain**: ECE 0.031 on test, i.e. the confidence the brief requires
  is worth something rather than being a decorative softmax maximum.

### 2.2 The synthetic shift is milder than expected

E1 loses only **0.049 macro-F1** (0.875 -> 0.826) and **0.080 hand IoU** (0.827 -> 0.747) under
the held-out pseudo-target transform. That is a real but modest drop, and it has a consequence
worth stating before the smartphone numbers arrive: **the pseudo-target is a weaker shift than a
real camera change is likely to be**, so it should be read as a lower bound on the domain gap and
a directional development signal, not as a stand-in for the phone result. If the smartphone drop
turns out to be much larger, that is evidence about the proxy, and the report should say so
rather than quietly dropping the comparison.

## 3. Ablations

*Pending — 33 configurations queued across two GPU hosts.* Tables will be generated by
`tools/make_report_tables.py` directly from the result JSONs.

## 4. Qualitative results

*Pending.*

## 5. Open questions this document must answer before the report is written

1. Does CPR cost anything in-domain? (E3 vs E1 on `rs_test`.)
2. How large is the real cross-camera drop, and which of the three tasks degrades most?
3. Which CPR stages carry the gain? (Block B leave-one-out.)
4. Does the mask-attended-pooling hypothesis hold — is its benefit larger on phone than on
   RealSense? (`e_nomap` vs `e3` across splits.)
5. Do any ablation deltas exceed the seed spread? Three seeds on `e1` and `e3` set that bar.
6. Does calibration survive the domain shift, or does the model become confidently wrong?
