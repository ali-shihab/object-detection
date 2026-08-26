# Design — Model, Losses, Augmentation, Experiments

**Scope owner:** this document. Architecture, loss composition, augmentation policy, the E3
cross-camera method, training recipe, evaluation protocol — and the *rationale* for each.
Requirements live in `00_PLAN.md`; dataset facts in `01_DATA.md`; measured outcomes in
`04_RESULTS.md`. No numbers that came out of a training run belong here.

**Last updated:** 2026-08-26 (Cycle 2 — corrections from implementation folded in)

---

## 1. What the model has to do

One RGB image in; four things out (LSA p5): a hand box `(xmin,ymin,xmax,ymax)`, a binary hand
mask, a gesture class from 10, and a gesture confidence. Trained on RealSense RGB only,
evaluated unchanged on smartphone RGB.

Two facts from `01_DATA.md` shape everything below:

- **One hand per image is safe.** Over 3,450 test masks thresholded at 128, 1.91 % contain more
  than one connected component and *zero* contain a second component larger than 0.5 % of the
  image. Single-instance decoding is therefore not a shortcut, it is a property of the data.
- **The hand is small and roughly centred.** Median hand area is 2.75 % of the image; the
  median box is 115x151 px at 640x480 (0.180 x 0.315 normalised), aspect w/h 0.76, centred at
  (0.448, 0.489). A stride-4 detection map is fine; a stride-8 one would put the median hand in
  ~14x19 cells, which is still workable but leaves less room for the smaller tail.

## 2. Input resolution

**384 x 288 (WxH).** Chosen because: it preserves the native 4:3 aspect exactly (no anisotropic
squash, which would corrupt the box-aspect statistics we rely on), it is divisible by 32 so the
stride-32 bottleneck is a clean 12x9, and it keeps the median hand box at 69x91 px — large
enough that a stride-4 head has real spatial support. 640x480 would cost ~2.8x the compute for a
hand that is already resolved at 384x288.

## 3. Architecture (`src/model.py`)

Everything is written from scratch as `torch.nn.Module` subclasses. No `torchvision.models`,
no `torchvision.ops`, no detection or segmentation library (LSA p10, p11; see `00_PLAN.md` s2).

### 3.1 Encoder — `HandNetEncoder`

Residual CNN, ~5 stages. `Block` = 3x3 conv -> Norm -> SiLU -> 3x3 conv -> Norm, plus identity
(or 1x1-projected) skip, then SiLU.

| Stage | Output stride | Channels | Blocks | Spatial (384x288 in) |
|---|---|---|---|---|
| stem | 2 | 24 | conv3x3 s2 + conv3x3 | 192 x 144 |
| s1 | 4 | 48 | 2 | 96 x 72 |
| s2 | 8 | 96 | 3 | 48 x 36 |
| s3 | 16 | 192 | 4 | 24 x 18 |
| s4 | 32 | 320 | 3 | 12 x 9 |

Norm layer is a constructor argument (`bn` | `gn` | `ibn`) so the normalisation ablation is a
config change, not a code fork. **Default is `bn`** — deliberately the conventional choice, so
that the E1 baseline is the one a reader expects and the normalisation result is reported as a
finding rather than baked in silently.

### 3.2 Segmentation decoder — U-Net style

Up-path from s4 with lateral concatenation of s3, s2, s1, stem (channels 160, 96, 64, 48), each
step = bilinear upsample -> concat -> two 3x3 conv-norm-SiLU. Final 1x1 conv to a single logit at
stride 2, bilinearly upsampled to 384x288. Skips matter here: the mask boundary is the thing the
Dice term is most sensitive to and the encoder's stride-32 tip cannot resolve finger gaps.

### 3.3 Detection head — centre-point, at stride 4

A from-scratch CenterNet-style head on the stride-4 feature map (96x72):

- `heat` — 1 channel, sigmoid. Gaussian-splatted ground truth at the box centre, radius from the
  box size (standard CornerNet radius rule).
- `size` — 2 channels, `log(w), log(h)` in input pixels. Log-space because box areas span roughly
  an order of magnitude (`01_DATA.md`).
- `off` — 2 channels, sub-cell centre offset in [0,1).

Decode: 3x3 max-pool peak suppression, take the single top peak, read `off` and `size` at it.
**Detection confidence = the peak heatmap value.**

**Head initialisation (added Cycle 2, after measurement).** Every final prediction conv is
initialised with `normal(0, 1e-3)` weights and a prior-matched bias: `-2.19` for the heatmap
(p=0.01) and `-3.56` for the segmentation logit (p=0.0275, the measured median hand-pixel
fraction). This is not cosmetic. A freshly-initialised BatchNorm network in *train* mode whitens
every stage, and the accumulated gain drove the stride-4 head logits to +/-27 — measured, not
assumed. The focal loss then charges ~9.2 nats for every cell the untrained head confidently and
wrongly calls foreground. Measured at 384x288 on a BN train-mode forward: with plain Kaiming
head init the total loss at step 0 is **13,455** (heatmap term 13,444); with the shipped init it
is **14.7** (heatmap term 9.3), and the isolated focal term on a hand-built target falls from
**788** to **3.7**. Weights are
`1e-3` rather than exactly zero because with `W = 0` the gradient reaching the encoder through
that head is identically zero for the first step.

*Why not regress four numbers from a global pooled vector?* Because that discards spatial
evidence, gives no calibrated detection confidence, and cannot generalise to a second hand.
A centre-point head costs ~3 extra conv layers and is a real detector. It also gives us a
confidence signal we can report and threshold, which direct regression does not.

### 3.4 Classification head — mask-attended pooling

```
g   = GAP(s4)                                  # 320-d, whole-image context
m   = sigmoid(seg_logits).detach()             # soft hand mask, gradient stopped
m8  = avg_pool(m -> stride 32 grid)            # 12x9, matches s4
a   = sum(s4 * m8) / (sum(m8) + eps)           # 320-d, hand-focused
z   = Linear(concat[g, a] -> 256) -> Norm -> SiLU -> Dropout(0.2) -> Linear(256 -> 10)
```

Confidence = `softmax(z).max()`.

*Rationale, and the hypothesis it encodes.* Background appearance is the part of the image most
strongly tied to the camera and the room; hand appearance is the part that carries the gesture.
Pooling encoder features under the predicted mask gives the classifier a descriptor whose support
is (mostly) hand pixels, so less of its evidence rides on camera-specific background statistics.
This is a **testable claim**, not decoration: ablation A-MAP in s8 removes `a` and keeps `g`, and
if the claim is right the gap between the two should be larger on the smartphone set than on the
RealSense set. The gradient into `m` is stopped so a bad early mask cannot destabilise the
classifier, and the *predicted* (not ground-truth) mask is used at train time so there is no
train/test mismatch.

## 4. Loss composition (`src/train.py`, helpers in `src/utils.py`)

All terms are built from PyTorch primitives; nothing is imported from a detection library.

| Term | Definition | Default weight |
|---|---|---|
| `L_heat` | Penalty-reduced focal loss on the centre heatmap (CornerNet form, alpha=2, beta=4), built on `binary_cross_entropy` | 1.0 |
| `L_size` | L1 on `log(w), log(h)` at the ground-truth centre cell only | 0.1 |
| `L_off` | L1 on the sub-cell offset at the centre cell only | 1.0 |
| `L_giou` | GIoU loss between the decoded box and the ground-truth box (our own implementation) | 1.0 |
| `L_seg` | `BCEWithLogitsLoss` + soft Dice, summed | 1.0 + 1.0 |
| `L_cls` | `CrossEntropyLoss(label_smoothing=0.05)` | 1.0 |

`L_giou` is included because the two required detection metrics *are* IoU-based; optimising
size/offset in L1 alone leaves the metric one step removed from the objective.

**Fixed weights are the default.** Learnable homoscedastic-uncertainty weighting (Kendall et al.)
is implemented behind `--loss-weighting uncertainty` and is reported as an ablation, not adopted
blind — it is one more thing that can silently destabilise a small-data run.

### 4.1 Sparse mask supervision

**Measured, not assumed:** the training release annotates a median of 2 frames per clip and
**14.1 % of frames overall** (2,899 of 20,550), while the test release annotates all 15 frames of
every clip (`01_DATA.md` s5.1). Every frame carries a clip-level gesture label.
The loss is written to handle either case: every sample carries a `has_mask` flag, and
`L_heat`, `L_size`, `L_off`, `L_giou`, `L_seg` are averaged **over flagged samples only**, while
`L_cls` is averaged over the whole batch. If annotation turns out to be sparse this is the
mechanism that still lets ~26k frames train the classifier; if it is dense the flag is all-ones
and the code path is unchanged.

## 5. Data pipeline (`src/dataloader.py`)

- **Packed format.** Raw release -> `pack_dataset.py` -> per-frame JPEG (quality 95) at 512x384
  for RGB, PNG for masks, plus a single `index.json` holding `(subject, gesture, clip, frame,
  has_mask, box)`. Reasons: the raw release is ~6.4 GB compressed and the Knuckles workspace
  quota has ~12 GB free (`00_PLAN.md` s4); and decoding one JPEG beats walking four directories.
  Packing at 512x384 (not 384x288) keeps headroom for random-resized-crop before the final
  resize. Strictly, the crop only avoids upsampling above area-scale 0.5625: at the bottom of the
  `U(0.5,1.0)` range the crop is 362x272 and the resize to 384x288 is a 1.06x upsample. That is
  negligible, but the claim is not absolute.
- **Ground-truth mask threshold: `>= 128`.** The released masks are *not* strictly {0,255} —
  0.0064 % of pixels lie in 1..254 (anti-aliased edges), affecting 23 % of masks. Thresholding at
  128 moves 0.11 % of foreground (`01_DATA.md` s2).
- **Boxes are derived from masks**, as the tight box of the thresholded foreground. This matches
  what the LSA brief itself specifies for the smartphone set (p6), so RealSense and smartphone
  ground truth are generated by one definition.
- **`.DS_Store` and any non-`frame_*` file is skipped explicitly** — 41 of them are scattered
  through the test release and will otherwise be globbed as frames.
- **Splits: by contributor.** 6 of the 30 de-duplicated contributors held out as validation
  (24 train / 6 val, fixed in `index.json` at pack time), the rest train. Never split by frame or by clip: at 3 FPS, consecutive frames in a clip have a median
  mask IoU of 0.649, so a frame-wise split leaks near-duplicates into validation and inflates
  every number reported.

## 6. Augmentation

**Geometric block** (applied jointly to image, mask and box, both experiments): random resized
crop (scale 0.5-1.0, aspect 3/4-4/3), rotation +/-10 deg, horizontal flip p=0.5.

> **Chirality check (resolved, Cycle 2 — flip is enabled at p=0.5).** All ten gestures are
> single-hand static poses whose *identity* survives mirroring: call, dislike, like, ok, one,
> palm, peace, rock, stop and three each name a finger configuration, not a handedness. None of
> the ten is distinguished from another by mirroring — the failure mode a flip could cause is a
> pair of classes that map onto each other, and no such pair exists in this label set. Since the
> brief fixes the *right* hand for capture, flipping also covers the left-hand case for free.

**Photometric block, E1 (baseline):** conventional colour jitter — brightness, contrast,
saturation, hue — the level of augmentation a standard implementation would use.

**Photometric block, E3:** Camera-Pipeline Randomisation, s7.

## 7. E3 — Camera-Pipeline Randomisation (CPR)

**The method.** Randomise the parameters of an approximate inverse-then-forward camera imaging
pipeline on every training frame, so the network sees the same hand rendered by many different
synthetic cameras. Photometric only: boxes and masks are untouched.

| # | Stage | What it models | Distribution |
|---|---|---|---|
| 1 | sRGB -> linear (`x^2.2`) | undo display encoding | always |
| 2 | White-balance gains on R and B | AWB differences | `U(0.80, 1.25)` |
| 3 | Exposure scale | metering differences | `2^U(-0.6, 0.6)` |
| 4 | 3x3 colour matrix, `I + N(0, 0.05)`, rows renormalised | sensor colour primaries / CCM | always |
| 5 | Signal-dependent noise `N(0, sqrt(a*x+b))` | sensor read + shot noise | p=0.5, `a~logU(1e-5,1e-3)`, `b~logU(1e-6,1e-4)` |
| 6 | linear -> sRGB with random gamma | tone encoding | `gamma ~ U(1.8, 2.6)` |
| 7 | Monotone 256-entry tone-curve LUT (5 control points, offsets `U(-0.10,0.10)`) | vendor tone mapping | always |
| 8 | Saturation `U(0.6,1.3)`, hue `U(-0.05,0.05)` | vendor colour rendering | always |
| 9 | Gaussian blur **or** unsharp mask, mutually exclusive | phone sharpening vs soft optics | p=0.5 / p=0.5 |
| 10 | Downsample by `U(1.0,2.0)` then upsample, **down kernel from {nearest, bilinear, area, bicubic, lanczos}, up kernel from the same set minus `area`** (cv2 degrades `INTER_AREA` to nearest when upscaling, which would double nearest's effective probability); the two legs draw independently, as real chains use different filters for capture- and viewing-scaling | resolution + resampling mismatch | p=0.5 |
| 11 | JPEG re-encode at quality `U(40,95)` | lossy pipeline artefacts | p=0.5 |
| 12 | +/-1 px R/B channel shift | chromatic aberration | p=0.2 |

**Why this and not a published domain-generalisation method.** The alternatives randomise a
*proxy* for the shift — random convolution kernels (RandConv, ICLR 2021), Fourier amplitude
spectra (APR-S, ICCV 2021), painting styles (Stylized-ImageNet, ICLR 2019) — and rely on the real
shift landing inside the induced span. CPR randomises the actual nuisance variables that separate
a RealSense D455 from a phone: white balance, tone curve, sharpening, denoising, resampling and
JPEG. It is also target-data-free (mandatory: LSA p8), costs ~1-3 ms/frame on CPU with zero GPU
overhead, changes no architecture, and cannot destabilise training. Each stage maps to a named
imaging step, which is what makes the stage-wise ablation in s8 meaningful rather than a
hyperparameter sweep. RandConv and APR-S are implemented as **competing** methods in the ablation
so the choice is defended with numbers, not assertion.

**Bidirectionality — an easy thing to get wrong.** Phone images are typically *better* than
RealSense frames: sharper, less noisy, more saturated, more aggressively tone-mapped. An
augmentation policy that only degrades (noise, blur, JPEG) covers "worse than training" and
leaves the actual direction of the shift uncovered. Stages 8-10 are deliberately two-sided —
sharpen as well as blur, saturate as well as desaturate.

**Corrections made during implementation** (each found by a test, not by inspection):

* **Stage 8 was narrowed from `U(0.6,1.5)` to `U(0.6,1.3)`.** The pseudo-target shift (s7.1)
  applies saturation x1.4; with the original upper bound the "held-out" shift was inside CPR's
  reachable set and target-free model selection would have been circular.
* **The pseudo-target tone curve had to be made stronger than CPR can draw.** A 2,000-draw
  search over stage 7 found a CPR curve within 0.015 (~4/255) of the first pseudo-target curve
  written. It now sits ~0.04 outside CPR's reachable band at both control points, and a test
  asserts that separation. **Honest residual overlap:** 2x resampling still lies inside stage
  10's kernel set, so what is genuinely held out is the tone curve and the specific
  four-operation composition, not every individual operation. The report should say this.
* **Stage 5's noise term is clamped.** `N(0, sqrt(a*x+b))` has a negative radicand whenever the
  colour matrix pushes a channel slightly below zero, which is a NaN generator as written.
* **`strength` interpolates only the continuous stages.** JPEG quality tops out at 95 (still
  lossy) and a chromatic shift is +/-1 px or nothing, so scaling those toward zero reaches
  "mildest", not "identity". `strength=0` switches the photometric block off entirely; the two
  discrete stages are removed via `cpr_stages`, not via strength.
* **Measured cost is 2.4-3.0 ms at 384x288 and 4.5-6.5 ms at 512x384**, not the 1-3 ms estimated
  here. CPR runs on the crop (~0.75 of the packed frame), so ~4-5 ms is the operating figure.
  Twelve workers (the value in `configs/base.yaml`) keep the GPU fed comfortably; the report quotes the measured number.

**Excluded by the brief:** anything that consumes target data — FDA, AdaBN, TENT, DANN,
CycleGAN-style translation, and test-time BN re-estimation. These are cited in the report as
excluded alternatives with the reason, not silently omitted.

### 7.1 Target-free validation protocol

We may not tune on the smartphone set (LSA p8), and tuning on it and then reporting it would make
every E3 number meaningless. Two development signals are used instead:

1. **Leave-contributors-out validation** — the held-out contributors from s5. Measures
   background, lighting and skin generalisation, but not sensor change.
2. **Synthetic pseudo-target** — held-out RealSense frames pushed through a transform family
   that never appears in training: a fixed phone-style tone curve, 2x bicubic up-resample,
   strong unsharp mask, saturation x1.4.

**What "held out" actually means here, measured rather than asserted.** Pushing a 256-level ramp
through the real `cpr()` and the real `pseudo_target_transform()` and comparing the transfer
curves they realise: over 200 CPR draws, the closest any draw comes to the pseudo-target curve,
measured as the maximum absolute deviation across the curve, is **0.067** (~17/255). But at
either *individual* control point the pseudo-target value is inside CPR's realisable range —
CPR spans [0.098, 0.447] at input 0.25 where the pseudo-target sits at 0.110, and
[0.503, 0.979] at 0.75 where it sits at 0.890. Point-wise separation would require a physically
absurd curve. **What is held out is therefore the curve as a whole plus the four-operation
composition, not each operation individually** — saturation x1.4 lies inside stage 8's range and
2x resampling inside stage 10's kernel set. An earlier version of this document claimed ~0.04
point-wise separation; that was measured on the bare tone LUT rather than on the composed
transform `cpr()` actually applies, and was wrong. The guarding test now measures the composed
curves (`tests/test_augment.py::test_pseudo_target_transform`).

**How it is used.** Every run reports its pseudo-target metrics alongside its RealSense ones, as
a target-free diagnostic. The runs that *select on* it — `--select-on pseudo` — are the CPR
strength sweep (`e3_strength05`, `e3_strength15`) and `e3_selpseudo`, which together answer
whether target-free selection changes the operating point at all. The headline E3 run selects on
RealSense validation, so the mandatory comparison does not depend on the proxy being good.

The real smartphone set is evaluated **once**, at the end, and the report says so explicitly.

## 8. Experiments and ablations

`00_PLAN.md` s3 holds the mandatory matrix (B0, B1, E1, E2, E3). The ablation grid below is what
earns the "experiments and innovation" marks; each row reports the full metric set on
RealSense-val, RealSense-test, pseudo-target and (once) smartphone.

**Block A — additive build-up.** A0 geometric only (`a0_none`) -> A1 + standard colour jitter
(`e1`) -> A2 + linear-ISP stages wb/exposure/ccm/gamma/tone (`a2_linear_isp`) -> A3 + degradation
stages noise/sharpblur/resample (`a3_degrade`) -> A4 + JPEG and chromatic aberration = full CPR
(`e3`).

**Block B — leave-one-out from A4.** B1 without tone/gamma/WB/CCM, B2 without noise, B3 without
blur/sharpen, B4 without resample+kernel randomisation, B5 without JPEG, **B6 without
saturation/hue, B7 without exposure and chromatic aberration**. Additive order confounds Block A;
Block B is what actually attributes the gain. B6 and B7 were added because B1-B5 left three of the
eleven stages (`exposure`, `satihue`, `chroma`) unattributed by any row — and "each stage maps to a
named imaging step and is individually accounted for" is the argument for CPR over a
hyperparameter sweep, so leaving three unaccounted would undercut it.

**Block C — normalisation, crossed with A0 and A4.** {GN(32), IBN-a} x {A0, A4}, with BN at
both points supplied by `a0_none` and `e3`. Queue rows: `c_gn`, `c_ibn` (A4) and `c_gn_a0`,
`c_ibn_a0` (A0). If
GN/IBN help at A0 but add nothing on top of A4, the augmentation has already absorbed the effect
— that is a reportable finding either way.

**Block D — competing randomisers.** D1 RandConv (k in {1,3}), D2 APR-S, D4 AugMix.
D4 is expected to underperform — AugMix deliberately *excludes* contrast, colour, brightness,
noise and blur to keep ImageNet-C a clean held-out test, and those are precisely our shift.
Reporting that is more useful than omitting it.

*D3 MixStyle was planned and is not implemented.* It is dropped rather than left in the plan: a
row in an ablation table that was never run is worse than an absent one. The three that are run
already span the three mechanisms that matter here (random filtering, frequency-domain
amplitude, and op-composition).

*AugMix deviation, applied and documented.* Five of AugMix's nine published ops are geometric,
and AugMix mixes several independently-warped branches, so there is no single geometry to apply
to a mask or box. Applying them image-only leaves the label describing content that has moved
(measured up to 25 px) — label noise that would corrupt D4's detection and segmentation targets
for reasons unrelated to AugMix. When a mask or box is present the draw is therefore restricted
to the four photometric ops (autocontrast, equalize, posterize, solarize). Those four still
exclude contrast, colour, brightness, sharpness, noise and blur, so the property D4 is cited for
survives the restriction.

**Block E — architecture ablations.** A-MAP: classification without mask-attended pooling
(s3.4 hypothesis). A-MT: single-task heads trained separately (isolates multi-task benefit).
A-UW: uncertainty loss weighting. A-GIOU: without the GIoU term.

**Seeds.** 3 seeds on A1/`e1` and A4/`e3` (the two references every delta is measured
against) and 2 on A0/`a0_none`; 1 seed elsewhere, stated in the table.
On a dataset this size, seed variance may exceed several Block-B effects, and a single-seed delta
reported without variance is the first thing a marker should attack.

## 9. Evaluation protocol (`src/evaluate.py`)

| Metric | Definition used | Note |
|---|---|---|
| Detection accuracy@0.5 IoU | fraction of images with `IoU(pred_box, gt_box) >= 0.5` | single predicted box = top heatmap peak |
| Mean bbox IoU | mean over images of `IoU(pred, gt)` | images with an empty GT mask are excluded and counted separately |
| Segmentation mIoU | mean of hand-IoU and background-IoU, averaged per image | hand-IoU also reported alone, since background IoU is ~0.97 by construction and flatters the mean |
| Dice | `2|A∩B| / (|A|+|B|)` on the hand class | per image, then averaged |
| Top-1 accuracy | argmax of gesture logits | |
| Macro-F1 | unweighted mean of per-class F1 over 10 classes | |
| Confusion matrix | 10x10, row-normalised version also plotted | |

Predicted mask threshold 0.5; ground-truth mask threshold 128. Metrics are computed per frame as
the brief requires, **and** aggregated per clip, because the 15 frames of a clip are correlated
(median consecutive mask IoU 0.649) so per-frame numbers have a smaller effective sample size than
their `n` suggests. Both are reported; per-frame is the headline.

## 10. Training recipe (starting point, to be confirmed in Cycle 3)

AdamW, lr 3e-4 with cosine decay and 3-epoch linear warmup, weight decay 0.05, batch 48 at
384x288, AMP (bf16), gradient clip 1.0, ~60 epochs, EMA of weights for evaluation, checkpoint
every epoch (host may be reclaimed — `00_PLAN.md` K5). Selection on RealSense-val macro-F1 +
mask mIoU, never on anything smartphone-derived.
