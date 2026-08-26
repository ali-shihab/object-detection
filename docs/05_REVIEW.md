# Adversarial review

**Scope owner:** this document. Findings from reviewing documented claims and code against the
artefacts and against the LSA coursework brief, per cycle, with resolutions.

**Last updated:** 2026-08-26

## Cycle 2 review — 2026-08-26

**Method:** Every module in `project_18006111_Shihab/src/`, `tools/`, `tests/` and `configs/` was
read, and every claim in `00_PLAN.md`, `01_DATA.md`, `02_DESIGN.md`, `03_IMPLEMENTATION.md`,
`README.md`, `requirements.txt`, `RECORDING_GUIDE.md` and `report/coursework1LSA_Shihab.tex` was
checked against the artefact that would prove it. Verification was by execution wherever possible:
all four test suites were run (10/10, 9/9, 17/17, 16/16 — all pass); `HandNetEncoder`/`HandNet`
were instantiated and every stride, channel count, block count, decoder width and head shape in
the s3.1/s3.2/s3.4 tables printed and compared; `MultiTaskLoss` was driven with `has_mask`
all-False, mixed and all-True batches and the gradients traced per head; the head-initialisation
claim was measured with and without the shipped init; a 6,000-draw search compared the tone
transfer function `cpr()` actually applies against `pseudo_target_transform`'s curve; a 480-frame
synthetic packed dataset was built and `src.train --dry-run`, `src.evaluate --split val` and
`src.visualise` were run end to end to confirm the required metric set, the confidence outputs and
the figures; the CLI surface was enumerated against `Config`'s fields; `--w-giou 0` was executed to
confirm rejection; photometric modes were measured for image/mask/box desynchronisation; and the
repository was grepped for every banned library and for pretrained-weight loading.

**Rules check, up front:** no banned library appears in the submitted model's path. `src/` imports
only `torch`, `numpy`, `cv2`, `PIL`, `matplotlib`, `yaml` — no `torchvision`, `timm`,
`segmentation_models_pytorch`, `detectron2`, `mmdet`, `ultralytics`, `albumentations` or `kornia`.
`requirements.txt` matches what is actually imported. `segment_anything` appears only as an
optional, non-default backend in `tools/annotate_smartphone.py` (see F10). No path exists by which
smartphone data can reach training or model selection (see the verified list).

### Findings

| # | Severity | Area | Claim / requirement | What is actually true | Evidence | Status |
|---|---|---|---|---|---|---|
| F1 | BLOCKER | Deliverable / Obj 2-4 | `README.md` s1 tree lists `smartphone_dataset/18006111_Shihab/` as "the collected smartphone test set (60 frames)" and `raw_videos/` as "the 20 source clips" | Neither exists. `smartphone_dataset/` holds only `RECORDING_GUIDE.md` and an **empty** `raw_videos/`; `weights/` and `results/` are empty. Objectives 2-4 and experiments E2/E3 are unsatisfiable as the tree stands | `ls -la project_18006111_Shihab/{smartphone_dataset,smartphone_dataset/raw_videos,results,weights}` — all empty | Open — known as K1 in `00_PLAN.md`; the README wording is the reviewable defect |
| F2 | MAJOR | E3 / s7.1 held-out claim | `02_DESIGN.md` s7 + `03_IMPLEMENTATION.md` I3: the pseudo-target curve "sits ~0.04 outside CPR's reachable band at both control points, and a test asserts that separation" | The map `cpr()` applies is stage 6 composed with stage 7 (`code = _U8_GRID ** (2/gamma)`, then `_tone_curve(rng, s, code)`), never `_tone_lut`. Over 6,000 draws that composed response reaches **0.1137** at input 0.251 (pseudo-target 0.1115 -> separation **0.0022**, not 0.04) and **0.8706** at 0.749 (pseudo-target 0.8885 -> 0.018). The guarding test measures `_tone_lut` — stage 7 on a *uniform* grid — a function `cpr()` never calls, so it does not cover the composition it claims to cover. The whole-curve separation does survive (min over draws of max abs diff = 0.055 for gamma+tone, 0.048 for full CPR, ~12/255), so the *family* is still held out | 6,000-draw sweep of `cpr(ramp, stages={"gamma","tone"})` and 1,500-draw sweep of full `cpr()` against `cv2.LUT(ramp, _PSEUDO_TONE_LUT)`; `tests/test_augment.py:331` reads `A._tone_lut(...)` | Open |
| F3 | MAJOR | Model selection | `02_DESIGN.md` s10: "Selection on RealSense-val macro-F1 + mask mIoU" | `_selection_score(m, "combined")` returns `(cls_macro_f1 + seg_iou_hand + det_acc@0.5) / 3` — three terms, and hand-IoU, not mIoU. Every checkpoint in the study is chosen by this rule | `src/train.py:_selection_score`; training log line `select/score = 0.00606` = `(0.01818 + 0 + 0)/3` from the end-to-end run | Open |
| F4 | MAJOR | E3 protocol | `02_DESIGN.md` s7.1: "**All CPR strength and variant selection is done against this**" (the pseudo-target) | No run selects on it. `configs/e1.yaml` and `configs/e3.yaml` both set `select_on: val`; no row of `scripts/queue_full.txt` passes `--select-on pseudo`; no row sweeps `--aug-strength`. The pseudo-target is only ever *reported*, by one post-training eval in `remote_experiment.sh` | configs + `queue_full.txt` + `remote_experiment.sh` | Open |
| F5 | MAJOR | Ablation grid | `02_DESIGN.md` s8 Block D lists "D3 MixStyle with contributor ID as pseudo-domain" | MixStyle is not implemented anywhere. `PHOTOMETRIC_MODES = ("none","jitter","cpr","randconv","aprs","augmix")`; `grep -rni mixstyle` over the whole repo returns nothing; no queue row | `src/augment.py:PHOTOMETRIC_MODES`; repo-wide grep; `queue_full.txt` has d1/d2/d4 only | Open |
| F6 | MAJOR | Ablation grid | `02_DESIGN.md` s8 Block C: "{BN, GN(32), IBN-a} x {A0, A4}"; Block A: A0 -> A1 -> A2 -> A3 -> A4 | `queue_full.txt` runs `c_gn`/`c_ibn` on `e3.yaml` (= A4) only — the A0 arm is absent, so the stated inference ("if GN/IBN help at A0 but add nothing on top of A4, the augmentation has absorbed the effect") cannot be drawn. Block A has rows for A0 (`a0_none`), A1 (`e1`) and A4 (`e3`) but none for A2 (linear-ISP stages only) or A3 (+ degradation stages) | `scripts/queue_full.txt` | Open |
| F7 | MAJOR | CLI / ablation reachability | `README.md` s4: "Any field can be overridden on the command line, which is how the ablations are run". `02_DESIGN.md` s8 Block E lists ablation **A-GIOU** ("without the GIoU term") | 14 of `Config`'s 37 fields have no CLI override: `geometric`, `w_heat`, `w_size`, `w_off`, **`w_giou`**, `w_bce`, `w_dice`, `w_cls`, `label_smoothing`, `warmup_epochs`, `min_lr_factor`, `grad_clip`, `amp`, `deterministic`. A-GIOU therefore cannot be launched as documented and has no queue row | `python -m src.train --w-giou 0 --out /tmp/x` -> `error: unrecognized arguments: --w-giou 0`; field-vs-parser enumeration | Open |
| F8 | MAJOR | Ablation validity | `02_DESIGN.md` s8 Block D presents D4 AugMix as a competing randomiser expected to underperform *because its published op set excludes the relevant nuisances* | The implementation adds a second, confounding cause: 5 of the 9 published ops are geometric (rotate, shear x/y, translate x/y) and `augmix()` warps the **image only**, while `AugmentPolicy.__call__` returns `mask` and `box` untouched. A D4 deficit will be attributable to two causes, not one. `src/augment.py` states the caveat in a docstring; `02_DESIGN.md` and `03_IMPLEMENTATION.md` do not | 60-draw measurement of intensity-centroid displacement from the unchanged GT box centre: augmix median 1.8 px / max **25.1 px**; jitter 0.7/0.7; cpr 0.7/2.7; randconv 0.7/6.8; aprs 0.7/4.4 | Open |
| F9 | MAJOR | Architecture doc | `02_DESIGN.md` s3.3: heatmap bias "`-2.19` for the heatmap (**p=0.01**)"; the same "p=0.01" is repeated in `src/model.py:_init_weights`'s comment | `sigmoid(-2.19) = 0.1007` — the prior is **p = 0.10**. `logit(0.01) = -4.595`. The `HEAT_BIAS_INIT` docstring flags the discrepancy correctly and `tests/test_model.py` prints "mean initial heat prob 0.100", so the design doc is contradicted by the code's own test | executed `torch.sigmoid(tensor(-2.19))`; `tests/test_model.py` output | Open |
| F10 | MAJOR | Rules compliance argument | `README.md` s8 and `tools/annotate_smartphone.py` docstring: "the brief explicitly sanctions a SAM-assisted annotation workflow", citing LSA p6 "you could use the annotation guide provided in coursework 1 release" | The quoted text sanctions an annotation *guide*, not Segment Anything. Against the brief's own rule — "if you are unsure whether a particular library is allowed, assume it is not" — this is an unsupported compliance argument on a rule-sensitive point. Separately, README s8's clause "no pretrained weights are loaded" is scoped "anywhere in `src/` or `tools/`", while `tools/annotate_smartphone.py:239` loads a SAM checkpoint. Mitigation already in place: `--backend` defaults to `grabcut`, which loads nothing | `README.md:138-146`; `tools/annotate_smartphone.py:4-31, 229-241, 372-373` | Open |
| F11 | MAJOR | Report requirement | Brief: a GitHub link to the code goes in the report (`00_PLAN.md` R13) | No GitHub URL anywhere in `report/coursework1LSA_Shihab.tex` or `report/refs.bib`, and none of the section comment blocks prompts the author to add one — so the scaffold will not catch the omission | `grep -in "github\|url{"` over both files -> no match | Open |
| F12 | MAJOR | Deliverable tree | `README.md` s3-s6 documents the reproduction path (`--config configs/e1.yaml`, `python tools/pack_dataset.py ...`) | `configs/`, `tools/`, `tests/` and `scripts/` sit *beside* `project_18006111_Shihab/`, not inside it. Submitting the folder the brief names leaves every documented command unrunnable and ships none of the ablation configs the report will cite. The six required `src/` filenames, `weights/`, `results/`, `requirements.txt` and `README.md` are all present and correctly named | device `find` of the tree; `README.md` s1 ("Tooling ... lives one level up, in `../tools/`") | Open |
| F13 | MAJOR | Obj 2 annotation | Brief: each smartphone image needs a class label, a binary mask **and a bounding box derived from the mask** | `annotate_smartphone.py` derives the box correctly but writes it only to `<out>/../qc/annotation_index.json` — i.e. `smartphone_dataset/qc/`, **outside** `smartphone_dataset/18006111_Shihab/`. The delivered dataset folder will contain `rgb/` and `annotation/` only. (`pack_dataset.py` re-derives boxes from masks, so nothing downstream breaks — this is a deliverable-completeness gap) | `tools/annotate_smartphone.py:389, 434-452` | Open |
| F14 | MINOR | Architecture doc | `02_DESIGN.md` s3.3 / `03_IMPLEMENTATION.md` I1: the initialisation defect put the loss "near **800** at step 0 instead of **3.7**" | Neither number reproduces. At 384x288 in BN train mode: shipped init total **14.74** (heat 9.26) on a noise batch and **14.80** on a photo-like batch; with plain Kaiming-initialised head convs, total **13,455** (heat 13,444) and **11,911**. The mechanism and its most specific claim are exactly right — max abs heat logit measured at **27.0**, matching "+/-27" — only the two loss figures are wrong | executed both initialisations through `MultiTaskLoss` on synthetic and photo-like batches | Open |
| F15 | MINOR | CPR stage table | `02_DESIGN.md` s7 row 10: "kernel drawn from {nearest, bilinear, area, bicubic, lanczos}" | Only the *downsample* leg draws from those five. The *upsample* leg draws from four: `_UP_KERNELS` drops `area`, because cv2 degrades `INTER_AREA` to nearest when upscaling. Rows 1-9, 11 and 12 all match the table exactly | `src/augment.py:_UP_KERNELS` and the stage-10 block | Open |
| F16 | MINOR | Data figure | `02_DESIGN.md` s4.1: "this is the mechanism that still lets **~26k frames** train the classifier" | The packed training release is **20,550** frames (`01_DATA.md` s5.1) | `01_DATA.md` s5.1 | Open |
| F17 | MINOR | Contributor count | `02_DESIGN.md` s5: "Roughly 6 of **31** contributors held out as validation" | 31 archive folders -> **30** after de-duplication; `01_DATA.md` s5.3 and `assign_splits` hold out 6 of 30 | `01_DATA.md` s5.1/s5.3 vs `02_DESIGN.md` s5 | Open |
| F18 | MINOR | Line-count table | `03_IMPLEMENTATION.md` s1 | Four of nine are off by more than rounding: `model.py` "~460" vs **475**; `evaluate.py` "~380" vs **366**; `pack_dataset.py` "~370" vs **401**; `annotate_smartphone.py` "~420" vs **464**. The other five are right: utils 479, augment 774, dataloader 272, train 539 ("~540"), visualise 440 | `wc -l` on all nine files | Open |
| F19 | MINOR | Figure count | `03_IMPLEMENTATION.md` s5: "all **eight** figure types render" | `visualise.py` defines **seven** plotting functions: confusion matrix, qualitative overlays, reliability, training curves, ablation bars, augmentation grid, domain gap. Five rendered on an end-to-end run with one result JSON; the other two need >= 2 results | `grep "^def plot_"`; executed `python -m src.visualise` on synthetic results -> 5 PNG/PDF pairs | Open |
| F20 | MINOR | Ghost file references | `tools/pack_dataset.py:63` "pure helpers (unit-tested in **tests/test_pack.py**)"; `requirements.txt:18` "**tests/test_evaluate.py** cross-checks our metrics against sklearn" | Neither file exists. `tests/` holds exactly test_utils, test_model, test_augment, test_dataloader. The sklearn cross-check lives in `tests/test_utils.py:334`. (`03_IMPLEMENTATION.md` s5 is honest that the pack checks are a scripted fixture — the two in-code references are the defect) | `ls tests/`; repo-wide grep | Open |
| F21 | MINOR | Config layering | `configs/e1.yaml` / `e3.yaml` header: "Experiment files override ONLY what differs, so any parameter absent from e1.yaml / e3.yaml is provably identical between them". `README.md` s4: they "differ in exactly one line (`photometric`)" | `configs/base.yaml` is loaded by nothing — `grep -rn "base.yaml"` over the repo returns no match, and `load_config` reads only the single `--config` path. e1.yaml and e3.yaml are **full copies** of base.yaml, so no parameter is absent and the layering described does not exist. `diff e1.yaml e3.yaml` shows **two** value lines differing (`photometric`, `tag`) plus the comment header. The substance holds today, but by duplication, not by construction | `grep`; `diff configs/e1.yaml configs/e3.yaml`; `src/train.py:load_config` | Open |
| F22 | MINOR | Pseudo-target eval | `pseudo_target_transform` docstring: "with a generator the unsharp amount and saturation get a +/-10 % jitter so a selection sweep sees a small neighbourhood rather than one point" | `evaluate_model` rebuilds `np.random.default_rng(1234)` **inside the per-batch loop**, so the draw sequence restarts every batch: the transform a frame receives is fixed by its position *within its batch*, hence by `--batch-size`, and every batch replays the same <= B draws. Training-time selection would use `eval_batch_size = batch_size = 48` while the CLI defaults to 32 (`remote_experiment.sh` takes that default), so a pseudo-target number reported after training is not the one selection saw | `src/evaluate.py:evaluate_model`; executed: batch 2's four draws reproduce batch 1's exactly, positions 0 and 2 differ | Open |
| F23 | MINOR | Doc-vs-code | `00_PLAN.md` s2: "Is `torchvision.transforms` allowed? **Used only for basic tensor/image ops**; the augmentation policy is our own code" | torchvision is not imported anywhere and is deliberately absent from `requirements.txt`. `00_PLAN.md` contradicts `requirements.txt`, `README.md` s8 and the code, on the one topic a marker will check first | repo-wide grep for `torchvision` -> only prose disclaimers | Open |
| F24 | MINOR | De-duplication | `01_DATA.md` s5.2 D2: "Contributors are de-duplicated by **content signature** before packing" | `subject_signature()` hashes sorted `(relative path, file size)` pairs only — not pixel content. The md5-of-every-PNG evidence cited in the same row came from a one-off `scripts/remote_check_dupes.sh`, not from what the packer runs. False positives are implausible here, but the mechanism is not what the row says | `tools/pack_dataset.py:subject_signature` | Open |
| F25 | MINOR | Worker count | `02_DESIGN.md` s7: "**Eight** workers keep the GPU fed comfortably"; `Config.num_workers` default is 8 | All three YAMLs set `num_workers: 12` | `configs/*.yaml` | Open |
| F26 | MINOR | Seeds | `02_DESIGN.md` s8: "**3 seeds on A0**, A4 and the final configuration" | `queue_full.txt` gives 3 seeds to `e1` (= A1, jitter) and `e3` (= A4) and **one** to `a0_none` (= A0) | `scripts/queue_full.txt` | Open |

### Verified-correct (checked and found sound)

**Rules and structure**
* No `torchvision`, `timm`, `segmentation_models_pytorch`, `detectron2`, `mmdet`, `ultralytics`,
  `albumentations` or `kornia` anywhere in `src/` or `tools/`. Third-party imports across the whole
  repo reduce to `torch`, `numpy`, `cv2`, `PIL`, `matplotlib`, `yaml` (+ `sklearn` in one test,
  `segment_anything` in the optional annotation backend) — exactly what `requirements.txt` lists,
  with the two dev-only entries correctly commented out.
* No pretrained weights in the model path: `HandNet._init_weights` is Kaiming/normal from scratch;
  nothing loads a checkpoint except `--resume` and `load_checkpoint`. `IMAGENET_MEAN/STD` are
  normalisation constants only.
* `src/` contains all six required filenames (`dataloader, model, train, evaluate, visualise,
  utils`) plus `augment.py`; `weights/`, `results/`, `requirements.txt`, `README.md` present;
  no supplied RealSense data is shipped. Report uses `\documentclass[conference]{IEEEtran}` with
  an unaltered `IEEEtran.cls` present.

**Architecture (`02_DESIGN.md` s3, verified by instantiating and printing shapes at 384x288)**
* Every row of the s3.1 table is exact: strides 2/4/8/16/32, channels 24/48/96/192/320, blocks
  2/3/4/3, spatial 192x144 / 96x72 / 48x36 / 24x18 / 12x9.
* Decoder widths 160/96/64/48 exactly as s3.2; seg logit emitted at stride 2 and bilinearly
  upsampled to the full 288x384; detection head at stride 4 (72x96) with `heat`(1)/`size`(2)/
  `off`(2); classifier first Linear has 640 inputs = concat[GAP(320), mask-attended(320)],
  hidden 256, dropout 0.2 — all as s3.4.
* The mask-attended pooling takes its mask from the model's **own** `out["seg"]`, `sigmoid`-ed and
  `.detach()`-ed, average-pooled to the s4 grid — exactly what s3.4 claims, including the
  train/test consistency argument. `tests/test_model.py` independently asserts detachment.
* `SEG_BIAS_INIT = -3.56` = `logit(0.0275)` to four decimals, matching the measured hand-pixel
  prior in `01_DATA.md` s2.4.
* Model is 10,009,544 parameters; A-MAP removes 81,920, as the test reports.

**Loss (`02_DESIGN.md` s4, s4.1 — verified by constructing the three batch types and running them)**
* All six terms present with the documented default weights (1.0 / 0.1 / 1.0 / 1.0 / 1.0+1.0 /
  1.0), composed from `binary_cross_entropy_with_logits`, `l1_loss`, `cross_entropy` and
  own-code GIoU. Nothing from a detection library.
* The s4.1 sparse-mask rule holds exactly. `has_mask` all-False: total = the classification term
  alone (2.2309), finite, `requires_grad=True`, `backward()` succeeds, **no NaN**, `seg_out` and
  `heat_head` receive `grad = None` while `cls_head` (1.21e3) and the encoder stem (7.09e1) do.
  Mixed (n=2) and all-True (n=4) batches give identical det/seg terms for identical targets,
  confirming the average is over flagged samples only while `L_cls` spans the whole batch.
* `focal_heatmap_loss` normalises by the positive count, so the term is comparable across batches
  with different annotation densities, as claimed.

**CPR (`02_DESIGN.md` s7 stage table, checked row by row against `src/augment.py`)**
* Rows 1-9, 11 and 12 all match: sRGB->linear always; WB `U(0.80,1.25)` on R and B; exposure
  `2^U(-0.6,0.6)`; CCM `I + N(0,0.05)` with rows renormalised; noise `p=0.5` with
  `a~logU(1e-5,1e-3)`, `b~logU(1e-6,1e-4)`; gamma `U(1.8,2.6)`; 5-knot monotone tone LUT with
  `U(-0.10,0.10)` offsets; saturation `U(0.6,1.3)` (the documented narrowing from 1.5) and hue
  `U(-0.05,0.05)`; blur/unsharp mutually exclusive at 0.5/0.5; JPEG `U(40,95)` at `p=0.5`;
  chroma +/-1 px at `p=0.2`. Only the up-leg kernel set deviates (F15).
* The noise radicand is clamped (`np.maximum(x, 0.0)`) — the documented NaN fix is real.
* `strength=0` short-circuits the block; the discrete stages are removed via `cpr_stages`, as s7
  says. `CPR_STAGES` has 11 toggleable names, matching `03_IMPLEMENTATION.md` s1.
* Measured cost matches the corrected figures: `cpr()` 2.95 ms at 384x288 and 5.07 ms at 512x384,
  inside the stated 2.4-3.0 / 4.5-6.5 ms bands.
* Block B's leave-one-out set is arithmetically complete: B1-B5 leave `exposure`, `satihue`,
  `chroma` unattributed, which is exactly what B6 and B7 were added to cover, and all seven rows
  exist in `queue_full.txt` with correct stage lists.

**Evaluation (`02_DESIGN.md` s9 — verified by running `src.evaluate` end to end on a synthetic pack)**
* Every required metric is produced, on **both** `frame` and `clip` blocks: `det_acc@0.5`,
  `mean_box_iou`, `seg_miou`, `seg_iou_hand`, `seg_iou_bg`, `seg_dice`, `cls_top1`,
  `cls_macro_f1`, plus a 10x10 `confusion_matrix` (and a row-normalised plot), per-class F1,
  `det_acc@0.75/0.9`, ECE and a reliability curve.
* **Gesture confidence is produced and reported**: `softmax(cls).max()` per frame, summarised as
  `mean_cls_conf`, `cls_conf_correct`, `cls_conf_incorrect`, and calibrated via ECE. Detection
  confidence (peak heatmap value) is reported alongside.
* **Metrics are computed on val as well as test**: `scripts/remote_experiment.sh` runs
  `src.evaluate --split val` and `--split test` for every queued run, and `train.py` logs the
  headline val block every epoch. Confirmed by executing `--split val` on the synthetic pack:
  160 frames / 40 clips, full metric block returned.
* Thresholds are exactly as documented: predicted mask `>= 0.5` in `seg_scores`; ground truth
  binarised at 128 in `dataloader.py` (`MASK_THRESH = 128`) and again after the resize, so the
  `>= 0.5` in `seg_scores` sees an already-binary target — the threshold is applied once per
  stage, not twice on the same quantity.
* Exclusion rules are implemented and counted, never silently zeroed: frames with `has_mask=False`
  are dropped from every detection and segmentation metric, and `n_boxes_scored` /
  `n_boxes_skipped` / `n_seg_scored` / `n_seg_skipped` report the denominator actually used
  (80/80 on a deliberately 50 %-annotated synthetic split).
* Qualitative mask-and-box overlays render (`plot_qualitative`), satisfying the brief's
  qualitative requirement.
* Bootstrap CIs resample clips, not frames; the point estimate is the real frame mean and does not
  move with the bootstrap seed.
* `MetricAccumulator`'s macro-F1 and confusion matrix are cross-checked against `sklearn` by a
  passing test; the clip block weights each frame `1/frames-in-clip` so `cls_top1` equals the mean
  of per-clip accuracies, also test-covered.

**Data pipeline (`01_DATA.md`, `tools/pack_dataset.py`)**
* Packer defaults are as documented: `--packed-size 512 384`, `--split-holdout 6`, `--seed 0`;
  JPEG quality 95 with `subsampling=0` (4:4:4) and PNG masks, matching `01_DATA.md` s5.4.
* Masks are resized **bilinearly then re-thresholded at 128** and the box is derived from the
  **resized** mask — both exactly as claimed, and `src/dataloader.py` repeats the same rule after
  its own resize so stored and loaded boxes cannot disagree.
* Non-`frame_*` files (including the 41 `.DS_Store`) are excluded by `FRAME_RE`, not by globbing.
* Wrapper descent works: `find_dataset_root` walks single-child directories up to six levels, which
  recovers the `25047621_Wu/dataset/25047621_Wu/...` case, and a zero-frame contributor prints an
  explicit warning.
* `--test` maps every contributor to a single `test` split; `assign_splits` is contributor-wise and
  deterministic in `seed`, and the split is written into `index.json` so it cannot drift.
* `01_DATA.md`'s test-set statistics are internally consistent with everything `02_DESIGN.md` s1/s2
  derives from them: median hand area 2.75 %, median box 115x151 px = 0.180x0.315 normalised,
  aspect 0.76, centre (0.448, 0.489), 1.91 % multi-component at threshold 128 with zero second
  components above 0.5 % of the image, and 115x151 -> 69x91 px at 384x288.

**Training and leakage**
* **No path exists for smartphone data to influence training or model selection.**
  `select_on` accepts only `val` or `pseudo`; the `pseudo` path applies `pseudo_target_transform`
  to the RealSense **val** loader, never to phone data; `train.py` never names the phone index;
  `remote_experiment.sh` deliberately omits phone evaluation and `remote_eval_phone.sh` is a
  separate, post-training script. `build_loaders` does construct a `test` loader from
  `test_index_path` (the RealSense test pack in all three configs), but `train.py` iterates only
  `train` and `val`, so it is never read.
* Splits are contributor-wise everywhere; a frame- or clip-wise split is impossible because the
  split map is keyed on `subject`.
* The geometric block transforms image, mask and box jointly, and recomputes the box from the
  rotated/cropped/flipped **mask** rather than from rotated corners; `tests/test_augment.py`
  asserts box/mask agreement to <= 1 px over 400 draws and that photometric ops leave mask and box
  bit-identical. Measured: jitter and cpr displace image content by <= 0.7 px median.
* Checkpoints are written to a temp file and `os.replace`d; every checkpoint carries the full
  config; every result JSON carries the checkpoint path, index path, training config and a
  SHA-256 prefix of the source files.
* Worker seeding is derived from `torch.initial_seed()` (per worker *and* per epoch) with the pid
  used only as a cache key, exactly as documented; four dataloader tests pin this, including the
  cv2 fork-deadlock regression.
* The end-to-end smoke run works: `src.train --dry-run 3` trains, validates, logs the val metric
  block and writes `best.pt`; `src.evaluate` and `src.visualise` then run off that checkpoint.

**Test coverage (`03_IMPLEMENTATION.md` s5) — all four suites executed**
* `tests/test_utils.py` **10/10 pass**, `tests/test_model.py` **all 9 checks pass**,
  `tests/test_augment.py` **17/17 pass**, `tests/test_dataloader.py` **16/16 pass**. The claimed
  pass counts are exact.
* `evaluate.py` imports nothing from `train.py`; the reverse edge is a single deferred import
  inside `main()` — verified.

**Smartphone protocol**
* `RECORDING_GUIDE.md` specifies 10 gestures x 2 clips x 3-5 s, right hand, landscape, locked
  AE/AF/AWB, ten distinct scenes — consistent with the brief. `annotate_smartphone.py` samples 3
  frames per clip from `n` equal windows (spacing guaranteed for a >= 3 s clip), writes
  `<gesture>/<clip>/{rgb,annotation}/frame_00N.png` — the CW1 structure minus depth — and warns
  if the total is not 60 or if a gesture is missing.

### Not checkable yet

* **Every number in `04_RESULTS.md` and the report.** No training has been run: `weights/`,
  `results/`, `report/figures/` and `report/generated/` are all empty, so `report/coursework1LSA_Shihab.tex`
  does not compile today (its `\input{generated/...}` targets do not exist). The macro names quoted
  in the section comments (`\EoneRsMacroF`, `\EoneRsHandIoU`, `\EoneRsDetAcc`, `\EtwoPhoneMacroF`,
  `\GainMacroF`) were checked against `tools/make_report_tables.py:macros()` and all resolve.
* **Report page count** (6 including references) — the `.tex` is a scaffold with no prose.
* **Anything requiring the smartphone data**: E2, E3, the E3-E2 gain, the QC flag rate, whether the
  annotation pipeline's masks are usable, and whether the `--crop-43` centre-crop (on by default,
  undocumented in `01_DATA.md`/`02_DESIGN.md`) interacts with the "same preprocessing" rule of E2.
* **Training-release mask anti-aliasing statistics** — `01_DATA.md` s6 already records this as
  unmeasured.
* **`tools/baseline_classical.py` (B0)** was not executed; it has no test suite and no queue row in
  `queue_full.txt`, only `scripts/remote_baseline.sh`.
* **Whether E3's gain survives seed variance** — the point of the 3-seed rows, which have not run.

---

## Cycle 2 resolutions — 2026-08-26

Every finding above was acted on the same day it was raised. Status below supersedes the
"Open" column in the tables.

| # | Sev | Resolution |
|---|---|---|
| F1 | BLOCKER | **Deferred to the user, not fixed in code.** The smartphone set requires a physical recording session; the capture protocol, the annotation pipeline (`tools/annotate_smartphone.py`) and the QC loop are all built and smoke-tested, so the turnaround once 20 clips exist is minutes. `README.md` s1 now reads as a description of the submitted tree; the tree is incomplete until the recording happens, and that is stated to the user rather than papered over. **Still open.** |
| F2 | MAJOR | **Fixed, and the claim weakened to what is true.** The guard test now pushes a 256-level ramp through the real `cpr()` and the real `pseudo_target_transform()` and compares the *composed* transfer curves, not the bare LUTs. Measured: over 200 CPR draws the closest whole-curve match is 0.067 max-deviation, but at either individual control point the pseudo-target lies inside CPR's range. `02_DESIGN.md` s7.1 now states exactly that, including which operations are genuinely held out (the curve shape and the four-op composition) and which are not (saturation 1.4, 2x resampling). |
| F3 | MAJOR | Fixed. `02_DESIGN.md` s10 now states the selection criterion the code implements: the equally-weighted mean of macro-F1, hand IoU and detection accuracy@0.5. |
| F4 | MAJOR | Fixed by making the claim true rather than by softening it. Three runs now select on the pseudo-target (`e3_strength05`, `e3_strength15`, `e3_selpseudo`), giving both a strength sweep and a direct test of whether target-free selection moves the operating point. s7.1 also now distinguishes *reported on* from *selected on*, and notes that the headline E3 selects on RealSense validation so the mandatory comparison does not rest on the proxy. |
| F5 | MAJOR | Fixed by deletion. MixStyle is removed from Block D with a note saying so — an unrun row in an ablation table is worse than an absent one. |
| F6 | MAJOR | Fixed. `a2_linear_isp`, `a3_degrade`, `c_gn_a0` and `c_ibn_a0` added to the queue, so Block A is a real ladder and Block C has both arms. |
| F7 | MAJOR | Fixed. `src/train.py` gains `--set KEY=VALUE` (YAML-parsed), which reaches **every** `Config` field. `e_no_giou` added to the queue as `--set w_giou=0.0`. `README.md` s4 documents it and no longer overstates the per-field flags. |
| F8 | MAJOR | Fixed properly rather than caveated. When a mask or box is present, AugMix draws only from its four photometric ops; the published set is kept for the classification-only case. Reasoning: AugMix mixes several independently-warped branches, so no single geometry exists to apply to the label, and image-only warping is label noise (measured up to 25 px), not a caveat. The four remaining ops still exclude contrast, colour, brightness, sharpness, noise and blur, which is the property D4 is cited for. Documented in `02_DESIGN.md` s8 and in the `augmix` docstring. |
| F9 | MAJOR | Fixed. `sigmoid(-2.19) = 0.1007`; the doc and comments now say `logit(0.1)`. |
| F10 | MAJOR | Fixed. `README.md` s8 no longer claims the brief "explicitly sanctions" SAM, and states plainly that `--backend sam` loads pretrained weights, that it is annotation tooling only, and that the default backend uses none. |
| F11 | MAJOR | Partially fixed. A repository placeholder and an instruction to fill it are now in `README.md` and in the report scaffold's introduction. **Creating the repository and pasting the URL is the user's action.** |
| F12 | MAJOR | Fixed. `tools/`, `configs/` and `tests/` moved inside `project_18006111_Shihab/`; the `sys.path` bootstrap in every tool and test updated; all four suites re-run and pass (10/10, 16/16, 17/17, all). Every `README.md` command now runs from the submitted tree. |
| F13 | MAJOR | Fixed. `tools/annotate_smartphone.py` writes `boxes.json` inside `smartphone_dataset/<studentno>_<surname>/`, alongside the masks it derives them from, with the coordinate convention stated in the file. The QC sidecar remains outside as working material. |
| F14 | MINOR | Fixed with the measured numbers: 13,455 total / 13,444 heatmap with plain Kaiming head init versus 14.7 / 9.3 shipped, and 788 -> 3.7 for the isolated focal term. |
| F15 | MINOR | Fixed. The s7 table now states that the up-leg draws from four kernels (no `INTER_AREA`) and that the two legs draw independently, with the reason for each. |
| F16, F17 | MINOR | Fixed: 20,550 frames, 30 contributors, 24 train / 6 val. |
| F18 | MINOR | Fixed: exact `wc -l` values, and the paths updated for the new layout. |
| F19 | MINOR | Fixed: seven plotting functions, five of which render from a single result JSON. |
| F20 | MINOR | Fixed: the `tests/test_pack.py` and `tests/test_evaluate.py` references now point at what exists (the pack regression fixture, and `tests/test_utils.py` for the sklearn cross-check). |
| F21 | MINOR | Fixed structurally. `load_config` resolves an `extends:` chain, and `e1.yaml`/`e3.yaml` are three-line overlays on `base.yaml`. `diff` between them is now one substantive line, by construction rather than by manual synchronisation. |
| F22 | MINOR | Fixed. The pseudo-target generator is constructed once per evaluation pass instead of once per batch, so the transform a frame receives no longer depends on `--batch-size`. |
| F23 | MINOR | Fixed. `00_PLAN.md` s2 now says torchvision is not used at all, and notes that the GPU venv contains an unused wheel left by the install command. |
| F24 | MINOR | Fixed. `01_DATA.md` s5.2 now says the packer de-duplicates on a `(path, size)` signature and attributes the pixel-content evidence to the one-off script that produced it. |
| F25 | MINOR | Fixed: twelve workers, matching `configs/base.yaml`. |
| F26 | MINOR | Fixed by adding runs: three seeds on `e1` and `e3`, two on `a0_none`. |

**Not fixed, and why:** F1 needs data only the user can produce; F11 needs a repository only the
user can create. Both are surfaced to the user rather than silently carried.
