# COMP0248 CW1 (LSA re-sit) — Master Plan

**Scope owner:** this document. Requirements, deliverables, experiment matrix, environment,
risks, decision log, document map. Results and analysis live in `04_RESULTS.md`; do not
duplicate numbers here.

**Source of truth:** `cw1/COMP0248_Coursework_1_LSA.pdf` (16 slides). Page references below
are slide numbers in that PDF. The non-LSA `COMP0248_Coursework_1.pdf` is the *original*
brief and is **superseded**; it is retained only because the LSA brief refers back to it for
the data-capture and annotation protocol (LSA p6-p7).

**Last updated:** 2026-08-26 (Cycle 2 — constraint rulings corrected against the code)

---

## 1. What the LSA coursework actually asks for

| # | Requirement | Source | Status |
|---|---|---|---|
| R1 | Multi-task deep model: **RGB-only** input -> hand bbox + binary hand mask + gesture class (10) | p3 (Obj 1), p5 | Design |
| R2 | Model must also output a **gesture confidence** (probability/score) | p5 | Design |
| R3 | Record + annotate a **smartphone RGB** dataset: 10 gestures x 2 clips x 3 frames = **60 images**, each with class label, binary mask, mask-derived bbox | p3 (Obj 2), p6 | **BLOCKED — see R3 note** |
| R4 | Smartphone dataset follows original CW1 folder structure, **minus** the depth folders | p7 | Pending R3 |
| R5 | **Experiment 1**: train on RealSense train, evaluate on RealSense test | p8 | Pending data |
| R6 | **Experiment 2**: zero-shot smartphone evaluation — no smartphone training data, no fine-tuning, **same preprocessing and same weights** as E1 | p8 | Pending R3 |
| R7 | **Experiment 3**: one method to improve smartphone performance **without any smartphone data in training**; report gain vs E2 | p4 (Obj 4), p8, p9 | Design |
| R8 | Metrics on **val and test** for **all** experiments: det acc@0.5 IoU, mean bbox IoU; seg mean IoU (hand vs bg), Dice; cls top-1, macro-F1, confusion matrix; qualitative overlays | p9 | Design |
| R9 | Own PyTorch code: custom `torch.nn.Module` architecture, own Dataset/DataLoader, own training loop, own loss composition, own evaluation scripts | p10 | Design |
| R10 | **No** high-level detection/segmentation frameworks (Ultralytics YOLO, Detectron2, MMDetection, segmentation_models_pytorch, pre-built Mask/Faster R-CNN fine-tunes, `maskrcnn_resnet50_fpn`) | p11 | Constraint |
| R11 | Deliverable tree `project_<studentno>_<surname>/` with `smartphone_dataset/`, `src/{dataloader,model,train,evaluate,visualise,utils}.py`, `weights/`, `results/`, `requirements.txt`, `README.md` | p12 | Scaffolded |
| R12 | **Do not** ship the supplied RealSense train or test datasets in the zip | p12 | Constraint |
| R13 | GitHub link to the code, with README, cited in the report | p12 | Pending |
| R14 | Report `coursework1LSA_<studentname>.pdf`, 6 pages max **including references**, unaltered IEEE Conference template | p13, p15 | User-authored |
| R15 | Code must run on the CS GPU servers | p15 | Constraint (we train on Knuckles) |

**Marking (p16):** Report 35 / Code and Implementation 35 / Experiments and innovation 30.

### R3 note — open blocker
The LSA smartphone dataset is a **new** artefact required by this re-sit. It is *not* the
RealSense contribution `dataset/18006111_Shihab/` that already exists on the Mac (720 MB, 10
gestures x 5 clips, with `depth/`, `depth_raw/`, `annotation/`) — that was the original CW1
Objective 1 and is part of the collated OneDrive release. LSA p2 describes the OneDrive link
as the *"Existing COMP0248 RealSense train and RGB-D dataset"*; LSA p2 separately defines
*"Test dataset 2: Smartphone RGB dataset collected and annotated by you"*. The two are
distinct. Resolution path is in `01_DATA.md` s5.

### Identity fields
- Student number: `18006111`; surname `Shihab`.
- Deliverable dir: `project_18006111_Shihab/`; report file `coursework1LSA_Shihab.pdf`.
- (`ucab274` is the user's UCL e-mail identifier, **not** the student number. The previous
  attempt's folder `project_ucab274_Shihab/` used the wrong identifier.)

---

## 2. Constraint interpretation (recorded so the report can defend it)

| Question | Ruling adopted | Rationale |
|---|---|---|
| Are ImageNet-pretrained backbones allowed? | **Primary submitted model is trained fully from scratch**; a pretrained-encoder variant is reported only as a clearly-labelled ablation. | p11 bans end-to-end det/seg frameworks and pre-built detectors, and says "if unsure, assume it is not". From-scratch keeps the submitted model unambiguously compliant while the ablation still answers the scientific question. |
| Is `torchvision.ops` (nms, box_iou, roi_align) allowed? | **Not used.** IoU / GIoU / box ops implemented in `src/utils.py`. | Removes any argument about borrowed detection machinery; costs ~40 lines. |
| Is `torchvision` allowed? | **Not used at all.** No module in `src/` or `tools/` imports it; `requirements.txt` does not list it; every augmentation, box and metric operation is written in this repository. | The ban in p11 is on end-to-end detection/segmentation frameworks, so `torchvision.transforms` would arguably be fine — but p11 also says "if you are unsure, assume it is not", and removing the dependency entirely costs about forty lines and removes the argument. (The GPU host's venv contains a `torchvision` wheel as a leftover of the install command; nothing imports it, and it is absent from `requirements.txt`.) |
| Is depth allowed as model input? | **No.** p3 says "RGB-only". Depth is used only for dataset analysis (e.g. reporting capture distance), never as a network input. | Explicit spec text; also required for cross-camera transfer to a phone. |
| Does E2 permit any re-tuning on phone images? | **No.** Same weights, same preprocessing, no BN re-estimation, no threshold search on the phone set. | p8 is explicit. |
| Does E3 permit test-time adaptation on unlabelled phone images? | **Avoided.** E3 is a *training-side* method only. | p8 says "without including any smartphone data in training"; TTA on the target set is arguably compliant but is a needless integrity risk. Recorded as future work instead. |

---

## 3. Experiment matrix

| ID | Model | Train data | Eval sets | Purpose |
|---|---|---|---|---|
| B0 | Classical baseline: skin-chroma segmentation + largest component -> mask -> bbox; crop -> small CNN classifier | RealSense train (classifier only) | RS-val, RS-test, Phone | Non-deep reference point required by "comparing also with a baseline" |
| B1 | Ours, single-task heads trained separately | RealSense train | RS-val, RS-test | Isolates the benefit of multi-task sharing |
| **E1** | Ours, multi-task, standard augmentation | RealSense train | RS-val, RS-test | **Mandatory** RealSense benchmark |
| **E2** | *E1 weights, unchanged* | — | Phone-test | **Mandatory** zero-shot cross-camera |
| **E3** | Ours, multi-task + camera-domain randomisation (CDR) | RealSense train **only** | RS-val, RS-test, Phone-test | **Mandatory** robustness improvement; gain reported vs E2 |
| A1..An | E3 ablations (each CDR component off; norm layer swap; mask-attended pooling off; loss-weighting scheme) | RealSense train | RS-val, Phone-test | Innovation marks; shows *which* part of E3 carries the gain |

Every row must report the full R8 metric set on every listed eval set. E3's headline number is
`E3(Phone) - E2(Phone)` per metric.

---

## 4. Environment and workflow

| Where | What runs there | Notes |
|---|---|---|
| MacBook (`~/workspace/.../cw1/lsa/`) | Authoring, docs, report, small CPU sanity tests, final artefact assembly | This is the deliverable's home; everything needed for submission is copied back here. |
| UCL Knuckles GPU host (`skate-l` and siblings) | Dataset staging, all training, all evaluation | Reached with `ssh-ucl-knuckles <host>-l`. Non-interactive command execution is wrapped by `lsa/scripts/knuckles_run.sh`. |

**Verified host facts (2026-08-25):** `skate-l` = NVIDIA RTX 4070 Ti SUPER, 16376 MiB, idle;
driver 590.44.01 / CUDA 13.1; Rocky Linux 9.8; system `python3` is 3.9.25; login shell is csh.

**Verified quotas (2026-08-25, `quota -s` as `ashihab`):**

| Filesystem | Used | Limit | Free |
|---|---|---|---|
| `evs2:/cs/student/msc` (= `$HOME`) | 9488 M | 10240 M | **~750 MB** |
| `evs2:/cs/student/project_msc/2025` (= workspace, `/cs/student/project_msc/2025/rai/ashihab`) | 90398 M | 100 G | **~12 GB** |

Consequence: **nothing large may touch `$HOME`.** venv, pip cache, dataset, checkpoints and
logs all live under the workspace path, with `XDG_CACHE_HOME`, `PIP_CACHE_HOME`, `TORCH_HOME`
and `HF_HOME` redirected there.

**12 GB is the hard budget for dataset + venv + checkpoints.** The mitigation is in
`01_DATA.md` s3: RGB and annotation files only (depth is never needed, R-ruling above), and a
resize-and-repack step that runs per gesture folder so raw downloads never all coexist.

---

## 5. Risk register

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| K1 | Smartphone dataset does not exist -> R3, R6, R7 all unsatisfiable | High | Fatal | Flagged to user in Cycle 1. Pipeline + recording spec built regardless so the turnaround after recording is minutes, not hours. |
| K2 | RealSense training release exceeds the 12 GB workspace budget | Medium | High | rclone `--include` filters exclude `depth/` and `depth_raw/`; per-gesture download -> repack -> delete loop; final packed set targeted at <2 GB. |
| K3 | Only 2 mask keyframes per training clip (original brief p6) -> few segmentation labels | Medium | Medium | Verify actual annotation density on download. The *test* release annotates all 15 frames/clip, so the train release may too. If sparse: classification loss on all frames, det/seg loss on annotated frames only (masked multi-task loss). |
| K4 | Near-duplicate frames within a clip inflate val scores | High | High | Split train/val **by contributing subject**, never by frame or clip. |
| K5 | Knuckles session killed for inactivity / host claimed by another user | Medium | Medium | Training runs under `nohup`/`setsid` writing to a log; the driver polls the log rather than holding a shell. Checkpoint every epoch so a lost host costs one epoch. |
| K6 | rclone OAuth blocked by UCL tenant admin consent | Medium | High | Fallback: browser download of the folder as zip, then `scp` Mac -> Knuckles. |
| K7 | From-scratch model underfits and E1 numbers are weak | Medium | Medium | Budget for a longer schedule, strong standard augmentation, and the multi-task auxiliary signal. Pretrained-encoder ablation quantifies the gap honestly. |
| K8 | Documentation drifts from code | Medium | Medium | Cycle 5 adversarial review re-derives every documented number from the artefacts on disk. |

---

## 6. Cycle workflow

Each cycle: **plan -> design -> implement -> document -> adversarial review -> report out.**
The adversarial review checks *content against reality* (does the code do what the doc says,
does the doc satisfy the LSA PDF), not prose quality. Findings and their resolutions are
appended to `05_REVIEW.md` with a cycle number.

## 7. Document map

| File | Owns |
|---|---|
| `00_PLAN.md` | Requirements, constraints, experiment matrix, environment, risks, decisions (this file) |
| `01_DATA.md` | Dataset provenance, download, statistics, splits, annotation protocol, smartphone set |
| `02_DESIGN.md` | Architecture, losses, augmentation, E3 method, training recipe, with rationale |
| `03_IMPLEMENTATION.md` | Code map, how to run, environment pinning, reproducibility, test coverage |
| `04_RESULTS.md` | Every metric, table, figure and analysis outcome; the report's evidence pack |
| `05_REVIEW.md` | Adversarial review findings per cycle and their resolutions |

## 8. Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-08-25 | Start fresh in `cw1/lsa/`, reusing nothing from `project_ucab274_Shihab/` | That tree is scaffolding only: `dataset/` holds a single `.gitkeep`, `results/` is empty, and it targets the *original* brief (RGB-D, no cross-camera objective) and the wrong student identifier. |
| 2026-08-25 | Train on `skate-l` | Verified idle 16 GB GPU; allow-listed by the user's connect script. |
| 2026-08-25 | RGB-only everywhere; depth excluded from download | LSA p3 mandates RGB-only input, and dropping depth resolves risk K2. |
| 2026-08-25 | Subject-wise train/val split | Frames within a clip are ~0.33 s apart at 3 FPS and are near-duplicates (K4). |
| 2026-08-26 | `tools/`, `configs/`, `tests/` moved **inside** `project_18006111_Shihab/` | The brief lists required directories, not exclusive ones. With them outside, every command in `README.md` failed when run from the tree that is actually submitted — a defect the Cycle 2 review found (F12). |
| 2026-08-26 | Second GPU host `uaru-l` added, queue split into two disjoint halves | 33 configurations at ~52 min each is ~28 h on one GPU. `uaru-l` was measured genuinely idle (141 MiB, 0 % util) where `tope-l`, `canada-l`, `nase-l`, `opah-l`, `pike-l`, `roach-l` and `rudd-l` were all in use by others. The halves must be disjoint: `remote_queue.sh` skips a run whose results exist, which does not prevent two hosts starting the same row simultaneously. |
| 2026-08-26 | Config files layer via an `extends:` key | `e1.yaml` and `e3.yaml` were full copies of `base.yaml`, so "they differ in exactly one line" was true only by manual synchronisation. They are now three-line overlays and the claim is structural (F21). |
