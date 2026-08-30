# COMP0248 Coursework 1 (LSA) — Cross-Camera Hand Gesture Detection, Segmentation and Classification

A single RGB-only network that, for one input image, predicts a hand bounding box, a binary hand
mask, a gesture class from ten, and a gesture confidence. It is trained only on Intel RealSense
D455 frames and evaluated unchanged on smartphone photographs.

Student number `18006111` (Shihab).

**Code repository:** https://github.com/ali-shihab/object-detection
The repository is public; the same link appears on page 1 of the report, as the brief requires
(LSA p12).

---

## 1. What is here

```
project_18006111_Shihab/
├── smartphone_dataset/
│   ├── 18006111_Shihab/          # the 60-frame smartphone test set (10 gestures x 2 clips x 3)
│   │   └── G01_call/clip02/{rgb,annotation}/frame_001.png ...
│   ├── raw_videos/               # source clips -- EXCLUDED from the submitted zip (size)
│   └── RECORDING_GUIDE.md        # the capture protocol these clips were shot to
├── src/
│   ├── utils.py                  # box ops, metrics, MetricAccumulator, seeding, IO
│   ├── model.py                  # HandNet: encoder + U-Net decoder + centre-point head + classifier
│   ├── augment.py                # geometric ops, Camera-Pipeline Randomisation, competing randomisers
│   ├── dataloader.py             # packed-dataset Dataset and loaders
│   ├── train.py                  # training loop, loss composition, EMA, checkpointing
│   ├── evaluate.py               # all required metrics, bootstrap CIs, calibration
│   └── visualise.py              # every figure in the report
├── tools/
│   ├── pack_dataset.py           # raw release -> packed training format
│   ├── annotate_smartphone.py    # clips -> frames, mask-derived boxes, QC sheets
│   ├── annotate_smartphone_sam.py # SAM pre-annotation of the smartphone masks (see s8)
│   ├── baseline_classical.py     # the non-deep baseline (skin + HOG + softmax regression)
│   └── make_report_tables.py     # result JSONs -> the report's LaTeX tables
├── configs/                      # base.yaml + one file per mandatory experiment
├── tests/                        # four suites, plain asserts, no pytest needed
├── weights/                      # e1_best.pt, e3_best.pt (EMA weights only; see s8)
├── results/                      # metric JSONs, comparison CSVs, figures
├── requirements.txt
└── README.md
```

Everything needed to reproduce the results is inside this directory: every command below runs
from this directory, in the tree that is handed in. The cluster orchestration scripts used to
drive the UCL GPU host are development infrastructure and are documented separately; they are
not required to run any of the above.

## 2. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Results in the report were produced with Python 3.11.14, torch 2.6.0+cu124, on one NVIDIA
RTX 4070 Ti SUPER (16 GB, sm_89) under Rocky Linux 9.8 — one of two UCL CS Knuckles GPU hosts the
queue was split across. The code
runs on CPU unchanged (pass `--device cpu`), just slowly.

## 3. Data preparation

The supplied RealSense datasets are **not** included in this submission, as the brief requires.
To reproduce from the released archives:

```bash
# RealSense train/val (RGB + masks only; the model never sees depth)
python tools/pack_dataset.py --src rgb_only.7z --out data/realsense_trainval \
    --packed-size 512 384 --split-holdout 6 --seed 0

# RealSense test
python tools/pack_dataset.py --src "Test data-COMP0248_Test_data_23" \
    --out data/realsense_test --packed-size 512 384 --test --subject-name test23

# smartphone test set (already packed in this submission; this is how it was made).
# Follows the Week-4 tutorial: SAM pre-annotation -> Label Studio refinement -> decode.
# Step 1 -- frame extraction from the recordings:
python tools/annotate_smartphone.py --videos smartphone_dataset/raw_videos \
    --out smartphone_dataset/18006111_Shihab
# Step 2 -- SAM pre-annotation (annotation tooling only; see s8):
python tools/annotate_smartphone_sam.py --src smartphone_dataset/18006111_Shihab \
    --out _scratch/ls_work --sam-checkpoint sam_vit_b_01ec64.pth
# Step 3 -- refine every mask by hand in Label Studio, then decode the JSON-MIN export.
#   The tutorial's convert_annotations_for_LS.py / process_LS_output.py do the two conversions.
#   label-studio and label-studio-converter cannot coexist in one environment: install
#   label-studio==1.13.1 in its own venv and run the converter elsewhere.
python tools/pack_dataset.py --src smartphone_dataset/18006111_Shihab \
    --out data/phone_test --packed-size 512 384 --test --subject-name 18006111_Shihab
```

Packing writes `rgb/*.jpg`, `ann/*.png` and one `index.json` holding the record list, the
mask-derived boxes and the **contributor-wise train/val split**, which is fixed at pack time so
every experiment reads an identical split.

## 4. Training

```bash
python -m src.train --config configs/e1.yaml --out runs/e1     # Experiment 1 baseline
python -m src.train --config configs/e3.yaml --out runs/e3     # Experiment 3 (+CPR)
```

`configs/e1.yaml` and `configs/e3.yaml` differ in exactly one line (`photometric`), so the
E3-vs-E2 comparison is a statement about the augmentation and nothing else. Any field can be
overridden on the command line, which is how the ablations are run:

```bash
python -m src.train --config configs/e3.yaml --out runs/b2_no_noise --cpr-stages wb,exposure,ccm,gamma,tone,satihue,sharpblur,resample,jpeg,chroma
python -m src.train --config configs/e1.yaml --out runs/gn --norm gn
python -m src.train --config configs/e1.yaml --out runs/no_map --no-mask-attn-pool
python -m src.train --config configs/e1.yaml --out runs/cls_only --heads cls
```

Frequently-used fields have their own flags; **every** field is reachable with `--set`, which
parses its value as a YAML scalar:

```bash
python -m src.train --config configs/e3.yaml --out runs/no_giou --set w_giou=0.0
python -m src.train --config configs/e1.yaml --out runs/warm5 --set warmup_epochs=5 --set min_lr_factor=0.05
```

This exists so that no ablation ever has to be run by hand-editing a config file: the command
line is recorded in `runs/<name>/config.json`, an edited config is not.

Training is resumable (`--resume auto`) and checkpoints every epoch — the GPU host can be
reclaimed at any time. `--dry-run N` runs N steps plus one evaluation, as a smoke test.

## 5. Evaluation

```bash
python -m src.evaluate --ckpt runs/e1/best.pt --index data/realsense_test/index.json \
    --split test --out results/e1_rs_test.json --examples 24 \
    --examples-out results/e1_examples.npz

# Experiment 2: the SAME checkpoint, unchanged, on the smartphone set.
# NOTE the filename: E2 is the E1 checkpoint, so its result file is e1_phone.json.
python -m src.evaluate --ckpt runs/e1/best.pt --index data/phone_test/index.json \
    --split test --out results/e1_phone.json

# Experiment 3 on the same smartphone set
python -m src.evaluate --ckpt runs/e3/best.pt --index data/phone_test/index.json \
    --split test --out results/e3_phone.json
```

Each result JSON carries every required metric per-frame **and** per-clip, bootstrap 95 %
confidence intervals over clips, a confusion matrix, calibration data, and a `config` block
recording the checkpoint, the index, the training hyper-parameters and a hash of the source
files that produced it.

## 6. Figures

```bash
python -m src.visualise --results results/*.json --curves runs/e1/log.jsonl \
    --examples results/e1_examples.npz --aug-source <a-frame.jpg> --out results/figures
```

## 7. Reproducibility notes

- `--seed` fixes Python, numpy and torch RNGs; `deterministic: true` additionally forces
  deterministic cuDNN kernels at some cost in speed.
- DataLoader workers are seeded per worker *and* per epoch, so augmentation is reproducible
  without every worker drawing the identical stream.
- The train/val split lives in `index.json`, not in the training code, so it cannot drift
  between runs.
- Model selection never touches smartphone data: checkpoints are selected on held-out RealSense
  contributors, optionally scored through a synthetic held-out shift (`--select-on pseudo`).
  The smartphone set is read once, at the end.

## 8. Rules compliance

The submitted model is written from scratch with `torch.nn` primitives. There is no
`torchvision.models`, no `torchvision.ops`, no `timm`, no `segmentation_models_pytorch`, no
Detectron2, MMDetection or Ultralytics anywhere in `src/` or `tools/`, and no pretrained weights
are loaded — the network is trained from random initialisation. Box IoU, GIoU, peak suppression,
Gaussian target rendering and every augmentation operation are implemented in this repository.
One qualification, stated rather than buried. The smartphone ground-truth masks were built with
`tools/annotate_smartphone_sam.py`, which loads a pretrained **Segment Anything** checkpoint and
uses **MediaPipe**'s hand landmarker to place SAM's prompt. Both are **annotation tooling**: they
run once, offline, to produce ground truth. Neither is imported by `src/`, runs at training or
inference time, or contributes to any reported prediction — the restrictions the brief imposes are
on the *submitted model*, which loads no pretrained weights of any kind.

This is the pipeline the course prescribes. The Week-4 tutorial
(`Realsense_datacollection_n_annotation.pdf`, Part 3) instructs students to pre-annotate with SAM3
in Colab and refine in Label Studio, and the RealSense masks this project trains and evaluates
against were produced that way. Two substitutions are recorded: SAM 1 stands in for SAM3, whose
weights are gated behind a manual approval form; and because SAM 1 is promptable by points and
boxes rather than text, MediaPipe supplies the prompt SAM3 would have taken from the word "hand".
MediaPipe contributes no pixels to any mask. Every one of the 60 masks was then reviewed by hand in
Label Studio and 42 were corrected.
