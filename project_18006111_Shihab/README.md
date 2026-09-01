# COMP0248 Coursework 1 (LSA) — Cross-Camera Hand Gesture Detection, Segmentation and Classification

A single RGB-only network that, for one input image, predicts a hand bounding box, a binary hand  
mask, a gesture class from ten, and a gesture confidence. It is trained only on Intel RealSense  
D455 frames and evaluated unchanged on smartphone photographs.

**Code repository:** [https://github.com/ali-shihab/object-detection](https://github.com/ali-shihab/object-detection)

---

## 1. What is here

```
project_18006111_Shihab/
├── smartphone_dataset/
│   ├── 18006111_Shihab/          # the 60-frame smartphone test set (10 gestures x 2 clips x 3)
│   │   └── G01_call/clip02/{rgb,annotation}/frame_001.png ...
│   └── raw_videos/               # source clips - excluded
record
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
├── configs/                      # base.yaml + e1.yaml and e3.yaml (E2 trains nothing)
├── tests/                        # four suites, plain asserts, no pytest needed
├── weights/
│   ├── e1_best.pt                # Experiment 1's checkpoint (run `e1`, EMA weights)
│   └── e3_best.pt                # Experiment 3's checkpoint (run `e3`, EMA weights)
├── results/
│   ├── <run>_<split>.json        # every metric, per run and split
│   ├── <run>_<split>_predictions.csv  # per-image box IoU, mask IoU, class, confidence
│   ├── figures/                  # confusion matrices, reliability diagrams, overlays, curves
│   └── runs/<run>/               # each run's resolved config.json and per-epoch log.jsonl
├── requirements.txt
└── README.md
```

`weights/` holds two checkpoints because the brief's Experiment 2 reuses Experiment 1's: E2 is
`e1_best.pt` evaluated on the smartphone set, not a third model. The names are the experiment's, not a run's - `e3_best.pt` is the checkpoint of the run named `e3`. (`results/runs/` also lists a
separate run called `e3_best`; that is an ablation, and its checkpoint is not shipped.)

The report shows one confusion matrix, because it is capped at six pages. The full set the brief asks for - every mandatory experiment on val, RealSense test and the smartphone set - is in `results/figures/cm_*.pdf`, alongside a reliability diagram per run.

Every command below runs from this directory. Everything needed to reproduce the results is  
here except the two things that cannot be: the supplied RealSense archives, and the packed indexes derived  
from them (§5). The cluster orchestration scripts that drove the UCL GPU host are development  
infrastructure and are not required to run anything here.

## 2. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

The first §3 command reads the release as a `.7z` archive and shells out to the `7z` binary
(`brew install p7zip` / `apt install p7zip-full`); pass an already-extracted directory instead and
it is not needed. Nothing else here shells out.

Results in the report were produced with Python 3.11.14, torch 2.6.0+cu124, on one NVIDIA RTX 4070 Ti SUPER (16 GB, sm_89) under Rocky Linux 9.8 - one of two UCL CS Knuckles GPU hosts the queue was split across. The code runs on CPU unchanged (pass `--device cpu`), just slowly.

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
# Step 1 needs smartphone_dataset/raw_videos/, which is excluded from the zip for size --
# the 60 extracted frames and their masks are shipped, so steps 1-3 need only be re-run to
# rebuild the set from scratch.
# Follows the Week-4 tutorial: SAM pre-annotation -> Label Studio refinement -> decode.
# Step 1 - frame extraction from the recordings:
python tools/annotate_smartphone.py --videos smartphone_dataset/raw_videos \
    --out smartphone_dataset/18006111_Shihab
# Step 2 - SAM pre-annotation (annotation tooling only; see s8):
python tools/annotate_smartphone_sam.py --src smartphone_dataset/18006111_Shihab \
    --out _scratch/ls_work --sam-checkpoint sam_vit_b_01ec64.pth
# Step 3 - refine every mask by hand in Label Studio, then decode the JSON-MIN export.
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

`configs/e1.yaml` and `configs/e3.yaml` differ in one substantive field, `photometric` - plus `tag`, which is only the run's name and is read by nothing. So the E3-vs-E2 comparison is a
statement about the augmentation and nothing else. Any field can be
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

This exists so that no ablation has to be run by hand-editing a config file: every run writes its fully resolved configuration - base file, config file and command-line overrides collapsed into one record - to `runs/<name>/config.json`, so a run is reproducible from its own record
rather than from whatever the config file says today. All 39 of those records ship, under
`results/runs/`.

Training is resumable (`--resume auto`) and checkpoints every epoch - the GPU host can be reclaimed at any time. `--dry-run N` runs N steps plus one evaluation, as a smoke test.

## 5. Evaluation

Evaluation reads a **packed index**, which is built by `tools/pack_dataset.py` and is not in
this zip: the RealSense indexes cannot ship (the brief forbids redistributing that data), and the
smartphone one is derived rather than stored. Build the smartphone index from the data that *is*
here, in one command, before running the two smartphone evaluations:

```bash
python tools/pack_dataset.py --src smartphone_dataset/18006111_Shihab \
    --out data/phone_test --packed-size 512 384 --test --subject-name 18006111_Shihab
```

The RealSense index needs the released archives first (§3). The commands below then run against
the shipped checkpoints; training writes its own to `runs/<name>/best.pt`, so substitute that path
to evaluate a model you have just trained.

```bash
python -m src.evaluate --ckpt weights/e1_best.pt --index data/realsense_test/index.json \
    --split test --out results/e1_rs_test.json --examples 24 \
    --examples-out results/e1_examples.npz

# Experiment 2: the SAME checkpoint, unchanged, on the smartphone set.
# NOTE the filename: E2 is the E1 checkpoint, so its result file is e1_phone.json.
python -m src.evaluate --ckpt weights/e1_best.pt --index data/phone_test/index.json \
    --split test --out results/e1_phone.json

# Experiment 3 on the same smartphone set
python -m src.evaluate --ckpt weights/e3_best.pt --index data/phone_test/index.json \
    --split test --out results/e3_phone.json
```

Each result JSON carries every required metric per-frame **and** per-clip, bootstrap 95 %
confidence intervals over clips, a confusion matrix, calibration data, and a `config` block
recording the checkpoint, the index, the training hyper-parameters and a hash of the source
files that produced it.

## 6. Figures

The overlay dumps (`*_examples.npz`, ~13 MB each) are stripped from the zip for size. `--examples` is optional - without it every figure but the qualitative overlay is drawn; the §5 evaluation command regenerates the dump it needs.

```bash
python -m src.visualise --results results/*.json --curves results/runs/e1/log.jsonl \
    --aug-source <a-frame.jpg> --out results/figures                       # all but Fig. 2

python -m src.evaluate --ckpt weights/e3_best.pt --index data/phone_test/index.json \
    --split test --out results/e3_phone.json --examples 24 \
    --examples-out results/e3_phone_examples.npz                           # then Fig. 2
python -m src.visualise --results results/*.json --examples results/e3_phone_examples.npz \
    --out results/figures
```

The report's tables are generated from the same JSONs, so no number in it is transcribed:

```bash
python tools/make_report_tables.py --results results --out <report-dir>/generated
```

Tests:

```bash
for t in tests/test_*.py; do python "$t" || break; done
```

Four suites of plain asserts: metrics and box ops against hand-worked values (and, if
`scikit-learn` is installed, against it), model shapes and parameter count, the dataloader's
sparse-mask handling, and every augmentation's invariants.

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

