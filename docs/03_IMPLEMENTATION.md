# Implementation — code map, environment, reproducibility, test coverage

**Scope owner:** this document. What each module does, how the pieces compose, how to run
anything, what is tested and what is not. Design *rationale* lives in `02_DESIGN.md`; measured
outcomes in `04_RESULTS.md`.

**Last updated:** 2026-08-26 (Cycle 2 — post-review: layout, line counts and coverage corrected)

---

## 1. Layout

Everything needed to reproduce a result lives **inside** `project_18006111_Shihab/`, which is the
directory that is submitted: `src/`, `tools/`, `configs/`, `tests/`, `smartphone_dataset/`,
`weights/`, `results/`, `requirements.txt`, `README.md`. The cluster orchestration in `scripts/`
and the working documents in `docs/` sit outside it — they are how the work was driven, not part
of what the brief asks for.

| Path | Lines | Role |
|---|---|---|
| `project_18006111_Shihab/src/utils.py` | 479 | `GESTURES`, normalisation constants, seeding, `mask_to_box`, `box_iou`, `box_giou_loss`, `gaussian_radius`, `draw_gaussian`, `seg_scores`, `MetricAccumulator`, `AverageMeter`, JSON IO |
| `project_18006111_Shihab/src/model.py` | 475 | `HandNet` (encoder, U-Net decoder, centre-point head, mask-attended classifier), `decode_detection` |
| `project_18006111_Shihab/src/augment.py` | 786 | geometric block, `jitter`, **CPR** (11 toggleable stages), `randconv`, `aprs`, `augmix`, `pseudo_target_transform`, `AugmentPolicy` |
| `project_18006111_Shihab/src/dataloader.py` | 272 | `HandGestureDataset`, `build_loaders`, worker seeding |
| `project_18006111_Shihab/src/train.py` | 555 | `Config`, focal/Dice/GIoU loss composition, sparse-mask rule, EMA, cosine schedule, checkpointing, per-epoch JSONL |
| `project_18006111_Shihab/src/evaluate.py` | 370 | `evaluate_model`, ECE, clip-level bootstrap CIs, `compare_runs`, CLI |
| `project_18006111_Shihab/src/visualise.py` | 440 | confusion matrix, qualitative overlays, reliability, training curves, ablation bars, augmentation grid, domain-gap bars |
| `project_18006111_Shihab/tools/pack_dataset.py` | 401 | raw release (7z or tree) -> packed JPEG/PNG + `index.json`; contributor de-duplication; wrapper descent |
| `project_18006111_Shihab/tools/annotate_smartphone.py` | 464 | clips -> sharp frame sampling -> mask proposal (GrabCut or SAM) -> mask-derived boxes -> CW folder tree + QC contact sheets |

Dependency direction is strictly one way:
`utils <- model <- {train, evaluate}`, `utils <- {augment, dataloader} <- {train, evaluate}`,
`{utils, evaluate, augment} <- visualise`. `evaluate.py` imports nothing from `train.py`; the
reverse edge is a single deferred import inside `main()`.

## 2. Environment

Python 3.11.14, torch 2.6.0+cu124, on one NVIDIA RTX 4070 Ti SUPER (16 GB, sm_89), Rocky Linux
9.8, UCL CS Knuckles host `skate-l`. Verified end to end before any training: CUDA available,
device correctly identified, a 2048x2048 matmul executed, 15.55 GB reported.

**Storage policy** (`00_PLAN.md` s4): the venv (5.4 GB) and all caches live on node-local
`/var/tmp` (178 GB free, no quota); the packed dataset, checkpoints and results live in the
quota-limited workspace; **nothing** is written to `$HOME` (~750 MB free).

**Job control.** `systemd --user` transient units with `loginctl enable-linger`. This matters:
the first attempt used `setsid nohup`, and every job died silently the instant the SSH session
closed, producing a zero-byte log and no error. Linger keeps the user manager alive after logout,
so `systemd-run --user --unit=cw1-<job> --collect` survives disconnection and is inspectable with
`systemctl --user status`.

## 3. Running things

Everything is driven from the Mac; no interactive session is held open.

| Script | Does |
|---|---|
| `scripts/knuckles_run.sh <host> <script> [--bg <name>]` | runs a local bash script on the GPU host; `--bg` detaches it as a systemd unit; `--tail <name> <n>` reads its log |
| `scripts/knuckles_scp.sh up\|down <host> <src> <dst>` | file transfer over the same two-hop path |
| `scripts/push_code.sh <host>` | tars the working tree and unpacks it into the workspace |
| `scripts/upload_data.sh <host>` | ships the raw archives (streamed, no local temp copy) |
| `scripts/remote_bootstrap.sh` | creates the venv, installs torch, verifies CUDA |
| `scripts/remote_prepare_data.sh` | extracts and packs both RealSense releases |
| `scripts/remote_experiment.sh <name> <config> [args]` | trains one configuration and evaluates it on RealSense val, test and pseudo-target |
| `scripts/remote_queue.sh <queue-file>` | runs a list of experiments back to back, skipping any that already have results |

The remote command string is base64-encoded before transport. The login shell on Knuckles is
**csh**, which cannot parse `> file 2>&1` and reports `Ambiguous output redirect` — encoding the
payload keeps bash semantics intact across both hops.

## 4. Reproducibility

- `utils.set_seed` seeds Python, numpy and torch; `deterministic: true` additionally forces
  deterministic cuDNN kernels.
- DataLoader workers are seeded from `(worker_id, torch.initial_seed())`, which decorrelates
  workers *and* varies per epoch. The process id is used only as a cache key for rebuilding the
  generator after a fork, never as seed entropy — putting it in the seed would make every run
  irreproducible, which is the opposite of the bug it guards against.
- The train/val split is written into `index.json` at pack time, so it cannot drift between runs
  or be accidentally re-randomised by a different seed at train time.
- Every checkpoint stores the full config; every result JSON stores the checkpoint path, the index
  path, the training config and a SHA-256 prefix of all source files.
- Checkpoints are written to a temp file and `os.replace`d, so a host reclaimed mid-write cannot
  leave a torn checkpoint.

## 5. Test coverage

Four suites, plain `assert` with a `__main__` runner, no pytest dependency.

| Suite | Passing | What it actually pins down |
|---|---|---|
| `tests/test_utils.py` | 10/10 | IoU/GIoU against hand-computed values; `MetricAccumulator` classification metrics cross-checked against `sklearn`; per-clip vs per-frame aggregation algebra; exclusion counting; Gaussian radius/splat |
| `tests/test_model.py` | all | output shapes at two input sizes; all three norm variants (asserting IBN places `InstanceNorm2d` in stages 1-3 and none in stage 4); every `heads` subset; `use_mask_attn_pool=False` has strictly fewer parameters; finite non-zero gradients; **`decode_detection` against a hand-built heatmap with a known peak**; bf16 autocast |
| `tests/test_augment.py` | 17/17 | **box/mask synchronisation after every geometric op** (the check that catches a desynchronisation that would silently poison every detection metric); photometric ops leave mask and box bit-identical; tone-curve monotonicity over 1000 draws; every CPR stage individually skippable; the pseudo-target shift is provably outside CPR's reachable band |
| `tests/test_dataloader.py` | 16/16 | shapes, dtypes, normalisation; `has_mask` false exactly when `ann` is null; `aug=None` byte-determinism; **two workers produce different augmentations of the same index while a fixed seed reproduces a batch**; a regression test for the cv2 fork deadlock |

**Not covered by automated tests:** `evaluate.py` and `visualise.py` are exercised end to end on
synthetic data (all eight figure types render; the training loop runs train -> validate ->
checkpoint on a synthetic packed dataset and the losses fall), but they have no dedicated unit
suite. `tools/pack_dataset.py` has a regression check for the two defects of `01_DATA.md` s5.2 —
a nested wrapper directory is descended, a byte-identical duplicate contributor is dropped — run
as a scripted fixture rather than a committed test file. Both gaps are listed in `05_REVIEW.md`.

## 6. Defects found and fixed during implementation

| # | Defect | How it surfaced | Fix |
|---|---|---|---|
| I1 | Detection-head logits reached +/-27 at initialisation **in train mode** (BatchNorm whitening gain), putting the focal loss near 800 at step 0 instead of 3.7 | first smoke run: `train/heat` = 647 averaged over six steps, while an eval-mode probe gave 4.2 | final prediction convs initialised `normal(0, 1e-3)` with prior-matched biases (`-2.19` heat, `-3.56` seg) |
| I2 | **OpenCV's thread pool is not fork-safe**: once the parent process has made a threaded `cv2` call, the first `cv2` call in a forked DataLoader worker deadlocks with no traceback and no timeout | the augmentation test suite hung for ten minutes | `cv2.setNumThreads(1)` at import in `dataloader.py` (and in the annotation tool). `setNumThreads(0)` does **not** fix it — 0 means "auto". A regression test reproduces the exact sequence |
| I3 | The pseudo-target "held-out" shift was reachable by CPR (a 2000-draw search found a CPR tone curve within 4/255 of it), making target-free model selection circular | a test written specifically to falsify the held-out claim | pseudo-target curve strengthened outside CPR's band; CPR stage 8 narrowed from `U(0.6,1.5)` to `U(0.6,1.3)`; residual overlap documented rather than hidden |
| I4 | `cv2.grabCut` asserts on an empty background sample set, and a skin seed that matches the whole frame produces a full-frame "hand" | annotation smoke test on a synthetic clip | seed rejected unless its area is in [0.2 %, 35 %]; a border ring is unconditionally marked sure-background; a >50 % result falls back to rectangle-initialised GrabCut |
| I5 | AugMix legitimately returns a near-identity image on some draws, so a single-draw "did the augmentation change anything" assertion is flaky | intermittent test failure | assertion moved to the median of nine draws, which still catches a mode that is a no-op on *every* draw |
