# Dataset Reference — COMP0248 CW1 LSA

**Scope owner:** this document. Dataset provenance, layout, statistics, splits, annotation
protocol. Model/architecture decisions live in `02_DESIGN.md`; results live in `04_RESULTS.md`.

**Last updated:** 2026-08-26 (s5 added: training release verified)

## 1. Sources and provenance

| Item | Value |
|---|---|
| Set covered here | Official RealSense **test** set (`test1`) |
| Source archive | OneDrive `COMP0248_Test_data_23.zip`, 1,960,543,214 bytes (recorded provenance; the zip is not on disk, so this size was **not** re-verified here) |
| Extracted path | `Test data-COMP0248_Test_data_23/` |
| Extracted size | 3,724,328,145 bytes (`du -sb`, verified) |
| File count | 14,071 files: 10,350 `.png`, 3,450 `.npy`, 230 `.json`, 41 `.DS_Store` |
| RealSense **training** set | **Not yet downloaded.** Everything in §2 is test-set only. No training statistics exist yet. |

Per-clip layout verified as `G{01..10}_{gesture}/clip{NN}/{rgb,depth,depth_raw,annotation}/frame_XXX.{png,npy}`
plus `depth_metadata.json`. 41 macOS `.DS_Store` files are interleaved in the tree (class dirs and some
clip dirs) — loaders must filter them.

## 2. RealSense test set (test1) — verified statistics

### 2.1 Inventory (n = 230 clips / 3,450 frames — exhaustive, no sampling)

| Quantity | Value |
|---|---|
| Gesture classes | 10 (`G01_call, G02_dislike, G03_like, G04_ok, G05_one, G06_palm, G07_peace, G08_rock, G09_stop, G10_three`) |
| Clips per class | 23 (identical set in every class) |
| Clip IDs | `clip01`, `clip04` … `clip25` — **`clip02` and `clip03` are absent from all 10 classes** |
| Frames per clip | 15 in all 230 clips (min = med = max = 15) |
| Total frames | 3,450 |
| Modality dirs present | `rgb`, `depth`, `depth_raw`, `annotation` present in **230/230** clips |
| `depth_metadata.json` | present in **230/230** clips |
| Frame index agreement | `rgb` ↔ `annotation` ↔ `depth` ↔ `depth_raw` indices identical in **230/230** clips; ids contiguous `001..015` |
| Deviating clips | **none** — 0 missing dirs, 0 index mismatches, 0 stray files, 0 odd filenames |

### 2.2 Class balance (exhaustive)

Perfectly uniform: every class has **23 clips / 345 frames** (10 × 345 = 3,450). No class imbalance
to correct for.

### 2.3 RGB images (n = 3,450 — every frame)

| Property | Result |
|---|---|
| Size | 640 × 480 in **3,450/3,450** |
| PIL mode | `RGB` in 3,450/3,450 |
| dtype / channels | `uint8` / 3 in 3,450/3,450 |
| Exceptions | **none** — nothing deviates from 640×480 uint8 RGB |

### 2.4 Annotation masks (n = 3,450 — every mask)

| Property | Result |
|---|---|
| Size | 640 × 480 in 3,450/3,450 (matches RGB exactly; 0 size mismatches) |
| PIL mode / channels / dtype | `L` / 1 channel / `uint8` in 3,450/3,450 |
| Empty masks (all-zero) | **0** |

**Not strictly {0,255}.** All 256 values occur somewhere in the corpus, but the intermediate band is
negligible in mass — a thin anti-aliased boundary, not a soft/probabilistic label:

| Value band | Pixels | Share of all 1,059,840,000 px |
|---|---|---|
| 0 (background) | 1,029,229,999 | 97.1118 % |
| 1–254 (intermediate) | 68,124 | 0.0064 % |
| 255 (hand) | 30,541,877 | 2.8817 % |

Per-mask intermediate-pixel count (n = 3,450): **792 masks (22.96 %) contain ≥ 1** intermediate pixel;
min 0, median 0, p95 72, max 756, mean 19.7. Thresholding at ≥128 vs >0 changes the foreground area not
at all in 2,662/3,450 masks, and by 9.8 px on average overall (**0.11 % of foreground**).

**Hand-pixel fraction** (mask area ÷ 307,200 px), n = 3,450, threshold `>0`:

| min | p05 | median | p95 | max | mean |
|---|---|---|---|---|---|
| 0.0001 (26 px) | 0.0159 | **0.0275** (8,448 px) | 0.0473 | 0.0671 (20,602 px) | 0.0289 |

Per class (n = 345 each), median area fraction ranges 0.0232 (`G05_one`) → 0.0359 (`G06_palm`);
minima drop to 0.0001 (`G10_three`) — see §2.8.

### 2.5 Boxes derived from masks (n = 3,450 non-empty; tight bbox of non-zero region)

| Statistic | min | p05 | median | p95 | max |
|---|---|---|---|---|---|
| box width (px) | 5 | 75 | 115 | 177 | 272 |
| box height (px) | 7 | 104 | 151 | 197 | 256 |
| box width (norm) | 0.0078 | 0.1172 | 0.1797 | 0.2759 | 0.4250 |
| box height (norm) | 0.0146 | 0.2167 | 0.3146 | 0.4104 | 0.5333 |
| box area fraction | 0.0001 | 0.0289 | 0.0562 | 0.1045 | 0.2267 |
| aspect ratio w/h | 0.1786 | 0.4913 | 0.7600 | 1.2734 | 2.6500 |
| centre cx (norm) | 0.1320 | 0.2406 | 0.4477 | 0.7364 | 0.9961 |
| centre cy (norm) | 0.1240 | 0.2865 | 0.4885 | 0.7604 | 0.9792 |
| mask area ÷ box area | 0.0836 | 0.3838 | 0.4937 | 0.6550 | 0.8786 |

- **Empty masks: 0/3,450.**
- **Border-touching (possible truncation): 45/3,450 = 1.30 %.** By edge: right 36, bottom 8, top 3,
  left 0. By class: `G09_stop` 13, `G10_three` 13, `G06_palm` 12, then ≤2 each for `G03/G05/G07/G08`.
- Boxes are consistently taller than wide (median aspect 0.76) and the hand fills ~49 % of its box.

### 2.6 Connected components (8-connectivity, n = 3,450)

| Threshold | 1 comp | 2 | 3 | 4 | 6 | 9 | multi-component rate |
|---|---|---|---|---|---|---|---|
| `mask > 0` | 3,287 | 128 | 27 | 5 | 2 | 1 | **163 = 4.72 %** |
| `mask >= 128` | 3,384 | 56 | 4 | 4 | 2 | 0 | **66 = 1.91 %** |

Fragmentation is almost entirely anti-aliasing specks, not second hands:

- Components ≥ 10 px: **3,436 masks have exactly 1**; 13 have 2; 1 has 3.
- Components ≥ 0.5 % of image: **0 masks have more than one.**
- Largest-component share of mask area: mean **0.99976**, median 1.0000, p05 1.0000, min 0.7846.
  ≥ 0.99 in **99.83 %** of masks (3,444/3,450); ≥ 0.999 in 99.45 %.
- The six lowest are all in one clip family: `G06_palm/clip08` frames 006/013/014/015 (0.785–0.852)
  and `G10_three/clip08` frames 010/013.

### 2.7 Depth (documentation only — not used by the model)

`depth_raw/*.npy` (n = 300: 10 classes × 10 clips × 3 frames): dtype **`uint16`**, shape **(480, 640)**
in 300/300. Global min 0, global max 65535. Per-frame max: min 1,340 / median 12,505 / max 65,535.
Zero (invalid) pixel fraction: min 0.029 / median 0.153 / max 0.391. Per-frame p99 median 2,747
(≈ 2.75 m). `depth/*.png` is an 8-bit `L`-mode 640×480 visualisation, not metric.

`depth_metadata.json` is **byte-identical (same MD5) across all 230 clips** — it carries no per-clip
information. Verbatim:

```json
{
    "depth_scale": 0.0010000000474974513,
    "unit": "meters per depth unit",
    "description": "Multiply raw depth values by depth_scale to get depth in meters"
}
```

### 2.8 Data-quality anomalies found

- **`clip08` is a scale outlier.** Median mask area fraction by clip ID (n = 150 frames each) ranges
  0.0143 (`clip08`) → 0.0433 (`clip15`); `clip08` is the smallest by a wide margin. 22 of the 38
  sub-0.5 %-area masks are in `clip08` (`G09_stop` 11, `G10_three` 11), plus `G06_palm/clip08` 9.
  Median hand depth in `G09_stop/clip08` is 1.01 m with median mask 900 px, vs 0.78 m / 8,391 px in
  `G09_stop/clip14` — distance alone (1.29×) does not explain the ~9× area drop.
- **38 masks (1.10 %) have foreground < 0.5 % of the image** (< 1,536 px). Three are extreme:
  `G10_three/clip09/frame_011` = 26 px, `G08_rock/clip08/frame_011` = 123 px,
  `G08_rock/clip08/frame_009` = 147 px.
- **Temporal redundancy** (n = 1,120 consecutive pairs from 80 clips): mask IoU(t, t+1) median 0.649,
  p05 0.187, p95 0.959; IoU ≥ 0.9 in 16.2 % of pairs. Box-centre shift median 0.0175 normalised units.
  Frames within a clip are correlated but are *not* near-duplicates.

### 2.9 Subject / session structure — **no subject identifier exists**

Nothing in the directory names, filenames, or `depth_metadata.json` identifies a subject: paths encode
only class and clip, and the metadata file is identical in all 230 clips. **The 5 holdout test subjects
cannot be recovered from the data.** Subject-wise analysis on the test set is not possible.

Filesystem mtimes (preserved through the archive; the extracted dir is dated Apr 8 while frames retain
Feb 2026 stamps) give a *partial, unreliable* session signal:

| Group | Clip IDs | mtime window | Interpretation |
|---|---|---|---|
| A | 14, 15, 16 | 2026-02-24 17:22–17:37 | recording-consistent |
| B | 18, 19, 20, 17 | 2026-02-24 22:38–23:02 | recording-consistent |
| C | 21, 22, 23, 24, 25 | 2026-02-26 20:50–21:13 | recording-consistent |
| D | 01, 05, 06, 07, 08, 09, 10, 11, 12, 13 | 2026-02-27 14:41–14:43 | **bulk copy — not capture times** |
| E | 04 | 2026-02-27 20:07 (4 s span) | **bulk copy — not capture times** |

Groups A–C show G01→G10 visited in order, ~10 s per 15-frame clip and 2–4 min between clips, which is
what a capture session looks like. Groups D–E write 10 clips concurrently in ≤90 s in directory-traversal
order — a file copy. So at most **12 of 23 clips carry a usable session hint, and it yields 3 groups,
not 5**. Do not use mtimes as subject labels.

## 3. Modelling implications

- **Single-instance detection is justified.** 0 empty masks, 0 masks with more than one component ≥ 0.5 %
  of the image, and the largest component holds ≥ 99 % of mask area in 99.83 % of frames. Exactly one
  hand per image is a safe assumption; a one-box-per-image head loses essentially nothing.
- **Binarise masks at ≥128 before use.** They are not clean {0,255}: 22.96 % of masks carry anti-aliased
  edge pixels. Thresholding costs 0.11 % of foreground area but halves spurious component count
  (4.72 % → 1.91 %) and removes 1-pixel specks that would otherwise pollute component-based logic.
- **No resizing needed for geometry consistency** — every RGB and mask is already 640×480 uint8 and
  pixel-aligned, so any resize is a modelling choice, not a repair. Masks are single-channel; do not
  assume 3-channel.
- **Box priors:** targets are small and portrait-shaped — median box 115×151 px (0.180 × 0.315
  normalised), median area 5.6 % of the image, median aspect 0.76, and 95 % of boxes are under 0.105
  area fraction. Anchors/priors tuned for large objects will mismatch; the useful aspect range is
  roughly 0.49–1.27.
- **Centre prior is weak but real** — cx/cy medians ≈ 0.45/0.49 with p05–p95 spanning 0.24–0.74 and
  0.29–0.76. Hands are broadly central; aggressive centre cropping would still clip the tails.
- **Hazards:** (a) `clip08` hands are ~3× smaller in area than typical, so per-clip error should be
  reported or the small-object regime will hide in the average; (b) 45 frames (1.30 %) touch the border
  (36 on the right edge) and are truncated — box regression targets there are lower bounds;
  (c) 38 masks under 1,536 px, three under 150 px, are near-degenerate supervision/eval targets;
  (d) frames within a clip are correlated (median consecutive IoU 0.649), so 3,450 frames are far fewer
  than 3,450 independent samples — aggregate metrics per clip when estimating variance;
  (e) `.DS_Store` files will break naive `os.listdir` globbing.

## 5. RealSense training/validation release — verified statistics

**Provenance.** OneDrive `.../COMP0248_CW1_hand_dataset/Training and Validation data/`, which
holds three items: `RGB_annotations_only/rgb_only.7z` (6,835,475,344 bytes, md5
`58db18e08d0090fe4efd6608911b4e7a`, 8.92 GB uncompressed, 24,729 files / 4,994 folders),
`RGB_depth_annotations/` (31 items, RGB + depth) and `late submission/` (two archives,
`25184095_Huang.zip` 372,183,632 B and `dataset_Garza.7z` 333,976,473 B).

**Only `rgb_only.7z` was used.** The LSA model is RGB-only (LSA p3), so the depth variant adds
nothing a network can consume and roughly 4x the bytes. The two late submissions were not
ingested: they are RGB+depth in two different archive formats, and two extra contributors out of
31 does not justify a separate normalisation path. This is a deliberate exclusion, recorded here
so the report can state the contributor count honestly.

### 5.1 Structure and volume (verified by packing the whole archive)

| Quantity | Value |
|---|---|
| Contributor folders in archive | 31 |
| Contributors after de-duplication | **30** |
| Clips | 1,500 |
| Frames (RGB) | **20,550** |
| Frames carrying a hand mask | **2,899 (14.1 %)** |
| Frames per gesture | 2,055, exactly uniform across all 10 |
| Frames per clip | median 15, max 15, min 2 |
| Annotated frames per clip | median 2, max 2, **0 for 50 clips** |
| Unreadable files / empty masks | 0 / 0 |
| Packed footprint (512x384 JPEG q95 + PNG) | 1.9 GB |

**Annotation density is the single most consequential difference from test1.** The training
release follows the original brief's rule of 2 mask keyframes per clip, while the test release
annotates all 15 frames of every clip. Any loss or metric that assumes a mask per frame is wrong
on 86 % of the training set. `02_DESIGN.md` s4.1 is the mechanism that handles it.

### 5.2 Three defects found, and what was done about each

| # | Defect | Evidence | Action |
|---|---|---|---|
| D1 | `25047621_Wu` submitted as `25047621_Wu/dataset/25047621_Wu/G01_call/...` — one extra wrapper directory | The first pack produced 30 contributors from 31 archive folders; `7z l` shows 1,014 paths under that contributor, i.e. a complete submission | `tools/pack_dataset.py:discover_clips` now descends wrapper directories, and prints an explicit `WARNING: zero frames` for any contributor that packs nothing. Recovers ~750 frames. |
| D2 | `25150455_Guan` and `25150455_Guan 2` are the same student number and **byte-identical**: 850 files each | `scripts/remote_check_dupes.sh` output: content-hash `e6c6bc8e...` and pixel-hash `0a9b4c48...` equal for both; individual frames md5-identical | Contributors are de-duplicated before packing by a **signature over the sorted `(relative path, file size)` pairs** — not by name, so a future duplicate under any name is caught, and not by pixel content, which would cost a full read of every file. The pixel-content equality quoted in the Evidence column was established once, separately, by `scripts/remote_check_dupes.sh`; the packer's cheaper signature is sufficient given that a path-and-size collision across 850 files is not a plausible accident. Packing both would have double-weighted that contributor and, under a different split seed, leaked a train subject into validation. |
| D3 | Three contributors (`25042819_Kong`, `25058518_Wang`, `25067254_Yi`) submitted **only the 2 annotated keyframes per clip**: 100 frames each rather than 750 | Per-subject frame counts are exactly {100, 750}: 27 contributors at 750, 3 at 100 | Kept as-is. They are valid data — 100 % annotated — and dropping them would discard 300 of the 2,899 mask-annotated frames. Recorded because it makes the subject-level frame counts non-uniform, which matters for the split. |

### 5.3 Train / validation split

Contributor-wise, fixed in `index.json` at pack time so every experiment in the study reads the
identical split. Six of the 30 contributors are held out (`--split-holdout 6 --seed 0`).
The split is *never* by frame or clip: consecutive frames of a clip are 0.33 s apart at 3 FPS and
have a median mask IoU of 0.649 (s2), so a frame-wise split would leak near-duplicates into
validation and inflate every reported number.

### 5.4 Storage format

Packed to 512x384, RGB as JPEG quality 95 with 4:4:4 chroma, masks as PNG, boxes stored in packed
pixels, all in one `index.json`. Two consequences worth stating in the report:

* **JPEG is lossy.** It is applied uniformly to train, validation and test, so no comparison in
  this study is confounded by it; and the smartphone set will itself be JPEG-derived, so the
  train/test compression statistics are closer with this choice than without it. PNG throughout
  would have cost 8.9 GB against a ~12 GB workspace quota.
* **Metrics are computed in the 512x384 frame,** not at the native 640x480. IoU and Dice are
  ratios and are invariant to a uniform rescale up to rasterisation error at the mask boundary.

### 5.5 What the model is trained on, in one line

20,550 RGB frames from 30 contributors x 10 gestures x 5 clips, of which 2,899 carry a hand mask
and a mask-derived box; validated on 6 held-out contributors; tested on 3,450 fully-annotated
frames from 5 unidentified holdout subjects.

## 6. Open questions / not yet verified

- ~~Training set entirely unverified~~ — **resolved in s5**: downloaded, packed and measured.
- **Training-set mask convention was not separately audited** for anti-aliasing the way test1 was
  (s2); the same `>= 128` threshold is applied, which is safe in either direction, but the
  intermediate-value fraction on the training masks is unmeasured.
- **Zip size 1,960,543,214 bytes not re-verified** — the archive is absent from disk; only the extracted
  tree (3,724,328,145 bytes) was measured.
- **Why `clip02` and `clip03` are missing** from every class is unknown (withheld? corrupt? renumbered?).
- **Whether "_23" in the archive name denotes the 23 clips per class** is a conjecture, not established.
- **Subject identity is unrecoverable** (§2.9). Which clips belong to which of the 5 holdout subjects,
  and whether the 3 mtime-derived session groups correspond to subjects, is unknown.
- **Annotation protocol** (tool, annotator count, whether wrist/forearm is included, inter-annotator
  agreement) is undocumented and not inferable from pixels alone.
- **RGB–depth extrinsic alignment** was not tested; masks were only checked against RGB geometry, and
  depth was never used to validate mask boundaries.
- **Whether any frame is duplicated** across clips or classes was not checked (no hashing pass run).
- **RGB content quality** (blur, exposure, background variety, subject appearance) was not measured —
  only geometry and dtype.
