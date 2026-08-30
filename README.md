# COMP0248 Coursework 1 (LSA) — Cross-Camera Hand Gesture Detection, Segmentation and Classification

An RGB-only multi-task network that predicts, for a single image, a hand bounding box, a binary
hand mask, a gesture class from ten, and a gesture confidence. Trained only on Intel RealSense
D455 frames; evaluated unchanged on smartphone photographs.

The submitted deliverable is [`project_18006111_Shihab/`](project_18006111_Shihab) — see its
[README](project_18006111_Shihab/README.md) for setup, training and evaluation commands.

| | |
|---|---|
| `project_18006111_Shihab/` | the submission: `src/`, `tools/`, `configs/`, `tests/`, `smartphone_dataset/`, `results/` |

This repository holds the coursework deliverable and nothing else. Two things a marker should
know are deliberately absent:

* **`weights/`** — the two trained checkpoints (`e1_best.pt`, `e3_best.pt`, 40 MB each) ship in
  the submission zip rather than in git. Every number in the report was produced from them, and
  `results/` carries the evaluation JSONs they produced.
* **The supplied RealSense datasets** — the brief forbids redistributing them;
  `project_18006111_Shihab/tools/pack_dataset.py` rebuilds the packed form from the released
  archives.

The supplied RealSense datasets are deliberately **not** in this repository, as the brief
requires; `project_18006111_Shihab/tools/pack_dataset.py` rebuilds the packed form from the
released archives.
