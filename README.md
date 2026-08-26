# COMP0248 Coursework 1 (LSA) — Cross-Camera Hand Gesture Detection, Segmentation and Classification

An RGB-only multi-task network that predicts, for a single image, a hand bounding box, a binary
hand mask, a gesture class from ten, and a gesture confidence. Trained only on Intel RealSense
D455 frames; evaluated unchanged on smartphone photographs.

The submitted deliverable is [`project_18006111_Shihab/`](project_18006111_Shihab) — see its
[README](project_18006111_Shihab/README.md) for setup, training and evaluation commands.

| | |
|---|---|
| `project_18006111_Shihab/` | the submission: `src/`, `tools/`, `configs/`, `tests/`, `smartphone_dataset/`, `weights/`, `results/` |
| `docs/` | working documents: plan, dataset reference, design, implementation, results, adversarial review |
| `report/` | the 6-page IEEE technical report, its generated tables and figures |
| `scripts/` | orchestration for the UCL CS GPU hosts (not part of the submission) |

The supplied RealSense datasets are deliberately **not** in this repository, as the brief
requires; `project_18006111_Shihab/tools/pack_dataset.py` rebuilds the packed form from the
released archives.
