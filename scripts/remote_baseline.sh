#!/usr/bin/env bash
# B0 classical baseline (skin -> HOG -> softmax). CPU-only; runs alongside a GPU job.
exec 2>&1
set -uo pipefail
WS=/cs/student/project_msc/2025/rai/ashihab/comp0248_lsa
SCRATCH=/var/tmp/cw1_$USER
PY="$SCRATCH/venv/cw1/bin/python"
export XDG_CACHE_HOME="$SCRATCH/cache" PYTHONUNBUFFERED=1
cd "$WS/code/project_18006111_Shihab" || exit 1
"$PY" tools/baseline_classical.py \
  --index "$WS/data/realsense_trainval/index.json" \
  --test-index "$WS/data/realsense_test/index.json" \
  --out "$WS/results/b0_classical" --epochs 400 --max-train 8000 || { echo "B0 FAILED"; exit 1; }
echo "BASELINE_DONE"
