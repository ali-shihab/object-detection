#!/usr/bin/env bash
# Unpack and pack both RealSense releases on the GPU host.
# Raw archives and extraction scratch stay on node-local disk (178 GB free);
# only the packed output lands in the quota-limited workspace.
exec 2>&1
set -uo pipefail

WS=/cs/student/project_msc/2025/rai/ashihab/comp0248_lsa
SCRATCH=/var/tmp/cw1_$USER
PY="$SCRATCH/venv/cw1/bin/python"
RAW="$SCRATCH/raw"
export XDG_CACHE_HOME="$SCRATCH/cache"

echo "=== [1/4] extract RealSense test tar ==="
if [ ! -d "$RAW/Test data-COMP0248_Test_data_23" ]; then
  tar -xf "$RAW/realsense_test.tar" -C "$RAW" || { echo "TAR FAILED"; exit 1; }
fi
find "$RAW/Test data-COMP0248_Test_data_23" -type f | wc -l

echo
echo "=== [2/4] pack RealSense test ==="
"$PY" "$WS/code/project_18006111_Shihab/tools/pack_dataset.py" \
  --src "$RAW/Test data-COMP0248_Test_data_23" \
  --out "$WS/data/realsense_test" \
  --packed-size 512 384 --test --subject-name test23 --workers 12 \
  || { echo "PACK TEST FAILED"; exit 1; }

echo
echo "=== [3/4] pack RealSense train/val from rgb_only.7z ==="
"$PY" "$WS/code/project_18006111_Shihab/tools/pack_dataset.py" \
  --src "$RAW/rgb_only.7z" \
  --out "$WS/data/realsense_trainval" \
  --packed-size 512 384 --split-holdout 6 --seed 0 --workers 12 \
  --scratch "$SCRATCH/packsrc" \
  || { echo "PACK TRAIN FAILED"; exit 1; }

echo
echo "=== [4/4] footprint ==="
du -sh "$WS/data"/* 2>/dev/null
quota -s 2>/dev/null | sed -n '1,6p'
echo "PREPARE_DATA_DONE"
