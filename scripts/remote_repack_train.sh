#!/usr/bin/env bash
# Re-pack the RealSense train/val release after two packer fixes:
#   * descend wrapper directories  -> recovers 25047621_Wu (packaged as dataset/25047621_Wu/...),
#     750 frames that the first pack silently dropped;
#   * drop byte-identical duplicate contributors -> removes "25150455_Guan 2", which is the same
#     850 files as 25150455_Guan and was double-weighting that contributor.
# The test pack is unaffected (single flat contributor) and is not rebuilt.
exec 2>&1
set -uo pipefail
WS=/cs/student/project_msc/2025/rai/ashihab/comp0248_lsa
SCRATCH=/var/tmp/cw1_$USER
PY="$SCRATCH/venv/cw1/bin/python"
export XDG_CACHE_HOME="$SCRATCH/cache"

rm -rf "$WS/data/realsense_trainval_v2"
"$PY" "$WS/code/project_18006111_Shihab/tools/pack_dataset.py" \
  --src "$SCRATCH/raw/rgb_only.7z" \
  --out "$WS/data/realsense_trainval_v2" \
  --packed-size 512 384 --split-holdout 6 --seed 0 --workers 12 \
  --scratch "$SCRATCH/packsrc" || { echo "PACK FAILED"; exit 1; }

echo "=== swap into place ==="
rm -rf "$WS/data/realsense_trainval_old"
[ -d "$WS/data/realsense_trainval" ] && mv "$WS/data/realsense_trainval" "$WS/data/realsense_trainval_old"
mv "$WS/data/realsense_trainval_v2" "$WS/data/realsense_trainval"
rm -rf "$WS/data/realsense_trainval_old"
du -sh "$WS/data"/*
quota -s 2>/dev/null | sed -n '4,5p'
echo "REPACK_DONE"
