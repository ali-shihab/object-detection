#!/usr/bin/env bash
# Train one configuration and evaluate it on every RealSense split.
#
#   remote_experiment.sh <run-name> <config> [extra train args...]
#
# Evaluation on the smartphone set is deliberately NOT here: the brief forbids letting phone
# data influence anything, and the cleanest way to guarantee that is for the training driver to
# have no path to it at all. Phone evaluation is a separate, explicit step run once at the end.
exec 2>&1
set -uo pipefail

NAME="${1:?run name}"; CFG="${2:?config}"; shift 2 || true
WS=/cs/student/project_msc/2025/rai/ashihab/comp0248_lsa
SCRATCH=/var/tmp/cw1_$USER
PY="$SCRATCH/venv/cw1/bin/python"
CODE="$WS/code/project_18006111_Shihab"
DATA="$WS/data"
RUN="$WS/runs/$NAME"
RES="$WS/results"
export XDG_CACHE_HOME="$SCRATCH/cache" PYTHONUNBUFFERED=1
mkdir -p "$RUN" "$RES"

cd "$CODE" || exit 1
echo "=== [$NAME] train ($(date +%T)) ==="
"$PY" -m src.train --config "$CODE/configs/$CFG" --out "$RUN" --resume auto "$@" || {
  echo "TRAIN FAILED for $NAME"; exit 1; }

for SPLIT_INDEX in "val:$DATA/realsense_trainval/index.json" "test:$DATA/realsense_test/index.json"; do
  SPLIT="${SPLIT_INDEX%%:*}"; IDX="${SPLIT_INDEX#*:}"
  echo "=== [$NAME] eval $SPLIT ($(date +%T)) ==="
  "$PY" -m src.evaluate --ckpt "$RUN/best.pt" --index "$IDX" --split "$SPLIT" \
      --out "$RES/${NAME}_rs_${SPLIT}.json" --examples 24 \
      --examples-out "$RES/${NAME}_rs_${SPLIT}_examples.npz" || echo "EVAL $SPLIT FAILED"
done

# Held-out synthetic shift: the target-free proxy for cross-camera behaviour (02_DESIGN.md s7.1).
echo "=== [$NAME] eval pseudo-target ($(date +%T)) ==="
"$PY" -m src.evaluate --ckpt "$RUN/best.pt" --index "$DATA/realsense_test/index.json" \
    --split test --pseudo-target --out "$RES/${NAME}_pseudo.json" || echo "EVAL PSEUDO FAILED"

du -sh "$RUN"
echo "EXPERIMENT_DONE $NAME"
