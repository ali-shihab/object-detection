#!/usr/bin/env bash
# Evaluate already-trained checkpoints on the smartphone test set.
#
# Separated from the training driver on purpose. The brief (LSA p8) requires Experiment 2 to use
# the same weights and the same preprocessing with no fine-tuning, and requires Experiment 3 to
# be trained without any smartphone data. Keeping phone evaluation in its own script, run once
# after all training is finished, makes that guarantee structural rather than a promise:
# nothing in the training path can read this data because the training path never names it.
exec 2>&1
set -uo pipefail
WS=/cs/student/project_msc/2025/rai/ashihab/comp0248_lsa
SCRATCH=/var/tmp/cw1_$USER
PY="$SCRATCH/venv/cw1/bin/python"
PHONE="$WS/data/phone_test/index.json"
export XDG_CACHE_HOME="$SCRATCH/cache" PYTHONUNBUFFERED=1

[ -f "$PHONE" ] || { echo "no smartphone pack at $PHONE -- run annotate_smartphone.py then pack_dataset.py --test"; exit 1; }
cd "$WS/code/project_18006111_Shihab" || exit 1

for RUN in "$@"; do
  CK="$WS/runs/$RUN/best.pt"
  [ -f "$CK" ] || { echo "missing checkpoint $CK"; continue; }
  echo "=== phone eval: $RUN ($(date +%T)) ==="
  "$PY" -m src.evaluate --ckpt "$CK" --index "$PHONE" --split test \
      --out "$WS/results/${RUN}_phone.json" --examples 24 \
      --examples-out "$WS/results/${RUN}_phone_examples.npz" || echo "FAILED $RUN"
done
echo "PHONE_EVAL_DONE"
