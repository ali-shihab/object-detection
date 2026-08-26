#!/usr/bin/env bash
# Regenerate every figure and every LaTeX table from whatever results exist right now.
# Safe to run repeatedly and while the queue is still going: it reads only result JSONs and
# training logs, and overwrites its own outputs.
exec 2>&1
set -uo pipefail
WS=/cs/student/project_msc/2025/rai/ashihab/comp0248_lsa
SCRATCH=/var/tmp/cw1_$USER
PY="$SCRATCH/venv/cw1/bin/python"
CODE="$WS/code/project_18006111_Shihab"
export XDG_CACHE_HOME="$SCRATCH/cache" PYTHONUNBUFFERED=1 MPLCONFIGDIR="$SCRATCH/cache/mpl"
mkdir -p "$WS/artifacts/figures" "$WS/artifacts/generated"
cd "$CODE" || exit 1

echo "=== tables ==="
"$PY" tools/make_report_tables.py --results "$WS/results" --out "$WS/artifacts/generated"

echo "=== figures ==="
# One representative RealSense frame for the augmentation grid, chosen deterministically.
SRC=$("$PY" - <<'PY'
import json, os
ix=json.load(open("/cs/student/project_msc/2025/rai/ashihab/comp0248_lsa/data/realsense_test/index.json"))
r=[x for x in ix["records"] if x["ann"]][len(ix["records"])//3]
print(os.path.join(ix["root"], r["rgb"]))
PY
)
echo "aug source: $SRC"
ARGS=(--results "$WS"/results/*.json --out "$WS/artifacts/figures" --aug-source "$SRC")
[ -f "$WS/runs/e1/log.jsonl" ] && ARGS+=(--curves "$WS/runs/e1/log.jsonl")
for E in "$WS"/results/e3_phone_examples.npz "$WS"/results/e1_rs_test_examples.npz; do
  [ -f "$E" ] && { ARGS+=(--examples "$E"); break; }
done
"$PY" -m src.visualise "${ARGS[@]}"

echo "=== outputs ==="
ls -1 "$WS/artifacts/figures"/*.png 2>/dev/null | xargs -n1 basename | tr '\n' ' '
echo; ls -1 "$WS/artifacts/generated"/*.tex 2>/dev/null | xargs -n1 basename | tr '\n' ' '
echo; echo FIGURES_DONE
