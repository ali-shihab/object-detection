#!/usr/bin/env bash
# Run a list of experiments back to back on one GPU, newline-separated on stdin as
#   <run-name>|<config>|<extra train args>
# One GPU, one job at a time: two 10M-param runs would fit in 16 GB but would contend for SMs
# and make every wall-clock number in the report incomparable.
exec 2>&1
set -uo pipefail
WS=/cs/student/project_msc/2025/rai/ashihab/comp0248_lsa
QUEUE="${1:?queue file}"

# Snapshot the queue onto node-local disk before iterating it.
# The queue file lives on NFS, and `while read < "$QUEUE"` holds an open file descriptor for the
# whole run -- hours. Any push_code.sh that replaces the file underneath yanks the inode away and
# the loop dies with "read error: Stale file handle", silently abandoning every remaining row.
# That happened once on uaru-l after e3_s1. Reading from a local copy makes a code push safe.
LOCAL=/var/tmp/cw1_$USER/_queue_$(basename "$QUEUE")
cp "$QUEUE" "$LOCAL" || exit 1
echo "queue snapshot: $LOCAL ($(grep -vc '^#' "$LOCAL") runs)"
QUEUE="$LOCAL"
while IFS='|' read -r name cfg extra; do
  [ -z "${name// }" ] && continue
  case "$name" in \#*) continue;; esac
  if [ -f "$WS/results/${name}_rs_test.json" ]; then
    echo "=== SKIP $name (already has results) ==="; continue
  fi
  echo "############ $name ($(date +%T)) ############"
  bash "$WS/code/scripts/remote_experiment.sh" "$name" "$cfg" $extra
done < "$QUEUE"
echo "QUEUE_DONE"
