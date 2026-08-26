#!/usr/bin/env bash
# Wait for the in-flight core queue (e1, e3) to finish, then run the full ablation queue.
# `remote_queue.sh` skips any run that already has a results JSON, so e1/e3 are not repeated.
# Chained rather than restarted so the currently-training e1 is not thrown away.
exec 2>&1
WS=/cs/student/project_msc/2025/rai/ashihab/comp0248_lsa
while systemctl --user is-active --quiet cw1-queue.service; do sleep 60; done
echo "core queue finished at $(date +%T); starting full queue"
bash "$WS/code/scripts/remote_queue.sh" "$WS/code/scripts/queue_full.txt"
