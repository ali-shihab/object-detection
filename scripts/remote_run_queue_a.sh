#!/usr/bin/env bash
exec 2>&1
WS=/cs/student/project_msc/2025/rai/ashihab/comp0248_lsa
while systemctl --user is-active --quiet cw1-queue.service; do sleep 60; done
echo "core queue finished at $(date +%T); starting queue A"
bash "$WS/code/scripts/remote_queue.sh" "$WS/code/scripts/queue_a.txt"
