#!/usr/bin/env bash
exec 2>&1
WS=/cs/student/project_msc/2025/rai/ashihab/comp0248_lsa
bash "$WS/code/scripts/remote_queue.sh" "$WS/code/scripts/queue_b.txt"
