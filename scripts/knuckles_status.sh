#!/usr/bin/env bash
# Uploaded and run on the remote host to report job status.
D=/var/tmp/cw1_$USER
echo "=== units ==="
systemctl --user list-units 'cw1-*' --all --no-pager 2>/dev/null | head -20
echo "=== logs dir ==="
ls -la "$D/logs" 2>/dev/null
echo "=== gpu ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
