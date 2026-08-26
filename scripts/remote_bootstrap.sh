#!/usr/bin/env bash
# Idempotent environment bootstrap for COMP0248 CW1 LSA on a Knuckles GPU host.
# Storage policy (see docs/00_PLAN.md s4):
#   $HOME               ~750 MB free  -> NOTHING goes here
#   workspace (quota)   ~12 GB free   -> packed dataset, checkpoints, results  (persistent)
#   /var/tmp (node-local, 193 GB)     -> venv, caches, raw dataset             (ephemeral)
exec 2>&1
set -uo pipefail

WS=/cs/student/project_msc/2025/rai/ashihab/comp0248_lsa
SCRATCH=/var/tmp/cw1_$USER

mkdir -p "$WS"/{data,runs,code,artifacts}
mkdir -p "$SCRATCH"/{venv,cache,raw,logs}

export UV_CACHE_DIR="$SCRATCH/cache/uv"
export UV_PYTHON_INSTALL_DIR="$SCRATCH/cache/uv-python"
export XDG_CACHE_HOME="$SCRATCH/cache"
export PIP_CACHE_DIR="$SCRATCH/cache/pip"
export TORCH_HOME="$SCRATCH/cache/torch"
export HF_HOME="$SCRATCH/cache/hf"
UV="$HOME/.local/bin/uv"

echo "=== uv version ==="; "$UV" --version

VENV="$SCRATCH/venv/cw1"
if [ ! -x "$VENV/bin/python" ]; then
  echo "=== creating venv (python 3.11) ==="
  "$UV" venv --python 3.11 "$VENV" || { echo "VENV CREATE FAILED"; exit 1; }
fi
echo "=== python ==="; "$VENV/bin/python" -V

echo "=== installing packages ==="
VIRTUAL_ENV="$VENV" "$UV" pip install --python "$VENV/bin/python" \
  "torch==2.6.0" "torchvision==0.21.0" --index-url https://download.pytorch.org/whl/cu124 \
  || { echo "TORCH INSTALL FAILED"; exit 1; }

VIRTUAL_ENV="$VENV" "$UV" pip install --python "$VENV/bin/python" \
  numpy pillow opencv-python-headless scipy scikit-learn matplotlib pyyaml tqdm pandas \
  || { echo "DEPS INSTALL FAILED"; exit 1; }

echo "=== verifying torch/CUDA ==="
"$VENV/bin/python" - <<'PY'
import torch, torchvision, platform
print("python      :", platform.python_version())
print("torch       :", torch.__version__)
print("torchvision :", torchvision.__version__)
print("cuda avail  :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device      :", torch.cuda.get_device_name(0))
    print("capability  :", torch.cuda.get_device_capability(0))
    x = torch.randn(2048, 2048, device="cuda")
    y = (x @ x).sum().item()
    print("matmul ok   :", isinstance(y, float))
    print("mem total GB:", round(torch.cuda.get_device_properties(0).total_memory/2**30, 2))
PY

echo "=== disk usage ==="
du -sh "$VENV" "$SCRATCH/cache" 2>/dev/null
quota -s 2>/dev/null | sed -n '1,10p'
echo "BOOTSTRAP_DONE"
