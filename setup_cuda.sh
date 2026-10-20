#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo "The reference environment requires Linux x86_64 and NVIDIA CUDA. No installation performed." >&2
    exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
    if [[ ! -x .bootstrap/bin/uv ]]; then
        "${BOOTSTRAP_PYTHON:-python3}" -m venv .bootstrap
        .bootstrap/bin/python -m pip install uv
    fi
    UV="$ROOT/.bootstrap/bin/uv"
else
    UV="$(command -v uv)"
fi
if [[ ! -e .venv ]]; then
    "$UV" python install 3.12.14
    "$UV" venv --seed --python 3.12.14 .venv
fi
.venv/bin/python -c 'import platform; assert platform.python_version() == "3.12.14", "Existing .venv must use Python 3.12.14; no environment was deleted."'
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements-lock.txt
.venv/bin/python -m pip check
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable: inspect nvidia-smi and driver compatibility"; x=torch.ones((4,4),device="cuda",requires_grad=True); (x@x).sum().backward(); torch.cuda.synchronize(); print("CUDA ready:",torch.cuda.get_device_name(0))'
echo "Ready. Run: bash run.sh reproduce --data-dir /absolute/path/to/chest_xray"
