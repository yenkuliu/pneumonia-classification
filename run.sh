#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONHASHSEED=820
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    echo "Python not found: $PYTHON. Run bash setup_cuda.sh first, or set PYTHON to an absolute interpreter path." >&2
    exit 1
fi
cd "$ROOT"
case "${1:-reproduce}" in
    reproduce) shift || true; exec "$PYTHON" -u reproduce.py "$@" ;;
    check) shift; exec "$PYTHON" -u reproduce.py --check-only "$@" ;;
    inference) shift; exec "$PYTHON" -u infer.py "$@" ;;
    plots) shift; exec "$PYTHON" -u plots.py "$@" ;;
    *) exec "$PYTHON" -u reproduce.py "$@" ;;
esac
