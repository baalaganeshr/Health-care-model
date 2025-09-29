#!/usr/bin/env bash
set -euo pipefail

mkdir -p /app/artifacts /app/models

if [ "$#" -eq 0 ]; then
  set -- python run.py --help
fi

whoami || true
python --version
python - <<'PY'
import torch, numpy
print(f"Torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
print(f"NumPy {numpy.__version__}")
PY

exec "$@"
