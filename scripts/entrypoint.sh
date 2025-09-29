#!/usr/bin/env bash
set -euo pipefail
mkdir -p /app/data /app/artifacts /app/models
echo "Python: $(python -V)"
python - <<'PY'
import torch, sys
print("Torch:", torch.__version__ if hasattr(torch,"__version__") else "not installed")
PY
exec "$@"
