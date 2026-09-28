#!/usr/bin/env bash
# Create a virtualenv and install clm-bench plus the CLM package (without its vLLM dependency).
#   ./setup.sh            # uses python3 on PATH
#   PYTHON=python3.12 ./setup.sh
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"

if command -v uv >/dev/null 2>&1; then
  uv venv .venv --python "$PYTHON"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  uv pip install -e .
  uv pip install --no-deps "contrastive-lm==0.1.0"
else
  "$PYTHON" -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install -U pip
  pip install -e .
  pip install --no-deps "contrastive-lm==0.1.0"
fi

python - <<'PY'
import torch, clm.heads, clm.schema, transformers
dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch {torch.__version__} | transformers {transformers.__version__} | device {dev}")
PY
echo "Done. Activate with: source .venv/bin/activate"
