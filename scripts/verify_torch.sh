#!/usr/bin/env bash
set -euo pipefail

# Verify enkai_tensor against a real libtorch install via Python wheels.
# CPU default: https://download.pytorch.org/whl/cpu
# For CUDA, set TORCH_INDEX_URL (e.g. https://download.pytorch.org/whl/cu121)
# Pin torch version to match tch crate expectations.

PYTHON=${PYTHON:-python3}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}
TORCH_VERSION=${TORCH_VERSION:-2.2.0+cpu}

$PYTHON -m pip install --upgrade pip
$PYTHON -m pip install "torch==${TORCH_VERSION}" --index-url "$TORCH_INDEX_URL"

export LIBTORCH_USE_PYTORCH=1
export PYTHON_SYS_EXECUTABLE=$PYTHON
TORCH_LIB_DIR=$($PYTHON -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
if [ ! -d "$TORCH_LIB_DIR" ]; then
  echo "Torch lib directory not found: $TORCH_LIB_DIR" >&2
  exit 1
fi
export LD_LIBRARY_PATH="$TORCH_LIB_DIR:${LD_LIBRARY_PATH:-}"

cargo test -p enkai_tensor --features torch

