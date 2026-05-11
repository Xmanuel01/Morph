#!/usr/bin/env bash
set -euo pipefail
PYTHON=${PYTHON:-python3.11}
TORCH_VERSION=${TORCH_VERSION:-2.2.0+cu121}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}

$PYTHON -m pip install --upgrade pip
$PYTHON -m pip install "torch==${TORCH_VERSION}" --index-url "$TORCH_INDEX_URL"

export LIBTORCH_USE_PYTORCH=1
export PYTHON_SYS_EXECUTABLE=$($PYTHON -c 'import sys; print(sys.executable)')
TORCH_LIB_DIR=$($PYTHON -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')
export LD_LIBRARY_PATH="$TORCH_LIB_DIR:${LD_LIBRARY_PATH:-}"
$PYTHON -c 'import json, torch; print(json.dumps({"torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "cuda_count": torch.cuda.device_count(), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))'

if [ "${RUN_CARGO_TORCH_TESTS:-0}" = "1" ]; then
  cargo test -p enkai_tensor --features torch --test cuda_llm_foundation -- --nocapture
fi
