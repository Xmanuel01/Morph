#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$ROOT"
PYTHON=${PYTHON:-python3.11}
INSTALL_PYTORCH_CUDA=${INSTALL_PYTORCH_CUDA:-0}
TORCH_VERSION=${TORCH_VERSION:-2.2.0+cu121}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}
BUILD_FIRST_PARTY_CUDA_KERNELS=${BUILD_FIRST_PARTY_CUDA_KERNELS:-0}
RUN_ROCM_SOURCE_BUILD=${RUN_ROCM_SOURCE_BUILD:-0}
RUN_METAL_SOURCE_BUILD=${RUN_METAL_SOURCE_BUILD:-0}
RUN_DISTRIBUTED_GPU_PROOF=${RUN_DISTRIBUTED_GPU_PROOF:-0}
RUN_FOUR_GPU_SOAK=${RUN_FOUR_GPU_SOAK:-0}
SKIP_PREFLIGHT=${SKIP_PREFLIGHT:-0}

if [ "$INSTALL_PYTORCH_CUDA" = "1" ]; then
  TORCH_VERSION="$TORCH_VERSION" TORCH_INDEX_URL="$TORCH_INDEX_URL" PYTHON="$PYTHON" scripts/setup_pytorch_cuda.sh
fi

FEATURES="torch"
if [ "$BUILD_FIRST_PARTY_CUDA_KERNELS" = "1" ]; then FEATURES="$FEATURES,cuda-kernels"; fi
if [ "$RUN_ROCM_SOURCE_BUILD" = "1" ]; then FEATURES="$FEATURES,rocm-kernels"; fi
if [ "$RUN_METAL_SOURCE_BUILD" = "1" ]; then FEATURES="$FEATURES,metal-kernels"; fi

if [ "$SKIP_PREFLIGHT" != "1" ]; then
  PREFLIGHT_ARGS=(scripts/preflight_v3_9_0_gpu_test.py --workspace . --python "$PYTHON")
  if [ "$BUILD_FIRST_PARTY_CUDA_KERNELS" = "1" ]; then PREFLIGHT_ARGS+=(--require-nvcc); fi
  if [ "$RUN_DISTRIBUTED_GPU_PROOF" = "1" ]; then PREFLIGHT_ARGS+=(--require-two-gpus); fi
  if [ "$RUN_FOUR_GPU_SOAK" = "1" ]; then PREFLIGHT_ARGS+=(--require-four-gpus); fi
  $PYTHON "${PREFLIGHT_ARGS[@]}"
fi

cargo build -p enkai_tensor --features "$FEATURES"
cargo test -p enkai_tensor --features "$FEATURES" --test cuda_kernel_manifest
cargo test -p enkai_tensor --features "$FEATURES" --test cuda_llm_foundation -- --nocapture
cargo build -p enkai
$PYTHON scripts/readiness_v3_9_0_cuda_llm_runtime_foundation.py --workspace . --python "$PYTHON"
$PYTHON scripts/verify_v3_9_0_cuda_llm_runtime_foundation.py --workspace .

if [ "$RUN_DISTRIBUTED_GPU_PROOF" = "1" ]; then
  DIST_FEATURES="$FEATURES,dist"
  cargo build -p enkai_tensor --features "$DIST_FEATURES"
  export ENKAI_ENABLE_DIST=1
  export ENKAI_RUN_MULTI_GPU_TESTS=1
  export ENKAI_SINGLE_GPU_GREEN=1
  if [ "$RUN_FOUR_GPU_SOAK" = "1" ]; then
    $PYTHON scripts/readiness_v3_9_0_distributed_gpu_execution.py --workspace . --python "$PYTHON" --run --run-soak4
  else
    $PYTHON scripts/readiness_v3_9_0_distributed_gpu_execution.py --workspace . --python "$PYTHON" --run
  fi
  $PYTHON scripts/verify_v3_9_0_distributed_gpu_execution.py --workspace .
fi
