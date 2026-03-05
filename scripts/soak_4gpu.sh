#!/usr/bin/env bash
set -euo pipefail

skip() {
  echo "SKIPPED: $1"
  exit 0
}

fail() {
  echo "FAIL: $1"
  exit 1
}

if [[ "${ENKAI_RUN_MULTI_GPU_TESTS:-0}" != "1" ]]; then
  skip "ENKAI_RUN_MULTI_GPU_TESTS not set to 1"
fi
if [[ "${ENKAI_SINGLE_GPU_GREEN:-0}" != "1" ]]; then
  skip "single-GPU gate not marked green (set ENKAI_SINGLE_GPU_GREEN=1 after soak pass)"
fi
if [[ "${ENKAI_ENABLE_DIST:-0}" != "1" ]]; then
  skip "ENKAI_ENABLE_DIST not set to 1"
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  skip "nvidia-smi not available"
fi

gpu_count="$(nvidia-smi -L | wc -l | tr -d '[:space:]')"
if [[ -z "$gpu_count" || "$gpu_count" -lt 4 ]]; then
  skip "fewer than 4 GPUs detected"
fi

launcher="${ENKAI_4GPU_LAUNCH_CMD:-}"
if [[ -z "$launcher" ]]; then
  skip "ENKAI_4GPU_LAUNCH_CMD not set"
fi

min_hours="${ENKAI_4GPU_MIN_HOURS:-3}"
nccl_timeout="${NCCL_TIMEOUT:-1800}"

echo "Starting 4-GPU soak harness (single-node)"
echo "Required minimum runtime (hours): $min_hours"
echo "NCCL timeout (sec): $nccl_timeout"
echo "NCCL guidance: set NCCL_ASYNC_ERROR_HANDLING=1 and NCCL_TIMEOUT >= 1800 for long runs."

start_epoch="$(date +%s)"
if ! eval "$launcher"; then
  fail "launcher exited non-zero"
fi
end_epoch="$(date +%s)"

hours="$(awk "BEGIN { printf \"%.2f\", ($end_epoch - $start_epoch) / 3600.0 }")"
if awk "BEGIN { exit !($hours < $min_hours) }"; then
  fail "runtime too short (${hours}h < ${min_hours}h)"
fi

echo "PASS: 4-GPU soak completed (${hours}h)"
