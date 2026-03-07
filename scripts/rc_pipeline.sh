#!/usr/bin/env sh
set -eu

verify_gpu="${VERIFY_GPU_EVIDENCE:-1}"
gpu_log_dir="${GPU_LOG_DIR:-artifacts/gpu}"
allow_missing_gpu="${ALLOW_MISSING_GPU_EVIDENCE:-0}"

if [ "$verify_gpu" != "1" ] && [ "$allow_missing_gpu" != "1" ]; then
  echo "[rc] VERIFY_GPU_EVIDENCE must be 1 for RC gating (or set ALLOW_MISSING_GPU_EVIDENCE=1 for dry-run)." >&2
  exit 1
fi

if [ "$allow_missing_gpu" = "1" ]; then
  echo "[rc] Running RC dry-run without mandatory GPU evidence."
  VERIFY_GPU_EVIDENCE=0 sh scripts/release_pipeline.sh
  python3 scripts/collect_release_evidence.py --gpu-log-dir "$gpu_log_dir"
  echo "[rc] RC dry-run completed."
  exit 0
fi

echo "[rc] Running full RC pipeline (GPU evidence required)..."
VERIFY_GPU_EVIDENCE=1 GPU_LOG_DIR="$gpu_log_dir" sh scripts/release_pipeline.sh
python3 scripts/collect_release_evidence.py --gpu-log-dir "$gpu_log_dir" --require-gpu
echo "[rc] RC pipeline passed with archived evidence."
