#!/usr/bin/env sh
set -eu

verify_gpu="${VERIFY_GPU_EVIDENCE:-0}"
gpu_log_dir="${GPU_LOG_DIR:-artifacts/gpu}"

echo "[v1.9] Running format gate..."
cargo fmt --all --check

echo "[v1.9] Running clippy gate..."
cargo clippy --workspace --all-targets -- -D warnings

echo "[v1.9] Running test gate..."
cargo test --workspace

echo "[v1.9] Running self-host corpus gate..."
cargo run -p enkai -- litec selfhost-ci enkai/tools/bootstrap/selfhost_corpus

if [ "$verify_gpu" = "1" ]; then
  echo "[v1.9] Verifying GPU gate evidence..."
  sh scripts/verify_gpu_gates.sh "$gpu_log_dir"
fi

echo "[v1.9] Release pipeline gates passed."
