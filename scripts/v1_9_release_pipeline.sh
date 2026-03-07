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

echo "[v1.9] Running docs contract consistency gate..."
python3 scripts/check_docs_consistency.py

echo "[v1.9] Running serve/frontend contract snapshot gate..."
cargo test -p enkai --bin enkai frontend::tests::contract_snapshots_match_reference_files

echo "[v1.9] Running self-host corpus gate..."
cargo run -p enkai -- litec selfhost-ci enkai/tools/bootstrap/selfhost_corpus

echo "[v1.9] Running self-host replacement fixed-point gate..."
cargo run -p enkai -- litec replace-check enkai/tools/bootstrap/selfhost_corpus --no-compare-stage0

if [ "$verify_gpu" = "1" ]; then
  echo "[v1.9] Verifying GPU gate evidence..."
  sh scripts/verify_gpu_gates.sh "$gpu_log_dir"
fi

echo "[v1.9] Release pipeline gates passed."
