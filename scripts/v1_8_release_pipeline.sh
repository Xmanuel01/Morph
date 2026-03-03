#!/usr/bin/env sh
set -eu

echo "[v1.8] Running format gate..."
cargo fmt --all --check

echo "[v1.8] Running clippy gate..."
cargo clippy --workspace --all-targets -- -D warnings

echo "[v1.8] Running test gate..."
cargo test --workspace

echo "[v1.8] Running self-host corpus gate..."
cargo run -p enkai -- litec selfhost-ci enkai/tools/bootstrap/selfhost_corpus

echo "[v1.8] Release pipeline gates passed."
