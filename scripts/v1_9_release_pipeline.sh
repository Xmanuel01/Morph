#!/usr/bin/env sh
set -eu

# Backward-compatible wrapper for the version-neutral release pipeline.
VERIFY_GPU_EVIDENCE="${VERIFY_GPU_EVIDENCE:-0}" \
GPU_LOG_DIR="${GPU_LOG_DIR:-artifacts/gpu}" \
SKIP_PACKAGE_CHECK="${SKIP_PACKAGE_CHECK:-0}" \
sh scripts/release_pipeline.sh
