#!/usr/bin/env sh
set -eu

# Backward-compatible wrapper for v2.0.0 RC pipeline.
ALLOW_MISSING_GPU_EVIDENCE="${ALLOW_MISSING_GPU_EVIDENCE:-0}" \
GPU_LOG_DIR="${GPU_LOG_DIR:-artifacts/gpu}" \
VERIFY_GPU_EVIDENCE="${VERIFY_GPU_EVIDENCE:-1}" \
sh scripts/rc_pipeline.sh
