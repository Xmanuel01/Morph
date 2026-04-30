#!/usr/bin/env sh
set -eu

# Backward-compatible wrapper for the v3.5.0 sign-off line.
exec sh scripts/rc_pipeline.sh "$@"
