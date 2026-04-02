#!/usr/bin/env sh
set -eu

# Backward-compatible wrapper for current release line.
exec sh scripts/rc_pipeline.sh "$@"
