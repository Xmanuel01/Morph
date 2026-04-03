#!/usr/bin/env sh
set -eu

profile="${1:-full}"
config="${2:-configs/enkai_50m.enk}"
output="${3:-artifacts/gpu/preflight.json}"
artifact_dir="${4:-artifacts/gpu}"

if command -v python3 >/dev/null 2>&1; then
  exec python3 scripts/gpu_preflight.py --profile "$profile" --config "$config" --output "$output" --artifact-dir "$artifact_dir"
fi

if command -v python >/dev/null 2>&1; then
  exec python scripts/gpu_preflight.py --profile "$profile" --config "$config" --output "$output" --artifact-dir "$artifact_dir"
fi

echo "python3/python not found" >&2
exit 1
