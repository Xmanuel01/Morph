#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
runner="$script_dir/gpu_harness.py"

if [[ ! -f "$runner" ]]; then
  echo "FAIL: missing harness runner $runner"
  exit 1
fi

if command -v python3 >/dev/null 2>&1; then
  exec python3 "$runner" soak4
fi

if command -v python >/dev/null 2>&1; then
  exec python "$runner" soak4
fi

echo "SKIPPED: python runtime not found (install python3)"
exit 0
