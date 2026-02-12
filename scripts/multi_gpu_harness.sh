#!/usr/bin/env bash
set -euo pipefail

if [[ "${ENKAI_RUN_MULTI_GPU_TESTS:-0}" != "1" ]]; then
  echo "skipped: ENKAI_RUN_MULTI_GPU_TESTS not set"
  exit 0
fi

echo "multi-gpu harness is scaffolded but disabled until single-GPU gate is green"
echo "When enabled, this script will run world_size=2 with fixed seed and compare loss/grad paths."
exit 0
