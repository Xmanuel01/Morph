#!/usr/bin/env sh
set -eu

log_dir="${1:-artifacts/gpu}"
single="$log_dir/single_gpu.log"
multi="$log_dir/multi_gpu.log"
four="$log_dir/soak_4gpu.log"

require_pattern() {
    path="$1"
    pattern="$2"
    message="$3"
    if [ ! -f "$path" ]; then
        echo "Missing log file: $path" >&2
        exit 1
    fi
    if ! grep -Eq "$pattern" "$path"; then
        echo "$message ($path)" >&2
        exit 1
    fi
}

require_pattern "$single" "status:[[:space:]]*PASS" "single GPU soak did not report PASS"
require_pattern "$single" "nan_or_inf:[[:space:]]*False" "single GPU soak reported non-finite values"
require_pattern "$single" "checkpoint_verified:[[:space:]]*True" "single GPU soak did not verify checkpoints"

require_pattern "$multi" "PASS:[[:space:]]*2-GPU DP correctness validated" "2-GPU harness did not report PASS"
require_pattern "$four" "PASS:[[:space:]]*4-GPU soak completed" "4-GPU soak did not report PASS"

echo "GPU gate evidence verified."
