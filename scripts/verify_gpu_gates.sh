#!/usr/bin/env sh
set -eu

log_dir="${1:-artifacts/gpu}"
single_log="$log_dir/single_gpu.log"
multi_log="$log_dir/multi_gpu.log"
four_log="$log_dir/soak_4gpu.log"

single_json="$log_dir/single_gpu_evidence.json"
multi_json="$log_dir/multi_gpu_evidence.json"
four_json="$log_dir/soak_4gpu_evidence.json"

has_pattern() {
    path="$1"
    pattern="$2"
    if [ ! -f "$path" ]; then
        return 1
    fi
    grep -Eq "$pattern" "$path"
}

require_single() {
    if has_pattern "$single_log" "status:[[:space:]]*PASS" &&
        has_pattern "$single_log" "nan_or_inf:[[:space:]]*False" &&
        has_pattern "$single_log" "checkpoint_verified:[[:space:]]*True"; then
        return 0
    fi
    if [ ! -f "$single_json" ]; then
        echo "Missing single GPU evidence (need $single_log or $single_json)" >&2
        exit 1
    fi
    has_pattern "$single_json" "\"status\"[[:space:]]*:[[:space:]]*\"PASS\"" ||
        { echo "single GPU evidence status is not PASS ($single_json)" >&2; exit 1; }
    has_pattern "$single_json" "\"nan_or_inf\"[[:space:]]*:[[:space:]]*false" ||
        { echo "single GPU evidence reports non-finite values ($single_json)" >&2; exit 1; }
    has_pattern "$single_json" "\"checkpoint_verified\"[[:space:]]*:[[:space:]]*true" ||
        { echo "single GPU evidence reports unverified checkpoint ($single_json)" >&2; exit 1; }
}

require_multi() {
    if has_pattern "$multi_log" "PASS:[[:space:]]*2-GPU DP correctness validated"; then
        return 0
    fi
    if [ ! -f "$multi_json" ]; then
        echo "Missing multi GPU evidence (need $multi_log or $multi_json)" >&2
        exit 1
    fi
    has_pattern "$multi_json" "\"status\"[[:space:]]*:[[:space:]]*\"PASS\"" ||
        { echo "2-GPU evidence status is not PASS ($multi_json)" >&2; exit 1; }
    has_pattern "$multi_json" "\"loss_parity\"[[:space:]]*:[[:space:]]*true" ||
        { echo "2-GPU loss parity check failed ($multi_json)" >&2; exit 1; }
    has_pattern "$multi_json" "\"grad_parity\"[[:space:]]*:[[:space:]]*true" ||
        { echo "2-GPU grad parity check failed ($multi_json)" >&2; exit 1; }
}

require_soak4() {
    if has_pattern "$four_log" "PASS:[[:space:]]*4-GPU soak completed"; then
        return 0
    fi
    if [ ! -f "$four_json" ]; then
        echo "Missing 4-GPU evidence (need $four_log or $four_json)" >&2
        exit 1
    fi
    has_pattern "$four_json" "\"status\"[[:space:]]*:[[:space:]]*\"PASS\"" ||
        { echo "4-GPU evidence status is not PASS ($four_json)" >&2; exit 1; }
}

require_single
require_multi
require_soak4

echo "GPU gate evidence verified."
