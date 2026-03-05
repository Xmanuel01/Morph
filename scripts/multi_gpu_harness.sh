#!/usr/bin/env bash
set -euo pipefail

skip() {
  echo "SKIPPED: $1"
  exit 0
}

fail() {
  echo "FAIL: $1"
  exit 1
}

if [[ "${ENKAI_RUN_MULTI_GPU_TESTS:-0}" != "1" ]]; then
  skip "ENKAI_RUN_MULTI_GPU_TESTS not set to 1"
fi

if [[ "${ENKAI_ENABLE_DIST:-0}" != "1" ]]; then
  skip "ENKAI_ENABLE_DIST not set to 1"
fi

if [[ "${ENKAI_SINGLE_GPU_GREEN:-0}" != "1" ]]; then
  skip "single-GPU gate not marked green (set ENKAI_SINGLE_GPU_GREEN=1 after soak pass)"
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  skip "nvidia-smi not available"
fi

gpu_count="$(nvidia-smi -L | wc -l | tr -d '[:space:]')"
if [[ -z "$gpu_count" || "$gpu_count" -lt 2 ]]; then
  skip "fewer than 2 GPUs detected"
fi

if ! command -v python3 >/dev/null 2>&1; then
  skip "python3 is required for JSON validation"
fi

launcher="${ENKAI_DP_LAUNCH_CMD:-}"
if [[ -z "$launcher" ]]; then
  skip "ENKAI_DP_LAUNCH_CMD not set; provide launcher that emits rank artifacts"
fi

work_dir="${ENKAI_DP_WORKDIR:-tmp/dp_harness}"
tol_loss="${ENKAI_DP_LOSS_TOL:-0.05}"
tol_grad="${ENKAI_DP_GRAD_TOL:-0.0001}"

mkdir -p "$work_dir"
dataset="$work_dir/deterministic.txt"
baseline_log="$work_dir/baseline.jsonl"
rank0_log="$work_dir/rank0.jsonl"
rank1_log="$work_dir/rank1.jsonl"
rank0_grad="$work_dir/rank0_grads.json"
rank1_grad="$work_dir/rank1_grads.json"

# Deterministic source batch set.
seq 1 200 | while read -r idx; do
  echo "sample $idx alpha beta gamma"
done >"$dataset"

if [[ -z "${ENKAI_BASELINE_CMD:-}" ]]; then
  skip "ENKAI_BASELINE_CMD not set"
fi

echo "Running baseline 1-GPU reference..."
if ! eval "${ENKAI_BASELINE_CMD}"; then
  fail "baseline command failed"
fi
if [[ ! -f "$baseline_log" ]]; then
  skip "baseline log not produced: $baseline_log"
fi

echo "Running 2-GPU launcher..."
if ! eval "$launcher"; then
  fail "launcher command failed"
fi

for artifact in "$rank0_log" "$rank1_log" "$rank0_grad" "$rank1_grad"; do
  if [[ ! -f "$artifact" ]]; then
    fail "missing artifact $artifact"
  fi
done

last_loss() {
  local path="$1"
  python3 - "$path" <<'PY'
import json
import sys
path = sys.argv[1]
last = None
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            last = line
if last is None:
    raise SystemExit("empty log")
obj = json.loads(last)
print(obj["loss"])
PY
}

base_loss="$(last_loss "$baseline_log")"
r0_loss="$(last_loss "$rank0_log")"
r1_loss="$(last_loss "$rank1_log")"

if ! python3 - "$base_loss" "$r0_loss" "$r1_loss" "$tol_loss" <<'PY'
import math
import sys
base, r0, r1, tol = map(float, sys.argv[1:5])
if math.fabs(r0 - base) > tol or math.fabs(r1 - base) > tol:
    raise SystemExit(
        f"loss mismatch vs baseline (base={base} r0={r0} r1={r1} tol={tol})"
    )
PY
then
  fail "loss mismatch vs baseline (base=$base_loss r0=$r0_loss r1=$r1_loss tol=$tol_loss)"
fi

if ! python3 - "$rank0_grad" "$rank1_grad" "$tol_grad" <<'PY'
import json
import math
import sys
g0_path, g1_path, tol = sys.argv[1], sys.argv[2], float(sys.argv[3])
with open(g0_path, "r", encoding="utf-8") as f:
    g0 = json.load(f)
with open(g1_path, "r", encoding="utf-8") as f:
    g1 = json.load(f)
if len(g0) != len(g1):
    raise SystemExit(f"gradient length mismatch ({len(g0)} vs {len(g1)})")
for idx, (a, b) in enumerate(zip(g0, g1)):
    if math.fabs(float(a) - float(b)) > tol:
        raise SystemExit(f"allreduce mismatch at idx={idx} (r0={a} r1={b} tol={tol})")
PY
then
  fail "allreduce mismatch detected"
fi

echo "PASS: 2-GPU DP correctness validated"
