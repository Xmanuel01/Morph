#!/usr/bin/env bash
set -euo pipefail

CONFIG="${ENKAI_CONFIG:-configs/enkai_50m.enk}"
KILL_STEP="${ENKAI_KILL_STEP:-1500}"
MAX_WAIT="${ENKAI_MAX_WAIT:-28800}"

echo "Using config: $CONFIG"
echo "Kill step: $KILL_STEP"

checkpoint_dir="checkpoints/enkai_50m"
log_path="${checkpoint_dir}/train_log.jsonl"

get_latest_step() {
  if [[ ! -f "$log_path" ]]; then
    echo 0
    return
  fi
  tail -n 1 "$log_path" | python - <<'PY'
import sys, json
line = sys.stdin.read().strip()
if not line:
    print(0); sys.exit(0)
try:
    obj = json.loads(line)
    print(int(obj.get("step", 0)))
except Exception:
    print(0)
PY
}

assert_log_health() {
  python - <<'PY'
import json, math
prev = -1
with open("checkpoints/enkai_50m/train_log.jsonl","r",encoding="utf8") as f:
    for line in f:
        line=line.strip()
        if not line:
            continue
        obj=json.loads(line)
        step=int(obj.get("step",0))
        loss=float(obj.get("loss",0.0))
        if math.isnan(loss) or math.isinf(loss):
            raise SystemExit(f"Non-finite loss at step {step}")
        if step < prev:
            raise SystemExit(f"Step went backwards: {step} < {prev}")
        prev=step
print("Log health OK")
PY
}

wait_until_step() {
  local target=$1
  local start=$(date +%s)
  while true; do
    step=$(get_latest_step)
    if [[ "$step" -ge "$target" ]]; then
      echo "$step"
      return
    fi
    sleep 5
    now=$(date +%s)
    if [[ $((now-start)) -gt "$MAX_WAIT" ]]; then
      echo "Timeout waiting for step $target (last=$step)" >&2
      exit 1
    fi
  done
}

echo "Starting training..."
enkai train "$CONFIG" &
pid=$!

step=$(wait_until_step "$KILL_STEP")
echo "Reached step $step, killing process..."
kill -9 "$pid" || true

echo "Restarting training..."
enkai train "$CONFIG" &
pid2=$!

resume_target=$((KILL_STEP + 50))
step2=$(wait_until_step "$resume_target")
echo "Resume confirmed at step $step2"
assert_log_health

kill -9 "$pid2" || true
echo "Soak test OK"
