#!/usr/bin/env bash
set -euo pipefail

CONFIG="${ENKAI_CONFIG:-configs/enkai_50m.enk}"
KILL_STEP="${ENKAI_KILL_STEP:-1500}"
POST_RESUME_STEPS="${ENKAI_POST_RESUME_STEPS:-50}"
MAX_WAIT="${ENKAI_MAX_WAIT:-28800}"
ARTIFACT_DIR="${ENKAI_GPU_ARTIFACT_DIR:-artifacts/gpu}"
CHECKPOINT_DIR="${ENKAI_CHECKPOINT_DIR:-checkpoints/enkai_50m}"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "SKIPPED: python runtime not found (install python3)"
  exit 0
fi

mkdir -p "$CHECKPOINT_DIR" "$ARTIFACT_DIR"
log_path="${CHECKPOINT_DIR}/train_log.jsonl"
single_log="${ARTIFACT_DIR}/single_gpu.log"
single_json="${ARTIFACT_DIR}/single_gpu_evidence.json"

echo "Using config: $CONFIG"
echo "Kill step: $KILL_STEP"
echo "Post-resume steps: $POST_RESUME_STEPS"

get_latest_step() {
  if [[ ! -f "$log_path" ]]; then
    echo 0
    return
  fi
  "$PYTHON_BIN" - "$log_path" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
last = 0
for raw in path.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
    except Exception:
        continue
    last = int(obj.get("step", 0))
print(last)
PY
}

get_last_loss() {
  if [[ ! -f "$log_path" ]]; then
    echo "null"
    return
  fi
  "$PYTHON_BIN" - "$log_path" <<'PY'
import json
import math
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
last = None
for raw in path.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
    except Exception:
        continue
    last = obj.get("loss")
if last is None:
    print("null")
else:
    print(float(last))
PY
}

assert_log_health() {
  "$PYTHON_BIN" - "$log_path" <<'PY'
import json
import math
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
prev = -1
for raw in path.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line:
        continue
    obj = json.loads(line)
    step = int(obj.get("step", 0))
    loss = float(obj.get("loss", 0.0))
    if math.isnan(loss) or math.isinf(loss):
        raise SystemExit(f"Non-finite loss at step {step}")
    if step < prev:
        raise SystemExit(f"Step went backwards: {step} < {prev}")
    prev = step
PY
}

wait_until_step() {
  local target="$1"
  local pid="$2"
  local start
  start="$(date +%s)"
  while true; do
    local step
    step="$(get_latest_step)"
    if [[ "$step" -ge "$target" ]]; then
      echo "$step"
      return
    fi
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      echo "Training process exited before step $target (last=$step)" >&2
      return 1
    fi
    sleep 5
    local now
    now="$(date +%s)"
    if [[ $((now - start)) -gt "$MAX_WAIT" ]]; then
      echo "Timeout waiting for step $target (last=$step)" >&2
      return 1
    fi
  done
}

latest_checkpoint_dir() {
  if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    return
  fi
  find "$CHECKPOINT_DIR" -maxdepth 1 -type d -name "step_*" | sort | tail -n 1
}

if [[ -f "$log_path" ]]; then
  rm -f "$log_path"
fi

echo "Starting training..."
enkai train "$CONFIG" >/dev/null 2>&1 &
pid="$!"

resumed_from_step=0
if ! step="$(wait_until_step "$KILL_STEP" "$pid")"; then
  kill -9 "$pid" >/dev/null 2>&1 || true
  echo "FAIL: unable to reach kill step" >&2
  exit 1
fi
resumed_from_step="$step"
kill -9 "$pid" >/dev/null 2>&1 || true

echo "Restarting training..."
enkai train "$CONFIG" >/dev/null 2>&1 &
pid2="$!"
resume_target=$((KILL_STEP + POST_RESUME_STEPS))
if ! step2="$(wait_until_step "$resume_target" "$pid2")"; then
  kill -9 "$pid2" >/dev/null 2>&1 || true
  echo "FAIL: resume did not reach target step" >&2
  exit 1
fi
kill -9 "$pid2" >/dev/null 2>&1 || true

assert_log_health
last_loss="$(get_last_loss)"
latest_ckpt="$(latest_checkpoint_dir || true)"
checkpoint_verified="False"
if [[ -n "${latest_ckpt:-}" ]]; then
  checkpoint_verified="True"
fi

nan_or_inf="False"
if ! "$PYTHON_BIN" - "$last_loss" <<'PY'
import math
import sys

raw = sys.argv[1]
if raw == "null":
    sys.exit(0)
loss = float(raw)
if math.isnan(loss) or math.isinf(loss):
    sys.exit(1)
sys.exit(0)
PY
  nan_or_inf="True"
fi

status="PASS"
if [[ "$nan_or_inf" == "True" || "$checkpoint_verified" != "True" ]]; then
  status="FAIL"
fi

last_step="$(get_latest_step)"
timestamp_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

{
  echo "timestamp_utc: $timestamp_utc"
  echo "status: $status"
  echo "last_step: $last_step"
  echo "last_loss: $last_loss"
  echo "resumed_from_step: $resumed_from_step"
  echo "nan_or_inf: $nan_or_inf"
  echo "checkpoint_verified: $checkpoint_verified"
  echo "checkpoint_path: ${latest_ckpt:-}"
  echo "evidence_json: $single_json"
} >"$single_log"

"$PYTHON_BIN" - "$single_json" "$timestamp_utc" "$status" "$last_step" "$last_loss" "$resumed_from_step" "$nan_or_inf" "$checkpoint_verified" "${latest_ckpt:-}" <<'PY'
import json
import sys

(
    out_path,
    ts,
    status,
    last_step,
    last_loss,
    resumed_from,
    nan_or_inf,
    checkpoint_verified,
    checkpoint_path,
) = sys.argv[1:]

def parse_num(raw):
    if raw == "null":
        return None
    try:
        return float(raw)
    except Exception:
        return None

doc = {
    "schema_version": 1,
    "gate": "single_gpu_soak",
    "timestamp_utc": ts,
    "status": status,
    "last_step": int(last_step),
    "last_loss": parse_num(last_loss),
    "resumed_from_step": int(resumed_from),
    "nan_or_inf": nan_or_inf == "True",
    "checkpoint_verified": checkpoint_verified == "True",
    "checkpoint_path": checkpoint_path or None,
}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(doc, f, indent=2)
    f.write("\n")
PY

cat "$single_log"
if [[ "$status" != "PASS" ]]; then
  exit 1
fi
echo "Soak test OK"
