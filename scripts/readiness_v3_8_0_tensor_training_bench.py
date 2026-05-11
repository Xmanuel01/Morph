#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_command(command: list[str], cwd: Path, env: dict[str, str]) -> dict[str, Any]:
    started = time.perf_counter()
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
    elapsed_ms = max(1, int((time.perf_counter() - started) * 1000))
    return {
        "command": command,
        "exit_code": result.returncode,
        "passed": result.returncode == 0,
        "elapsed_ms": elapsed_ms,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def parse_metric_lines(stdout: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for raw in stdout.splitlines():
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            continue
        try:
            metrics[key] = json.loads(value)
        except json.JSONDecodeError:
            metrics[key] = value
    return metrics


def python_reference_training(iterations: int) -> dict[str, Any]:
    x = [[1.0, 0.5, -1.0, 2.0], [0.25, -0.75, 1.5, 0.5]]
    targets = [0, 1]
    w = [[0.1, -0.2, 0.3, -0.4], [-0.1, 0.2, -0.3, 0.4]]
    b = [0.0, 0.0]
    started = time.perf_counter()
    final_loss = 0.0
    for _ in range(iterations):
        logits = []
        for row in x:
            logits.append([
                sum(row[j] * w[class_id][j] for j in range(4)) + b[class_id]
                for class_id in range(2)
            ])
        loss = 0.0
        grad_logits = [[0.0, 0.0], [0.0, 0.0]]
        for row_id, row in enumerate(logits):
            max_logit = max(row)
            exp_values = [math.exp(value - max_logit) for value in row]
            denom = sum(exp_values)
            probs = [value / denom for value in exp_values]
            loss -= math.log(max(probs[targets[row_id]], 1e-12))
            for class_id in range(2):
                grad_logits[row_id][class_id] = (probs[class_id] - (1.0 if class_id == targets[row_id] else 0.0)) / 2.0
        final_loss = loss / 2.0
        grad_w = [[0.0 for _ in range(4)] for _ in range(2)]
        grad_b = [0.0, 0.0]
        for row_id, row in enumerate(x):
            for class_id in range(2):
                grad_b[class_id] += grad_logits[row_id][class_id]
                for feature in range(4):
                    grad_w[class_id][feature] += grad_logits[row_id][class_id] * row[feature]
        for class_id in range(2):
            for feature in range(4):
                w[class_id][feature] -= 0.01 * grad_w[class_id][feature]
            b[class_id] -= 0.01 * grad_b[class_id]
    elapsed_ms = max(1, int((time.perf_counter() - started) * 1000))
    return {
        "elapsed_ms": elapsed_ms,
        "iterations": iterations,
        "loss": final_loss,
        "iterations_per_sec": iterations / (elapsed_ms / 1000.0),
    }


def training_source(iterations: int) -> str:
    return f'''
import std::io
import std::json
import std::tensor

policy default ::
    allow io.write
::

fn emit(name: String, value) ::
    io.stdout_write_text(name)
    io.stdout_write_text("=")
    io.stdout_write_text(json.stringify(value))
    io.stdout_write_text("\\n")
::

fn main() ::
    tensor.memory_clear_limit()
    tensor.memory_reset_peak()
    let x := tensor.from_array([1.0, 0.5, -1.0, 2.0, 0.25, -0.75, 1.5, 0.5], [2, 4])
    let targets := tensor.from_array([0.0, 1.0], [2])
    let w := tensor.requires_grad(tensor.from_array([0.1, -0.2, 0.3, -0.4, -0.1, 0.2, -0.3, 0.4], [2, 4]))
    let b := tensor.requires_grad(tensor.from_array([0.0, 0.0], [2]))
    mut state := tensor.adamw_state()
    mut step := 0
    mut final_loss := 0.0
    while step < {iterations} ::
        let logits := tensor.linear(x, w, b)
        let loss := tensor.cross_entropy(logits, targets)
        final_loss := tensor.get_flat(loss, 0)
        tensor.backward(loss)
        tensor.clip_grad_norm([w, b], 1.0, 0.000000001)
        let gw := tensor.grad(w)
        let gb := tensor.grad(b)
        state := tensor.adamw_step_multi([w, b], [gw, gb], state, 0.01, 0.9, 0.999, 0.00000001, 0.0)
        tensor.zero_grad_multi([w, b])
        step := step + 1
    ::
    emit("loss", final_loss)
    emit("steps", step)
    emit("peak_memory_bytes", tensor.memory_peak())
    emit("current_memory_bytes", tensor.memory_current())
::

main()
'''.strip() + "\n"


def eval_source(iterations: int) -> str:
    return f'''
import std::io
import std::json
import std::tensor

policy default ::
    allow io.write
::

fn emit(name: String, value) ::
    io.stdout_write_text(name)
    io.stdout_write_text("=")
    io.stdout_write_text(json.stringify(value))
    io.stdout_write_text("\\n")
::

fn main() ::
    tensor.memory_clear_limit()
    tensor.memory_reset_peak()
    let q := tensor.from_array([1.0, 0.0, 0.0, 1.0, 0.5, -0.5, 0.25, 0.75, 1.5, 0.25, -0.25, 0.5, 0.0, 1.0, 0.5, -1.0], [1, 4, 4])
    let k := tensor.from_array([1.0, 0.0, 0.0, 1.0, 0.5, -0.5, 0.25, 0.75, 1.5, 0.25, -0.25, 0.5, 0.0, 1.0, 0.5, -1.0], [1, 4, 4])
    let v := tensor.from_array([0.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 6.0], [1, 4, 4])
    mut step := 0
    mut checksum := 0.0
    while step < {iterations} ::
        let out := tensor.attention(q, k, v)
        let reduced := tensor.sum(tensor.sum(out, 2, false), 1, false)
        checksum := checksum + tensor.get_flat(reduced, 0)
        step := step + 1
    ::
    emit("checksum", checksum)
    emit("steps", step)
    emit("peak_memory_bytes", tensor.memory_peak())
::

main()
'''.strip() + "\n"


def oom_source() -> str:
    return '''
import std::tensor

fn main() ::
    let base := tensor.memory_current()
    tensor.memory_set_limit(base + 8)
    tensor.from_array([1.0, 2.0], [2])
::

main()
'''.strip() + "\n"


def run_enkai_case(
    enkai_bin: Path,
    workspace: Path,
    env: dict[str, str],
    work_root: Path,
    name: str,
    source: str,
    estimated_ops: int,
) -> dict[str, Any]:
    source_path = work_root / f"{name}.enk"
    source_path.write_text(source, encoding="utf-8")
    check = run_command([str(enkai_bin), "check", str(source_path)], workspace, env)
    run = run_command([str(enkai_bin), "run", str(source_path)], workspace, env)
    metrics = parse_metric_lines(run["stdout_tail"])
    return {
        "name": name,
        "source": str(source_path),
        "check": check,
        "run": run,
        "metrics": metrics,
        "estimated_ops": estimated_ops,
        "estimated_ops_per_sec": estimated_ops / (run["elapsed_ms"] / 1000.0),
        "passed": check["passed"] and run["passed"] and bool(metrics),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate v3.8.0 first-party tensor training benchmark evidence.")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--enkai-bin", default=None)
    parser.add_argument("--output", default="artifacts/readiness/v3_8_0_tensor_training_bench.json")
    parser.add_argument("--iterations", type=int, default=16)
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    enkai_bin = Path(args.enkai_bin).resolve() if args.enkai_bin else (workspace / "target" / "debug" / ("enkai.exe" if os.name == "nt" else "enkai"))
    output = workspace / args.output
    work_root = workspace / "artifacts" / "v3_8_0_tensor_training_bench"
    work_root.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.setdefault("ENKAI_STD", str((workspace / "std").resolve()))

    iterations = max(1, args.iterations)
    cases = [
        run_enkai_case(enkai_bin, workspace, env, work_root, "training_autodiff_optimizer", training_source(iterations), iterations * 128),
        run_enkai_case(enkai_bin, workspace, env, work_root, "eval_attention", eval_source(iterations), iterations * 512),
    ]
    python_reference = python_reference_training(iterations)

    oom_path = work_root / "oom_limit.enk"
    oom_path.write_text(oom_source(), encoding="utf-8")
    oom_check = run_command([str(enkai_bin), "check", str(oom_path)], workspace, env)
    oom_run = run_command([str(enkai_bin), "run", str(oom_path)], workspace, env)
    oom_case = {
        "name": "memory_limit_oom",
        "source": str(oom_path),
        "check": oom_check,
        "run": oom_run,
        "expected_error": "would exceed tensor memory limit",
        "passed": oom_check["passed"]
        and (not oom_run["passed"])
        and "would exceed tensor memory limit" in oom_run["stderr_tail"],
    }

    all_cases = cases + [oom_case]
    passed = all(case["passed"] for case in all_cases)
    payload = {
        "schema_version": 1,
        "version": "v3.8.0",
        "scope": "first_party_tensor_training_benchmark",
        "contract": {
            "requires": [
                "training/autodiff/optimizer case checks and runs",
                "eval attention case checks and runs",
                "peak memory evidence emitted for executable cases",
                "bounded OOM path fails deterministically",
                "Python reference timing archived for comparison context",
            ],
            "non_claims": [
                "not a GPU benchmark",
                "not PyTorch parity",
                "CLI process startup is included in Enkai elapsed time",
            ],
        },
        "enkai_bin": str(enkai_bin),
        "iterations": iterations,
        "python_reference": python_reference,
        "cases": all_cases,
        "passed": passed,
    }
    write_json(output, payload)
    print(json.dumps({"passed": passed, "output": str(output)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
