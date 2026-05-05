#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_config(path: Path, payload: dict[str, Any]) -> None:
    escaped = json.dumps(payload).replace('\\', '\\\\').replace('"', '\\"')
    source = f'fn main() ::\n    return json.parse("{escaped}")\n::\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def run_command(command: list[str], cwd: Path, env: dict[str, str]) -> dict[str, Any]:
    started = time.perf_counter()
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return {
        "command": command,
        "exit_code": result.returncode,
        "passed": result.returncode == 0,
        "elapsed_ms": elapsed_ms,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def python_reference_train(config: dict[str, Any], dataset_lines: list[str]) -> dict[str, Any]:
    vocab: dict[str, int] = {}
    tokens: list[int] = []
    for line in dataset_lines:
        for word in line.split():
            if word not in vocab:
                vocab[word] = len(vocab)
            tokens.append(vocab[word])
    vocab_size = max(int(config["vocab_size"]), len(vocab) or 1)
    hidden_size = int(config["hidden_size"])
    seq_len = int(config["seq_len"])
    batch_size = int(config["batch_size"])
    lr = float(config["lr"])
    max_steps = int(config["max_steps"])
    embed_len = vocab_size * hidden_size
    out_len = hidden_size * vocab_size
    bias_offset = embed_len + out_len
    param_count = embed_len + out_len + vocab_size
    params = [0.001 * ((i % 7) - 3) for i in range(param_count)]

    def project_logits(token_id: int) -> list[float]:
        emb_start = token_id * hidden_size
        logits = [params[bias_offset + j] for j in range(vocab_size)]
        for j in range(vocab_size):
            total = logits[j]
            for h in range(hidden_size):
                total += params[emb_start + h] * params[embed_len + h * vocab_size + j]
            logits[j] = total
        return logits

    start = time.perf_counter()
    total_tokens = 0
    final_loss = 0.0
    for step in range(max_steps):
        loss_sum = 0.0
        micro_batches = 0
        for batch_index in range(batch_size):
            base = (step * batch_size + batch_index) % max(1, len(tokens) - 1)
            window = [tokens[(base + offset) % len(tokens)] for offset in range(seq_len + 1)]
            total_tokens += seq_len
            batch_loss = 0.0
            for token, target in zip(window[:-1], window[1:]):
                logits = project_logits(token)
                max_logit = max(logits)
                exp_logits = [math.exp(value - max_logit) for value in logits]
                exp_sum = sum(exp_logits)
                prob = max(exp_logits[target] / exp_sum, 1e-9)
                batch_loss += -math.log(prob)
            loss_sum += batch_loss / max(1, seq_len)
            micro_batches += 1
        final_loss = loss_sum / max(1, micro_batches)
        for i in range(len(params)):
            params[i] -= lr * 0.00001 * ((i % 5) - 2)
    elapsed_ms = max(1, int((time.perf_counter() - start) * 1000))
    peak_memory_bytes_est = param_count * 12 + batch_size * seq_len * 8
    return {
        "elapsed_ms": elapsed_ms,
        "tokens": total_tokens,
        "tokens_per_sec": total_tokens / (elapsed_ms / 1000.0),
        "loss": final_loss,
        "peak_memory_bytes_est": peak_memory_bytes_est,
    }


def directory_size_bytes(root: Path) -> int:
    total = 0
    if root.exists():
        for path in root.rglob("*"):
            if path.is_file():
                total += path.stat().st_size
    return total


def workload_config(base_config: dict[str, Any], workload: dict[str, Any], dataset_path: Path, checkpoint_dir: Path) -> dict[str, Any]:
    cfg = dict(base_config)
    cfg.update(workload.get("config_overrides", {}))
    model = dict(base_config.get("model", {}))
    model.update(workload.get("model", {}))
    cfg["model"] = model
    cfg["dataset_path"] = str(dataset_path)
    cfg["checkpoint_dir"] = str(checkpoint_dir)
    cfg["tokenizer_train"] = {
        "path": str(dataset_path),
        "vocab_size": int(cfg["tokenizer_train"]["vocab_size"]),
    }
    if "hidden_size" in model:
        cfg["hidden_size"] = int(model["hidden_size"])
    if "vocab_size" in model:
        cfg["vocab_size"] = int(model["vocab_size"])
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate broader realistic AI workload benchmark evidence.")
    parser.add_argument("--enkai-bin", required=True)
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--suite", default="bench/suites/v3_7_0_ai_runtime_realistic_workloads.json")
    parser.add_argument("--output", default="artifacts/readiness/v3_7_0_ai_runtime_realistic_workloads.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = Path(args.workspace).resolve()
    enkai_bin = Path(args.enkai_bin).resolve()
    suite = read_json((workspace / args.suite).resolve())
    output_path = (workspace / args.output).resolve()
    inventory = read_json(workspace / "artifacts" / "readiness" / "strict_selfhost_dependency_inventory.json")

    work_root = workspace / "artifacts" / "v3_7_0_ai_runtime_realistic_workloads"
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.setdefault("ENKAI_STD", str((workspace / "std").resolve()))

    base_config = dict(suite["base_config"])
    policy = suite["benchmark_policy"]
    workloads_out: list[dict[str, Any]] = []
    all_passed = True

    for workload in suite["workloads"]:
        name = workload["name"]
        case_root = work_root / name
        dataset_lines = workload["dataset_lines"] * max(1, int(workload.get("dataset_repeat", 1)))
        dataset_path = case_root / "data.txt"
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_path.write_text("\n".join(dataset_lines) + "\n", encoding="utf-8")

        accel_cfg = workload_config(base_config, workload, dataset_path, case_root / "enkai_accel_ckpt")
        accel_cfg["suite_id"] = suite["suite"]
        accel_train_path = case_root / "enkai_accel_train.enk"
        write_config(accel_train_path, accel_cfg)
        accel_train = run_command([str(enkai_bin), "train", str(accel_train_path)], workspace, env)
        accel_report = None
        eval_check = None
        eval_report = None
        if accel_train["passed"]:
            accel_report = read_json(Path(accel_cfg["checkpoint_dir"]) / "ai_runtime_report.json")
            eval_check = run_command([str(enkai_bin), "eval", str(accel_train_path)], workspace, env)
            if eval_check["passed"]:
                eval_report = read_json(Path(accel_cfg["checkpoint_dir"]) / "ai_runtime_report.json")

        cpu_cfg = dict(accel_cfg)
        cpu_cfg["backend"] = "cpu"
        cpu_cfg["checkpoint_dir"] = str(case_root / "cpu_ckpt")
        cpu_path = case_root / "cpu_train.enk"
        write_config(cpu_path, cpu_cfg)
        cpu_train = run_command([str(enkai_bin), "train", str(cpu_path)], workspace, env)
        cpu_report = read_json(Path(cpu_cfg["checkpoint_dir"]) / "ai_runtime_report.json") if cpu_train["passed"] else None

        native_cfg = dict(accel_cfg)
        native_cfg["backend"] = "native"
        native_cfg["checkpoint_dir"] = str(case_root / "native_ckpt")
        native_path = case_root / "native_train.enk"
        write_config(native_path, native_cfg)
        native_train = run_command([str(enkai_bin), "train", str(native_path)], workspace, env)
        native_report = read_json(Path(native_cfg["checkpoint_dir"]) / "ai_runtime_report.json") if native_train["passed"] else None

        python_report = python_reference_train(accel_cfg, dataset_lines)

        passed = bool(accel_train["passed"] and eval_check and eval_check["passed"] and cpu_train["passed"] and native_train["passed"])
        comparison = {}
        gates = {}
        if passed and accel_report and cpu_report and native_report and eval_report:
            accel_elapsed = max(1, int(accel_report["elapsed_ms"]))
            cpu_elapsed = max(1, int(cpu_report["elapsed_ms"]))
            python_elapsed = max(1, int(python_report["elapsed_ms"]))
            accel_checkpoint_bytes = directory_size_bytes(Path(accel_cfg["checkpoint_dir"]))
            cpu_checkpoint_bytes = max(1, directory_size_bytes(Path(cpu_cfg["checkpoint_dir"])))
            eval_tokens = int(accel_cfg["batch_size"]) * int(accel_cfg["seq_len"]) * int(eval_report.get("eval_batches", 0))
            eval_elapsed_ms = max(1, int(eval_report["elapsed_ms"]))
            eval_tokens_per_sec = eval_tokens / (eval_elapsed_ms / 1000.0)
            min_eval_tokens_per_sec = float(
                workload.get("regression_overrides", {}).get(
                    "min_eval_tokens_per_sec",
                    policy["regression_limits"]["min_eval_tokens_per_sec"],
                )
            )
            comparison = {
                "enkai_vs_python_speedup": round(python_elapsed / accel_elapsed, 6),
                "enkai_vs_cpu_speedup": round(cpu_elapsed / accel_elapsed, 6),
                "enkai_vs_native_ratio": round(accel_elapsed / max(1, int(native_report["elapsed_ms"])), 6),
                "training_time_reduction_vs_python_pct": round((1.0 - (accel_elapsed / python_elapsed)) * 100.0, 6),
                "eval_tokens_per_sec": round(eval_tokens_per_sec, 6),
            }
            gates = {
                "speedup_vs_python_passed": comparison["enkai_vs_python_speedup"] >= float(policy["regression_limits"]["min_speedup_vs_python"]),
                "speedup_vs_cpu_passed": comparison["enkai_vs_cpu_speedup"] >= float(policy["regression_limits"]["min_speedup_vs_cpu"]),
                "peak_memory_passed": int(accel_report["peak_memory_bytes_est"]) <= int(cpu_report["peak_memory_bytes_est"] * float(policy["regression_limits"]["max_peak_memory_vs_cpu_factor"])),
                "checkpoint_overhead_passed": accel_checkpoint_bytes <= int(cpu_checkpoint_bytes * float(policy["regression_limits"]["max_checkpoint_overhead_vs_cpu_factor"])),
                "eval_throughput_passed": eval_tokens_per_sec >= min_eval_tokens_per_sec,
            }
            passed = passed and all(gates.values())
        else:
            passed = False

        workloads_out.append({
            "name": name,
            "category": workload["category"],
            "passed": passed,
            "train_success": accel_train["passed"],
            "eval_success": bool(eval_check and eval_check["passed"]),
            "config": {
                "hidden_size": accel_cfg["hidden_size"],
                "seq_len": accel_cfg["seq_len"],
                "batch_size": accel_cfg["batch_size"],
                "max_steps": accel_cfg["max_steps"],
                "model": accel_cfg["model"],
            },
            "benchmark": {
                "enkai_accel": accel_report,
                "cpu_scalar_baseline": cpu_report,
                "python_baseline": python_report,
                "native_comparison": native_report,
                "eval_report": eval_report,
                "comparisons": comparison,
                "regression_gates": gates,
                "regression_thresholds": {
                    "min_eval_tokens_per_sec": workload.get("regression_overrides", {}).get(
                        "min_eval_tokens_per_sec",
                        policy["regression_limits"]["min_eval_tokens_per_sec"],
                    )
                },
            },
            "artifacts": {
                "workspace": str(case_root),
                "dataset_path": str(dataset_path),
                "accel_train_stderr_tail": accel_train["stderr_tail"],
                "accel_eval_stderr_tail": "" if eval_check is None else eval_check["stderr_tail"],
            },
        })
        all_passed = all_passed and passed

    speedups_python = [float(item["benchmark"]["comparisons"]["enkai_vs_python_speedup"]) for item in workloads_out if item["benchmark"]["comparisons"]]
    speedups_cpu = [float(item["benchmark"]["comparisons"]["enkai_vs_cpu_speedup"]) for item in workloads_out if item["benchmark"]["comparisons"]]
    eval_tps = [float(item["benchmark"]["comparisons"].get("eval_tokens_per_sec", 0.0)) for item in workloads_out if item["benchmark"]["comparisons"]]

    summary = {
        "schema_version": 1,
        "verified_contract_version": "v3.7.0",
        "suite": suite["suite"],
        "all_passed": all_passed,
        "machine_profile": suite["machine_profile"],
        "policy": policy,
        "rust_free_shipped_path": bool(inventory["policy"]["rust_free_shipped_path"]),
        "workload_count": len(workloads_out),
        "passed_workload_count": sum(1 for item in workloads_out if item["passed"]),
        "aggregate": {
            "min_speedup_vs_python": min(speedups_python) if speedups_python else 0.0,
            "median_speedup_vs_python": statistics.median(speedups_python) if speedups_python else 0.0,
            "min_speedup_vs_cpu": min(speedups_cpu) if speedups_cpu else 0.0,
            "median_speedup_vs_cpu": statistics.median(speedups_cpu) if speedups_cpu else 0.0,
            "min_eval_tokens_per_sec": min(eval_tps) if eval_tps else 0.0,
        },
        "workloads": workloads_out,
        "artifacts": {
            "workspace": str(work_root),
            "suite_path": str((workspace / args.suite).resolve()),
        },
    }
    write_json(output_path, summary)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
