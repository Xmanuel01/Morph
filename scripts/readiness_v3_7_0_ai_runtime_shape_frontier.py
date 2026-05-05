#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
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


def apply_shape_overrides(base_config: dict[str, Any], shape: dict[str, Any], checkpoint_dir: Path) -> dict[str, Any]:
    cfg = dict(base_config)
    for key, value in shape.get("config_overrides", {}).items():
        cfg[key] = value
    model = dict(base_config.get("model", {}))
    model.update(shape["model"])
    cfg["model"] = model
    if "hidden_size" in model:
        cfg["hidden_size"] = int(model["hidden_size"])
    if "vocab_size" in model:
        cfg["vocab_size"] = int(model["vocab_size"])
    cfg["checkpoint_dir"] = str(checkpoint_dir)
    return cfg


def run_command(command: list[str], cwd: Path, env: dict[str, str]) -> dict[str, Any]:
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
    return {
        "command": command,
        "exit_code": result.returncode,
        "passed": result.returncode == 0,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the v3.7.0 model-frontier and latency proof.")
    parser.add_argument("--enkai-bin", required=True)
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--suite", default="bench/suites/v3_7_0_ai_runtime_foundation.json")
    parser.add_argument("--output", default="artifacts/readiness/v3_7_0_ai_runtime_shape_frontier.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.workspace).resolve()
    enkai_bin = Path(args.enkai_bin).resolve()
    suite = read_json((repo_root / args.suite).resolve())
    output_path = (repo_root / args.output).resolve()

    work_root = repo_root / "artifacts" / "v3_7_0_ai_runtime_shape_frontier"
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    dataset_lines = suite["dataset_lines"] * max(1, int(suite.get("dataset_repeat", 1)))
    dataset_path = work_root / "data.txt"
    dataset_path.write_text("\n".join(dataset_lines) + "\n", encoding="utf-8")

    env = dict(os.environ)
    env.setdefault("ENKAI_STD", str((repo_root / "std").resolve()))

    base_config = dict(suite["base_config"])
    base_config["dataset_path"] = str(dataset_path)
    base_config["checkpoint_dir"] = str(work_root / "base_ckpt")
    base_config["tokenizer_train"] = {
        "path": str(dataset_path),
        "vocab_size": int(base_config["tokenizer_train"]["vocab_size"]),
    }

    base_config_path = work_root / "base_train.enk"
    write_config(base_config_path, base_config)
    base_train = run_command([str(enkai_bin), "train", str(base_config_path)], repo_root, env)
    if not base_train["passed"]:
        write_json(output_path, {"all_passed": False, "failure": base_train})
        return 1

    base_train_report = read_json(Path(base_config["checkpoint_dir"]) / "ai_runtime_report.json")

    resume_train = run_command([str(enkai_bin), "train", str(base_config_path)], repo_root, env)
    if not resume_train["passed"]:
        write_json(output_path, {"all_passed": False, "failure": resume_train})
        return 1
    resume_train_report = read_json(Path(base_config["checkpoint_dir"]) / "ai_runtime_report.json")

    eval_run = run_command([str(enkai_bin), "eval", str(base_config_path)], repo_root, env)
    if not eval_run["passed"]:
        write_json(output_path, {"all_passed": False, "failure": eval_run})
        return 1
    eval_report = read_json(Path(base_config["checkpoint_dir"]) / "ai_runtime_report.json")

    train_elapsed = max(1, int(base_train_report["elapsed_ms"]))
    resume_elapsed = max(1, int(resume_train_report["elapsed_ms"]))
    eval_elapsed = max(1, int(eval_report["elapsed_ms"]))
    latency_gates = suite.get("latency_gates", {})
    resume_factor = float(latency_gates.get("resume_vs_train_factor", 0.25))
    eval_factor = float(latency_gates.get("eval_vs_train_factor", 0.3))
    throughput_gates = suite.get("throughput_gates", {})
    checkpoint_bytes = int(resume_train_report.get("checkpoint_bytes") or base_train_report.get("checkpoint_bytes") or 0)
    eval_tokens = int(base_config["batch_size"]) * int(base_config["seq_len"]) * int(eval_report.get("eval_batches", 0))
    resume_checkpoint_bytes_per_sec = checkpoint_bytes / max(resume_elapsed / 1000.0, 1e-6)
    eval_tokens_per_sec = eval_tokens / max(eval_elapsed / 1000.0, 1e-6)

    shape_results = []
    all_shapes_passed = True
    for shape in suite.get("shape_frontier", []):
        shape_name = shape["name"]
        cfg = apply_shape_overrides(base_config, shape, work_root / f"{shape_name}_ckpt")
        cfg_path = work_root / f"{shape_name}.enk"
        write_config(cfg_path, cfg)
        train_check = run_command([str(enkai_bin), "train", str(cfg_path)], repo_root, env)
        eval_check = run_command([str(enkai_bin), "eval", str(cfg_path)], repo_root, env) if train_check["passed"] else {"passed": False, "stderr_tail": "train failed"}
        passed = bool(train_check["passed"] and eval_check["passed"])
        all_shapes_passed = all_shapes_passed and passed
        report = None
        if passed:
            report = read_json(Path(cfg["checkpoint_dir"]) / "ai_runtime_report.json")
        shape_results.append({
            "name": shape_name,
            "passed": passed,
            "hidden_size": cfg.get("hidden_size"),
            "model": cfg["model"],
            "kernel": None if report is None else report.get("kernel"),
            "worker_count": None if report is None else report.get("worker_count"),
            "loss": None if report is None else report.get("loss"),
            "train_stderr_tail": train_check.get("stderr_tail"),
            "eval_stderr_tail": eval_check.get("stderr_tail"),
        })

    summary = {
        "schema_version": 1,
        "verified_contract_version": "v3.7.0",
        "all_passed": all_shapes_passed,
        "model_frontier": {
            "all_shapes_passed": all_shapes_passed,
            "shapes": shape_results,
        },
        "latency_baselines": {
            "train_elapsed_ms": train_elapsed,
            "resume_elapsed_ms": resume_elapsed,
            "eval_elapsed_ms": eval_elapsed,
            "resume_vs_train_factor": round(resume_elapsed / train_elapsed, 6),
            "eval_vs_train_factor": round(eval_elapsed / train_elapsed, 6),
            "resume_latency_gate_passed": resume_elapsed <= max(1, int(train_elapsed * resume_factor)),
            "eval_latency_gate_passed": eval_elapsed <= max(1, int(train_elapsed * eval_factor)),
            "resume_worker_count": resume_train_report.get("worker_count"),
            "eval_worker_count": eval_report.get("worker_count"),
        },
        "throughput_baselines": {
            "checkpoint_bytes": checkpoint_bytes,
            "resume_checkpoint_bytes_per_sec": round(resume_checkpoint_bytes_per_sec, 3),
            "eval_tokens": eval_tokens,
            "eval_tokens_per_sec": round(eval_tokens_per_sec, 3),
            "resume_checkpoint_throughput_gate_passed": resume_checkpoint_bytes_per_sec
            >= float(throughput_gates.get("min_resume_checkpoint_bytes_per_sec", 0.0)),
            "eval_throughput_gate_passed": eval_tokens_per_sec
            >= float(throughput_gates.get("min_eval_tokens_per_sec", 0.0)),
        },
        "artifacts": {
            "workspace": str(work_root),
            "base_train_config": str(base_config_path),
            "base_train_report": str(Path(base_config["checkpoint_dir"]) / "ai_runtime_report.json"),
        },
    }
    summary["all_passed"] = bool(
        summary["model_frontier"]["all_shapes_passed"]
        and summary["latency_baselines"]["resume_latency_gate_passed"]
        and summary["latency_baselines"]["eval_latency_gate_passed"]
        and summary["throughput_baselines"]["resume_checkpoint_throughput_gate_passed"]
        and summary["throughput_baselines"]["eval_throughput_gate_passed"]
    )
    write_json(output_path, summary)
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
