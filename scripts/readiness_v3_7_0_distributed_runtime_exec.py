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
    escaped = json.dumps(payload).replace("\\", "\\\\").replace('"', '\\"')
    source = f'fn main() ::\n    return json.parse("{escaped}")\n::\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


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
    parser = argparse.ArgumentParser(description="Execute the first v3.7.0 distributed-runtime preview.")
    parser.add_argument("--enkai-bin", required=True)
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--suite", default="bench/suites/v3_7_0_ai_runtime_foundation.json")
    parser.add_argument("--output", default="artifacts/readiness/v3_7_0_distributed_runtime_exec.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.workspace).resolve()
    enkai_bin = Path(args.enkai_bin).resolve()
    suite = read_json((repo_root / args.suite).resolve())
    output_path = (repo_root / args.output).resolve()

    dist = suite["distributed_preview"]
    world_size = int(dist["world_size"])
    work_root = repo_root / "artifacts" / "v3_7_0_distributed_runtime_exec"
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    dataset_lines = suite["dataset_lines"] * max(1, int(suite.get("dataset_repeat", 1)))
    dataset_path = work_root / "data.txt"
    dataset_path.write_text("\n".join(dataset_lines) + "\n", encoding="utf-8")

    env = dict(os.environ)
    env.setdefault("ENKAI_STD", str((repo_root / "std").resolve()))
    if dist.get("require_dist_opt_in", False):
        env["ENKAI_ENABLE_DIST"] = "1"

    base_config = dict(suite["base_config"])
    base_config["dataset_path"] = str(dataset_path)
    base_config["tokenizer_train"] = {
        "path": str(dataset_path),
        "vocab_size": int(base_config["tokenizer_train"]["vocab_size"]),
    }
    base_config["world_size"] = world_size
    base_config["dist"] = {
        "topology": dist["topology"],
        "rendezvous": "env://",
        "retry_budget": 1,
        "device_map": list(range(world_size)),
        "preview_mode": dist["execution_mode"],
    }

    rank_reports: list[dict[str, Any]] = []
    all_passed = True
    min_rank_tokens = int(dist.get("min_rank_tokens", 1))
    for rank in range(world_size):
        cfg = dict(base_config)
        cfg["rank"] = rank
        cfg["checkpoint_dir"] = str(work_root / f"rank{rank}_ckpt")
        cfg_path = work_root / f"rank{rank}.enk"
        write_config(cfg_path, cfg)

        train_run = run_command([str(enkai_bin), "train", str(cfg_path)], repo_root, env)
        train_report = None
        if train_run["passed"]:
            train_report = read_json(Path(cfg["checkpoint_dir"]) / "ai_runtime_report.json")
        eval_run = run_command([str(enkai_bin), "eval", str(cfg_path)], repo_root, env) if train_run["passed"] else {"passed": False, "stdout_tail": "", "stderr_tail": "train failed"}
        eval_report = None
        if eval_run["passed"]:
            eval_report = read_json(Path(cfg["checkpoint_dir"]) / "ai_runtime_report.json")

        rank_passed = bool(
            train_run["passed"]
            and eval_run["passed"]
            and train_report is not None
            and eval_report is not None
            and train_report.get("executed_backend") == "enkai_accel"
            and train_report.get("world_size") == world_size
            and train_report.get("rank") == rank
            and int(train_report.get("tokens", 0)) >= min_rank_tokens
        )
        all_passed = all_passed and rank_passed
        rank_reports.append({
            "rank": rank,
            "passed": rank_passed,
            "train_exit_code": train_run["exit_code"],
            "eval_exit_code": eval_run["exit_code"],
            "train_tokens": None if train_report is None else train_report.get("tokens"),
            "train_worker_count": None if train_report is None else train_report.get("worker_count"),
            "eval_loss": None if eval_report is None else eval_report.get("loss"),
            "world_size": None if train_report is None else train_report.get("world_size"),
            "executed_backend": None if train_report is None else train_report.get("executed_backend"),
            "checkpoint_dir": cfg["checkpoint_dir"],
            "train_stderr_tail": train_run.get("stderr_tail"),
            "eval_stderr_tail": eval_run.get("stderr_tail"),
        })

    summary = {
        "schema_version": 1,
        "verified_contract_version": "v3.7.0",
        "all_passed": all_passed,
        "execution_mode": dist["execution_mode"],
        "topology": dist["topology"],
        "world_size": world_size,
        "synchronized_gradients": False,
        "rank_reports": rank_reports,
        "artifacts": {
            "workspace": str(work_root),
            "dataset_path": str(dataset_path),
        },
    }
    write_json(output_path, summary)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
