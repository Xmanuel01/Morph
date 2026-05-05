#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute networked long-context v3.7.0 workload proof.")
    parser.add_argument("--enkai-bin", required=True)
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--suite", default="bench/suites/v3_7_0_ai_runtime_foundation.json")
    parser.add_argument("--output", default="artifacts/readiness/v3_7_0_networked_long_context_exec.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.workspace).resolve()
    suite_path = (repo_root / args.suite).resolve()
    suite = read_json(suite_path)
    spec = suite["networked_long_context_exec"]
    work_root = repo_root / "artifacts" / "v3_7_0_networked_long_context_exec"
    proxy_suite_path = work_root / "suite.json"

    proxy_suite = dict(suite)
    overrides = spec.get("config_overrides", {})
    proxy_base_config = dict(suite["base_config"])
    proxy_base_config.update({
        "seq_len": int(overrides["seq_len"]),
        "batch_size": int(overrides["batch_size"]),
        "hidden_size": int(overrides["hidden_size"]),
        "max_steps": int(overrides["max_steps"]),
        "save_every": int(overrides["save_every"]),
        "eval_steps": int(overrides["eval_steps"]),
        "oom_budget_bytes": int(overrides["oom_budget_bytes"]),
        "model": dict(spec["model"]),
    })
    proxy_suite["base_config"] = proxy_base_config
    proxy_suite["dataset_repeat"] = int(overrides.get("dataset_repeat", suite.get("dataset_repeat", 1)))
    proxy_suite["networked_rendezvous_exec"] = {
        "execution_mode": spec["execution_mode"],
        "topology": spec["topology"],
        "world_size": int(spec["world_size"]),
        "rendezvous": spec["rendezvous"],
        "retry_budget": int(spec.get("retry_budget", 3)),
        "require_dist_opt_in": True,
        "min_rank_tokens": int(spec.get("min_rank_tokens", 1)),
        "throughput_gates": spec["throughput_gates"],
    }
    write_json(proxy_suite_path, proxy_suite)

    output_path = (repo_root / args.output).resolve()
    runner = repo_root / "scripts" / "readiness_v3_7_0_networked_rendezvous_exec.py"
    result = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--enkai-bin",
            str(Path(args.enkai_bin).resolve()),
            "--workspace",
            str(repo_root),
            "--suite",
            str(proxy_suite_path.relative_to(repo_root)),
            "--output",
            str(output_path.relative_to(repo_root)),
        ],
        cwd=repo_root,
        text=True,
    )
    if result.returncode != 0:
        return result.returncode

    report = read_json(output_path)
    report["long_context_contract"] = {
        "source_suite": str(suite_path),
        "source_spec": "networked_long_context_exec",
        "seq_len": int(overrides["seq_len"]),
        "world_size": int(spec["world_size"]),
        "model_preset": spec["model"]["preset"],
    }
    report["all_passed"] = bool(
        report.get("all_passed")
        and int(report.get("world_size", 0)) == int(spec["world_size"])
        and int(overrides["seq_len"]) >= 32
        and report.get("baseline", {}).get("merged_replay", {}).get("passed") is True
        and report.get("baseline", {}).get("throughput", {}).get("networked_gradient_bytes_gate_passed") is True
    )
    write_json(output_path, report)
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
