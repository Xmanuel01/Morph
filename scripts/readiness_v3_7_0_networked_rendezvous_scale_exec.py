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
    parser = argparse.ArgumentParser(description="Execute the frozen world_size=4 networked rendezvous tranche.")
    parser.add_argument("--enkai-bin", required=True)
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--suite", default="bench/suites/v3_7_0_ai_runtime_foundation.json")
    parser.add_argument("--output", default="artifacts/readiness/v3_7_0_networked_rendezvous_scale_exec.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.workspace).resolve()
    suite_path = (repo_root / args.suite).resolve()
    suite = read_json(suite_path)
    design = suite["networked_rendezvous_scale_design"]
    work_root = repo_root / "artifacts" / "v3_7_0_networked_rendezvous_scale_exec"
    proxy_suite_path = work_root / "suite.json"

    proxy_suite = dict(suite)
    proxy_base_config = dict(suite["base_config"])
    proxy_base_config.update(
        {
            "seq_len": 8,
            "batch_size": 3,
            "hidden_size": 32,
            "max_steps": 4,
            "save_every": 1,
            "eval_steps": 2,
            "oom_budget_bytes": 16777216,
            "model": {
                "vocab_size": int(suite["base_config"]["vocab_size"]),
                "hidden_size": 32,
                "device": "cpu",
            },
        }
    )
    proxy_suite["base_config"] = proxy_base_config
    proxy_suite["dataset_repeat"] = 2
    proxy_suite["networked_rendezvous_exec"] = {
        "execution_mode": "networked-sync-preview",
        "topology": design["topology"],
        "world_size": int(design["world_size"]),
        "rendezvous": design["rendezvous"],
        "retry_budget": int(design.get("retry_budget", 3)),
        "require_dist_opt_in": True,
        "min_rank_tokens": int(design.get("min_rank_tokens", 1)),
        "throughput_gates": design.get("throughput_gates", {
            "min_combined_train_tokens_per_sec": 60,
            "min_combined_eval_tokens_per_sec": 1000,
            "min_checkpoint_merge_bytes_per_sec": 5000000,
            "min_networked_gradient_bytes": 1
        }),
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
    report["scale_contract"] = {
        "source_suite": str(suite_path),
        "source_design": "networked_rendezvous_scale_design",
        "execution_widened_from_world_size": int(suite["networked_rendezvous_exec"]["world_size"]),
        "preconditions": design.get("preconditions", []),
    }
    report["all_passed"] = bool(
        report.get("all_passed")
        and report.get("world_size") == int(design["world_size"])
        and report.get("topology") == design["topology"]
        and str(report.get("execution_mode")) == "networked-sync-preview"
        and "bounded_contract_before_execution_widening" in design.get("preconditions", [])
    )
    write_json(output_path, report)
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
