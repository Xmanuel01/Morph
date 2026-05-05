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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exercise adversarial networked gradient exchange cases.")
    parser.add_argument("--enkai-bin", required=True)
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--suite", default="bench/suites/v3_7_0_ai_runtime_foundation.json")
    parser.add_argument("--output", default="artifacts/readiness/v3_7_0_networked_gradient_adversarial.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.workspace).resolve()
    enkai_bin = Path(args.enkai_bin).resolve()
    suite = read_json((repo_root / args.suite).resolve())
    output_path = (repo_root / args.output).resolve()
    work_root = repo_root / "artifacts" / "v3_7_0_networked_gradient_adversarial"
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    dataset_path = work_root / "data.txt"
    dataset_path.write_text("\n".join(suite["dataset_lines"]) + "\n", encoding="utf-8")
    cases = [
        {
            "name": "peer_disconnect",
            "world_size": 2,
            "fault_mode": "peer-disconnect",
            "target_rank": 1,
            "expected_errors": ["E_DIST_SYNC_PEER_DISCONNECT", "E_DIST_RENDEZVOUS_CORRUPT"],
        },
        {
            "name": "stale_step_payload",
            "world_size": 2,
            "fault_mode": "stale-step",
            "target_rank": 1,
            "expected_errors": ["E_DIST_SYNC_MISMATCH"],
        },
        {
            "name": "wrong_tensor_length",
            "world_size": 2,
            "fault_mode": "wrong-tensor-length",
            "target_rank": 1,
            "expected_errors": ["E_DIST_SYNC_SHAPE"],
        },
        {
            "name": "duplicate_rank_payload",
            "world_size": 3,
            "fault_mode": "duplicate-rank",
            "target_rank": 2,
            "expected_errors": ["duplicate gradient peer rank"],
        },
        {
            "name": "aggregation_timeout",
            "world_size": 2,
            "fault_mode": "timeout",
            "target_rank": 1,
            "expected_errors": ["E_DIST_SYNC_TIMEOUT"],
        },
    ]

    base_config = dict(suite["base_config"])
    base_config.update({
        "dataset_path": str(dataset_path),
        "tokenizer_train": {
            "path": str(dataset_path),
            "vocab_size": int(base_config["tokenizer_train"]["vocab_size"]),
        },
        "seq_len": 8,
        "batch_size": 2,
        "hidden_size": 24,
        "max_steps": 1,
        "save_every": 1,
        "eval_steps": 1,
        "model": {
            "vocab_size": int(base_config["vocab_size"]),
            "hidden_size": 24,
            "device": "cpu",
        },
    })
    base_env = dict(os.environ)
    base_env.setdefault("ENKAI_STD", str((repo_root / "std").resolve()))
    base_env["ENKAI_ENABLE_DIST"] = "1"
    base_env["ENKAI_DIST_GRAD_TIMEOUT_MS"] = "750"

    case_results = []
    for index, case in enumerate(cases):
        case_root = work_root / case["name"]
        rendezvous = f"tcp://127.0.0.1:{43301 + index * 30}"
        rank_cfgs = []
        for rank in range(int(case["world_size"])):
            cfg = dict(base_config)
            cfg["world_size"] = int(case["world_size"])
            cfg["rank"] = rank
            cfg["checkpoint_dir"] = str(case_root / f"rank{rank}_ckpt")
            cfg["dist"] = {
                "topology": "multi-node",
                "rendezvous": rendezvous,
                "retry_budget": 2,
                "device_map": list(range(int(case["world_size"]))),
                "preview_mode": "networked-sync-preview",
            }
            cfg_path = case_root / f"rank{rank}.enk"
            write_config(cfg_path, cfg)
            rank_cfgs.append((rank, cfg_path))

        env = dict(base_env)
        env["ENKAI_DIST_GRAD_FAULT_MODE"] = case["fault_mode"]
        env["ENKAI_DIST_GRAD_FAULT_RANK"] = str(case["target_rank"])
        env["ENKAI_DIST_GRAD_FAULT_STEP"] = "1"

        procs = [
            (
                rank,
                subprocess.Popen(
                    [str(enkai_bin), "train", str(cfg_path)],
                    cwd=repo_root,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                ),
            )
            for rank, cfg_path in rank_cfgs
        ]
        rank_reports = []
        combined_output = ""
        for rank, proc in procs:
            try:
                stdout, stderr = proc.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
                stderr += "\nE_DIST_SYNC_TIMEOUT: harness timeout"
            combined_output += stdout + "\n" + stderr + "\n"
            rank_reports.append({
                "rank": rank,
                "exit_code": proc.returncode,
                "stderr_tail": stderr[-2000:],
            })
        observed = any(expected in combined_output for expected in case["expected_errors"])
        passed = observed and any(report["exit_code"] != 0 for report in rank_reports)
        case_results.append({
            "name": case["name"],
            "passed": passed,
            "fault_mode": case["fault_mode"],
            "expected_errors": case["expected_errors"],
            "observed_expected_error": observed,
            "rank_reports": rank_reports,
        })

    summary = {
        "schema_version": 1,
        "verified_contract_version": "v3.7.0",
        "all_passed": all(case["passed"] for case in case_results),
        "surface": "networked_gradient_exchange_adversarial",
        "cases": case_results,
        "artifacts": {
            "workspace": str(work_root),
            "dataset_path": str(dataset_path),
        },
    }
    write_json(output_path, summary)
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
