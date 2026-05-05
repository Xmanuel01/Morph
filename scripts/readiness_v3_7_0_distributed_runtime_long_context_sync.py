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
    parser = argparse.ArgumentParser(description="Execute longer-context synchronized distributed workloads.")
    parser.add_argument("--enkai-bin", required=True)
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--suite", default="bench/suites/v3_7_0_ai_runtime_foundation.json")
    parser.add_argument("--output", default="artifacts/readiness/v3_7_0_distributed_runtime_long_context_sync.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.workspace).resolve()
    suite_path = (repo_root / args.suite).resolve()
    suite = read_json(suite_path)
    work_root = repo_root / "artifacts" / "v3_7_0_distributed_runtime_long_context_sync"
    proxy_suite_path = work_root / "suite.json"

    proxy_suite = dict(suite)
    proxy_suite["distributed_sync_preview"] = suite["distributed_sync_long_context_preview"]
    proxy_suite["distributed_sync_shape_frontier"] = suite["distributed_sync_long_context_frontier"]
    write_json(proxy_suite_path, proxy_suite)

    output_path = (repo_root / args.output).resolve()
    runner = repo_root / "scripts" / "readiness_v3_7_0_distributed_runtime_sync.py"
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
    seq_lengths = [
        int(shape.get("config_overrides", {}).get("seq_len", suite["base_config"]["seq_len"]))
        for shape in suite["distributed_sync_long_context_frontier"]
    ]
    report["long_context_contract"] = {
        "source_suite": str(suite_path),
        "source_frontier": "distributed_sync_long_context_frontier",
        "min_seq_len": min(seq_lengths),
        "max_seq_len": max(seq_lengths),
        "cases": [shape["name"] for shape in suite["distributed_sync_long_context_frontier"]],
    }
    report["all_passed"] = bool(
        report.get("all_passed")
        and report.get("world_size") == int(suite["distributed_sync_long_context_preview"]["world_size"])
        and min_seq_lengths_passed(report, min(seq_lengths))
    )
    write_json(output_path, report)
    return 0 if report["all_passed"] else 1


def min_seq_lengths_passed(report: dict[str, Any], min_seq_len: int) -> bool:
    for case in report.get("shape_envelope", {}).get("cases", []):
        for rank in case.get("rank_reports", []):
            tokens = int(rank.get("train_tokens") or 0)
            if tokens < min_seq_len:
                return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
