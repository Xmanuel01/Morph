#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the first executable v3.7.0 distributed-runtime tranche.")
    parser.add_argument("--contract", default="enkai/contracts/v3_7_0_distributed_runtime_exec.json")
    parser.add_argument("--output", default="artifacts/readiness/v3_7_0_distributed_runtime_exec_verify.json")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    contract_path = (repo_root / args.contract).resolve()
    contract = read_json(contract_path)
    report_path = (repo_root / contract["output_report"]).resolve()
    report = read_json(report_path)
    failures: list[str] = []

    if not report.get("all_passed"):
        failures.append("all_passed expected True, got False")
    if report.get("execution_mode") != "rank-sharded-preview":
        failures.append(f"execution_mode expected 'rank-sharded-preview', got {report.get('execution_mode')!r}")
    if report.get("synchronized_gradients") is not False:
        failures.append("synchronized_gradients expected False")
    rank_reports = report.get("rank_reports", [])
    if len(rank_reports) != int(report.get("world_size", 0)):
        failures.append("rank report count does not match world_size")
    for rel_path, snippets in contract["required_text_snippets"].items():
        text = (repo_root / rel_path).read_text(encoding="utf-8-sig")
        for snippet in snippets:
            if snippet not in text:
                failures.append(f"missing snippet in {rel_path}: {snippet}")

    output = {
        "schema_version": 1,
        "contract": str(contract_path),
        "report": str(report_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "all_passed": not failures,
        "failures": failures,
    }
    output_path = (repo_root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return 0 if output["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
