#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the realistic AI workload benchmark matrix.")
    parser.add_argument("--contract", default="enkai/contracts/v3_7_0_ai_runtime_realistic_workloads.json")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def get_path(obj: Any, dotted_path: str) -> Any:
    current = obj
    for part in dotted_path.split('.'):
        if isinstance(current, dict):
            current = current[part]
        else:
            raise KeyError(dotted_path)
    return current


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    contract_path = (repo_root / args.contract).resolve()
    contract = read_json(contract_path)
    report_path = (repo_root / contract["required_report"]).resolve()
    report = read_json(report_path)
    output_path = (repo_root / (args.output or contract["output_report"])).resolve()

    failures: list[str] = []
    for dotted_path, expected in contract["required_status_fields"].items():
        try:
            actual = get_path(report, dotted_path)
        except Exception:
            failures.append(f"missing {dotted_path}")
            continue
        if actual != expected:
            failures.append(f"{dotted_path} expected {expected!r}, got {actual!r}")

    aggregate = report.get("aggregate", {})
    for key, expected in contract["aggregate_gates"].items():
        actual = float(aggregate.get(key, 0.0))
        if actual < float(expected):
            failures.append(f"aggregate.{key} expected >= {expected}, got {actual}")

    for workload in report.get("workloads", []):
        if not workload.get("passed"):
            failures.append(f"workload {workload.get('name')} did not pass")
        gates = workload.get("benchmark", {}).get("regression_gates", {})
        for gate_name, gate_value in gates.items():
            if gate_value is not True:
                failures.append(f"workload {workload.get('name')} gate {gate_name} expected True")

    for rel_path, snippets in contract.get("required_text_snippets", {}).items():
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return 0 if output["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
