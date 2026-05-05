#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the v3.7.0 AI runtime foundation surface.")
    parser.add_argument("--contract", default="enkai/contracts/v3_7_0_ai_runtime_foundation.json")
    parser.add_argument("--output", default="artifacts/readiness/v3_7_0_ai_runtime_foundation_verify.json")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def get_path(obj: Any, dotted_path: str) -> Any:
    current = obj
    for part in dotted_path.split('.'):
        if isinstance(current, dict):
            if part not in current:
                raise KeyError(dotted_path)
            current = current[part]
        elif isinstance(current, list):
            index = int(part)
            current = current[index]
        else:
            raise KeyError(dotted_path)
    return current


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    contract_path = (repo_root / args.contract).resolve()
    output_path = (repo_root / args.output).resolve()
    contract = read_json(contract_path)
    report_path = (repo_root / contract["required_report"]).resolve()
    report = read_json(report_path)
    failures: list[str] = []

    for field_path, expected in contract["required_status_fields"].items():
        try:
            actual = get_path(report, field_path)
        except Exception:
            failures.append(f"missing field {field_path}")
            continue
        if actual != expected:
            failures.append(f"{field_path} expected {expected!r}, got {actual!r}")

    if report.get("benchmark", {}).get("comparisons", {}).get("enkai_vs_python_speedup", 0) <= 0:
        failures.append("enkai_vs_python_speedup must be > 0")
    if report.get("benchmark", {}).get("enkai_accel", {}).get("peak_memory_bytes_est", 0) <= 0:
        failures.append("enkai_accel peak_memory_bytes_est must be > 0")

    for rel_path, snippets in contract.get("required_text_snippets", {}).items():
        path = (repo_root / rel_path).resolve()
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            failures.append(f"missing required file {rel_path}")
            continue
        for snippet in snippets:
            if snippet not in text:
                failures.append(f"missing snippet in {rel_path}: {snippet}")

    result = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": str(contract_path),
        "report": str(report_path),
        "all_passed": not failures,
        "failures": failures,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok" if result["all_passed"] else "failed", "output": str(output_path), "all_passed": result["all_passed"]}, separators=(",", ":")))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
