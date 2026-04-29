#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the v3.3.0 model and registry surface."
    )
    parser.add_argument(
        "--contract",
        default="enkai/contracts/selfhost_model_and_registry_surface_v3_3_0.json",
        help="Path to the model-and-registry surface contract JSON",
    )
    parser.add_argument(
        "--output",
        default="artifacts/readiness/strict_selfhost_model_and_registry_surface.json",
        help="Path to write the verification report JSON",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    contract_path = (repo_root / args.contract).resolve()
    output_path = (repo_root / args.output).resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    slice_report_path = (repo_root / contract["required_slice_report"]).resolve()
    slice_report = json.loads(slice_report_path.read_text(encoding="utf-8"))

    failures: list[str] = []
    checks_by_id = {entry.get("id"): entry for entry in slice_report.get("checks", [])}
    for check_id in contract["required_checks"]:
        if check_id not in checks_by_id:
            failures.append(f"missing required slice check {check_id}")

    if not slice_report.get("all_passed", False):
        failures.append("model/registry slice report is not green")

    native_expectations = contract["required_native_behavior"]
    native_entry = checks_by_id.get("train-native", {})
    for field, expected in native_expectations.items():
        if native_entry.get(field) != expected:
            failures.append(
                f"train-native.{field} expected {expected}, got {native_entry.get(field)}"
            )

    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": str(contract_path),
        "slice_report": str(slice_report_path),
        "all_passed": not failures,
        "failures": failures,
        "checks": slice_report.get("checks", []),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok" if report["all_passed"] else "failed",
                "output": str(output_path),
                "all_passed": report["all_passed"],
            },
            separators=(",", ":"),
        )
    )
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
