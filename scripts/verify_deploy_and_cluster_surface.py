#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the v3.3.0 deploy and cluster surface."
    )
    parser.add_argument(
        "--contract",
        default="enkai/contracts/selfhost_deploy_and_cluster_surface_v3_3_0.json",
        help="Path to the deploy-and-cluster surface contract JSON",
    )
    parser.add_argument(
        "--output",
        default="artifacts/readiness/strict_selfhost_deploy_and_cluster_surface.json",
        help="Path to write the verification report JSON",
    )
    return parser.parse_args()


def get_path(obj: Any, dotted_path: str) -> Any:
    current = obj
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(dotted_path)
        current = current[part]
    return current


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    contract_path = (repo_root / args.contract).resolve()
    output_path = (repo_root / args.output).resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    slice_report_path = (repo_root / contract["required_slice_report"]).resolve()
    slice_report = json.loads(slice_report_path.read_text(encoding="utf-8"))

    failures: list[str] = []
    cases = slice_report.get("cases", {})
    validations = slice_report.get("validations", {})

    if not slice_report.get("all_passed", False):
        failures.append("systems control-plane slice report is not green")

    for case_id in contract["required_cases"]:
        entry = cases.get(case_id)
        if not isinstance(entry, dict):
            failures.append(f"missing required case {case_id}")
            continue
        if not entry.get("ok", False):
            failures.append(f"{case_id} did not pass")

    for dotted_path, expected in contract["required_validations"].items():
        try:
            actual = get_path(validations, dotted_path)
        except KeyError:
            failures.append(f"missing validation path {dotted_path}")
            continue
        if actual != expected:
            failures.append(
                f"{dotted_path} expected {expected!r}, got {actual!r}"
            )

    for field, fragment in contract["required_stdout_fragments"].items():
        actual = validations.get(field)
        if not isinstance(actual, str):
            failures.append(f"missing stdout validation {field}")
            continue
        if fragment not in actual:
            failures.append(
                f"{field} missing required fragment {fragment!r}"
            )

    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": str(contract_path),
        "slice_report": str(slice_report_path),
        "all_passed": not failures,
        "failures": failures,
        "cases": {case_id: cases.get(case_id) for case_id in contract["required_cases"]},
        "validations": {
            dotted_path: (
                get_path(validations, dotted_path)
                if dotted_path in contract["required_validations"]
                else None
            )
            for dotted_path in contract["required_validations"]
        },
        "stdout_fragments": {
            field: validations.get(field) for field in contract["required_stdout_fragments"]
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
