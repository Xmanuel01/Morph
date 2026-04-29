#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the strict-selfhost tensor backend surface."
    )
    parser.add_argument(
        "--contract",
        default="enkai/contracts/selfhost_tensor_backend_v3_3_0.json",
        help="Path to the tensor-backend contract JSON",
    )
    parser.add_argument(
        "--output",
        default="artifacts/readiness/strict_selfhost_tensor_backend_surface.json",
        help="Path to write the verification report JSON",
    )
    return parser.parse_args()


def get_path(obj: Any, dotted_path: str) -> Any:
    current = obj
    for part in dotted_path.split("."):
        if isinstance(current, dict):
            if part not in current:
                raise KeyError(dotted_path)
            current = current[part]
        elif isinstance(current, list):
            try:
                index = int(part)
            except ValueError as exc:
                raise KeyError(dotted_path) from exc
            if index < 0 or index >= len(current):
                raise KeyError(dotted_path)
            current = current[index]
        else:
            raise KeyError(dotted_path)
    return current


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    contract_path = (repo_root / args.contract).resolve()
    output_path = (repo_root / args.output).resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    report_cache: dict[str, dict[str, Any]] = {}
    failures: list[str] = []

    for report_rel in contract["required_reports"]:
        report_path = (repo_root / report_rel).resolve()
        try:
            report_cache[report_rel] = json.loads(report_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            failures.append(f"missing required report {report_rel}")

    for composite_key, expected in contract["required_status_fields"].items():
        report_rel, field_path = composite_key.split(":", 1)
        report = report_cache.get(report_rel)
        if report is None:
            continue
        try:
            actual = get_path(report, field_path)
        except KeyError:
            failures.append(f"missing field {field_path} in {report_rel}")
            continue
        if actual != expected:
            failures.append(
                f"{report_rel}:{field_path} expected {expected!r}, got {actual!r}"
            )

    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": str(contract_path),
        "all_passed": not failures,
        "failures": failures,
        "reports": {
            report_rel: report_cache.get(report_rel) for report_rel in contract["required_reports"]
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
