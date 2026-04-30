#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the v3.4.0 install host-matrix baseline scope."
    )
    parser.add_argument(
        "--contract",
        default="enkai/contracts/v3_4_0_install_host_matrix_baseline.json",
        help="Path to the v3.4.0 install host-matrix contract JSON",
    )
    parser.add_argument(
        "--output",
        default="artifacts/readiness/v3_4_0_install_host_matrix_baseline.json",
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
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    failures: list[str] = []
    json_cache: dict[str, dict[str, Any]] = {}
    text_cache: dict[str, str] = {}

    for rel_path in contract.get("required_files", []):
        if not (repo_root / rel_path).is_file():
            failures.append(f"missing required file {rel_path}")

    for composite_key, expected in contract.get("required_json_fields", {}).items():
        rel_path, field_path = composite_key.split(":", 1)
        payload = json_cache.get(rel_path)
        if payload is None:
            path = repo_root / rel_path
            if not path.is_file():
                failures.append(f"missing required JSON file {rel_path}")
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            json_cache[rel_path] = payload
        try:
            actual = get_path(payload, field_path)
        except Exception:
            failures.append(f"missing field {field_path} in {rel_path}")
            continue
        if actual != expected:
            failures.append(
                f"{rel_path}:{field_path} expected {expected!r}, got {actual!r}"
            )

    for rel_path, snippets in contract.get("required_text_snippets", {}).items():
        text = text_cache.get(rel_path)
        if text is None:
            path = repo_root / rel_path
            if not path.is_file():
                failures.append(f"missing required text file {rel_path}")
                continue
            text = path.read_text(encoding="utf-8")
            text_cache[rel_path] = text
        for snippet in snippets:
            if snippet not in text:
                failures.append(f"{rel_path} missing required snippet: {snippet}")

    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": str(contract_path),
        "all_passed": not failures,
        "failures": failures,
        "verified_scope": contract.get("scope"),
        "verified_contract_version": contract.get("contract_version"),
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
