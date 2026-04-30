#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the v3.4.0 compatibility-only storage/data baseline scope."
    )
    parser.add_argument(
        "--contract",
        default="enkai/contracts/v3_4_0_compatibility_storage_data_baseline.json",
        help="Path to the v3.4.0 compatibility storage/data contract JSON",
    )
    parser.add_argument(
        "--output",
        default="artifacts/readiness/v3_4_0_compatibility_storage_data_baseline.json",
        help="Path to write the verification report JSON",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    contract_path = (repo_root / args.contract).resolve()
    output_path = (repo_root / args.output).resolve()
    contract = load_json(contract_path)

    failures: list[str] = []

    for rel_path in contract.get("required_files", []):
        if not (repo_root / rel_path).is_file():
            failures.append(f"missing required file {rel_path}")

    inventory_path = repo_root / "artifacts/readiness/strict_selfhost_dependency_inventory.json"
    data_surface_path = repo_root / "artifacts/readiness/strict_selfhost_data_registry_protocols_surface.json"
    readiness_path = repo_root / "artifacts/readiness/strict_selfhost.json"
    blockers_path = repo_root / "artifacts/readiness/strict_selfhost_blockers.json"
    next_step_path = repo_root / "artifacts/readiness/v3_4_0_zero_rust_next_step_baseline.json"

    if inventory_path.is_file():
        inventory = load_json(inventory_path)
        policy = inventory.get("policy", {})
        summary = inventory.get("summary", {})
        components = inventory.get("components", [])
        if policy.get("sqlite_policy") != "compatibility_only_outside_strict_selfhost_release_blockers":
            failures.append("strict_selfhost dependency inventory has unexpected sqlite policy")
        if summary.get("strict_selfhost_cpu_complete") is not True:
            failures.append("strict_selfhost dependency inventory no longer reports cpu complete")
        data_component = next((c for c in components if c.get("id") == "data_registry_protocols"), None)
        if data_component is None:
            failures.append("data_registry_protocols component missing from strict_selfhost dependency inventory")
        else:
            if data_component.get("status") != "done":
                failures.append("data_registry_protocols component is not done")
            if "sqlite_binding" not in data_component.get("native_components", []):
                failures.append("data_registry_protocols component no longer tracks sqlite_binding")
            notes = str(data_component.get("notes", ""))
            if "SQLite-backed compatibility paths globally" not in notes:
                failures.append("data_registry_protocols notes no longer describe broader SQLite replacement work")

    for artifact_path, label in [
        (data_surface_path, "strict_selfhost data registry surface"),
        (readiness_path, "strict_selfhost readiness"),
        (blockers_path, "strict_selfhost blockers"),
        (next_step_path, "v3.4.0 zero-Rust next-step baseline"),
    ]:
        if artifact_path.is_file():
            payload = load_json(artifact_path)
            if payload.get("all_passed") is not True:
                failures.append(f"{label} artifact is not green")

    for rel_path, snippets in contract.get("required_text_snippets", {}).items():
        path = repo_root / rel_path
        if not path.is_file():
            failures.append(f"missing required text file {rel_path}")
            continue
        text = path.read_text(encoding="utf-8")
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
