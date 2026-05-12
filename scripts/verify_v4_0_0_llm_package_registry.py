#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify v4.0 LLM package registry ecosystem evidence.")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--contract", default="enkai/contracts/v4_0_0_llm_package_registry.json")
    parser.add_argument("--input", default="artifacts/readiness/v4_0_0_llm_package_registry.json")
    parser.add_argument("--output", default="artifacts/readiness/v4_0_0_llm_package_registry_verify.json")
    args = parser.parse_args()

    root = Path(args.workspace).resolve()
    contract = read_json(root / args.contract)
    evidence = read_json(root / args.input)
    failures: list[str] = []

    if evidence.get("schema_version") != 1:
        failures.append("evidence schema_version must be 1")
    if evidence.get("contract_version") != contract.get("contract_version"):
        failures.append("contract version mismatch")
    if evidence.get("scope") != contract.get("scope"):
        failures.append("scope mismatch")
    if evidence.get("all_passed") is not True:
        failures.append("evidence all_passed expected True")
    gates = evidence.get("gates", {})
    for gate in contract.get("required_gates", []):
        if gates.get(gate) is not True:
            failures.append(f"required gate failed: {gate}")
    for rel in contract.get("required_artifacts", []):
        if not (root / rel).is_file():
            failures.append(f"missing required artifact: {rel}")
    claims = evidence.get("production_claims", {})
    if claims.get("llm_package_registry_ecosystem_proven") is not True:
        failures.append("llm_package_registry_ecosystem_proven expected True")
    if claims.get("pytorch_core_execution_dependency") is not False:
        failures.append("PyTorch must not be marked as a core execution dependency")
    if claims.get("native_postinstall_hooks_allowed") is not False:
        failures.append("native postinstall hooks must remain forbidden")
    if not evidence.get("manifest_digest") or not evidence.get("lock_digest"):
        failures.append("manifest and lock digests are required")

    result = {
        "schema_version": 1,
        "contract": str((root / args.contract).resolve()),
        "input": str((root / args.input).resolve()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "all_passed": not failures,
        "failures": failures,
    }
    write_json(root / args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
