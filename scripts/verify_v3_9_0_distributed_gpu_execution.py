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
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def evidence_failure(label: str, result: dict[str, Any]) -> str:
    if result.get("error"):
        return str(result["error"])
    status = result.get("status") or "not PASS"
    reason = result.get("reason")
    if reason:
        return f"{label} evidence did not pass ({status}: {reason})"
    return f"{label} evidence did not pass ({status})"


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify v3.9.0 distributed GPU execution evidence.")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--contract", default="enkai/contracts/v3_9_0_distributed_gpu_execution.json")
    parser.add_argument("--input", default="artifacts/readiness/v3_9_0_distributed_gpu_execution.json")
    parser.add_argument("--output", default="artifacts/readiness/v3_9_0_distributed_gpu_execution_verify.json")
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

    two_rank = evidence.get("two_rank", {})
    if two_rank.get("passed") is not True:
        failures.append(evidence_failure("two_rank", two_rank if isinstance(two_rank, dict) else {}))
    payload = two_rank.get("payload", {}) if isinstance(two_rank.get("payload"), dict) else {}
    expected_status = contract["two_rank_requirements"]["required_status"]
    if payload.get("status") != expected_status:
        failures.append(f"two_rank status expected {expected_status}, found {payload.get('status')}")
    if payload.get("status") == expected_status:
        if int(payload.get("world_size", 0)) != int(contract["two_rank_requirements"]["world_size"]):
            failures.append("two_rank world_size mismatch")
        checks = payload.get("checks", {})
        for check in contract["two_rank_requirements"]["required_checks"]:
            if checks.get(check) is not True:
                failures.append(f"two_rank check failed: {check}")
        if len(payload.get("ranks", [])) != 2:
            failures.append("two_rank ranks length expected 2")
    claims = evidence.get("production_claims", {})
    if claims.get("distributed_gpu_execution_proven") is not True:
        failures.append("distributed_gpu_execution_proven expected True")
    if claims.get("claim_without_hardware_evidence") is not False:
        failures.append("claim_without_hardware_evidence must be False")

    result = {
        "schema_version": 1,
        "contract": str((root / args.contract).resolve()),
        "input": str((root / args.input).resolve()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "all_passed": not failures,
        "failures": failures,
    }
    write_json(root / args.output, result)
    print(json.dumps(result, indent=2))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
