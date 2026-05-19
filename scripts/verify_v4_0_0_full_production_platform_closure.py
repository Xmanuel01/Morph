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


def nested_get(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify v4.0.0 full production platform umbrella closure.")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--contract", default="enkai/contracts/v4_0_0_full_production_platform_closure.json")
    parser.add_argument("--output", default="artifacts/readiness/v4_0_0_full_production_platform_closure.json")
    args = parser.parse_args()

    root = Path(args.workspace).resolve()
    contract = read_json(root / args.contract)
    failures: list[str] = []
    surfaces: list[dict[str, Any]] = []

    for surface in contract["required_surfaces"]:
        artifact_rel = surface["artifact"]
        artifact_path = root / artifact_rel
        has_claim = bool(surface.get("claim"))
        record: dict[str, Any] = {
            "id": surface["id"],
            "artifact": artifact_rel,
            "exists": artifact_path.exists(),
            "all_passed": False,
            "claim_required": has_claim,
            "claim_passed": not has_claim,
            "failures": [],
        }
        if not artifact_path.exists():
            failures.append(f"{surface['id']}: missing artifact {artifact_rel}")
            surfaces.append(record)
            continue
        payload = read_json(artifact_path)
        record["all_passed"] = payload.get("all_passed") is True
        if has_claim:
            claim_path = f"production_claims.{surface['claim']}"
            record["claim_passed"] = nested_get(payload, claim_path) is True
        else:
            claim_path = None
        payload_failures = payload.get("failures")
        if isinstance(payload_failures, list):
            record["failures"] = payload_failures[:20]
        if surface.get("requires_all_passed") and not record["all_passed"]:
            failures.append(f"{surface['id']}: artifact is not green")
        if has_claim and not record["claim_passed"]:
            failures.append(f"{surface['id']}: production claim {surface['claim']} is not true")
        if nested_get(payload, "production_claims.claim_without_hardware_evidence") is True:
            failures.append(f"{surface['id']}: claim_without_hardware_evidence is true")
        surfaces.append(record)

    blocked_surfaces = [surface for surface in surfaces if not surface["all_passed"] or not surface["claim_passed"]]
    open_blockers = {
        surface["id"]: surface.get("failures", [])
        for surface in blocked_surfaces
    }

    result = {
        "schema_version": 1,
        "contract_version": contract["contract_version"],
        "scope": contract["scope"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": str((root / args.contract).resolve()),
        "surfaces": surfaces,
        "blocked_surfaces": [surface["id"] for surface in blocked_surfaces],
        "open_blockers": open_blockers,
        "closure_policy": {
            "broad_platform_claim_allowed": not failures,
            "partial_surface_override_allowed": False,
            "hardware_or_external_evidence_required": True,
        },
        "production_claims": {
            "full_production_platform_proven": not failures,
            "claim_without_all_surfaces_green": False,
            "blocked_partials_remaining": bool(failures),
        },
        "all_passed": not failures,
        "failures": failures,
    }
    write_json(root / args.output, result)
    print(json.dumps({"all_passed": result["all_passed"], "failures": failures, "output": args.output}, indent=2))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
