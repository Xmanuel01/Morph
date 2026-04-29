#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the readiness_and_validation surface is complete."
    )
    parser.add_argument(
        "--contract",
        default=str(
            Path(__file__).resolve().parents[1]
            / "enkai"
            / "contracts"
            / "selfhost_readiness_validation_v3_3_0.json"
        ),
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require_all_passed(payload: dict, label: str) -> None:
    if payload.get("all_passed") is not True:
        raise AssertionError(f"{label} is not all_passed")


def main() -> int:
    args = parse_args()
    contract_path = Path(args.contract).resolve()
    root = contract_path.parents[2]
    output_path = Path(args.output).resolve()
    contract = load_json(contract_path)
    required_child_status = contract["required_child_status"]

    rv_path = root / "artifacts" / "readiness" / "strict_selfhost_readiness_validation_slices.json"
    strict_path = root / "artifacts" / "readiness" / "strict_selfhost.json"
    blockers_path = root / "artifacts" / "readiness" / "strict_selfhost_blockers.json"
    suite_path = root / "artifacts" / "readiness" / "strict_selfhost_validation_suite_dispatch_slice.json"
    release_path = root / "artifacts" / "readiness" / "strict_selfhost_release_evidence_collection_slice.json"

    try:
        rv_payload = load_json(rv_path)
        surfaces = rv_payload.get("surfaces", [])
        if not surfaces:
            raise AssertionError("readiness/validation surfaces report is empty")
        bad = [
            surface["id"]
            for surface in surfaces
            if surface.get("status") != required_child_status
        ]
        if bad:
            raise AssertionError(
                f"surfaces not in status {required_child_status}: {', '.join(bad)}"
            )

        strict_payload = load_json(strict_path)
        blockers_payload = load_json(blockers_path)
        suite_payload = load_json(suite_path)
        release_payload = load_json(release_path)

        require_all_passed(strict_payload, "strict_selfhost")
        require_all_passed(blockers_payload, "strict_selfhost_blockers")
        require_all_passed(suite_payload, "validation_suite_dispatch")
        require_all_passed(release_payload, "release_evidence_collection")

        payload = {
            "all_passed": True,
            "contract": str(contract_path),
            "readiness_validation_slices": str(rv_path),
            "verified_checks": contract["required_checks"],
            "child_status": required_child_status,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": "ok", "output": str(output_path)}, separators=(",", ":")))
        return 0
    except Exception as exc:  # noqa: BLE001
        payload = {"all_passed": False, "contract": str(contract_path), "error": str(exc)}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": "error", "output": str(output_path), "error": str(exc)}, separators=(",", ":")))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
