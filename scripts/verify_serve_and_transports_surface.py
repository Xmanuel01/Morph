#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the serve_and_transports surface is complete."
    )
    parser.add_argument(
        "--contract",
        default=str(
            Path(__file__).resolve().parents[1]
            / "enkai"
            / "contracts"
            / "selfhost_serve_and_transports_v3_3_0.json"
        ),
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    contract_path = Path(args.contract).resolve()
    root = contract_path.parents[2]
    output_path = Path(args.output).resolve()

    try:
        transport = load_json(root / "artifacts" / "readiness" / "strict_selfhost_systems_transport_slice.json")
        strict_selfhost = load_json(root / "artifacts" / "readiness" / "strict_selfhost.json")
        if transport.get("all_passed") is not True:
            raise AssertionError("strict_selfhost_systems_transport_slice is not all_passed")
        if strict_selfhost.get("all_passed") is not True:
            raise AssertionError("strict_selfhost is not all_passed")
        payload = {
            "all_passed": True,
            "contract": str(contract_path),
            "verified_checks": ["strict_selfhost_systems_transport_slice", "strict_selfhost"],
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
