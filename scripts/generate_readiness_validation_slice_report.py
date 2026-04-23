#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a readiness/validation self-host tranche report from the contract."
    )
    parser.add_argument("--contract", required=True, help="Path to the readiness/validation slice contract JSON")
    parser.add_argument("--output", required=True, help="Path to write the generated report JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract_path = Path(args.contract).resolve()
    output_path = Path(args.output).resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    surfaces = sorted(
        contract.get("surfaces", []),
        key=lambda item: (item.get("priority", 9999), item.get("id", "")),
    )
    status_counts = Counter(item.get("status", "unknown") for item in surfaces)
    next_peelable_surface = None
    for surface in surfaces:
        if surface.get("status") not in {"contract_driven", "complete"}:
            next_peelable_surface = surface.get("id")
            break
    if next_peelable_surface is None and surfaces:
        next_peelable_surface = surfaces[0].get("id")
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": str(contract_path),
        "profile": contract.get("profile"),
        "version_line": contract.get("version_line"),
        "summary": {
            "surface_count": len(surfaces),
            "status_counts": dict(status_counts),
            "next_peelable_surface": next_peelable_surface,
        },
        "surfaces": surfaces,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(output_path),
                "next_peelable_surface": report["summary"]["next_peelable_surface"],
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
