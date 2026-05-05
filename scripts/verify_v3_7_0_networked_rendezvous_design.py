#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the v3.7.0 networked rendezvous design freeze.")
    parser.add_argument("--contract", default="enkai/contracts/v3_7_0_networked_rendezvous_design.json")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    contract_path = (repo_root / args.contract).resolve()
    contract = read_json(contract_path)
    suite = read_json((repo_root / contract["suite"]).resolve())
    sync_report = read_json((repo_root / contract["required_sync_report"]).resolve())
    design = suite.get("networked_rendezvous_design", {})
    failures: list[str] = []

    if not sync_report.get("all_passed"):
        failures.append("required synchronized preview report is not green")
    if design.get("execution_frozen") is not True:
        failures.append("networked_rendezvous_design.execution_frozen expected True")
    if design.get("process_model") != "multi-process":
        failures.append("networked_rendezvous_design.process_model expected 'multi-process'")
    if design.get("topology") != "multi-node":
        failures.append("networked_rendezvous_design.topology expected 'multi-node'")
    if int(design.get("world_size", 0)) < 2:
        failures.append("networked_rendezvous_design.world_size expected >= 2")
    rendezvous = str(design.get("rendezvous", ""))
    if not rendezvous.startswith("tcp://"):
        failures.append("networked_rendezvous_design.rendezvous expected tcp:// scheme")

    for rel_path, snippets in contract["required_text_snippets"].items():
        text = (repo_root / rel_path).read_text(encoding="utf-8-sig")
        for snippet in snippets:
            if snippet not in text:
                failures.append(f"missing snippet in {rel_path}: {snippet}")

    output = {
        "schema_version": 1,
        "contract": str(contract_path),
        "suite": str((repo_root / contract["suite"]).resolve()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "all_passed": not failures,
        "execution_widened": False,
        "design": design,
        "failures": failures,
    }
    output_path = (repo_root / (args.output or contract["output_report"])).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return 0 if output["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
