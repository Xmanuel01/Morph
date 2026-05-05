#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify adversarial AI runtime input corruption coverage.")
    parser.add_argument("--contract", default="enkai/contracts/v3_7_0_ai_runtime_adversarial_inputs.json")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    contract_path = (repo_root / args.contract).resolve()
    contract = read_json(contract_path)
    report = read_json((repo_root / contract["required_report"]).resolve())
    failures: list[str] = []
    if report.get("all_passed") is not True:
        failures.append("adversarial report is not green")
    if report.get("base_train_passed") is not True:
        failures.append("base train precondition did not pass")
    for case_name, case in report.get("cases", {}).items():
        if case.get("passed") is not True:
            failures.append(f"case {case_name} did not pass")
    for rel_path, snippets in contract.get("required_text_snippets", {}).items():
        text = (repo_root / rel_path).read_text(encoding="utf-8-sig")
        for snippet in snippets:
            if snippet not in text:
                failures.append(f"missing snippet in {rel_path}: {snippet}")
    output = {
        "schema_version": 1,
        "contract": str(contract_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "all_passed": not failures,
        "failures": failures,
    }
    output_path = (repo_root / (args.output or contract["output_report"])).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return 0 if output["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
