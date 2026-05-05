#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify adversarial peer behavior coverage for networked rendezvous.")
    parser.add_argument("--contract", default="enkai/contracts/v3_7_0_networked_rendezvous_peer_adversarial.json")
    parser.add_argument("--output", default="artifacts/readiness/v3_7_0_networked_rendezvous_peer_adversarial_verify.json")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    contract_path = (repo_root / args.contract).resolve()
    contract = read_json(contract_path)
    report_path = (repo_root / contract["output_report"]).resolve()
    report = read_json(report_path)
    failures: list[str] = []

    if report.get("all_passed") is not True:
        failures.append("all_passed expected True")
    observed_cases = {case.get("name"): case for case in report.get("cases", [])}
    for required in contract["required_cases"]:
        case = observed_cases.get(required)
        if case is None:
            failures.append(f"missing adversarial case {required}")
        elif case.get("passed") is not True:
            failures.append(f"adversarial case {required} did not pass")
        elif case.get("observed_expected_error") is not True:
            failures.append(f"adversarial case {required} did not observe expected error")

    for rel_path, snippets in contract.get("required_text_snippets", {}).items():
        text = (repo_root / rel_path).read_text(encoding="utf-8-sig")
        for snippet in snippets:
            if snippet not in text:
                failures.append(f"missing snippet in {rel_path}: {snippet}")

    output = {
        "schema_version": 1,
        "contract": str(contract_path),
        "report": str(report_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "all_passed": not failures,
        "failures": failures,
    }
    output_path = (repo_root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return 0 if output["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
