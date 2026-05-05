#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify networked v3.7.0 throughput regression gates.")
    parser.add_argument("--contract", default="enkai/contracts/v3_7_0_networked_throughput_regressions.json")
    parser.add_argument("--output", default="artifacts/readiness/v3_7_0_networked_throughput_regressions.json")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def assert_case_gates(name: str, case: dict[str, Any], failures: list[str]) -> dict[str, Any]:
    throughput = case.get("throughput", {})
    summary = {
        "name": name,
        "passed": bool(
            case.get("passed") is True
            and throughput.get("combined_train_gate_passed") is True
            and throughput.get("combined_eval_gate_passed") is True
            and throughput.get("checkpoint_merge_gate_passed") is True
            and throughput.get("networked_gradient_bytes_gate_passed") is True
        ),
        "combined_train_tokens_per_sec": throughput.get("combined_train_tokens_per_sec"),
        "combined_eval_tokens_per_sec": throughput.get("combined_eval_tokens_per_sec"),
        "checkpoint_merge_bytes_per_sec": throughput.get("checkpoint_merge_bytes_per_sec"),
        "networked_gradient_bytes": throughput.get("networked_gradient_bytes"),
    }
    if not summary["passed"]:
        failures.append(f"{name} throughput gates did not pass")
    if int(throughput.get("networked_gradient_bytes", 0)) <= 0:
        failures.append(f"{name} did not record networked gradient bytes")
    return summary


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    contract_path = (repo_root / args.contract).resolve()
    contract = read_json(contract_path)
    failures: list[str] = []
    summaries: list[dict[str, Any]] = []

    for name, rel_path in contract["required_reports"].items():
        report = read_json((repo_root / rel_path).resolve())
        if report.get("all_passed") is not True:
            failures.append(f"{name} report all_passed expected True")
        for case_name in ("baseline", "fault_injection"):
            summaries.append(assert_case_gates(f"{name}.{case_name}", report.get(case_name, {}), failures))

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
        "cases": summaries,
        "failures": failures,
    }
    output_path = (repo_root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return 0 if output["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
