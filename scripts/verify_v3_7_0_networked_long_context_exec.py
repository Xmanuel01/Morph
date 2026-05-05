#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify networked long-context execution proof.")
    parser.add_argument("--contract", default="enkai/contracts/v3_7_0_networked_long_context_exec.json")
    parser.add_argument("--output", default="artifacts/readiness/v3_7_0_networked_long_context_exec_verify.json")
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
    if int(report.get("world_size", 0)) != int(contract["world_size"]):
        failures.append(f"world_size expected {contract['world_size']}")
    if int(report.get("long_context_contract", {}).get("seq_len", 0)) < int(contract["min_seq_len"]):
        failures.append(f"seq_len expected >= {contract['min_seq_len']}")
    for case_name in ("baseline", "fault_injection"):
        case = report.get(case_name, {})
        if case.get("passed") is not True:
            failures.append(f"{case_name}.passed expected True")
        if case.get("identical_checkpoint_semantics") is not True:
            failures.append(f"{case_name}.identical_checkpoint_semantics expected True")
        if case.get("merged_replay", {}).get("passed") is not True:
            failures.append(f"{case_name}.merged_replay.passed expected True")
        throughput = case.get("throughput", {})
        for gate in (
            "combined_train_gate_passed",
            "combined_eval_gate_passed",
            "checkpoint_merge_gate_passed",
            "networked_gradient_bytes_gate_passed",
        ):
            if throughput.get(gate) is not True:
                failures.append(f"{case_name}.throughput.{gate} expected True")
        for rank_report in case.get("rank_reports", []):
            if rank_report.get("networked_gradient_exchange") is not True:
                failures.append(f"{case_name}.rank[{rank_report.get('rank')}].networked_gradient_exchange expected True")
            if int(rank_report.get("networked_gradient_bytes") or 0) <= 0:
                failures.append(f"{case_name}.rank[{rank_report.get('rank')}].networked_gradient_bytes expected > 0")

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
