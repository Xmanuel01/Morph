#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the world_size=4 networked rendezvous execution proof.")
    parser.add_argument("--contract", default="enkai/contracts/v3_7_0_networked_rendezvous_scale_exec.json")
    parser.add_argument("--output", default="artifacts/readiness/v3_7_0_networked_rendezvous_scale_exec_verify.json")
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
        failures.append(f"world_size expected {contract['world_size']}, got {report.get('world_size')!r}")
    if report.get("execution_mode") != "networked-sync-preview":
        failures.append("execution_mode expected networked-sync-preview")
    if report.get("baseline", {}).get("passed") is not True:
        failures.append("baseline.passed expected True")
    if report.get("baseline", {}).get("identical_checkpoint_semantics") is not True:
        failures.append("baseline identical checkpoint semantics expected True")
    if report.get("baseline", {}).get("merged_replay", {}).get("passed") is not True:
        failures.append("baseline merged checkpoint replay expected True")
    if report.get("baseline", {}).get("throughput", {}).get("combined_train_gate_passed") is not True:
        failures.append("baseline combined train throughput gate expected True")
    if report.get("baseline", {}).get("throughput", {}).get("combined_eval_gate_passed") is not True:
        failures.append("baseline combined eval throughput gate expected True")
    if report.get("baseline", {}).get("throughput", {}).get("checkpoint_merge_gate_passed") is not True:
        failures.append("baseline checkpoint merge throughput gate expected True")
    if report.get("baseline", {}).get("throughput", {}).get("networked_gradient_bytes_gate_passed") is not True:
        failures.append("baseline networked gradient byte gate expected True")
    if report.get("fault_injection", {}).get("passed") is not True:
        failures.append("fault_injection.passed expected True")
    if int(report.get("fault_injection", {}).get("total_retry_count", 0)) < int(contract.get("min_retry_count", 1)):
        failures.append("fault_injection.total_retry_count below minimum")
    if report.get("fault_injection", {}).get("fault_injection_observed") is not True:
        failures.append("fault_injection_observed expected True")
    if report.get("fault_injection", {}).get("identical_checkpoint_semantics") is not True:
        failures.append("fault injection checkpoint semantics expected True")
    if report.get("fault_injection", {}).get("merged_replay", {}).get("passed") is not True:
        failures.append("fault injection merged checkpoint replay expected True")
    for case_name in ("baseline", "fault_injection"):
        ranks = report.get(case_name, {}).get("rank_reports", [])
        if len(ranks) != int(contract["world_size"]):
            failures.append(f"{case_name}.rank_reports expected {contract['world_size']} entries")
        for rank_report in ranks:
            if rank_report.get("passed") is not True:
                failures.append(f"{case_name}.rank[{rank_report.get('rank')}].passed expected True")
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
