#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the synchronized distributed-runtime v3.7.0 tranche.")
    parser.add_argument("--contract", default="enkai/contracts/v3_7_0_distributed_runtime_sync.json")
    parser.add_argument("--output", default="artifacts/readiness/v3_7_0_distributed_runtime_sync_verify.json")
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

    if not report.get("all_passed"):
        failures.append("all_passed expected True, got False")
    if report.get("execution_mode") != "synchronized-grad-preview":
        failures.append(f"execution_mode expected 'synchronized-grad-preview', got {report.get('execution_mode')!r}")
    if report.get("synchronized_gradients") is not True:
        failures.append("synchronized_gradients expected True")
    shape_envelope = report.get("shape_envelope", {})
    if shape_envelope.get("all_shapes_passed") is not True:
        failures.append("shape_envelope.all_shapes_passed expected True")
    cases = shape_envelope.get("cases", [])
    if not cases:
        failures.append("shape_envelope.cases expected non-empty")
    world_size = int(report.get("world_size", 0))
    for case in cases:
        if case.get("identical_rank_checkpoints") is not True:
            failures.append(f"{case.get('name')} identical_rank_checkpoints expected True")
        if not case.get("merged_replay", {}).get("passed"):
            failures.append(f"{case.get('name')} merged_replay.passed expected True")
        if len(case.get("rank_reports", [])) != world_size:
            failures.append(f"{case.get('name')} rank report count does not match world_size")
    if report.get("distributed_throughput", {}).get("all_gates_passed") is not True:
        failures.append("distributed_throughput.all_gates_passed expected True")

    for rel_path, snippets in contract["required_text_snippets"].items():
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
