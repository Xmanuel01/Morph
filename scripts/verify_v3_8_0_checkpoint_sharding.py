#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify v3.8.0 checkpoint sharding proof.")
    parser.add_argument("--contract", default="enkai/contracts/v3_8_0_checkpoint_sharding.json")
    parser.add_argument("--output", default="artifacts/readiness/v3_8_0_checkpoint_sharding_verify.json")
    return parser.parse_args()

def main() -> int:
    args = parse_args(); root = Path(__file__).resolve().parents[1]
    contract = read_json(root / args.contract); report = read_json(root / contract["output_report"])
    failures = []
    if report.get("all_passed") is not True: failures.append("all_passed expected True")
    if int(report.get("world_size", 0)) != int(contract["world_size"]): failures.append("world_size mismatch")
    if report.get("identical_checkpoint_semantics") is not True: failures.append("identical_checkpoint_semantics expected True")
    if report.get("merged_replay", {}).get("passed") is not True: failures.append("merged_replay.passed expected True")
    if report.get("corruption_case", {}).get("passed") is not True: failures.append("corruption_case.passed expected True")
    if report.get("throughput_gates", {}).get("shard_copy_gate_passed") is not True: failures.append("shard_copy_gate_passed expected True")
    for rel, snippets in contract["required_text_snippets"].items():
        text = (root / rel).read_text(encoding="utf-8-sig")
        for snippet in snippets:
            if snippet not in text: failures.append(f"missing snippet in {rel}: {snippet}")
    output = {"schema_version": 1, "contract": str((root / args.contract).resolve()), "report": str((root / contract["output_report"]).resolve()), "generated_at_utc": datetime.now(timezone.utc).isoformat(), "all_passed": not failures, "failures": failures}
    out = root / args.output; out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return 0 if output["all_passed"] else 1
if __name__ == "__main__": raise SystemExit(main())
