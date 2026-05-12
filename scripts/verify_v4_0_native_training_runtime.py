#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--contract", default="enkai/contracts/v4_0_native_training_runtime.json")
    parser.add_argument("--input", default="artifacts/readiness/v4_0_native_training_runtime.json")
    parser.add_argument("--output", default="artifacts/readiness/v4_0_native_training_runtime_verify.json")
    args = parser.parse_args()
    root = Path(args.workspace).resolve()
    contract = json.loads((root / args.contract).read_text(encoding="utf-8-sig"))
    report = json.loads((root / args.input).read_text(encoding="utf-8-sig"))
    failures = []
    if report.get("contract_version") != contract.get("contract_version"):
        failures.append("contract version mismatch")
    if not report.get("all_passed"):
        failures.append("readiness report is not green")
    evidence = report.get("enkai_native_evidence", {})
    for metric in contract.get("required_metrics", []):
        if metric == "fusion_checksum_delta_abs" and evidence.get("matmul_bias_relu", {}).get("checksum_delta_abs") is None:
            failures.append(metric)
        if metric == "fusion_loss_delta_abs" and evidence.get("softmax_cross_entropy", {}).get("loss_delta_abs") is None:
            failures.append(metric)
        if metric.startswith("memory_") and not evidence.get("memory"):
            failures.append(metric)
        if metric.startswith("training_") and not evidence.get("training"):
            failures.append(metric)
        if metric.startswith("benchmark_"):
            name = metric.removeprefix("benchmark_")
            if evidence.get("benchmarks", {}).get(name) is None:
                failures.append(metric)
    payload = {
        "schema_version": 1,
        "contract": str((root / args.contract).resolve()),
        "input": str((root / args.input).resolve()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "all_passed": not failures,
        "failures": failures,
    }
    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
