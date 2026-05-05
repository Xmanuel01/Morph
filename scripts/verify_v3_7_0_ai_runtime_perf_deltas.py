#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the v3.7.0 performance-delta tranche.")
    parser.add_argument("--contract", default="enkai/contracts/v3_7_0_ai_runtime_perf_deltas.json")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def get_path(obj: Any, dotted_path: str) -> Any:
    current = obj
    for part in dotted_path.split('.'):
        if isinstance(current, dict):
            if part not in current:
                raise KeyError(dotted_path)
            current = current[part]
        elif isinstance(current, list):
            current = current[int(part)]
        else:
            raise KeyError(dotted_path)
    return current


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    contract_path = (repo_root / args.contract).resolve()
    contract = read_json(contract_path)
    report_path = (repo_root / contract["required_report"]).resolve()
    report = read_json(report_path)
    output_path = (repo_root / (args.output or contract["output_report"])).resolve()

    failures: list[str] = []
    for dotted_path, expected in contract["required_status_fields"].items():
        try:
            actual = get_path(report, dotted_path)
        except KeyError:
            failures.append(f"missing {dotted_path}")
            continue
        if actual != expected:
            failures.append(f"{dotted_path} expected {expected!r}, got {actual!r}")

    try:
        python_speedup = float(get_path(report, "benchmark.comparisons.enkai_vs_python_speedup"))
        cpu_speedup = float(get_path(report, "benchmark.comparisons.enkai_vs_cpu_speedup"))
        native_ratio = float(get_path(report, "benchmark.comparisons.enkai_vs_native_ratio"))
        training_reduction = float(get_path(report, "benchmark.comparisons.training_time_reduction_vs_python_pct"))
    except (KeyError, TypeError, ValueError) as err:
        failures.append(f"invalid benchmark comparison payload: {err}")
        python_speedup = cpu_speedup = native_ratio = training_reduction = 0.0

    if python_speedup <= 1.0:
        failures.append(f"enkai_vs_python_speedup must be > 1.0, got {python_speedup}")
    if cpu_speedup <= 1.0:
        failures.append(f"enkai_vs_cpu_speedup must be > 1.0, got {cpu_speedup}")

    native_requested = report["benchmark"]["native_comparison"]["requested_backend"]
    native_executed = report["benchmark"]["native_comparison"]["executed_backend"]
    if native_requested != "native":
        failures.append(f"native comparison requested_backend must be 'native', got {native_requested!r}")
    if native_executed not in {"native", "enkai_accel"}:
        failures.append(f"native comparison executed_backend must be 'native' or 'enkai_accel', got {native_executed!r}")

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
        "summary": {
            "enkai_vs_python_speedup": python_speedup,
            "enkai_vs_cpu_speedup": cpu_speedup,
            "enkai_vs_native_ratio": native_ratio,
            "training_time_reduction_vs_python_pct": training_reduction,
            "peak_memory_regression_passed": report["benchmark"]["regression_gates"]["peak_memory_regression_passed"],
            "checkpoint_overhead_regression_passed": report["benchmark"]["regression_gates"]["checkpoint_overhead_regression_passed"],
            "native_requested_backend": native_requested,
            "native_executed_backend": native_executed,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return 0 if output["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
