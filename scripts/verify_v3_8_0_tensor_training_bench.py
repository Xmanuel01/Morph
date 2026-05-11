#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify v3.8.0 tensor training benchmark evidence.")
    parser.add_argument("--input", default="artifacts/readiness/v3_8_0_tensor_training_bench.json")
    parser.add_argument("--output", default="artifacts/readiness/v3_8_0_tensor_training_bench_verify.json")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    evidence = read_json(input_path)
    failures: list[str] = []

    if evidence.get("schema_version") != 1:
        failures.append("schema_version must be 1")
    if evidence.get("version") != "v3.8.0":
        failures.append("version must be v3.8.0")
    if evidence.get("scope") != "first_party_tensor_training_benchmark":
        failures.append("scope mismatch")
    if not evidence.get("passed"):
        failures.append("evidence did not pass")

    cases = evidence.get("cases", [])
    names = {case.get("name") for case in cases}
    for required in ["training_autodiff_optimizer", "eval_attention", "memory_limit_oom"]:
        if required not in names:
            failures.append(f"missing case {required}")

    for case in cases:
        name = case.get("name")
        if not case.get("passed"):
            failures.append(f"case {name} failed")
        if name != "memory_limit_oom":
            metrics = case.get("metrics", {})
            if case.get("estimated_ops_per_sec", 0) <= 0:
                failures.append(f"case {name} missing positive throughput")
            if metrics.get("peak_memory_bytes", 0) <= 0:
                failures.append(f"case {name} missing peak memory evidence")
            if case.get("run", {}).get("elapsed_ms", 0) <= 0:
                failures.append(f"case {name} missing elapsed time")
        else:
            stderr = case.get("run", {}).get("stderr_tail", "")
            if "would exceed tensor memory limit" not in stderr:
                failures.append("OOM case did not include deterministic memory limit error")

    python_ref = evidence.get("python_reference", {})
    if python_ref.get("iterations_per_sec", 0) <= 0:
        failures.append("missing positive Python reference timing")

    result = {
        "schema_version": 1,
        "input": str(input_path),
        "passed": not failures,
        "failures": failures,
    }
    write_json(output_path, result)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
