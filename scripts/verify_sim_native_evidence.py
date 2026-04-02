#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def fail(message: str) -> None:
    raise SystemExit(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--run-report", required=True)
    parser.add_argument("--profile-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    summary_path = Path(args.summary)
    run_report_path = Path(args.run_report)
    profile_report_path = Path(args.profile_report)
    output_path = Path(args.output)

    summary = read_json(summary_path)
    run_report = read_json(run_report_path)
    profile_report = read_json(profile_report_path)

    if summary.get("schema_version") != 1:
        fail("simulation native evidence summary schema_version must be 1")
    if run_report.get("command") != "sim.run":
        fail("simulation native run evidence must come from sim.run")
    if profile_report.get("case") != summary.get("profile_case"):
        fail("simulation native profile evidence case id mismatch")
    if profile_report.get("status") != "ok":
        fail("simulation native profile evidence status must be ok")

    result = run_report.get("result")
    if result != [summary.get("expected_value"), summary.get("expected_buffer_len")]:
        fail("simulation native run result does not match expected tuple")

    counters = profile_report.get("counters")
    if not isinstance(counters, dict):
        fail("simulation native profile counters must be an object")
    ffi_calls = int(counters.get("ffi_calls", 0))
    native_function_calls = int(counters.get("native_function_calls", 0))
    marshal_in_bytes = int(counters.get("marshal_in_bytes", 0))
    if ffi_calls < 4:
        fail("simulation native profile ffi_calls must be >= 4")
    if native_function_calls < 4:
        fail("simulation native profile native_function_calls must be >= 4")
    if marshal_in_bytes < 3:
        fail("simulation native profile marshal_in_bytes must be >= 3")

    timing = profile_report.get("timing_ms")
    if not isinstance(timing, dict):
        fail("simulation native profile timing_ms must be an object")
    if float(timing.get("native_calls", 0.0)) <= 0.0:
        fail("simulation native profile native_calls timing must be > 0")

    verification = {
        "schema_version": 1,
        "summary": str(summary_path),
        "run_report": str(run_report_path),
        "profile_report": str(profile_report_path),
        "all_passed": True,
        "ffi_calls": ffi_calls,
        "native_function_calls": native_function_calls,
        "marshal_in_bytes": marshal_in_bytes,
    }
    write_json(output_path, verification)
    return 0


if __name__ == "__main__":
    sys.exit(main())
