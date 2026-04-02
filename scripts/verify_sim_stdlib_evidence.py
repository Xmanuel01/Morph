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
        fail("simulation stdlib evidence summary schema_version must be 1")
    if run_report.get("command") != "sim.run":
        fail("simulation stdlib run evidence must come from sim.run")
    if profile_report.get("case") != summary.get("profile_case"):
        fail("simulation stdlib profile evidence case id mismatch")
    if profile_report.get("status") != "ok":
        fail("simulation stdlib profile evidence status must be ok")
    if run_report.get("result") != summary.get("expected_result"):
        fail("simulation stdlib run result does not match expected result")

    counters = profile_report.get("counters")
    if not isinstance(counters, dict):
        fail("simulation stdlib profile counters must be an object")
    if int(counters.get("ffi_calls", 0)) != 0:
        fail("simulation stdlib profile ffi_calls must be 0")
    if int(counters.get("native_function_calls", 0)) <= 0:
        fail("simulation stdlib profile native_function_calls must be > 0")
    if int(counters.get("opcode_dispatch", 0)) <= 0:
        fail("simulation stdlib profile opcode_dispatch must be > 0")
    if int(counters.get("object_allocations", 0)) <= 0:
        fail("simulation stdlib profile object_allocations must be > 0")

    timing = profile_report.get("timing_ms")
    if not isinstance(timing, dict):
        fail("simulation stdlib profile timing_ms must be an object")
    if float(timing.get("native_calls", 0.0)) != 0.0:
        fail("simulation stdlib profile native_calls timing must be 0")
    if float(timing.get("vm_exec", 0.0)) <= 0.0:
        fail("simulation stdlib profile vm_exec timing must be > 0")

    verification = {
        "schema_version": 1,
        "summary": str(summary_path),
        "run_report": str(run_report_path),
        "profile_report": str(profile_report_path),
        "all_passed": True,
        "opcode_dispatch": counters.get("opcode_dispatch", 0),
        "object_allocations": counters.get("object_allocations", 0),
    }
    write_json(output_path, verification)
    return 0


if __name__ == "__main__":
    sys.exit(main())
