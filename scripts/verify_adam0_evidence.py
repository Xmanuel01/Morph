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
        fail("adam0 smoke summary schema_version must be 1")
    if run_report.get("command") != "sim.run":
        fail("adam0 run evidence must come from sim.run")
    if profile_report.get("case") != summary.get("profile_case"):
        fail("adam0 profile evidence case id mismatch")
    if profile_report.get("status") != "ok":
        fail("adam0 profile evidence status must be ok")

    result = run_report.get("result")
    if not isinstance(result, dict):
        fail("adam0 run result must be an object snapshot")
    entities = result.get("entities")
    if not isinstance(entities, list):
        fail("adam0 snapshot entities must be a list")
    if len(entities) != int(summary.get("target_agents", 0)):
        fail("adam0 snapshot entity count mismatch")
    if int(result.get("agent_count", -1)) != int(summary.get("target_agents", 0)):
        fail("adam0 snapshot agent_count mismatch")
    if int(result.get("shard_count", -1)) != int(summary.get("target_shards", 0)):
        fail("adam0 snapshot shard_count mismatch")
    if int(result.get("joined_total", -1)) != int(summary.get("target_joined_total", 0)):
        fail("adam0 snapshot joined_total mismatch")
    if int(result.get("report_total", 0)) <= 0:
        fail("adam0 snapshot report_total must be > 0")
    if int(result.get("sparse_edges", 0)) <= 0:
        fail("adam0 snapshot sparse_edges must be > 0")
    if int(result.get("feature_nnz", 0)) <= 0:
        fail("adam0 snapshot feature_nnz must be > 0")
    if not isinstance(result.get("queue"), list) or len(result.get("queue")) != 0:
        fail("adam0 snapshot queue must be fully drained")
    if not isinstance(result.get("log"), list) or len(result.get("log")) <= 0:
        fail("adam0 snapshot log must contain dispatched events")

    counters = profile_report.get("counters")
    if not isinstance(counters, dict):
        fail("adam0 profile counters must be an object")
    if summary.get("require_native_accel") is True and int(counters.get("ffi_calls", 0)) <= 0:
        fail("adam0 profile ffi_calls must be > 0 when native accel is required")
    if int(counters.get("native_function_calls", 0)) <= 0:
        fail("adam0 profile native_function_calls must be > 0")
    if int(counters.get("opcode_dispatch", 0)) <= 0:
        fail("adam0 profile opcode_dispatch must be > 0")
    if int(counters.get("sim_coroutines_spawned", 0)) <= 0:
        fail("adam0 profile sim_coroutines_spawned must be > 0")
    if int(counters.get("sim_coroutine_emits", 0)) <= 0:
        fail("adam0 profile sim_coroutine_emits must be > 0")

    timing = profile_report.get("timing_ms")
    if not isinstance(timing, dict):
        fail("adam0 profile timing_ms must be an object")
    if summary.get("require_native_accel") is True and float(timing.get("native_calls", 0.0)) <= 0.0:
        fail("adam0 profile native_calls timing must be > 0 when native accel is required")
    if float(timing.get("vm_exec", 0.0)) <= 0.0:
        fail("adam0 profile vm_exec timing must be > 0")

    verification = {
        "schema_version": 1,
        "summary": str(summary_path),
        "run_report": str(run_report_path),
        "profile_report": str(profile_report_path),
        "all_passed": True,
        "entity_count": len(entities),
        "log_count": len(result.get("log", [])),
        "ffi_calls": counters.get("ffi_calls", 0),
        "sim_coroutines_spawned": counters.get("sim_coroutines_spawned", 0),
        "sim_coroutine_emits": counters.get("sim_coroutine_emits", 0),
    }
    write_json(output_path, verification)
    return 0


if __name__ == "__main__":
    sys.exit(main())
