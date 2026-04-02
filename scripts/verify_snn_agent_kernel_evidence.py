#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def fail(message: str) -> None:
    raise SystemExit(message)


def expect_close(actual, expected, *, label: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-9):
        fail(f"{label} mismatch: expected {expected}, got {actual}")


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
        fail("snn agent kernel summary schema_version must be 1")
    if run_report.get("command") != "sim.run":
        fail("snn agent kernel run evidence must come from sim.run")
    if profile_report.get("case") != summary.get("profile_case"):
        fail("snn agent kernel profile case mismatch")
    if profile_report.get("status") != "ok":
        fail("snn agent kernel profile status must be ok")

    result = run_report.get("result")
    if not isinstance(result, dict):
        fail("snn agent kernel run result must be an object")

    expected = summary.get("expected")
    if not isinstance(expected, dict):
        fail("snn agent kernel expected summary must be an object")

    for key in ("nearest", "occupancy", "neighbors", "sense", "action", "rand_match", "spikes_0", "spikes_1", "spikes_2", "synapse_nnz"):
        if result.get(key) != expected.get(key):
            fail(f"snn agent kernel result field {key} mismatch")

    expect_close(result.get("reward"), expected.get("reward"), label="reward")

    potentials = result.get("potentials")
    if not isinstance(potentials, list) or len(potentials) != int(expected.get("potential_len", -1)):
        fail("snn agent kernel potentials length mismatch")

    rand_float = result.get("rand_float")
    if not isinstance(rand_float, (int, float)) or not (0.0 <= float(rand_float) < 1.0):
        fail("snn agent kernel rand_float must be within [0, 1)")

    state = result.get("agent_state")
    if not isinstance(state, dict):
        fail("snn agent kernel agent_state must be an object")
    body = state.get("body")
    memory = state.get("memory")
    position = state.get("position")
    if not isinstance(body, dict) or body.get("kind") != expected.get("agent_kind"):
        fail("snn agent kernel agent body mismatch")
    if not isinstance(memory, dict) or memory.get("epoch") != expected.get("agent_epoch"):
        fail("snn agent kernel agent memory mismatch")
    if not isinstance(position, dict):
        fail("snn agent kernel agent position must be an object")
    expect_close(position.get("x"), expected["agent_position"]["x"], label="agent position x")
    expect_close(position.get("y"), expected["agent_position"]["y"], label="agent position y")
    expect_close(state.get("reward"), expected.get("agent_reward_after_take"), label="agent reward after take")
    if int(state.get("actions_pending", -1)) != int(expected.get("pending_actions", -1)):
        fail("snn agent kernel pending actions mismatch")
    if int(state.get("senses_pending", -1)) != int(expected.get("pending_senses", -1)):
        fail("snn agent kernel pending senses mismatch")

    counters = profile_report.get("counters")
    if not isinstance(counters, dict):
        fail("snn agent kernel profile counters must be an object")
    if summary.get("require_native_accel") is True and int(counters.get("ffi_calls", 0)) <= 0:
        fail("snn agent kernel ffi_calls must be > 0 when native accel is required")
    if int(counters.get("native_function_calls", 0)) <= 0:
        fail("snn agent kernel native_function_calls must be > 0")
    if int(counters.get("opcode_dispatch", 0)) <= 0:
        fail("snn agent kernel opcode_dispatch must be > 0")
    if int(counters.get("object_allocations", 0)) <= 0:
        fail("snn agent kernel object_allocations must be > 0")

    timing = profile_report.get("timing_ms")
    if not isinstance(timing, dict):
        fail("snn agent kernel timing_ms must be an object")
    if summary.get("require_native_accel") is True and float(timing.get("native_calls", 0.0)) <= 0.0:
        fail("snn agent kernel native_calls timing must be > 0")
    if float(timing.get("vm_exec", 0.0)) <= 0.0:
        fail("snn agent kernel vm_exec timing must be > 0")

    verification = {
        "schema_version": 1,
        "summary": str(summary_path),
        "run_report": str(run_report_path),
        "profile_report": str(profile_report_path),
        "all_passed": True,
        "ffi_calls": counters.get("ffi_calls", 0),
        "native_function_calls": counters.get("native_function_calls", 0),
        "opcode_dispatch": counters.get("opcode_dispatch", 0),
    }
    write_json(output_path, verification)
    return 0


if __name__ == "__main__":
    sys.exit(main())
