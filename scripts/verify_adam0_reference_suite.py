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
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    summary_path = Path(args.summary)
    output_path = Path(args.output)
    summary = read_json(summary_path)

    if summary.get("schema_version") != 1:
        fail("adam0 reference suite summary schema_version must be 1")
    if summary.get("suite") != "adam0_reference_v2_7_1":
        fail("adam0 reference suite id mismatch")

    hardware = summary.get("hardware_assumptions")
    if not isinstance(hardware, dict):
        fail("adam0 reference suite hardware assumptions must be an object")
    for key in ("bounded_envelope", "reference_host", "notes"):
        if not isinstance(hardware.get(key), str) or not hardware.get(key):
            fail(f"adam0 reference suite missing hardware assumption field: {key}")

    cases = summary.get("cases")
    if not isinstance(cases, list) or len(cases) != 3:
        fail("adam0 reference suite must contain exactly 3 benchmark cases")

    expected_ids = ["adam0_baseline_100", "adam0_stress_1000", "adam0_target_10000"]
    expected_agents = {
        "adam0_baseline_100": 100,
        "adam0_stress_1000": 1000,
        "adam0_target_10000": 10000,
    }

    verification_cases = []
    for case in cases:
        case_id = case.get("id")
        if case_id not in expected_ids:
            fail(f"unexpected adam0 case id: {case_id}")
        if int(case.get("target_agents", 0)) != expected_agents[case_id]:
            fail(f"{case_id}: target_agents mismatch")

        run_report = read_json(Path(case["run_report"]))
        profile_report = read_json(Path(case["profile_report"]))
        snapshot_report = read_json(Path(case["snapshot_report"]))
        replay_report = read_json(Path(case["replay_report"]))

        if run_report.get("command") != "sim.run":
            fail(f"{case_id}: run report must come from sim.run")
        if profile_report.get("case") != case_id or profile_report.get("status") != "ok":
            fail(f"{case_id}: profile report mismatch")
        if replay_report.get("command") != "sim.replay":
            fail(f"{case_id}: replay report must come from sim.replay")

        result = run_report.get("result")
        replay_result = replay_report.get("result")
        if not isinstance(result, dict) or not isinstance(snapshot_report, dict) or not isinstance(replay_result, dict):
            fail(f"{case_id}: result/snapshot/replay payloads must be objects")

        expected_agent_count = expected_agents[case_id]
        if int(result.get("agent_count", -1)) != expected_agent_count:
            fail(f"{case_id}: agent_count mismatch")
        if int(result.get("joined_total", -1)) != expected_agent_count:
            fail(f"{case_id}: joined_total mismatch")
        if int(result.get("report_total", 0)) <= 0:
            fail(f"{case_id}: report_total must be > 0")
        if int(result.get("action_total", 0)) < expected_agent_count:
            fail(f"{case_id}: action_total must be >= agent_count")
        if int(result.get("dispatch_total", 0)) <= 0:
            fail(f"{case_id}: dispatch_total must be > 0")
        if int(result.get("sparse_edges", 0)) <= 0 or int(result.get("feature_nnz", 0)) <= 0:
            fail(f"{case_id}: sparse evidence must be > 0")
        if not isinstance(result.get("entities"), list) or len(result.get("entities")) != expected_agent_count:
            fail(f"{case_id}: snapshot entities must match target agent count")
        if not isinstance(result.get("queue"), list) or len(result.get("queue")) != 0:
            fail(f"{case_id}: queue must be fully drained")
        if not isinstance(result.get("log"), list) or len(result.get("log")) <= 0:
            fail(f"{case_id}: event log must be non-empty")
        if not isinstance(result.get("hardware_class"), str) or not result.get("hardware_class"):
            fail(f"{case_id}: hardware_class must be present")
        if not isinstance(result.get("reference_host"), str) or not result.get("reference_host"):
            fail(f"{case_id}: reference_host must be present")

        if not isinstance(replay_result.get("entities"), list) or len(replay_result.get("entities")) != expected_agent_count:
            fail(f"{case_id}: replay entities must match target agent count")
        if not isinstance(replay_result.get("queue"), list) or len(replay_result.get("queue")) != 0:
            fail(f"{case_id}: replay queue must be fully drained")

        counters = profile_report.get("counters")
        timing = profile_report.get("timing_ms")
        if not isinstance(counters, dict) or not isinstance(timing, dict):
            fail(f"{case_id}: profile counters/timing must be objects")
        if case.get("require_native_accel") is True and int(counters.get("ffi_calls", 0)) <= 0:
            fail(f"{case_id}: ffi_calls must be > 0")
        if int(counters.get("native_function_calls", 0)) <= 0:
            fail(f"{case_id}: native_function_calls must be > 0")
        if int(counters.get("sim_coroutines_spawned", 0)) <= 0:
            fail(f"{case_id}: sim_coroutines_spawned must be > 0")
        if int(counters.get("sim_coroutine_emits", 0)) <= 0:
            fail(f"{case_id}: sim_coroutine_emits must be > 0")
        if float(timing.get("native_calls", 0.0)) <= 0.0:
            fail(f"{case_id}: native_calls timing must be > 0")
        if float(timing.get("vm_exec", 0.0)) <= 0.0:
            fail(f"{case_id}: vm_exec timing must be > 0")

        verification_cases.append(
            {
                "id": case_id,
                "agent_count": expected_agent_count,
                "report_total": int(result.get("report_total", 0)),
                "dispatch_total": int(result.get("dispatch_total", 0)),
                "sparse_edges": int(result.get("sparse_edges", 0)),
                "ffi_calls": counters.get("ffi_calls", 0),
                "native_function_calls": counters.get("native_function_calls", 0),
                "sim_coroutines_spawned": counters.get("sim_coroutines_spawned", 0),
                "native_ms": timing.get("native_calls", 0.0),
                "vm_exec_ms": timing.get("vm_exec", 0.0),
            }
        )

    payload = {
        "schema_version": 1,
        "summary": str(summary_path),
        "all_passed": True,
        "cases": verification_cases,
    }
    write_json(output_path, payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
