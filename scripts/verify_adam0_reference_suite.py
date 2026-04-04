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


def normalize_value(value):
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, list):
        return [normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_value(item) for key, item in value.items()}
    return value


def canonical_json(value):
    return json.dumps(normalize_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hash_json(value) -> str:
    import hashlib

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def projection_hash(snapshot: dict) -> str:
    projected = {
        "entities": snapshot.get("entities", []),
        "queue": snapshot.get("queue", []),
        "log": snapshot.get("log", []),
        "seed": snapshot.get("seed"),
        "now": snapshot.get("now"),
        "next_seq": snapshot.get("next_seq"),
        "max_events": snapshot.get("max_events"),
    }
    return hash_json(projected)


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

    suite_path_value = summary.get("suite_path")
    if not isinstance(suite_path_value, str) or not suite_path_value:
        fail("adam0 reference suite summary must include suite_path")
    suite_path = Path(suite_path_value)
    if not suite_path.exists():
        fail(f"adam0 reference suite definition not found: {suite_path}")
    suite = read_json(suite_path)

    if summary.get("suite") != suite.get("suite"):
        fail("adam0 reference suite id mismatch")

    hardware = summary.get("hardware_assumptions")
    if not isinstance(hardware, dict):
        fail("adam0 reference suite hardware assumptions must be an object")
    for key in ("bounded_envelope", "reference_host", "notes"):
        if not isinstance(hardware.get(key), str) or not hardware.get(key):
            fail(f"adam0 reference suite missing hardware assumption field: {key}")

    cases = summary.get("cases")
    suite_cases = suite.get("cases")
    if not isinstance(cases, list) or not isinstance(suite_cases, list):
        fail("adam0 reference suite cases must be arrays")
    if len(cases) != len(suite_cases):
        fail("adam0 reference suite case count mismatch")

    expected_cases = {str(case["id"]): case for case in suite_cases}
    verification_cases = []
    for case in cases:
        case_id = case.get("id")
        if case_id not in expected_cases:
            fail(f"unexpected adam0 case id: {case_id}")
        case_def = expected_cases[case_id]
        expected_agents = int(case_def["target_agents"])
        expected_shard_size = int(case_def["target_shard_size"])
        config = case_def.get("config", {})
        rounds = int(config.get("rounds", 0))
        required_native_counters = case_def.get("required_native_counters", {})

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

        if int(result.get("agent_count", -1)) != expected_agents:
            fail(f"{case_id}: agent_count mismatch")
        if int(result.get("joined_total", -1)) != expected_agents:
            fail(f"{case_id}: joined_total mismatch")
        if int(result.get("shard_size", -1)) != expected_shard_size:
            fail(f"{case_id}: shard_size mismatch")
        if int(result.get("report_total", 0)) != int(result.get("shard_count", 0)) * rounds:
            fail(f"{case_id}: report_total does not match shard_count * rounds")
        if int(result.get("action_total", 0)) != expected_agents * rounds:
            fail(f"{case_id}: action_total mismatch")
        if int(result.get("dispatch_total", 0)) <= 0:
            fail(f"{case_id}: dispatch_total must be > 0")
        if int(result.get("sparse_edges", 0)) <= 0 or int(result.get("feature_nnz", 0)) <= 0:
            fail(f"{case_id}: sparse evidence must be > 0")
        if not isinstance(result.get("entities"), list) or len(result.get("entities")) != expected_agents:
            fail(f"{case_id}: result entities must match target agent count")
        if not isinstance(result.get("queue"), list) or len(result.get("queue")) != 0:
            fail(f"{case_id}: queue must be fully drained")
        if not isinstance(result.get("log"), list) or len(result.get("log")) <= 0:
            fail(f"{case_id}: event log must be non-empty")
        if not isinstance(result.get("hardware_class"), str) or not result.get("hardware_class"):
            fail(f"{case_id}: hardware_class must be present")
        if not isinstance(result.get("reference_host"), str) or not result.get("reference_host"):
            fail(f"{case_id}: reference_host must be present")

        if snapshot_report != result:
            fail(f"{case_id}: snapshot report must match sim.run result")
        if not isinstance(replay_result.get("entities"), list) or len(replay_result.get("entities")) != expected_agents:
            fail(f"{case_id}: replay entities must match target agent count")
        if not isinstance(replay_result.get("queue"), list) or len(replay_result.get("queue")) != 0:
            fail(f"{case_id}: replay queue must be fully drained")

        result_hash = hash_json(result)
        snapshot_hash = hash_json(snapshot_report)
        replay_hash = hash_json(replay_result)
        state_hash = projection_hash(result)
        replay_state_hash = projection_hash(replay_result)
        event_log_hash = hash_json(result.get("log", []))
        if result_hash != snapshot_hash:
            fail(f"{case_id}: result hash must match snapshot hash")
        if state_hash != replay_state_hash:
            fail(f"{case_id}: replay state hash must match snapshot state hash")

        counters = profile_report.get("counters")
        timing = profile_report.get("timing_ms")
        if not isinstance(counters, dict) or not isinstance(timing, dict):
            fail(f"{case_id}: profile counters/timing must be objects")
        ffi_calls = int(counters.get("ffi_calls", 0) or 0)
        native_function_calls = int(counters.get("native_function_calls", 0) or 0)
        marshal_copy_ops = int(counters.get("marshal_copy_ops", 0) or 0)
        kernel_checks = {}
        kernel_native_total = 0
        for counter_name, minimum in required_native_counters.items():
            observed = int(counters.get(counter_name, 0) or 0)
            kernel_native_total += observed
            passed = observed >= int(minimum)
            kernel_checks[counter_name] = {
                "minimum": int(minimum),
                "observed": observed,
                "passed": passed,
            }
            if not passed:
                fail(f"{case_id}: required native counter {counter_name} below minimum")
        marshal_copy_ratio = (marshal_copy_ops / ffi_calls) if ffi_calls > 0 else 0.0
        max_marshal_copy_ratio = case_def.get("max_marshal_copy_ratio")
        if max_marshal_copy_ratio is not None and marshal_copy_ratio > float(max_marshal_copy_ratio):
            fail(f"{case_id}: marshal copy ratio exceeded threshold")
        if case_def.get("require_kernel_native_dominance") and kernel_native_total <= marshal_copy_ops:
            fail(f"{case_id}: kernel native calls do not dominate marshal copy operations")
        if case_def.get("require_native_accel") is True and ffi_calls <= 0:
            fail(f"{case_id}: ffi_calls must be > 0")
        if native_function_calls <= 0:
            fail(f"{case_id}: native_function_calls must be > 0")
        if int(counters.get("sim_coroutines_spawned", 0) or 0) <= 0:
            fail(f"{case_id}: sim_coroutines_spawned must be > 0")
        if int(counters.get("sim_coroutine_emits", 0) or 0) <= 0:
            fail(f"{case_id}: sim_coroutine_emits must be > 0")
        if float(timing.get("native_calls", 0.0) or 0.0) <= 0.0:
            fail(f"{case_id}: native_calls timing must be > 0")
        if float(timing.get("vm_exec", 0.0) or 0.0) <= 0.0:
            fail(f"{case_id}: vm_exec timing must be > 0")

        verification_cases.append(
            {
                "id": case_id,
                "agent_count": expected_agents,
                "report_total": int(result.get("report_total", 0)),
                "dispatch_total": int(result.get("dispatch_total", 0)),
                "sparse_edges": int(result.get("sparse_edges", 0)),
                "ffi_calls": ffi_calls,
                "native_function_calls": native_function_calls,
                "sim_coroutines_spawned": int(counters.get("sim_coroutines_spawned", 0) or 0),
                "native_ms": float(timing.get("native_calls", 0.0) or 0.0),
                "vm_exec_ms": float(timing.get("vm_exec", 0.0) or 0.0),
                "marshal_copy_ratio": marshal_copy_ratio,
                "kernel_native_calls_total": kernel_native_total,
                "kernel_counter_checks": kernel_checks,
                "result_hash": result_hash,
                "snapshot_hash": snapshot_hash,
                "replay_hash": replay_hash,
                "state_hash": state_hash,
                "replay_state_hash": replay_state_hash,
                "event_log_hash": event_log_hash,
            }
        )

    payload = {
        "schema_version": 1,
        "summary": str(summary_path),
        "suite": summary.get("suite"),
        "suite_path": str(suite_path),
        "all_passed": True,
        "cases": verification_cases,
    }
    write_json(output_path, payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
