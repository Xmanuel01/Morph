#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


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


def apply_case_env(base: dict[str, str], config: dict[str, object]) -> dict[str, str]:
    env = dict(base)
    mapping = {
        "case_id": "ENKAI_ADAM0_CASE_ID",
        "agent_count": "ENKAI_ADAM0_AGENT_COUNT",
        "shard_size": "ENKAI_ADAM0_SHARD_SIZE",
        "rounds": "ENKAI_ADAM0_ROUNDS",
        "fanout": "ENKAI_ADAM0_FANOUT",
        "pool_capacity": "ENKAI_ADAM0_POOL_CAPACITY",
        "max_events": "ENKAI_ADAM0_MAX_EVENTS",
        "seed": "ENKAI_ADAM0_SEED",
        "neighbor_radius_milli": "ENKAI_ADAM0_NEIGHBOR_RADIUS_MILLI",
        "move_step_milli": "ENKAI_ADAM0_MOVE_STEP_MILLI",
        "spacing_milli": "ENKAI_ADAM0_SPACING_MILLI",
        "hardware_class": "ENKAI_ADAM0_HARDWARE_CLASS",
        "reference_host": "ENKAI_ADAM0_REFERENCE_HOST",
    }
    for key, env_key in mapping.items():
        value = config.get(key)
        if value is None:
            raise SystemExit(f"adam0 reference config missing required key: {key}")
        env[env_key] = str(value)
    return env


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enkai-bin", required=True)
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--suite", default="bench/suites/adam0_reference_v2_9_4.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    enkai_bin = Path(args.enkai_bin)
    suite_path = workspace / args.suite
    output = workspace / args.output

    if not enkai_bin.exists():
        raise SystemExit(f"enkai binary not found: {enkai_bin}")
    if not suite_path.exists():
        raise SystemExit(f"adam0 suite definition not found: {suite_path}")

    suite = read_json(suite_path)
    script_target = workspace / str(suite.get("script", ""))
    if not script_target.exists():
        raise SystemExit(f"adam0 reference target not found: {script_target}")

    env = dict(os.environ)
    env["ENKAI_SIM_ACCEL"] = "1"
    env.setdefault("ENKAI_STD", str((workspace / "std").resolve()))

    cases_out = []
    for case in suite.get("cases", []):
        case_id = str(case["id"])
        config = case.get("config")
        if not isinstance(config, dict):
            raise SystemExit(f"{case_id}: case config must be an object")
        case_env = apply_case_env(env, config)

        run_report = workspace / str(case["run_report"])
        profile_report = workspace / str(case["profile_report"])
        snapshot_report = workspace / str(case["snapshot_report"])
        replay_report = workspace / str(case["replay_report"])
        for artifact in (run_report, profile_report, snapshot_report, replay_report):
            ensure_parent(artifact)

        run([str(enkai_bin), "sim", "run", "--output", str(run_report), str(script_target)], workspace, case_env)
        run_payload = read_json(run_report)
        if run_payload.get("command") != "sim.run":
            raise SystemExit(f"{case_id}: sim run report command mismatch")
        result = run_payload.get("result")
        if not isinstance(result, dict):
            raise SystemExit(f"{case_id}: sim run result must be an object")
        write_json(snapshot_report, result)

        run(
            [
                str(enkai_bin),
                "sim",
                "profile",
                "--case",
                case_id,
                "--output",
                str(profile_report),
                str(script_target),
            ],
            workspace,
            case_env,
        )
        profile_payload = read_json(profile_report)
        if profile_payload.get("case") != case_id:
            raise SystemExit(f"{case_id}: sim profile case mismatch")
        if profile_payload.get("status") != "ok":
            raise SystemExit(f"{case_id}: sim profile status was not ok")

        run(
            [
                str(enkai_bin),
                "sim",
                "replay",
                "--snapshot",
                str(snapshot_report),
                "--steps",
                "0",
                "--output",
                str(replay_report),
            ],
            workspace,
            case_env,
        )
        replay_payload = read_json(replay_report)
        if replay_payload.get("command") != "sim.replay":
            raise SystemExit(f"{case_id}: sim replay report command mismatch")
        replay_result = replay_payload.get("result")
        if not isinstance(replay_result, dict):
            raise SystemExit(f"{case_id}: sim replay result must be an object")

        counters = profile_payload.get("counters", {})
        timing = profile_payload.get("timing_ms", {})
        total_ms = float(timing.get("total", 0.0) or 0.0)
        native_ms = float(timing.get("native_calls", 0.0) or 0.0)
        native_share_pct = (native_ms / total_ms * 100.0) if total_ms > 0.0 else 0.0
        kernel_native_counters = {
            "sim_sparse_native_calls": int(counters.get("sim_sparse_native_calls", 0) or 0),
            "sim_event_native_calls": int(counters.get("sim_event_native_calls", 0) or 0),
            "sim_pool_native_calls": int(counters.get("sim_pool_native_calls", 0) or 0),
            "sim_spatial_native_calls": int(counters.get("sim_spatial_native_calls", 0) or 0),
            "sim_snn_native_calls": int(counters.get("sim_snn_native_calls", 0) or 0),
        }
        kernel_native_total = sum(kernel_native_counters.values())
        marshal_copy_ops = int(counters.get("marshal_copy_ops", 0) or 0)
        ffi_calls = int(counters.get("ffi_calls", 0) or 0)
        marshal_copy_ratio = (marshal_copy_ops / ffi_calls) if ffi_calls > 0 else 0.0
        result_hash = hash_json(result)
        snapshot_hash = hash_json(read_json(snapshot_report))
        replay_hash = hash_json(replay_result)
        state_hash = projection_hash(result)
        replay_state_hash = projection_hash(replay_result)
        event_log_hash = hash_json(result.get("log", []))
        cases_out.append(
            {
                "id": case_id,
                "target_agents": case["target_agents"],
                "target_shard_size": case["target_shard_size"],
                "require_native_accel": case.get("require_native_accel", False),
                "required_native_counters": case.get("required_native_counters", {}),
                "require_kernel_native_dominance": case.get("require_kernel_native_dominance", False),
                "max_marshal_copy_ratio": case.get("max_marshal_copy_ratio"),
                "config": config,
                "run_report": str(run_report),
                "profile_report": str(profile_report),
                "snapshot_report": str(snapshot_report),
                "replay_report": str(replay_report),
                "result_hash": result_hash,
                "snapshot_hash": snapshot_hash,
                "replay_hash": replay_hash,
                "state_hash": state_hash,
                "replay_state_hash": replay_state_hash,
                "event_log_hash": event_log_hash,
                "result": {
                    "agent_count": result.get("agent_count"),
                    "shard_count": result.get("shard_count"),
                    "joined_total": result.get("joined_total"),
                    "report_total": result.get("report_total"),
                    "action_total": result.get("action_total"),
                    "spike_total": result.get("spike_total"),
                    "dispatch_total": result.get("dispatch_total"),
                    "sparse_edges": result.get("sparse_edges"),
                    "feature_nnz": result.get("feature_nnz"),
                    "hardware_class": result.get("hardware_class"),
                    "reference_host": result.get("reference_host"),
                },
                "profile_breakdown": {
                    "ffi_calls": ffi_calls,
                    "native_function_calls": int(counters.get("native_function_calls", 0) or 0),
                    "sim_coroutines_spawned": int(counters.get("sim_coroutines_spawned", 0) or 0),
                    "sim_coroutine_emits": int(counters.get("sim_coroutine_emits", 0) or 0),
                    "sim_coroutine_next_waits": int(counters.get("sim_coroutine_next_waits", 0) or 0),
                    "marshal_in_bytes": int(counters.get("marshal_in_bytes", 0) or 0),
                    "marshal_out_bytes": int(counters.get("marshal_out_bytes", 0) or 0),
                    "marshal_copy_ops": marshal_copy_ops,
                    "native_ms": native_ms,
                    "vm_exec_ms": float(timing.get("vm_exec", 0.0) or 0.0),
                    "total_ms": total_ms,
                    "native_share_pct": native_share_pct,
                    "marshal_copy_ratio": marshal_copy_ratio,
                    "kernel_native_counters": kernel_native_counters,
                    "kernel_native_calls_total": kernel_native_total,
                },
            }
        )

    payload = {
        "schema_version": 1,
        "suite": suite.get("suite"),
        "suite_path": str(suite_path),
        "description": suite.get("description"),
        "script": str(script_target),
        "hardware_assumptions": suite.get("hardware_assumptions", {}),
        "cases": cases_out,
        "all_passed": True,
    }
    write_json(output, payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
