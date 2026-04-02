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
    parser.add_argument("--suite", default="bench/suites/adam0_reference_v2_7_1.json")
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

        counters = profile_payload.get("counters", {})
        timing = profile_payload.get("timing_ms", {})
        total_ms = float(timing.get("total", 0.0) or 0.0)
        native_ms = float(timing.get("native_calls", 0.0) or 0.0)
        native_share_pct = (native_ms / total_ms * 100.0) if total_ms > 0.0 else 0.0
        cases_out.append(
            {
                "id": case_id,
                "target_agents": case["target_agents"],
                "target_shard_size": case["target_shard_size"],
                "require_native_accel": case["require_native_accel"],
                "config": config,
                "run_report": str(run_report),
                "profile_report": str(profile_report),
                "snapshot_report": str(snapshot_report),
                "replay_report": str(replay_report),
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
                },
                "profile_breakdown": {
                    "ffi_calls": counters.get("ffi_calls", 0),
                    "native_function_calls": counters.get("native_function_calls", 0),
                    "sim_coroutines_spawned": counters.get("sim_coroutines_spawned", 0),
                    "sim_coroutine_emits": counters.get("sim_coroutine_emits", 0),
                    "marshal_in_bytes": counters.get("marshal_in_bytes", 0),
                    "marshal_out_bytes": counters.get("marshal_out_bytes", 0),
                    "marshal_copy_ops": counters.get("marshal_copy_ops", 0),
                    "native_ms": native_ms,
                    "vm_exec_ms": timing.get("vm_exec", 0.0),
                    "total_ms": total_ms,
                    "native_share_pct": native_share_pct,
                },
            }
        )

    payload = {
        "schema_version": 1,
        "suite": suite.get("suite"),
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
