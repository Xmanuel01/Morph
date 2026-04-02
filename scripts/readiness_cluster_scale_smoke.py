#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_cluster_inputs(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.txt").write_text("hello\ncluster\nsimulation\n", encoding="utf-8")
    (root / "ckpt").mkdir(parents=True, exist_ok=True)
    (root / "sim_cluster.enk").write_text(
        """import json
import std::sim
fn main() ::
    let world := sim.make_seeded(16, 4242)
    sim.entity_set(world, 1, json.parse("{\\\"partition\\\":1,\\\"state\\\":\\\"boot\\\"}"))
    sim.schedule(world, 0.5, json.parse("{\\\"event\\\":\\\"sense\\\",\\\"rank\\\":1}"))
    sim.schedule(world, 1.0, json.parse("{\\\"event\\\":\\\"step\\\",\\\"rank\\\":2}"))
    sim.run(world, 2)
    return sim.snapshot(world)
::
main()
""",
        encoding="utf-8",
    )
    config = root / "cluster_config.enk"
    config.write_text(
        """import json
fn main() ::
    return json.parse("{\\"config_version\\":1,\\"backend\\":\\"native\\",\\"vocab_size\\":8,\\"hidden_size\\":4,\\"seq_len\\":4,\\"batch_size\\":2,\\"lr\\":0.1,\\"dataset_path\\":\\"data.txt\\",\\"checkpoint_dir\\":\\"ckpt\\",\\"max_steps\\":2,\\"save_every\\":1,\\"log_every\\":1,\\"tokenizer_train\\":{\\"path\\":\\"data.txt\\",\\"vocab_size\\":8},\\"world_size\\":2,\\"rank\\":0,\\"workload\\":\\"simulation\\",\\"dist\\":{\\"topology\\":\\"multi-node\\",\\"rendezvous\\":\\"tcp://127.0.0.1:29500\\",\\"retry_budget\\":2,\\"device_map\\":[0,1],\\"hosts\\":[\\"node-a\\",\\"node-b\\"],\\"host_map\\":[0,1]},\\"simulation\\":{\\"target\\":\\"sim_cluster.enk\\",\\"partition_count\\":2,\\"total_steps\\":6,\\"step_window\\":2,\\"snapshot_interval\\":2,\\"route_policy\\":\\"deterministic-ring\\",\\"recovery_dir\\":\\"recovery\\",\\"seed\\":17}}")
::
main()
""",
        encoding="utf-8",
    )
    return config


def clear_fault_injection_markers() -> None:
    temp_root = Path(tempfile.gettempdir())
    for marker in temp_root.glob("enkai_cluster_fail_once_rank*_after*.marker"):
        try:
            marker.unlink()
        except FileNotFoundError:
            continue


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enkai-bin", required=True)
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    enkai_bin = Path(args.enkai_bin)
    output = workspace / args.output
    if not enkai_bin.exists():
        raise SystemExit(f"enkai binary not found: {enkai_bin}")

    root = workspace / "artifacts" / "cluster_scale"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    clear_fault_injection_markers()
    config = write_cluster_inputs(root)

    validate_report = root / "validate.json"
    plan_report = root / "plan.json"
    run_report = root / "run.json"

    env = dict(os.environ)
    env["ENKAI_SIM_ACCEL"] = "1"
    env["ENKAI_ENABLE_DIST"] = "1"
    env["ENKAI_CLUSTER_INJECT_FAIL_ONCE_RANK"] = "1"
    env["ENKAI_CLUSTER_INJECT_FAIL_ONCE_AFTER_WINDOWS"] = "1"
    env["ENKAI_CLUSTER_INJECT_FAIL_ONCE_EXIT_CODE"] = "75"

    run(
        [str(enkai_bin), "cluster", "validate", str(config), "--json", "--output", str(validate_report)],
        workspace,
        env,
    )
    run(
        [str(enkai_bin), "cluster", "plan", str(config), "--json", "--output", str(plan_report)],
        workspace,
        env,
    )
    run(
        [str(enkai_bin), "cluster", "run", str(config), "--json", "--output", str(run_report)],
        workspace,
        env,
    )

    validate_payload = read_json(validate_report)
    plan_payload = read_json(plan_report)
    run_payload = read_json(run_report)

    simulation = run_payload.get("simulation", {})
    summary = {
        "schema_version": 1,
        "status": "ok",
        "config": str(config),
        "validate_report": str(validate_report),
        "plan_report": str(plan_report),
        "run_report": str(run_report),
        "recovery_dir": simulation.get("recovery_dir"),
        "topology": run_payload.get("topology"),
        "world_size": run_payload.get("world_size"),
        "route_policy": simulation.get("route_policy"),
        "resumed_partitions": simulation.get("resumed_partitions"),
        "rank_count": len(run_payload.get("rank_reports", [])),
        "hosts": [rank.get("host") for rank in plan_payload.get("rank_plans", [])],
        "validate_ok": validate_payload.get("ok", False),
    }
    write_json(output, summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
