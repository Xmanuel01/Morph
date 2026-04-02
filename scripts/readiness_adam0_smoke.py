#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROFILE_CASE = "readiness_adam0_100"


def run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enkai-bin", required=True)
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    enkai_bin = Path(args.enkai_bin)
    output = workspace / args.output
    env = dict(os.environ)
    env["ENKAI_SIM_ACCEL"] = "1"

    if not enkai_bin.exists():
        raise SystemExit(f"enkai binary not found: {enkai_bin}")

    script = workspace / "examples" / "adam0_100.enk"
    if not script.exists():
        raise SystemExit(f"adam0 example not found: {script}")

    sim_dir = workspace / "artifacts" / "sim"
    sim_dir.mkdir(parents=True, exist_ok=True)
    run_report = sim_dir / "adam0_100_run.json"
    profile_report = sim_dir / "adam0_100_profile.json"

    run([str(enkai_bin), "sim", "run", "--output", str(run_report), str(script)], workspace, env)
    run_payload = read_json(run_report)
    if run_payload.get("command") != "sim.run":
        raise SystemExit("adam0 smoke run report command mismatch")

    run(
        [
            str(enkai_bin),
            "sim",
            "profile",
            "--case",
            PROFILE_CASE,
            "--output",
            str(profile_report),
            str(script),
        ],
        workspace,
        env,
    )
    profile_payload = read_json(profile_report)
    if profile_payload.get("case") != PROFILE_CASE:
        raise SystemExit("adam0 smoke profile case mismatch")
    if profile_payload.get("status") != "ok":
        raise SystemExit("adam0 smoke profile status was not ok")

    summary = {
        "schema_version": 1,
        "script": str(script),
        "run_report": str(run_report),
        "profile_report": str(profile_report),
        "profile_case": PROFILE_CASE,
        "require_native_accel": True,
        "target_agents": 100,
        "target_shards": 10,
        "target_joined_total": 100,
    }
    write_json(output, summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
