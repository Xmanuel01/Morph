#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    result = subprocess.run(cmd, cwd=cwd)
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

    if not enkai_bin.exists():
        raise SystemExit(f"enkai binary not found: {enkai_bin}")

    sim_dir = workspace / "artifacts" / "sim"
    sim_dir.mkdir(parents=True, exist_ok=True)
    run_report = sim_dir / "smoke_run.json"
    replay_report = sim_dir / "smoke_replay.json"
    profile_report = sim_dir / "smoke_profile.json"
    snapshot_path = sim_dir / "smoke_snapshot.json"

    with tempfile.TemporaryDirectory(prefix="enkai_sim_smoke_") as tmp:
        root = Path(tmp)
        script = root / "sim_smoke.enk"
        script.write_text(
            (
                "import std::sim\n"
                "fn main() ::\n"
                "    let world := sim.make_seeded(16, 4242)\n"
                "    sim.entity_set(world, 1, 99)\n"
                "    sim.schedule(world, 0.5, 5)\n"
                "    sim.schedule(world, 1.0, 10)\n"
                "    sim.step(world)?\n"
                "    return sim.snapshot(world)\n"
                "::\n"
                "main()\n"
            ),
            encoding="utf-8",
        )

        run([str(enkai_bin), "sim", "run", "--output", str(run_report), str(script)], workspace)
        run_payload = read_json(run_report)
        snapshot = run_payload.get("result")
        if not isinstance(snapshot, dict):
            raise SystemExit("sim smoke run result was not a JSON object snapshot")
        for key in ("seed", "now", "queue", "log", "entities"):
            if key not in snapshot:
                raise SystemExit(f"sim smoke snapshot missing key: {key}")
        snapshot_path.write_text(
            json.dumps(snapshot, separators=(",", ":")),
            encoding="utf-8",
        )

        run(
            [
                str(enkai_bin),
                "sim",
                "profile",
                "--case",
                "readiness_sim_smoke",
                "--output",
                str(profile_report),
                str(script),
            ],
            workspace,
        )
        profile_payload = read_json(profile_report)
        if profile_payload.get("case") != "readiness_sim_smoke":
            raise SystemExit("sim smoke profile case id mismatch")

        run(
            [
                str(enkai_bin),
                "sim",
                "replay",
                "--snapshot",
                str(snapshot_path),
                "--steps",
                "2",
                "--output",
                str(replay_report),
            ],
            workspace,
        )
        replay_payload = read_json(replay_report)
        replay_result = replay_payload.get("result")
        if not isinstance(replay_result, dict):
            raise SystemExit("sim smoke replay result was not a JSON object snapshot")
        if len(replay_result.get("log", [])) < len(snapshot.get("log", [])):
            raise SystemExit("sim smoke replay log did not advance deterministically")

        summary = {
            "schema_version": 1,
            "script": str(script),
            "run_report": str(run_report),
            "profile_report": str(profile_report),
            "replay_report": str(replay_report),
            "snapshot": str(snapshot_path),
            "seed": snapshot["seed"],
            "run_pending_events": len(snapshot.get("queue", [])),
            "replay_log_len": len(replay_result.get("log", [])),
        }
        write_json(output, summary)

    return 0


if __name__ == "__main__":
    sys.exit(main())
