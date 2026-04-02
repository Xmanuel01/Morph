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
    parser.add_argument("--replay-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    summary_path = Path(args.summary)
    run_path = Path(args.run_report)
    profile_path = Path(args.profile_report)
    replay_path = Path(args.replay_report)
    output_path = Path(args.output)

    summary = read_json(summary_path)
    run_report = read_json(run_path)
    profile_report = read_json(profile_path)
    replay_report = read_json(replay_path)

    if summary.get("schema_version") != 1:
        fail("simulation evidence summary schema_version must be 1")
    if run_report.get("command") != "sim.run":
        fail("simulation run evidence must come from sim.run")
    if replay_report.get("command") != "sim.replay":
        fail("simulation replay evidence must come from sim.replay")
    if profile_report.get("case") != "readiness_sim_smoke":
        fail("simulation profile evidence case id mismatch")

    run_result = run_report.get("result")
    replay_result = replay_report.get("result")
    if not isinstance(run_result, dict):
        fail("simulation run result must be an object snapshot")
    if not isinstance(replay_result, dict):
        fail("simulation replay result must be an object snapshot")

    if run_result.get("seed") != summary.get("seed"):
        fail("simulation summary seed does not match run snapshot")
    if replay_result.get("seed") != summary.get("seed"):
        fail("simulation replay snapshot seed mismatch")

    run_log = run_result.get("log", [])
    replay_log = replay_result.get("log", [])
    if not isinstance(run_log, list) or not isinstance(replay_log, list):
        fail("simulation log payloads must be arrays")
    if len(run_log) == 0:
        fail("simulation run log is empty")
    if len(replay_log) < len(run_log):
        fail("simulation replay log did not preserve prior events")
    if replay_result.get("now", 0) < run_result.get("now", 0):
        fail("simulation replay time regressed")

    run_queue = run_result.get("queue", [])
    replay_queue = replay_result.get("queue", [])
    if not isinstance(run_queue, list) or not isinstance(replay_queue, list):
        fail("simulation queue payloads must be arrays")
    if len(replay_queue) > len(run_queue):
        fail("simulation replay queue unexpectedly grew")

    payload = {
        "schema_version": 1,
        "status": "ok",
        "seed": summary.get("seed"),
        "run_log_len": len(run_log),
        "replay_log_len": len(replay_log),
        "run_queue_len": len(run_queue),
        "replay_queue_len": len(replay_queue),
        "profile_case": profile_report.get("case"),
        "run_report": str(run_path),
        "profile_report": str(profile_path),
        "replay_report": str(replay_path),
    }
    write_json(output_path, payload)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as err:
        print(f"error: {err}", file=sys.stderr)
        raise SystemExit(1)
