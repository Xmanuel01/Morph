#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, separators=(",", ":")) for row in rows), encoding="utf-8")


def run_cmd(args: list[str], cwd: Path) -> dict[str, Any]:
    proc = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    return {
        "args": args,
        "exit_code": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v3.8.0 worker persistence/scheduling proof.")
    parser.add_argument("--enkai-bin", required=True)
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--suite", default="bench/suites/v3_8_0_worker_checkpoint.json")
    parser.add_argument("--output", default="artifacts/readiness/v3_8_0_worker_persistence_scheduling.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.workspace).resolve()
    enkai = str(Path(args.enkai_bin).resolve())
    suite = read_json((root / args.suite).resolve())
    spec = suite["worker_persistence_scheduling"]
    output = (root / args.output).resolve()
    work = root / "artifacts" / "v3_8_0_worker_persistence_scheduling"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    state = work / "state"
    queue = spec["queue"]
    ok_handler = work / "ok.enk"
    fail_handler = work / "fail.enk"
    ok_handler.write_text('fn main() ::\n    return 0\n::\nmain()\n', encoding="utf-8")
    fail_handler.write_text('fn main() ::\n    missing_symbol()\n::\nmain()\n', encoding="utf-8")

    enqueue_retry = run_cmd([
        enkai, "worker", "enqueue", "--queue", queue, "--dir", str(state),
        "--payload", '{"kind":"retry"}', "--id", "retry-job", "--max-attempts", "2",
        "--retry-delay-ms", str(spec["retry_delay_ms"]), "--json", "--output", str(work / "enqueue_retry.json")
    ], root)
    fail_once = run_cmd([
        enkai, "worker", "run", "--queue", queue, "--dir", str(state), "--handler", str(fail_handler),
        "--once", "--json", "--output", str(work / "fail_once.json")
    ], root)
    immediate_idle = run_cmd([
        enkai, "worker", "run", "--queue", queue, "--dir", str(state), "--handler", str(ok_handler),
        "--once", "--json", "--output", str(work / "immediate_idle.json")
    ], root)
    time.sleep((int(spec["retry_delay_ms"]) + 150) / 1000.0)
    retry_replay = run_cmd([
        enkai, "worker", "run", "--queue", queue, "--dir", str(state), "--handler", str(ok_handler),
        "--json", "--output", str(work / "retry_replay.json")
    ], root)

    queue_root = state / "queues" / queue
    pending_path = queue_root / "pending.jsonl"
    inflight_path = queue_root / "inflight.jsonl"
    scheduled_path = queue_root / "scheduled.jsonl"
    state_path = queue_root / "queue_state.json"

    enqueue_stale = run_cmd([
        enkai, "worker", "enqueue", "--queue", queue, "--dir", str(state),
        "--payload", '{"kind":"stale"}', "--id", "stale-job", "--max-attempts", "2",
        "--json", "--output", str(work / "enqueue_stale.json")
    ], root)
    pending = read_jsonl(pending_path)
    stale = next(row for row in pending if row["id"] == "stale-job")
    pending = [row for row in pending if row["id"] != "stale-job"]
    now_ms = int(time.time() * 1000)
    stale["attempts"] = 1
    stale["lease_id"] = "proof-stale-lease"
    stale["leased_ms"] = now_ms - int(spec["stale_lease_age_ms"])
    stale["lease_deadline_ms"] = now_ms - 1
    inflight = read_jsonl(inflight_path)
    inflight.append(stale)
    write_jsonl(pending_path, pending)
    write_jsonl(inflight_path, inflight)

    reclaim_run = run_cmd([
        enkai, "worker", "run", "--queue", queue, "--dir", str(state), "--handler", str(ok_handler),
        "--once", "--visibility-timeout-ms", str(spec["visibility_timeout_ms"]), "--json", "--output", str(work / "reclaim_run.json")
    ], root)

    enqueue_manifest_path = work / "enqueue_manifest.json"
    run_manifest_path = work / "run_manifest.json"
    manifest_enqueue = run_cmd([
        enkai, "systems", "worker-manifest", "enqueue", "--queue", queue, "--dir", str(state),
        "--payload", '{"kind":"manifest"}', "--max-attempts", "3", "--retry-delay-ms", "333",
        "--json", "--output", str(enqueue_manifest_path)
    ], root)
    manifest_run = run_cmd([
        enkai, "systems", "worker-manifest", "run", "--queue", queue, "--dir", str(state),
        "--handler", str(ok_handler), "--visibility-timeout-ms", "77", "--json", "--output", str(run_manifest_path)
    ], root)

    fail_report = read_json(work / "fail_once.json") if (work / "fail_once.json").is_file() else {}
    idle_report = read_json(work / "immediate_idle.json") if (work / "immediate_idle.json").is_file() else {}
    replay_report = read_json(work / "retry_replay.json") if (work / "retry_replay.json").is_file() else {}
    reclaim_report = read_json(work / "reclaim_run.json") if (work / "reclaim_run.json").is_file() else {}
    queue_state = read_json(state_path) if state_path.is_file() else {}
    enqueue_manifest = read_json(enqueue_manifest_path) if enqueue_manifest_path.is_file() else {}
    run_manifest = read_json(run_manifest_path) if run_manifest_path.is_file() else {}

    delayed_retry = {
        "passed": bool(
            enqueue_retry["exit_code"] == 0
            and fail_once["exit_code"] == 0
            and fail_report.get("status") == "requeued"
            and fail_report.get("requeued") == 1
            and read_jsonl(scheduled_path) == []
            and immediate_idle["exit_code"] == 0
            and idle_report.get("status") in {"idle", "drained"}
            and retry_replay["exit_code"] == 0
            and replay_report.get("acked", 0) >= 1
        ),
        "fail_report": fail_report,
        "immediate_idle_report": idle_report,
        "retry_replay_report": replay_report,
    }
    stale_reclaim = {
        "passed": bool(
            enqueue_stale["exit_code"] == 0
            and reclaim_run["exit_code"] == 0
            and reclaim_report.get("reclaimed_stale") == 1
            and reclaim_report.get("acked") == 1
            and int(queue_state.get("stale_reclaimed_count", 0)) >= 1
        ),
        "reclaim_report": reclaim_report,
        "queue_state": queue_state,
    }
    manifest_projection = {
        "passed": bool(
            manifest_enqueue["exit_code"] == 0
            and manifest_run["exit_code"] == 0
            and enqueue_manifest.get("retry_policy", {}).get("delay_ms") == 333
            and run_manifest.get("run_policy", {}).get("visibility_timeout_ms") == 77
            and enqueue_manifest.get("backend_kind") == spec["expected_backend_kind"]
            and run_manifest.get("backend_kind") == spec["expected_backend_kind"]
        ),
        "enqueue_manifest": enqueue_manifest,
        "run_manifest": run_manifest,
    }
    summary = {
        "schema_version": 1,
        "verified_contract_version": "v3.8.0",
        "all_passed": delayed_retry["passed"] and stale_reclaim["passed"] and manifest_projection["passed"],
        "suite": suite["suite"],
        "queue_backend_kind": spec["expected_backend_kind"],
        "delayed_retry": delayed_retry,
        "stale_inflight_reclaim": stale_reclaim,
        "manifest_projection": manifest_projection,
        "commands": {
            "enqueue_retry": enqueue_retry,
            "fail_once": fail_once,
            "immediate_idle": immediate_idle,
            "retry_replay": retry_replay,
            "enqueue_stale": enqueue_stale,
            "reclaim_run": reclaim_run,
            "manifest_enqueue": manifest_enqueue,
            "manifest_run": manifest_run,
        },
        "artifacts": {"workspace": str(work), "state_dir": str(state)},
    }
    write_json(output, summary)
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
