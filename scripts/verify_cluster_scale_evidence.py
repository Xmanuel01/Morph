#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    summary_path = Path(args.summary)
    output_path = Path(args.output)
    payload = read_json(summary_path)

    issues: list[str] = []
    if payload.get("status") != "ok":
        issues.append("summary status must be ok")
    if not payload.get("validate_ok"):
        issues.append("cluster validate report did not mark ok=true")
    if payload.get("topology") != "multi-node":
        issues.append("cluster topology must be multi-node")
    if int(payload.get("world_size", 0)) != 2:
        issues.append("cluster world_size must be 2")
    if payload.get("route_policy") != "deterministic-ring":
        issues.append("cluster route_policy must be deterministic-ring")
    if int(payload.get("rank_count", 0)) != 2:
        issues.append("cluster run must produce two rank reports")
    if int(payload.get("resumed_partitions", 0)) < 1:
        issues.append("cluster run must exercise snapshot recovery on at least one partition")

    for key in ("validate_report", "plan_report", "run_report"):
        path = payload.get(key)
        if not path or not Path(path).is_file():
            issues.append(f"missing file for {key}")

    if payload.get("plan_report") and Path(payload["plan_report"]).is_file():
        plan = read_json(Path(payload["plan_report"]))
        if plan.get("workload_kind") != "simulation":
            issues.append("cluster plan workload_kind must be simulation")
        if plan.get("host_map") != [0, 1]:
            issues.append("cluster host_map must be [0, 1]")
        hosts = [rank.get("host") for rank in plan.get("rank_plans", [])]
        if hosts != ["node-a", "node-b"]:
            issues.append(f"cluster hosts mismatch: {hosts}")

    if payload.get("run_report") and Path(payload["run_report"]).is_file():
        run = read_json(Path(payload["run_report"]))
        if not run.get("all_passed", False):
            issues.append("cluster run did not pass")
        if run.get("execution_mode") != "supervised":
            issues.append("cluster run execution_mode must be supervised")
        simulation = run.get("simulation", {})
        if simulation.get("total_completed_steps") != 12:
            issues.append("cluster run total_completed_steps must be 12")
        recovered = 0
        retried = 0
        for rank in run.get("rank_reports", []):
            if rank.get("status") != "ok":
                issues.append(f"rank {rank.get('rank')} status was not ok")
            if int(rank.get("completed_steps", 0)) != 6:
                issues.append(f"rank {rank.get('rank')} completed_steps must be 6")
            if not rank.get("final_snapshot"):
                issues.append(f"rank {rank.get('rank')} missing final snapshot")
            for key in ("window_reports", "stdout_logs", "stderr_logs"):
                entries = rank.get(key, [])
                if not isinstance(entries, list) or not entries:
                    issues.append(f"rank {rank.get('rank')} missing {key}")
            if rank.get("recovered_from_snapshot"):
                recovered += 1
            if int(rank.get("retries_used", 0)) > 0:
                retried += 1
        if recovered < 1:
            issues.append("expected at least one rank to recover from snapshot")
        if retried < 1:
            issues.append("expected at least one rank to use retry budget")

    result = {
        "schema_version": 1,
        "status": "ok" if not issues else "failed",
        "passed": not issues,
        "issues": issues,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
