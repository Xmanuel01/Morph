#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the v3.7.0 threaded acceleration determinism proof.")
    parser.add_argument("--enkai-bin", required=True)
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--output", default="artifacts/readiness/v3_7_0_ai_runtime_threaded_determinism.json")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    if path.is_dir():
        digest = hashlib.sha256()
        for child in sorted(p for p in path.rglob("*") if p.is_file()):
            digest.update(str(child.relative_to(path)).encode("utf-8"))
            with child.open("rb") as handle:
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
        return digest.hexdigest()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def normalized_report_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "suite_id": report.get("suite_id"),
        "command": report.get("command"),
        "requested_backend": report.get("requested_backend"),
        "executed_backend": report.get("executed_backend"),
        "fallback_reason": report.get("fallback_reason"),
        "kernel": report.get("kernel"),
        "worker_count": report.get("worker_count"),
        "success": report.get("success"),
        "step": report.get("step"),
        "tokens": report.get("tokens"),
        "eval_batches": report.get("eval_batches"),
        "loss": report.get("loss"),
        "ppl": report.get("ppl"),
        "peak_memory_bytes_est": report.get("peak_memory_bytes_est"),
        "config_hash": report.get("config_hash"),
        "error_code": report.get("error_code"),
        "error_message": report.get("error_message"),
    }


def run_readiness(repo_root: Path, enkai_bin: Path, output_path: Path) -> dict[str, Any]:
    cmd = [
        "py",
        "-3",
        str(repo_root / "scripts" / "readiness_v3_7_0_ai_runtime_foundation.py"),
        "--enkai-bin",
        str(enkai_bin),
        "--output",
        str(output_path),
    ]
    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    payload = read_json(output_path)
    train_report_path = Path(payload["artifacts"]["train_report"])
    train_report = read_json(train_report_path)
    checkpoint_hash = None
    latest_checkpoint = train_report.get("latest_checkpoint_path")
    if latest_checkpoint:
        checkpoint_hash = sha256_file(Path(latest_checkpoint))
    payload["_train_report_hash"] = sha256_json(normalized_report_payload(train_report))
    payload["_checkpoint_hash"] = checkpoint_hash
    payload["_train_report"] = train_report
    return payload


def main() -> int:
    args = parse_args()
    repo_root = Path(args.workspace).resolve()
    enkai_bin = Path(args.enkai_bin).resolve()
    output_path = (repo_root / args.output).resolve()

    run_a_path = repo_root / "artifacts" / "readiness" / "v3_7_0_ai_runtime_foundation_run_a.json"
    run_b_path = repo_root / "artifacts" / "readiness" / "v3_7_0_ai_runtime_foundation_run_b.json"

    run_a = run_readiness(repo_root, enkai_bin, run_a_path)
    run_b = run_readiness(repo_root, enkai_bin, run_b_path)

    worker_a = int(run_a["benchmark"]["enkai_accel"]["worker_count"])
    worker_b = int(run_b["benchmark"]["enkai_accel"]["worker_count"])
    cpu_count = os.cpu_count() or 1
    threaded_expected = cpu_count > 1
    threaded_engaged = worker_a > 1 and worker_b > 1 if threaded_expected else worker_a >= 1 and worker_b >= 1

    output = {
        "schema_version": 1,
        "verified_contract_version": "v3.7.0",
        "all_passed": True,
        "runs": {
            "run_a": {
                "all_passed": run_a["all_passed"],
                "enkai_elapsed_ms": run_a["benchmark"]["enkai_accel"]["elapsed_ms"],
                "worker_count": worker_a,
                "train_report_hash": run_a["_train_report_hash"],
                "checkpoint_hash": run_a["_checkpoint_hash"],
            },
            "run_b": {
                "all_passed": run_b["all_passed"],
                "enkai_elapsed_ms": run_b["benchmark"]["enkai_accel"]["elapsed_ms"],
                "worker_count": worker_b,
                "train_report_hash": run_b["_train_report_hash"],
                "checkpoint_hash": run_b["_checkpoint_hash"],
            },
        },
        "threading": {
            "cpu_count": cpu_count,
            "threaded_expected": threaded_expected,
            "threaded_path_engaged": threaded_engaged,
            "worker_count_match": worker_a == worker_b,
            "worker_count": worker_a,
            "kernel_match": run_a["benchmark"]["enkai_accel"]["kernel"] == run_b["benchmark"]["enkai_accel"]["kernel"],
            "kernel": run_a["benchmark"]["enkai_accel"]["kernel"],
        },
        "determinism": {
            "loss_match": run_a["benchmark"]["enkai_accel"]["loss"] == run_b["benchmark"]["enkai_accel"]["loss"],
            "python_loss_match": run_a["benchmark"]["python_baseline"]["loss"] == run_b["benchmark"]["python_baseline"]["loss"],
            "cpu_loss_match": run_a["benchmark"]["cpu_scalar_baseline"]["loss"] == run_b["benchmark"]["cpu_scalar_baseline"]["loss"],
            "train_report_hash_match": run_a["_train_report_hash"] == run_b["_train_report_hash"],
            "checkpoint_hash_match": run_a["_checkpoint_hash"] == run_b["_checkpoint_hash"],
            "checkpoint_bytes_match": run_a["benchmark"]["enkai_accel"]["checkpoint_bytes"] == run_b["benchmark"]["enkai_accel"]["checkpoint_bytes"],
        },
        "artifacts": {
            "run_a_report": str(run_a_path),
            "run_b_report": str(run_b_path),
        },
    }
    output["all_passed"] = all([
        output["runs"]["run_a"]["all_passed"],
        output["runs"]["run_b"]["all_passed"],
        output["threading"]["threaded_path_engaged"],
        output["threading"]["worker_count_match"],
        output["threading"]["kernel_match"],
        output["determinism"]["loss_match"],
        output["determinism"]["python_loss_match"],
        output["determinism"]["cpu_loss_match"],
        output["determinism"]["train_report_hash_match"],
        output["determinism"]["checkpoint_hash_match"],
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return 0 if output["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
