#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_command(command: list[str], cwd: Path, env: dict[str, str], timeout: int) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        proc = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, timeout=timeout)
        return {
            "command": command,
            "exit_code": proc.returncode,
            "passed": proc.returncode == 0,
            "elapsed_ms": max(1, int((time.perf_counter() - started) * 1000)),
            "stdout_tail": proc.stdout[-8000:],
            "stderr_tail": proc.stderr[-8000:],
        }
    except Exception as exc:
        return {
            "command": command,
            "exit_code": 1,
            "passed": False,
            "elapsed_ms": max(1, int((time.perf_counter() - started) * 1000)),
            "stdout_tail": "",
            "stderr_tail": repr(exc),
        }


def artifact_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "passed": False, "error": f"missing artifact: {path}"}
    try:
        payload = read_json(path)
    except Exception as exc:
        return {"exists": True, "passed": False, "error": repr(exc)}
    status = payload.get("status")
    return {
        "exists": True,
        "passed": status == "PASS",
        "status": status,
        "reason": payload.get("reason") or payload.get("failure_reason"),
        "payload": payload,
    }


def failed_artifact_message(label: str, result: dict[str, Any]) -> str:
    if result.get("error"):
        return str(result["error"])
    status = result.get("status") or "not PASS"
    reason = result.get("reason")
    if reason:
        return f"{label} evidence {status}: {reason}"
    return f"{label} evidence did not pass"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run/collect v3.9.0 distributed GPU execution proof.")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--contract", default="enkai/contracts/v3_9_0_distributed_gpu_execution.json")
    parser.add_argument("--output", default="artifacts/readiness/v3_9_0_distributed_gpu_execution.json")
    parser.add_argument("--run", action="store_true", help="Run the 2-rank GPU harness before collecting evidence.")
    parser.add_argument("--run-soak4", action="store_true", help="Run the 4-GPU soak harness before collecting evidence.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout-sec", type=int, default=3600)
    args = parser.parse_args()

    root = Path(args.workspace).resolve()
    contract = read_json(root / args.contract)
    env = os.environ.copy()
    env.setdefault("ENKAI_ENABLE_DIST", "1")
    env.setdefault("ENKAI_RUN_MULTI_GPU_TESTS", "1")
    env.setdefault("ENKAI_SINGLE_GPU_GREEN", "1")
    env.setdefault("ENKAI_GPU_ARTIFACT_DIR", str((root / "artifacts" / "gpu").resolve()))

    commands: list[dict[str, Any]] = []
    if args.run:
        commands.append(run_command([args.python, "scripts/gpu_harness.py", "multi"], root, env, args.timeout_sec))
    if args.run_soak4:
        commands.append(run_command([args.python, "scripts/gpu_harness.py", "soak4"], root, env, max(args.timeout_sec, 14400)))

    multi_path = root / contract["two_rank_requirements"]["required_artifact"]
    soak_path = root / contract["four_rank_soak_requirements"]["required_artifact"]
    multi = artifact_status(multi_path)
    soak = artifact_status(soak_path)

    failures: list[str] = []
    if not multi.get("passed"):
        failures.append(failed_artifact_message("2-rank distributed GPU", multi))
    multi_payload = multi.get("payload", {}) if isinstance(multi.get("payload"), dict) else {}
    if multi_payload and multi_payload.get("status") == contract["two_rank_requirements"]["required_status"]:
        if int(multi_payload.get("world_size", 0)) != int(contract["two_rank_requirements"]["world_size"]):
            failures.append("2-rank evidence world_size mismatch")
        checks = multi_payload.get("checks", {})
        for check in contract["two_rank_requirements"]["required_checks"]:
            if checks.get(check) is not True:
                failures.append(f"2-rank distributed check failed: {check}")
        if len(multi_payload.get("ranks", [])) != 2:
            failures.append("2-rank evidence must include exactly two rank reports")
        artifacts = multi_payload.get("artifacts", {})
        for key in ["rank0_grads", "rank1_grads", "stdout_rank0", "stderr_rank0", "stdout_rank1", "stderr_rank1"]:
            path = artifacts.get(key)
            if not path or not Path(path).exists():
                failures.append(f"missing archived distributed artifact: {key}")

    if args.run_soak4 or os.environ.get("ENKAI_REQUIRE_4GPU_SOAK") == "1":
        if not soak.get("passed"):
            failures.append(failed_artifact_message("4-GPU soak", soak))
        soak_payload = soak.get("payload", {}) if isinstance(soak.get("payload"), dict) else {}
        if soak_payload and float(soak_payload.get("runtime_hours", 0.0)) < float(contract["four_rank_soak_requirements"]["minimum_hours_default"]):
            failures.append("4-GPU soak runtime below contract minimum")

    payload = {
        "schema_version": 1,
        "contract_version": contract["contract_version"],
        "scope": contract["scope"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {"platform": platform.platform(), "python": args.python, "cwd": str(root)},
        "commands": commands,
        "two_rank": multi,
        "four_rank_soak": soak,
        "production_claims": {
            "distributed_gpu_execution_proven": bool(not failures and multi.get("passed")),
            "four_gpu_soak_proven": bool(soak.get("passed")),
            "claim_without_hardware_evidence": False,
        },
        "all_passed": not failures,
        "failures": failures,
    }
    write_json(root / args.output, payload)
    print(json.dumps({"all_passed": payload["all_passed"], "failures": failures, "output": args.output}, indent=2))
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
