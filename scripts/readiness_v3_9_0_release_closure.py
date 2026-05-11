#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run(command: list[str], cwd: Path, timeout: int) -> dict:
    started = time.perf_counter()
    try:
        proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=timeout)
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run v3.9.0 release closure checks.")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--output", default="artifacts/readiness/v3_9_0_release_closure.json")
    parser.add_argument("--skip-workspace-tests", action="store_true")
    args = parser.parse_args()
    root = Path(args.workspace).resolve()

    checks = {
        "cargo_build_enkai": run(["cargo", "build", "-p", "enkai"], root, 900),
        "api_lock_verify": run([sys.executable, "scripts/verify_v3_9_0_package_model_api_lock.py", "--workspace", "."], root, 120),
        "cuda_manifest_tests": run(["cargo", "test", "-p", "enkai_tensor", "--no-default-features", "--test", "cuda_kernel_manifest"], root, 300),
    }
    if args.skip_workspace_tests:
        checks["cargo_test_workspace"] = {"passed": False, "skipped": True, "reason": "explicitly skipped"}
    else:
        checks["cargo_test_workspace"] = run(
            ["cargo", "test", "--workspace", "--", "--test-threads=1"],
            root,
            7200,
        )

    failures = [name for name, check in checks.items() if check.get("passed") is not True]
    result = {
        "schema_version": 1,
        "contract_version": "v3.9.0",
        "scope": "release_closure",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "all_passed": not failures,
        "failures": failures,
        "checks": checks,
    }
    write_json(root / args.output, result)
    print(json.dumps({"all_passed": result["all_passed"], "failures": failures, "output": args.output}, indent=2))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
