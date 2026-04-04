#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def run_command(command: list[str], cwd: Path, env: dict[str, str]) -> dict:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    return {
        "command": command,
        "exit_code": result.returncode,
        "passed": result.returncode == 0,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def write_json(path: Path, payload: object) -> None:
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
    ffi_safety_artifact = workspace / "artifacts/validation/ffi_safety.json"

    env = dict(os.environ)
    env.setdefault("ENKAI_STD", str((workspace / "std").resolve()))
    env.setdefault("CARGO_BUILD_JOBS", "1")

    checks = []
    checks.append(
        {
            "id": "ffi_native_build",
            **run_command(["cargo", "build", "-p", "enkai_native"], workspace, env),
        }
    )
    checks.append(
        {
            "id": "ffi_safety_validation",
            **run_command(
                [
                    str(enkai_bin),
                    "validate",
                    "ffi-safety",
                    "--json",
                    "--output",
                    str(ffi_safety_artifact),
                ],
                workspace,
                env,
            ),
            "artifact": str(ffi_safety_artifact),
        }
    )

    targeted_tests = [
        (
            "ffi_native_invalid_handle",
            ["cargo", "test", "-p", "enkai_native", "invalid_handle_access_is_counted_without_crashing"],
        ),
        (
            "ffi_native_double_free",
            ["cargo", "test", "-p", "enkai_native", "double_free_of_handles_is_ignored_and_counted"],
        ),
        (
            "ffi_wrong_handle_kind",
            ["cargo", "test", "-p", "enkairt", "--test", "ffi", "ffi_wrong_handle_kind_is_rejected_and_counted"],
        ),
        (
            "ffi_fault_injection",
            ["cargo", "test", "-p", "enkairt", "--test", "ffi", "ffi_fault_injection_errors_are_stable"],
        ),
        (
            "sim_corrupted_replay",
            ["cargo", "test", "-p", "enkairt", "--test", "interpreter", "sim_restore_rejects_corrupted_snapshots"],
        ),
    ]
    for check_id, command in targeted_tests:
        checks.append({"id": check_id, **run_command(command, workspace, env)})

    payload = {
        "schema_version": 1,
        "validation": "runtime_safety",
        "checks": checks,
        "all_passed": all(check["passed"] for check in checks),
    }
    write_json(output, payload)
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
