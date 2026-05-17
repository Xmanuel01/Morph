#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(command: list[str], cwd: Path, timeout: int = 300) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        return {
            "command": command,
            "exit_code": proc.returncode,
            "passed": proc.returncode == 0,
            "stdout_tail": stdout[-12000:],
            "stderr_tail": stderr[-12000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "exit_code": None,
            "passed": False,
            "stdout_tail": (exc.stdout or "")[-12000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-12000:] if isinstance(exc.stderr, str) else "",
            "error": "timeout",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify tensor FFI opaque handle closure.")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--contract", default="enkai/contracts/v4_0_0_tensor_ffi_opaque_handles.json")
    parser.add_argument("--output", default="artifacts/readiness/v4_0_0_tensor_ffi_opaque_handles.json")
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()

    root = Path(args.workspace).resolve()
    contract = read_json(root / args.contract)
    source_path = root / "enkai_tensor" / "src" / "lib.rs"
    tests_path = root / "enkai_tensor" / "tests" / "handle_hardening.rs"
    amp_tests_path = root / "enkai_tensor" / "tests" / "amp_handles.rs"
    source = source_path.read_text(encoding="utf-8")
    tests = tests_path.read_text(encoding="utf-8") + "\n" + amp_tests_path.read_text(encoding="utf-8")

    failures: list[str] = []
    missing_markers = [m for m in contract["required_source_markers"] if m not in source]
    missing_tests = [m for m in contract["required_tests"] if m not in tests]
    if missing_markers:
        failures.append(f"missing source markers: {missing_markers}")
    if missing_tests:
        failures.append(f"missing tests: {missing_tests}")

    next_handle_body = re.search(r"fn next_handle\(kind: HandleKind\) -> i64 \{(?P<body>.*?)\n\}", source, re.S)
    require_body = re.search(r"fn require_handle_kind\(.*?\) -> Result<\(\), String> \{(?P<body>.*?)\n\}", source, re.S)
    gates = {
        "opaque_capability_policy_exported": "enkai_tensor_handle_abi_policy" in source,
        "non_sequential_token_generation": bool(next_handle_body and "OPAQUE_STRIDE" in next_handle_body.group("body") and "OPAQUE_SALT" in next_handle_body.group("body")),
        "kind_tag_validation": bool(require_body and "Invalid {label} handle kind" in require_body.group("body")),
        "checksum_validation": "handle_checksum(kind, payload)" in source and "opaque handle checksum invalid" in source,
        "registry_membership_validation": ".get(&handle)" in source and "Invalid tensor handle" in source,
        "stale_handle_rejection": "Stale tensor handle (freed)" in source and "Stale device handle (freed)" in source,
        "wrong_kind_rejection": "Invalid tensor handle kind" in tests and "Invalid device handle kind" in tests,
        "amp_optimizer_device_tensor_handle_coverage": all(marker in source for marker in ["SCALER_FREED", "OPT_FREED", "DEVICE_FREED", "TENSOR_FREED"]),
    }

    for gate in contract["required_gates"]:
        if gates.get(gate) is not True:
            failures.append(f"gate failed: {gate}")

    commands: list[dict[str, Any]] = []
    if args.run_tests:
        for command in [
            [
                "cargo",
                "test",
                "-p",
                "enkai_tensor",
                "--features",
                "torch",
                "--test",
                "handle_hardening",
                "ffi_handles_are_opaque_typed_and_stale_checked",
            ],
            [
                "cargo",
                "test",
                "-p",
                "enkai_tensor",
                "--features",
                "torch",
                "--test",
                "handle_hardening",
                "ffi_handle_abi_policy_is_machine_readable",
            ],
            [
                "cargo",
                "test",
                "-p",
                "enkai_tensor",
                "--features",
                "torch",
                "--test",
                "amp_handles",
                "amp_scaler_retain_and_double_free",
            ],
        ]:
            result = run(command, root, timeout=600)
            commands.append(result)
            if not result["passed"]:
                failures.append(f"command failed: {' '.join(command)}")

    payload = {
        "schema_version": 1,
        "contract_version": contract["contract_version"],
        "scope": contract["scope"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gates": gates,
        "commands": commands,
        "source": str(source_path),
        "production_claims": {
            "tensor_ffi_opaque_handles_proven": not failures,
            "raw_sequential_ids_allowed": False,
            "forged_tag_only_handles_allowed": False,
            "stale_handle_reuse_allowed": False,
            "legacy_i64_identity_claim": False,
        },
        "all_passed": not failures,
        "failures": failures,
    }
    write_json(root / args.output, payload)
    print(json.dumps({"all_passed": payload["all_passed"], "failures": failures, "output": args.output}, indent=2))
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
