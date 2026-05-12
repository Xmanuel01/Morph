#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(command: list[str], cwd: Path, timeout: int = 180) -> dict[str, Any]:
    proc = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return {
        "command": command,
        "exit_code": proc.returncode,
        "passed": proc.returncode == 0,
        "stdout_tail": proc.stdout[-8000:],
        "stderr_tail": proc.stderr[-8000:],
    }


def artifact_ok(path: Path) -> tuple[bool, dict[str, Any] | None, str | None]:
    if not path.is_file():
        return False, None, f"missing artifact: {path}"
    try:
        payload = read_json(path)
    except Exception as exc:
        return False, None, f"invalid json {path}: {exc}"

    mode = payload.get("mode")
    if mode == "grpc":
        refs = [
            payload.get("probe"),
            payload.get("server_log"),
            payload.get("conversation_state"),
        ]
        refs_ok = all(ref and Path(ref).is_file() for ref in refs)
        ok = payload.get("schema_version") == 1 and payload.get("serve_exit_code") == 0 and refs_ok
        return ok, payload, None if ok else f"grpc smoke summary incomplete: {path}"

    if mode == "mobile":
        refs = [
            payload.get("report"),
            payload.get("sdk_snapshot"),
            payload.get("app_json"),
            payload.get("package_json"),
        ]
        refs_ok = all(ref and Path(ref).is_file() for ref in refs)
        ok = payload.get("schema_version") == 1 and refs_ok
        return ok, payload, None if ok else f"mobile smoke summary incomplete: {path}"

    status = payload.get("status")
    passed = payload.get("passed")
    success = payload.get("success")
    all_passed = payload.get("all_passed")
    ok = status == "ok" or passed is True or success is True or all_passed is True
    return ok, payload, None if ok else f"artifact not green: {path}"


def contains_all(text: str, markers: list[str]) -> list[str]:
    return [marker for marker in markers if marker not in text]


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify bounded MySQL/gRPC/mobile app platform closure.")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--contract", default="enkai/contracts/v4_0_0_app_platform_closure.json")
    parser.add_argument("--output", default="artifacts/readiness/v4_0_0_app_platform_closure.json")
    parser.add_argument("--enkai-bin", default="target/debug/enkai.exe")
    parser.add_argument("--run", action="store_true", help="Regenerate gRPC and mobile evidence before verification.")
    args = parser.parse_args()

    root = Path(args.workspace).resolve()
    contract = read_json(root / args.contract)
    enkai_bin = (root / args.enkai_bin).resolve()
    failures: list[str] = []
    commands: list[dict[str, Any]] = []

    if args.run:
        if not enkai_bin.is_file():
            failures.append(f"enkai binary missing for --run: {enkai_bin}")
        else:
            commands.append(run([
                str(enkai_bin), "--version"
            ], root, timeout=60))
            commands.append(run([
                "py", "scripts/readiness_grpc_smoke.py", "--enkai-bin", str(enkai_bin), "--workspace", ".", "--output", "artifacts/readiness/grpc_smoke.json"
            ], root, timeout=240))
            commands.append(run([
                "py", "scripts/verify_grpc_evidence.py", "--summary", "artifacts/readiness/grpc_smoke.json", "--probe", "artifacts/grpc/probe.json", "--server-log", "artifacts/grpc/server.jsonl", "--conversation-state", "artifacts/grpc/conversation_state.json", "--output", "artifacts/readiness/grpc_evidence_verify.json"
            ], root, timeout=120))
            commands.append(run([
                "py", "scripts/readiness_mobile_scaffold_smoke.py", "--enkai-bin", str(enkai_bin), "--workspace", ".", "--output", "artifacts/readiness/deploy_mobile_smoke.json"
            ], root, timeout=240))
            commands.append(run([
                "py", "scripts/verify_mobile_scaffold_evidence.py", "--summary", "artifacts/readiness/deploy_mobile_smoke.json", "--report", "artifacts/readiness/deploy_mobile.json", "--sdk-snapshot", "artifacts/mobile/sdk_api.snapshot.json", "--app-json", "artifacts/mobile/app.json", "--package-json", "artifacts/mobile/package.json", "--output", "artifacts/readiness/deploy_mobile_evidence_verify.json"
            ], root, timeout=120))
            for result in commands:
                if not result["passed"]:
                    failures.append(f"command failed: {' '.join(result['command'])}")

    std_db = (root / "std/db.enk").read_text(encoding="utf-8")
    native = (root / "enkai_native/src/lib.rs").read_text(encoding="utf-8")
    vm = (root / "enkairt/src/vm.rs").read_text(encoding="utf-8")
    ffi_tests = (root / "enkairt/tests/ffi_modules.rs").read_text(encoding="utf-8")

    required_std = [f"pub fn {name}" for name in contract["mysql_policy"]["required_std_functions"]]
    missing_std = contains_all(std_db, required_std)
    missing_native = contains_all(native, [
        "db_mysql_open",
        "db_mysql_close",
        "db_mysql_transaction_begin",
        "db_mysql_transaction_commit",
        "db_mysql_transaction_rollback",
        "db_mysql_exec",
        "db_mysql_exec_many",
        "db_mysql_query",
        "mysql_transaction_apis_fail_closed_for_invalid_handles",
    ])
    missing_vm = contains_all(vm, [
        "db_mysql_transaction_begin",
        "db_mysql_transaction_commit",
        "db_mysql_transaction_rollback",
        "db_mysql_exec_many",
    ])
    missing_tests = contains_all(ffi_tests, ["std_db_mysql_transaction_apis_fail_closed_for_invalid_handle"])
    if missing_std:
        failures.append(f"std::db missing MySQL APIs: {missing_std}")
    if missing_native:
        failures.append(f"native MySQL surface missing markers: {missing_native}")
    if missing_vm:
        failures.append(f"VM policy surface missing MySQL markers: {missing_vm}")
    if missing_tests:
        failures.append(f"MySQL fail-closed tests missing markers: {missing_tests}")

    artifact_results: dict[str, Any] = {}
    for rel in contract["required_artifacts"]:
        if rel == args.output:
            continue
        ok, payload, error = artifact_ok(root / rel)
        artifact_results[rel] = {"ok": ok, "error": error, "payload": payload}
        if not ok:
            failures.append(error or f"artifact failed: {rel}")

    grpc_probe = read_json(root / "artifacts/grpc/probe.json") if (root / "artifacts/grpc/probe.json").is_file() else {}
    mobile_package = read_json(root / "artifacts/mobile/package.json") if (root / "artifacts/mobile/package.json").is_file() else {}
    gates = {
        "mysql_transaction_api_surface": not missing_std and not missing_native and not missing_vm,
        "mysql_fail_closed_without_live_service": not missing_tests,
        "grpc_health_ready_chat_stream": grpc_probe.get("health", {}).get("status") == "ok" and grpc_probe.get("ready", {}).get("status") == "ready" and grpc_probe.get("stream", [{}])[-1].get("event") == "done",
        "grpc_conversation_persistence": artifact_results.get("artifacts/readiness/grpc_evidence_verify.json", {}).get("ok") is True,
        "mobile_strict_deploy_validation": artifact_results.get("artifacts/readiness/deploy_mobile_evidence_verify.json", {}).get("ok") is True,
        "mobile_sdk_snapshot_and_package": "expo" in json.dumps(mobile_package),
        "no_unproven_external_service_claim": os.environ.get("ENKAI_MYSQL_URL") is None,
    }
    for gate in contract["required_gates"]:
        if gates.get(gate) is not True:
            failures.append(f"gate failed: {gate}")

    payload = {
        "schema_version": 1,
        "contract_version": contract["contract_version"],
        "scope": contract["scope"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "commands": commands,
        "gates": gates,
        "artifacts": artifact_results,
        "production_claims": {
            "mysql_api_closure_proven": gates["mysql_transaction_api_surface"] and gates["mysql_fail_closed_without_live_service"],
            "live_mysql_service_proven": False,
            "grpc_local_end_to_end_proven": gates["grpc_health_ready_chat_stream"] and gates["grpc_conversation_persistence"],
            "mobile_scaffold_package_closure_proven": gates["mobile_strict_deploy_validation"] and gates["mobile_sdk_snapshot_and_package"],
            "external_production_deployment_claimed": False,
        },
        "all_passed": not failures,
        "failures": failures,
    }
    write_json(root / args.output, payload)
    print(json.dumps({"all_passed": payload["all_passed"], "failures": failures, "output": args.output}, indent=2))
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
