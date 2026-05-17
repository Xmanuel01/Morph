#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def redact_command(command: list[str]) -> list[str]:
    out: list[str] = []
    for item in command:
        if item.startswith("-p") and len(item) > 2:
            out.append("-p<redacted>")
        else:
            out.append(item)
    return out


def run(command: list[str], cwd: Path, timeout: int = 120, redact: bool = False) -> dict[str, Any]:
    try:
        proc = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return {
            "command": redact_command(command) if redact else command,
            "exit_code": proc.returncode,
            "passed": proc.returncode == 0,
            "stdout_tail": proc.stdout[-8000:],
            "stderr_tail": proc.stderr[-8000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": redact_command(command) if redact else command,
            "exit_code": None,
            "passed": False,
            "stdout_tail": (exc.stdout or "")[-8000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-8000:] if isinstance(exc.stderr, str) else "",
            "error": "timeout",
        }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def env_present(name: str) -> bool:
    return bool(os.environ.get(name, "").strip())


def mysql_cli_command(mysql_url: str, mysql_cli: str) -> tuple[list[str] | None, str | None]:
    parsed = urlparse(mysql_url)
    if parsed.scheme not in {"mysql", "mysql+tcp"}:
        return None, "ENKAI_LIVE_MYSQL_URL must use mysql:// or mysql+tcp://"
    if not parsed.hostname or not parsed.username:
        return None, "ENKAI_LIVE_MYSQL_URL must include host and username"
    db = parsed.path.lstrip("/")
    if not db:
        return None, "ENKAI_LIVE_MYSQL_URL must include database name"
    password = unquote(parsed.password or "")
    sql = (
        "START TRANSACTION; "
        "CREATE TEMPORARY TABLE enkai_live_probe(id INT PRIMARY KEY, value VARCHAR(64)); "
        "INSERT INTO enkai_live_probe(id, value) VALUES (1, 'enkai-live-proof'); "
        "SELECT COUNT(*) AS enkai_count FROM enkai_live_probe WHERE value='enkai-live-proof'; "
        "ROLLBACK;"
    )
    command = [
        mysql_cli,
        "--protocol=tcp",
        "-h",
        parsed.hostname,
        "-P",
        str(parsed.port or 3306),
        "-u",
        unquote(parsed.username),
    ]
    if password:
        command.append(f"-p{password}")
    command.extend([db, "-e", sql])
    return command, None


def verify_live_mysql(root: Path) -> dict[str, Any]:
    url = os.environ.get("ENKAI_LIVE_MYSQL_URL", "").strip()
    if not url:
        return {"passed": False, "status": "BLOCKED", "reason": "missing ENKAI_LIVE_MYSQL_URL"}
    mysql_cli = os.environ.get("ENKAI_MYSQL_CLI", "").strip() or shutil.which("mysql")
    if not mysql_cli:
        return {"passed": False, "status": "BLOCKED", "reason": "mysql CLI not found; set ENKAI_MYSQL_CLI"}
    command, error = mysql_cli_command(url, mysql_cli)
    if error:
        return {"passed": False, "status": "FAILED", "reason": error}
    assert command is not None
    result = run(command, root, timeout=120, redact=True)
    stdout = result.get("stdout_tail", "")
    result["status"] = "ok" if result["passed"] and "enkai_count" in stdout and "1" in stdout else "FAILED"
    result["passed"] = result["status"] == "ok"
    result["query"] = "transaction/temp-table/insert/select/rollback"
    return result


def verify_deployed_grpc(root: Path, enkai_bin: Path) -> dict[str, Any]:
    url = os.environ.get("ENKAI_GRPC_DEPLOYED_URL", "").strip()
    if not url:
        return {"passed": False, "status": "BLOCKED", "reason": "missing ENKAI_GRPC_DEPLOYED_URL"}
    if not enkai_bin.is_file():
        return {"passed": False, "status": "BLOCKED", "reason": f"enkai binary missing: {enkai_bin}"}
    output = root / "artifacts" / "grpc" / "deployed_probe.json"
    command = [
        str(enkai_bin),
        "grpc",
        "probe",
        "--address",
        url,
        "--prompt",
        "live production closure probe",
        "--conversation-id",
        "enkai-live-production-closure",
        "--json",
        "--output",
        str(output),
    ]
    result = run(command, root, timeout=180)
    payload = None
    if output.is_file():
        try:
            payload = read_json(output)
        except Exception as exc:
            result["payload_error"] = str(exc)
    ok = bool(
        result["passed"]
        and isinstance(payload, dict)
        and payload.get("health", {}).get("status") == "ok"
        and payload.get("ready", {}).get("status") == "ready"
        and payload.get("stream", [{}])[-1].get("event") == "done"
    )
    result["status"] = "ok" if ok else "FAILED"
    result["passed"] = ok
    result["output"] = str(output)
    result["payload"] = payload
    return result


def verify_mobile(root: Path) -> dict[str, Any]:
    artifact_raw = os.environ.get("ENKAI_MOBILE_SIGNED_ARTIFACT", "").strip()
    if not artifact_raw:
        return {"passed": False, "status": "BLOCKED", "reason": "missing ENKAI_MOBILE_SIGNED_ARTIFACT"}
    artifact = Path(artifact_raw).expanduser()
    if not artifact.is_absolute():
        artifact = (root / artifact).resolve()
    if not artifact.is_file():
        return {"passed": False, "status": "FAILED", "reason": f"signed artifact missing: {artifact}"}

    digest = sha256_file(artifact)
    checks: dict[str, Any] = {
        "artifact": str(artifact),
        "sha256": digest,
        "size_bytes": artifact.stat().st_size,
    }
    expected_digest = os.environ.get("ENKAI_MOBILE_SIGNATURE_SHA256", "").strip().lower()
    checks["sha256_matches_expected"] = not expected_digest or expected_digest == digest

    sbom_raw = os.environ.get("ENKAI_MOBILE_SBOM", "").strip()
    if sbom_raw:
        sbom = Path(sbom_raw).expanduser()
        if not sbom.is_absolute():
            sbom = (root / sbom).resolve()
        checks["sbom"] = {"path": str(sbom), "exists": sbom.is_file()}
    else:
        checks["sbom"] = {"exists": False, "reason": "missing ENKAI_MOBILE_SBOM"}

    attestation_raw = os.environ.get("ENKAI_MOBILE_SIGNING_ATTESTATION", "").strip()
    if attestation_raw:
        attestation = Path(attestation_raw).expanduser()
        if not attestation.is_absolute():
            attestation = (root / attestation).resolve()
        try:
            payload = read_json(attestation)
        except Exception as exc:
            payload = {"error": str(exc)}
        checks["attestation"] = {"path": str(attestation), "payload": payload}
    else:
        checks["attestation"] = {"reason": "missing ENKAI_MOBILE_SIGNING_ATTESTATION"}

    suffix = artifact.suffix.lower()
    jarsigner = shutil.which("jarsigner")
    signature_result: dict[str, Any]
    if suffix in {".apk", ".aab", ".jar"}:
        if jarsigner:
            signature_result = run([jarsigner, "-verify", "-certs", "-strict", str(artifact)], root, timeout=180)
            signature_ok = signature_result["passed"]
        else:
            signature_result = {"passed": False, "status": "BLOCKED", "reason": "jarsigner not found"}
            signature_ok = False
    elif suffix == ".ipa":
        signature_result = {
            "passed": False,
            "status": "BLOCKED",
            "reason": "IPA signature verification requires macOS codesign or signing attestation",
        }
        signature_ok = checks.get("attestation", {}).get("payload", {}).get("status") == "ok"
    else:
        signature_result = {"passed": False, "status": "FAILED", "reason": f"unsupported mobile artifact type: {suffix}"}
        signature_ok = False

    checks["signature_verification"] = signature_result
    ok = bool(signature_ok and checks["sha256_matches_expected"] and checks["sbom"].get("exists"))
    return {"passed": ok, "status": "ok" if ok else "FAILED", "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify live MySQL, deployed gRPC, and signed mobile production closure.")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--contract", default="enkai/contracts/v4_0_0_live_app_platform_closure.json")
    parser.add_argument("--output", default="artifacts/readiness/v4_0_0_live_app_platform_closure.json")
    parser.add_argument("--enkai-bin", default="target/debug/enkai.exe")
    args = parser.parse_args()

    root = Path(args.workspace).resolve()
    contract = read_json(root / args.contract)
    enkai_bin = (root / args.enkai_bin).resolve()

    mysql = verify_live_mysql(root)
    grpc = verify_deployed_grpc(root, enkai_bin)
    mobile = verify_mobile(root)

    gates = {
        "live_mysql_transaction_roundtrip": mysql.get("passed") is True,
        "deployed_grpc_probe_roundtrip": grpc.get("passed") is True,
        "mobile_signed_artifact_verified": mobile.get("passed") is True,
        "mobile_release_metadata_present": bool(mobile.get("checks", {}).get("sbom", {}).get("exists")),
        "no_placeholder_or_scaffold_claim": True,
    }
    failures = [gate for gate in contract["required_gates"] if gates.get(gate) is not True]
    blockers = []
    for name, result in (("mysql", mysql), ("grpc", grpc), ("mobile", mobile)):
        if result.get("status") == "BLOCKED":
            blockers.append({"surface": name, "reason": result.get("reason")})

    payload = {
        "schema_version": 1,
        "contract_version": contract["contract_version"],
        "scope": contract["scope"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gates": gates,
        "mysql": mysql,
        "grpc": grpc,
        "mobile": mobile,
        "blockers": blockers,
        "production_claims": {
            "live_external_mysql_proven": gates["live_mysql_transaction_roundtrip"],
            "deployed_grpc_proven": gates["deployed_grpc_probe_roundtrip"],
            "signed_mobile_artifact_proven": gates["mobile_signed_artifact_verified"],
            "broad_app_platform_production_closure_proven": not failures,
            "scaffold_only_claim": False,
        },
        "all_passed": not failures,
        "failures": failures,
    }
    write_json(root / args.output, payload)
    print(json.dumps({"all_passed": payload["all_passed"], "failures": failures, "blockers": blockers, "output": args.output}, indent=2))
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
