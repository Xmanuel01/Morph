#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from socket import create_connection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the native std/accel self-host CPU slice without the enkai_native library."
    )
    parser.add_argument("--enkai-bin", required=True, help="Path to the built enkai executable")
    parser.add_argument(
        "--contract",
        default="enkai/contracts/selfhost_native_std_accel_v3_3_0.json",
        help="Path to the native-std/accel slice contract JSON",
    )
    parser.add_argument(
        "--output",
        default="artifacts/readiness/strict_selfhost_native_std_accel_slice.json",
        help="Path to write the verification report JSON",
    )
    return parser.parse_args()


def run_command(command: list[str], cwd: Path, extra_env: dict[str, str] | None = None) -> dict[str, object]:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env=env,
    )
    return {
        "command": command,
        "cwd": str(cwd),
        "exit_code": completed.returncode,
        "ok": completed.returncode == 0,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def native_library_name() -> str:
    if os.name == "nt":
        return "enkai_native.dll"
    if sys.platform == "darwin":
        return "libenkai_native.dylib"
    return "libenkai_native.so"


def candidate_native_paths(root: Path, enkai_bin: Path) -> list[Path]:
    library_name = native_library_name()
    candidates: list[Path] = []
    if enkai_bin.parent.is_dir():
        candidates.append(enkai_bin.parent / library_name)
        candidates.append(enkai_bin.parent / "deps" / library_name)
    candidates.append(root / "target" / "debug" / library_name)
    candidates.append(root / "target" / "debug" / "deps" / library_name)
    for child in sorted(root.iterdir()):
        if child.is_dir() and child.name.startswith("target"):
            candidates.append(child / "debug" / library_name)
            candidates.append(child / "debug" / "deps" / library_name)
    seen: set[Path] = set()
    out: list[Path] = []
    for candidate in candidates:
        if candidate.is_file():
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                out.append(resolved)
    return out


def hide_native_paths(paths: list[Path]) -> list[tuple[Path, Path]]:
    hidden: list[tuple[Path, Path]] = []
    for path in paths:
        backup = path.with_name(path.name + ".nativeproof.bak")
        if backup.exists():
            backup.unlink()
        path.rename(backup)
        hidden.append((path, backup))
    return hidden


def restore_native_paths(hidden: list[tuple[Path, Path]]) -> None:
    for original, backup in reversed(hidden):
        if backup.exists():
            backup.rename(original)


def read_json_if_exists(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_project(root: Path) -> Path:
    project = root / "entrypoint_project"
    (project / "src").mkdir(parents=True, exist_ok=True)
    (project / "tests").mkdir(parents=True, exist_ok=True)
    (project / "enkai.toml").write_text("name = 'native-std-proof'\nversion = '0.1.0'\n", encoding="utf-8")
    (project / "src" / "main.enk").write_text(
        "fn main() -> Int ::\n    return 0\n::\n\nmain()\n", encoding="utf-8"
    )
    (project / "tests" / "main.enk").write_text(
        "fn main() -> Int ::\n    return 0\n::\n", encoding="utf-8"
    )
    return project


def response_status(raw: bytes) -> int:
    line = raw.decode("utf-8", errors="replace").splitlines()[0] if raw else ""
    try:
        return int(line.split()[1])
    except Exception:
        return 0


def response_body(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace")
    marker = "\r\n\r\n"
    if marker in text:
        return text.split(marker, 1)[1]
    return ""


def read_http_response(port: int, request: bytes, *, timeout_s: float = 3.0) -> tuple[bytes, str | None]:
    response = b""
    connect_error = None
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with create_connection(("127.0.0.1", port), timeout=0.25) as sock:
                sock.sendall(request)
                while True:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    response += chunk
                connect_error = None
                break
        except OSError as err:
            connect_error = str(err)
            time.sleep(0.05)
    return response, connect_error


def prepare_backend_service(root: Path) -> Path:
    backend_dir = root / "backend_service"
    (backend_dir / "contracts").mkdir(parents=True, exist_ok=True)
    (backend_dir / "src").mkdir(parents=True, exist_ok=True)
    (backend_dir / "contracts" / "backend_api.snapshot.json").write_text("{}\n", encoding="utf-8")
    (backend_dir / "src" / "main.enk").write_text(
        "fn main() ::\n    return 0\n::\nmain()\n", encoding="utf-8"
    )
    return backend_dir


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    enkai_bin = Path(args.enkai_bin).resolve()
    contract_path = (root / args.contract).resolve() if not Path(args.contract).is_absolute() else Path(args.contract).resolve()
    output_path = (root / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output).resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    native_candidates = candidate_native_paths(root, enkai_bin)

    with tempfile.TemporaryDirectory(prefix="enkai_native_std_slice_") as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        project = prepare_project(temp_dir)
        backend_dir = prepare_backend_service(temp_dir)
        reports = temp_dir / "reports"
        reports.mkdir(parents=True, exist_ok=True)

        hidden_paths: list[tuple[Path, Path]] = []
        checks: list[dict[str, object]] = []
        all_passed = True
        required_entrypoints = {str(item) for item in contract.get("required_entrypoints", [])}
        try:
            hidden_paths = hide_native_paths(native_candidates)

            run_report = reports / "run_backend.json"
            run_result = run_command(
                [str(enkai_bin), "run", "--runtime-backend", "selfhost", str(root / "examples" / "hello" / "main.enk")],
                root,
                {"ENKAI_RUN_BACKEND_REPORT": str(run_report)},
            )
            run_backend = read_json_if_exists(run_report)
            run_ok = run_result["ok"] and run_backend is not None and run_backend.get("backend") == contract.get("required_runtime_backend")
            checks.append({
                "id": "run",
                "native_hidden": bool(hidden_paths),
                "result": run_result,
                "backend": run_backend,
                "passed": run_ok,
            })
            if "run" in required_entrypoints:
                all_passed &= run_ok

            check_report = reports / "check_backend.json"
            check_result = run_command(
                [str(enkai_bin), "check", str(project / "src" / "main.enk")],
                root,
                {"ENKAI_CHECK_BACKEND_REPORT": str(check_report)},
            )
            check_backend = read_json_if_exists(check_report)
            check_ok = check_result["ok"] and check_backend is not None and check_backend.get("backend") == contract.get("required_runtime_backend")
            checks.append({
                "id": "check",
                "result": check_result,
                "backend": check_backend,
                "passed": check_ok,
            })
            if "check" in required_entrypoints:
                all_passed &= check_ok

            build_report = reports / "build_backend.json"
            build_result = run_command(
                [str(enkai_bin), "build", str(project)],
                root,
                {"ENKAI_BUILD_BACKEND_REPORT": str(build_report)},
            )
            build_backend = read_json_if_exists(build_report)
            build_ok = build_result["ok"] and build_backend is not None and build_backend.get("backend") == contract.get("required_runtime_backend")
            checks.append({
                "id": "build",
                "result": build_result,
                "backend": build_backend,
                "passed": build_ok,
            })
            if "build" in required_entrypoints:
                all_passed &= build_ok

            test_report = reports / "test_backend.json"
            test_result = run_command(
                [str(enkai_bin), "test", str(project)],
                root,
                {"ENKAI_TEST_BACKEND_REPORT": str(test_report)},
            )
            test_backend = read_json_if_exists(test_report)
            test_entries = [] if not isinstance(test_backend, dict) else test_backend.get("entries", [])
            compiler_only = bool(contract.get("require_test_compiler_backend_only", False))
            test_ok = bool(
                test_result["ok"]
                and isinstance(test_backend, dict)
                and test_backend.get("failed") == 0
                and test_entries
                and all(
                    isinstance(row, dict)
                    and row.get("compiler_backend") == contract.get("required_runtime_backend")
                    and (
                        compiler_only
                        or row.get("runtime_backend") == contract.get("required_runtime_backend")
                    )
                    for row in test_entries
                )
            )
            checks.append({
                "id": "test",
                "result": test_result,
                "backend": test_backend,
                "passed": test_ok,
            })
            if "test" in required_entrypoints:
                all_passed &= test_ok

            serve_manifest = reports / "serve_manifest.json"
            serve_manifest_result = run_command(
                [
                    str(enkai_bin),
                    "systems",
                    "serve-manifest",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "18087",
                    str(backend_dir),
                    "--json",
                    "--output",
                    str(serve_manifest),
                ],
                root,
            )
            connect_error = None
            status = 0
            body = ""
            serve_proc = subprocess.Popen(
                [str(enkai_bin), "systems", "serve-exec", "--manifest", str(serve_manifest)],
                cwd=str(root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            try:
                request = b"GET /api/v1/health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"
                raw_response, connect_error = read_http_response(18087, request)
                status = response_status(raw_response)
                body = response_body(raw_response)
                serve_ok = (
                    serve_manifest_result["ok"]
                    and serve_manifest.is_file()
                    and connect_error is None
                    and status == 200
                    and "ok" in body.lower()
                )
            finally:
                serve_proc.terminate()
                try:
                    serve_stdout, serve_stderr = serve_proc.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    serve_proc.kill()
                    serve_stdout, serve_stderr = serve_proc.communicate()
            checks.append({
                "id": "serve-generic-health",
                "manifest_result": serve_manifest_result,
                "connect_error": connect_error,
                "http_status": status,
                "http_body": body,
                "serve_stdout": serve_stdout,
                "serve_stderr": serve_stderr,
                "passed": serve_ok,
            })
            all_passed &= serve_ok

        finally:
            restore_native_paths(hidden_paths)

    payload = {
        "schema_version": 1,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "contract": str(contract_path),
        "native_candidates": [str(path) for path in native_candidates],
        "hidden_native_paths": [str(original) for original, _ in hidden_paths],
        "all_passed": all_passed,
        "checks": checks,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok" if all_passed else "failed", "output": str(output_path), "all_passed": all_passed}, separators=(",", ":")))
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
