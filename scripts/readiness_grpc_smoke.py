#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enkai-bin", required=True)
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    enkai_bin = Path(args.enkai_bin)
    output = Path(args.output)
    if not enkai_bin.exists():
        raise SystemExit(f"enkai binary not found: {enkai_bin}")

    artifacts_dir = workspace / "artifacts" / "grpc"
    shutil.rmtree(artifacts_dir, ignore_errors=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="enkai_readiness_grpc_") as tmp:
        root = Path(tmp)
        target = root / "grpc-service"
        (target / "src").mkdir(parents=True, exist_ok=True)
        (target / "enkai.toml").write_text(
            "[package]\nname = \"grpc-service\"\nversion = \"0.1.0\"\n",
            encoding="utf-8",
        )
        (target / "src" / "main.enk").write_text(
            "import std::env\nimport std::time\n\n"
            "policy default ::\n"
            "    allow env\n"
            "::\n\n"
            "fn main() ::\n"
            "    let contract_mode := env.get(\"ENKAI_CONTRACT_TEST_MODE\")\n"
            "    if contract_mode != none ::\n"
            "        let until_ms := time.now_ms() + 3000\n"
            "        while time.now_ms() < until_ms ::\n"
            "            let tick_ms := time.now_ms()\n"
            "        ::\n"
            "        return\n"
            "    ::\n"
            "    while true ::\n"
            "        let tick_ms := time.now_ms()\n"
            "    ::\n"
            "::\n\nmain()\n",
            encoding="utf-8",
        )

        env_values = os.environ.copy()
        http_port = free_port()
        grpc_port = free_port()
        env_values.update(
            {
                "ENKAI_CONTRACT_TEST_MODE": "1",
                "ENKAI_STD": str((workspace / "std").resolve()),
                "ENKAI_SERVE_HOST": "127.0.0.1",
                "ENKAI_SERVE_PORT": str(http_port),
                "ENKAI_GRPC_HOST": "127.0.0.1",
                "ENKAI_GRPC_PORT": str(grpc_port),
                "ENKAI_CONVERSATION_DIR": str(artifacts_dir),
                "ENKAI_LOG_PATH": str(artifacts_dir / "server.jsonl"),
            }
        )

        proc = subprocess.Popen(
            [
                str(enkai_bin),
                "serve",
                "--host",
                "127.0.0.1",
                "--port",
                str(http_port),
                "--grpc-host",
                "127.0.0.1",
                "--grpc-port",
                str(grpc_port),
                str(target),
            ],
            cwd=workspace,
            env=env_values,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        probe_path = artifacts_dir / "probe.json"
        probe_error = None
        for _ in range(40):
            time.sleep(0.25)
            probe = subprocess.run(
                [
                    str(enkai_bin),
                    "grpc",
                    "probe",
                    "--address",
                    f"http://127.0.0.1:{grpc_port}",
                    "--api-version",
                    "v1",
                    "--prompt",
                    "grpc smoke",
                    "--output",
                    str(probe_path),
                ],
                cwd=workspace,
                env=env_values,
                capture_output=True,
                text=True,
            )
            if probe.returncode == 0:
                probe_error = None
                break
            probe_error = probe.stderr.strip() or probe.stdout.strip() or f"exit {probe.returncode}"
        else:
            proc.kill()
            stdout, stderr = proc.communicate(timeout=5)
            raise SystemExit(f"gRPC probe failed: {probe_error}\nstdout:\n{stdout}\nstderr:\n{stderr}")

        try:
            stdout, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate(timeout=5)

        payload = {
            "schema_version": 1,
            "mode": "grpc",
            "service_dir": str(target),
            "http_port": http_port,
            "grpc_port": grpc_port,
            "probe": str(probe_path),
            "server_log": str(artifacts_dir / "server.jsonl"),
            "conversation_state": str(artifacts_dir / "conversation_state.json"),
            "conversation_backup": str(artifacts_dir / "conversation_state.backup.json"),
            "serve_exit_code": proc.returncode,
            "serve_stdout": stdout,
            "serve_stderr": stderr,
        }
        write_json(workspace / output, payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
