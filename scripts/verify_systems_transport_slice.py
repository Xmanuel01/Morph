#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from socket import create_connection
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the v3.3.0 self-host systems transport manifest slice."
    )
    parser.add_argument("--enkai-bin", required=True, help="Path to enkai executable")
    parser.add_argument(
        "--contract",
        default="enkai/contracts/selfhost_systems_transport_v3_3_0.json",
        help="Path to the transport-slice contract JSON",
    )
    parser.add_argument(
        "--output",
        default="artifacts/readiness/strict_selfhost_systems_transport_slice.json",
        help="Path to write the verification report JSON",
    )
    return parser.parse_args()


def run_command(command: list[str], cwd: Path, extra_env: dict[str, str] | None = None) -> dict:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
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


def ensure(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


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


def read_http_response(port: int, request: bytes, *, timeout_s: float = 2.0) -> tuple[bytes, str | None]:
    response = b""
    connect_error = None
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with create_connection(("127.0.0.1", port), timeout=0.2) as sock:
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


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    enkai_bin = Path(args.enkai_bin).resolve()
    contract_path = (repo_root / args.contract).resolve()
    output_path = (repo_root / args.output).resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory(prefix="enkai_systems_transport_") as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        registry = temp_dir / "registry"
        version_dir = registry / "chat" / "v1.0.0"
        (version_dir / "checkpoint").mkdir(parents=True, exist_ok=True)
        (registry / "chat" / ".active_version").write_text("v1.0.0\n", encoding="utf-8")

        base_manifest = temp_dir / "serve_manifest_base.json"
        base_result = run_command(
            [
                str(enkai_bin),
                "systems",
                "serve-manifest",
                "--host",
                "127.0.0.1",
                "--port",
                "8080",
                "--grpc-host",
                "127.0.0.1",
                "--grpc-port",
                "9090",
                "--trace-vm",
                "--trace-net",
                "examples/hello/main.enk",
                "--json",
                "--output",
                str(base_manifest),
            ],
            repo_root,
        )

        single_manifest = temp_dir / "serve_manifest_single.json"
        single_result = run_command(
            [
                str(enkai_bin),
                "systems",
                "serve-manifest",
                "--host",
                "127.0.0.1",
                "--port",
                "8080",
                "--grpc-host",
                "127.0.0.1",
                "--grpc-port",
                "9090",
                "--registry",
                str(registry),
                "--model",
                "chat",
                "examples/hello/main.enk",
                "--json",
                "--output",
                str(single_manifest),
            ],
            repo_root,
        )

        multi_manifest = temp_dir / "serve_manifest_multi.json"
        multi_result = run_command(
            [
                str(enkai_bin),
                "systems",
                "serve-manifest",
                "--host",
                "127.0.0.1",
                "--port",
                "8080",
                "--grpc-host",
                "127.0.0.1",
                "--grpc-port",
                "9090",
                "--multi-model",
                "--registry",
                str(registry),
                "examples/hello/main.enk",
                "--json",
                "--output",
                str(multi_manifest),
            ],
            repo_root,
        )

        backend_dir = temp_dir / "backend_service"
        (backend_dir / "contracts").mkdir(parents=True, exist_ok=True)
        (backend_dir / "src").mkdir(parents=True, exist_ok=True)
        (backend_dir / "contracts" / "backend_api.snapshot.json").write_text(
            "{}\n", encoding="utf-8"
        )
        (backend_dir / "src" / "main.enk").write_text(
            "fn main() ::\n    return 0\n::\nmain()\n", encoding="utf-8"
        )
        backend_manifest = temp_dir / "backend_service_manifest.json"
        backend_manifest_result = run_command(
            [
                str(enkai_bin),
                "systems",
                "serve-manifest",
                "--host",
                "127.0.0.1",
                "--port",
                "18081",
                "--grpc-host",
                "127.0.0.1",
                "--grpc-port",
                "19081",
                str(backend_dir),
                "--json",
                "--output",
                str(backend_manifest),
            ],
            repo_root,
        )
        backend_exec = subprocess.Popen(
            [
                str(enkai_bin),
                "systems",
                "serve-exec",
                "--manifest",
                str(backend_manifest),
            ],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env={**os.environ, "ENKAI_CONTRACT_TEST_MODE": "1"},
        )
        backend_response, backend_connect_error = read_http_response(
            18081,
            b"GET /api/v1/health HTTP/1.1\r\nHost: localhost\r\nx-enkai-api-version: v1\r\nConnection: close\r\n\r\n",
        )
        grpc_probe_output = temp_dir / "grpc_probe.json"
        grpc_probe_result = run_command(
            [
                str(enkai_bin),
                "grpc",
                "probe",
                "--address",
                "http://127.0.0.1:19081",
                "--api-version",
                "v1",
                "--prompt",
                "hello from grpc contract",
                "--json",
                "--output",
                str(grpc_probe_output),
            ],
            repo_root,
        )
        backend_stdout, backend_stderr = backend_exec.communicate(timeout=10)

        generic_target = temp_dir / "generic_service.enk"
        generic_target.write_text(
            "fn main() -> Int ::\n    return 7\n::\nmain()\n", encoding="utf-8"
        )
        generic_public = temp_dir / "public"
        generic_public.mkdir(parents=True, exist_ok=True)
        (generic_public / "index.html").write_text(
            "<h1>hello static</h1>", encoding="utf-8"
        )
        generic_manifest = temp_dir / "generic_service_manifest.json"
        generic_manifest_result = run_command(
            [
                str(enkai_bin),
                "systems",
                "serve-manifest",
                "--host",
                "127.0.0.1",
                "--port",
                "18082",
                str(generic_target),
                "--json",
                "--output",
                str(generic_manifest),
            ],
            repo_root,
        )
        generic_exec = subprocess.Popen(
            [
                str(enkai_bin),
                "systems",
                "serve-exec",
                "--manifest",
                str(generic_manifest),
            ],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env={**os.environ, "ENKAI_CONTRACT_TEST_MODE": "1"},
        )
        generic_response, generic_connect_error = read_http_response(
            18082,
            b"GET /api/v1/invoke HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
        )
        static_response, static_connect_error = read_http_response(
            18082,
            b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
        )
        if static_connect_error is not None:
            failures = [f"static service request failed: {static_connect_error}"]
            report = {
                "schema_version": 1,
                "profile": contract.get("profile"),
                "contract": str(contract_path),
                "contract_version": contract.get("contract_version"),
                "all_passed": False,
                "cases": {},
                "validations": {},
                "failures": failures,
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            return 1
        generic_stdout, generic_stderr = generic_exec.communicate(timeout=10)

        route_dir = temp_dir / "route_service"
        (route_dir / "routes" / "get" / "users").mkdir(parents=True, exist_ok=True)
        (route_dir / "routes" / "get" / "posts" / "[post_id]" / "comments").mkdir(
            parents=True, exist_ok=True
        )
        (route_dir / "routes" / "any" / "files" / "...path").mkdir(
            parents=True, exist_ok=True
        )
        (route_dir / "routes" / "post").mkdir(parents=True, exist_ok=True)
        (route_dir / "src").mkdir(parents=True, exist_ok=True)
        (route_dir / "src" / "main.enk").write_text(
            "fn main() ::\n    return 0\n::\nmain()\n",
            encoding="utf-8",
        )
        (route_dir / "routes" / "get" / "ping.enk").write_text(
            "fn main() ::\n\
    let resp := json.parse(\"{}\")\n\
    let body := json.parse(\"{}\")\n\
    body.route := \"ping\"\n\
    resp.body := body\n\
    return resp\n\
::\n\
main()\n",
            encoding="utf-8",
        )
        (route_dir / "routes" / "get" / "users" / "[id].enk").write_text(
            "import std::env\n\
policy default ::\n\
    allow env\n\
::\n\
fn main() ::\n\
    let resp := json.parse(\"{}\")\n\
    let body := json.parse(\"{}\")\n\
    body.route := \"user_show\"\n\
    body.id := env.get(\"ENKAI_SERVE_ROUTE_PARAM_id\")?\n\
    resp.body := body\n\
    return resp\n\
::\n\
main()\n",
            encoding="utf-8",
        )
        (route_dir / "routes" / "post" / "echo.enk").write_text(
            "import std::env\n\
policy default ::\n\
    allow env\n\
::\n\
fn main() ::\n\
    let resp := json.parse(\"{}\")\n\
    let headers := json.parse(\"{}\")\n\
    let body := json.parse(\"{}\")\n\
    headers.x_route_mode := \"post\"\n\
    body.echo := env.get(\"ENKAI_SERVE_REQUEST_BODY\")?\n\
    resp.status := 201\n\
    resp.headers := headers\n\
    resp.body := body\n\
    return resp\n\
::\n\
main()\n",
            encoding="utf-8",
        )
        (route_dir / "routes" / "any" / "status.enk").write_text(
            "fn main() ::\n\
    let resp := json.parse(\"{}\")\n\
    let body := json.parse(\"{}\")\n\
    body.route := \"any_status\"\n\
    resp.body := body\n\
    return resp\n\
::\n\
main()\n",
            encoding="utf-8",
        )
        (route_dir / "routes" / "get" / "posts" / "[post_id]" / "comments" / "[comment_id].enk").write_text(
            "import std::env\n\
policy default ::\n\
    allow env\n\
::\n\
fn main() ::\n\
    let resp := json.parse(\"{}\")\n\
    let body := json.parse(\"{}\")\n\
    body.route := \"nested_comment\"\n\
    body.post := env.get(\"ENKAI_SERVE_ROUTE_PARAM_post_id\")?\n\
    body.comment := env.get(\"ENKAI_SERVE_ROUTE_PARAM_comment_id\")?\n\
    resp.body := body\n\
    return resp\n\
::\n\
main()\n",
            encoding="utf-8",
        )
        (route_dir / "routes" / "any" / "files" / "...path" / "index.enk").write_text(
            "import std::env\n\
policy default ::\n\
    allow env\n\
::\n\
fn main() ::\n\
    let resp := json.parse(\"{}\")\n\
    let body := json.parse(\"{}\")\n\
    body.route := \"catchall\"\n\
    body.path := env.get(\"ENKAI_SERVE_ROUTE_PARAM_path\")?\n\
    resp.body := body\n\
    return resp\n\
::\n\
main()\n",
            encoding="utf-8",
        )
        route_manifest = temp_dir / "route_service_manifest.json"
        route_manifest_result = run_command(
            [
                str(enkai_bin),
                "systems",
                "serve-manifest",
                "--host",
                "127.0.0.1",
                "--port",
                "18083",
                str(route_dir),
                "--json",
                "--output",
                str(route_manifest),
            ],
            repo_root,
        )
        route_exec = subprocess.Popen(
            [
                str(enkai_bin),
                "systems",
                "serve-exec",
                "--manifest",
                str(route_manifest),
            ],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env={**os.environ, "ENKAI_CONTRACT_TEST_MODE": "1"},
        )
        route_ping_response, route_ping_connect_error = read_http_response(
            18083,
            b"GET /api/v1/ping HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
        )
        route_user_response, route_user_connect_error = read_http_response(
            18083,
            b"GET /api/v1/users/42 HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
        )
        route_post_response, route_post_connect_error = read_http_response(
            18083,
            b"POST /api/v1/echo HTTP/1.1\r\nHost: localhost\r\nContent-Length: 10\r\nConnection: close\r\n\r\nhello body",
        )
        route_any_response, route_any_connect_error = read_http_response(
            18083,
            b"PATCH /api/v1/status HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
        )
        route_nested_response, route_nested_connect_error = read_http_response(
            18083,
            b"GET /api/v1/posts/post-1/comments/abc123 HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
        )
        route_catchall_response, route_catchall_connect_error = read_http_response(
            18083,
            b"GET /api/v1/files/folder%201/report.txt HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
        )
        route_stdout, route_stderr = route_exec.communicate(timeout=10)

        report = {
            "schema_version": 1,
            "profile": contract.get("profile"),
            "contract": str(contract_path),
            "contract_version": contract.get("contract_version"),
            "all_passed": False,
            "cases": {
                "base": base_result,
                "single": single_result,
                "multi": multi_result,
                "backend_manifest": backend_manifest_result,
                "generic_manifest": generic_manifest_result,
                "route_manifest": route_manifest_result,
                "backend_service": {
                    "exit_code": backend_exec.returncode,
                    "ok": backend_exec.returncode == 0,
                    "stdout": backend_stdout,
                    "stderr": backend_stderr,
                    "connect_error": backend_connect_error,
                },
                "grpc_probe": grpc_probe_result,
                "generic_service": {
                    "exit_code": generic_exec.returncode,
                    "ok": generic_exec.returncode == 0,
                    "stdout": generic_stdout,
                    "stderr": generic_stderr,
                    "connect_error": generic_connect_error,
                },
                "static_service": {
                    "http_status": response_status(static_response),
                    "body": response_body(static_response),
                },
                "route_service": {
                    "exit_code": route_exec.returncode,
                    "ok": route_exec.returncode == 0,
                    "stdout": route_stdout,
                    "stderr": route_stderr,
                    "ping_connect_error": route_ping_connect_error,
                    "user_connect_error": route_user_connect_error,
                    "post_connect_error": route_post_connect_error,
                    "any_connect_error": route_any_connect_error,
                    "nested_connect_error": route_nested_connect_error,
                    "catchall_connect_error": route_catchall_connect_error,
                    "ping_status": response_status(route_ping_response),
                    "ping_body": response_body(route_ping_response),
                    "user_status": response_status(route_user_response),
                    "user_body": response_body(route_user_response),
                    "post_status": response_status(route_post_response),
                    "post_body": response_body(route_post_response),
                    "any_status": response_status(route_any_response),
                    "any_body": response_body(route_any_response),
                    "nested_status": response_status(route_nested_response),
                    "nested_body": response_body(route_nested_response),
                    "catchall_status": response_status(route_catchall_response),
                    "catchall_body": response_body(route_catchall_response),
                    "post_raw": route_post_response.decode("utf-8", errors="replace"),
                },
            },
            "validations": {},
        }

        failures: list[str] = []
        ensure(base_result["ok"], "base serve-manifest command failed", failures)
        ensure(single_result["ok"], "single-model serve-manifest command failed", failures)
        ensure(multi_result["ok"], "multi-model serve-manifest command failed", failures)
        ensure(backend_manifest_result["ok"], "backend service manifest command failed", failures)
        ensure(backend_exec.returncode == 0, "backend service exec command failed", failures)
        ensure(grpc_probe_result["ok"], "gRPC probe command failed", failures)
        ensure(generic_manifest_result["ok"], "generic service manifest command failed", failures)
        ensure(generic_exec.returncode == 0, "generic service exec command failed", failures)
        ensure(route_manifest_result["ok"], "route service manifest command failed", failures)
        ensure(route_exec.returncode == 0, "route service exec command failed", failures)

        if not failures:
            base_payload = json.loads(base_manifest.read_text(encoding="utf-8"))
            single_payload = json.loads(single_manifest.read_text(encoding="utf-8"))
            multi_payload = json.loads(multi_manifest.read_text(encoding="utf-8"))
            generic_payload = json.loads(generic_manifest.read_text(encoding="utf-8"))
            route_payload = json.loads(route_manifest.read_text(encoding="utf-8"))
            grpc_probe_payload = json.loads(grpc_probe_output.read_text(encoding="utf-8"))

            report["validations"] = {
                "base_manifest": base_payload,
                "single_manifest": single_payload,
                "multi_manifest": multi_payload,
                "generic_manifest": generic_payload,
                "route_manifest": route_payload,
                "grpc_probe": grpc_probe_payload,
                "execution_mode": contract["required_execution_mode"],
                "backend_service_http_status": response_status(backend_response),
                "backend_service_body": response_body(backend_response),
                "generic_service_http_status": response_status(generic_response),
                "generic_service_body": response_body(generic_response),
                "static_service_http_status": response_status(static_response),
                "static_service_body": response_body(static_response),
                "route_service_ping_http_status": response_status(route_ping_response),
                "route_service_ping_body": response_body(route_ping_response),
                "route_service_user_http_status": response_status(route_user_response),
                "route_service_user_body": response_body(route_user_response),
                "route_service_post_http_status": response_status(route_post_response),
                "route_service_post_body": response_body(route_post_response),
                "route_service_any_http_status": response_status(route_any_response),
                "route_service_any_body": response_body(route_any_response),
                "route_service_nested_http_status": response_status(route_nested_response),
                "route_service_nested_body": response_body(route_nested_response),
                "route_service_catchall_http_status": response_status(route_catchall_response),
                "route_service_catchall_body": response_body(route_catchall_response),
            }

            ensure(base_payload["model"]["mode"] == "none", "base manifest mode must be none", failures)
            ensure(
                single_payload["model"]["mode"] == "single",
                "single manifest mode must be single",
                failures,
            )
            ensure(
                multi_payload["model"]["mode"] == "multi",
                "multi manifest mode must be multi",
                failures,
            )
            ensure(
                base_payload["runtime_flags"] == contract["required_runtime_flags"],
                "base manifest runtime flags do not match contract",
                failures,
            )
            for key in contract["required_http_env_keys"]:
                ensure(key in base_payload["env_projection"], f"missing HTTP env key {key}", failures)
            for key in contract["required_grpc_env_keys"]:
                ensure(key in base_payload["env_projection"], f"missing gRPC env key {key}", failures)
            for key in contract.get("required_runtime_env_keys", []):
                ensure(
                    key in base_payload["env_projection"],
                    f"missing runtime env key {key}",
                    failures,
                )
            for field in contract.get("required_http_runtime_fields", []):
                ensure(
                    field in base_payload.get("http_runtime", {}),
                    f"base manifest missing explicit http runtime field {field}",
                    failures,
                )
            for key in contract["required_model_env_keys"]["single"]:
                ensure(key in single_payload["env_projection"], f"missing single-model env key {key}", failures)
            for key in contract["required_model_env_keys"]["multi"]:
                ensure(key in multi_payload["env_projection"], f"missing multi-model env key {key}", failures)
            for field in contract.get("required_grpc_runtime_fields", []):
                ensure(
                    field in base_payload.get("grpc_runtime", {}),
                    f"base manifest missing explicit grpc runtime field {field}",
                    failures,
                )
            ensure(
                response_status(backend_response) == contract["required_backend_service_status"],
                "backend service HTTP status did not match contract",
                failures,
            )
            ensure(
                contract["required_backend_service_body_fragment"] in response_body(backend_response),
                "backend service response body did not match contract",
                failures,
            )
            ensure(
                grpc_probe_payload["health"]["status"] == contract["required_grpc_health_status"],
                "gRPC health status did not match contract",
                failures,
            )
            ensure(
                grpc_probe_payload["ready"]["status"] == contract["required_grpc_ready_status"],
                "gRPC ready status did not match contract",
                failures,
            )
            ensure(
                contract["required_grpc_chat_reply_fragment"] in grpc_probe_payload["chat"]["reply"],
                "gRPC chat reply did not match contract",
                failures,
            )
            ensure(
                len(grpc_probe_payload["stream"]) == contract["required_grpc_stream_event_count"],
                "gRPC stream event count did not match contract",
                failures,
            )
            ensure(
                response_status(generic_response) == contract["required_generic_service_status"],
                "generic service HTTP status did not match contract",
                failures,
            )
            ensure(
                contract["required_generic_service_body_fragment"] in response_body(generic_response),
                "generic service response body did not match contract",
                failures,
            )
            ensure(
                response_status(static_response) == contract["required_static_service_status"],
                "static service HTTP status did not match contract",
                failures,
            )
            ensure(
                contract["required_static_service_body_fragment"] in response_body(static_response),
                "static service response body did not match contract",
                failures,
            )
            ensure(
                response_status(route_ping_response) == contract["required_route_service_status"],
                "route service ping HTTP status did not match contract",
                failures,
            )
            ensure(
                "\"route\":\"ping\"" in response_body(route_ping_response),
                "route service ping response body did not match contract",
                failures,
            )
            ensure(
                response_status(route_user_response) == contract["required_route_service_status"],
                "route service dynamic HTTP status did not match contract",
                failures,
            )
            ensure(
                contract["required_route_service_body_fragment"] in response_body(route_user_response),
                "route service dynamic response body did not match contract",
                failures,
            )
            ensure(
                response_status(route_post_response) == contract["required_post_route_service_status"],
                "route service POST HTTP status did not match contract",
                failures,
            )
            ensure(
                contract["required_post_route_service_body_fragment"] in response_body(route_post_response),
                "route service POST response body did not match contract",
                failures,
            )
            ensure(
                f'{contract["required_post_route_service_header"]}: post'
                in route_post_response.decode("utf-8", errors="replace").lower(),
                "route service POST response header did not match contract",
                failures,
            )
            ensure(
                response_status(route_any_response) == contract["required_any_route_service_status"],
                "route service ANY-method HTTP status did not match contract",
                failures,
            )
            ensure(
                contract["required_any_route_service_body_fragment"] in response_body(route_any_response),
                "route service ANY-method response body did not match contract",
                failures,
            )
            ensure(
                response_status(route_nested_response) == contract["required_nested_route_service_status"],
                "route service nested dynamic HTTP status did not match contract",
                failures,
            )
            ensure(
                contract["required_nested_route_service_body_fragment"] in response_body(route_nested_response),
                "route service nested dynamic response body did not match contract",
                failures,
            )
            ensure(
                response_status(route_catchall_response) == contract["required_catchall_route_service_status"],
                "route service catchall HTTP status did not match contract",
                failures,
            )
            ensure(
                contract["required_catchall_route_service_body_fragment"] in response_body(route_catchall_response),
                "route service catchall response body did not match contract",
                failures,
            )

        report["all_passed"] = not failures
        report["failures"] = failures
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "status": "ok" if report["all_passed"] else "failed",
                    "output": str(output_path),
                    "all_passed": report["all_passed"],
                },
                separators=(",", ":"),
            )
        )
        return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
