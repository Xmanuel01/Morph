#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HTTP serving throughput benchmark")
    parser.add_argument("--impl", choices=["enkai", "python"], required=True)
    parser.add_argument("--requests", type=int, default=120)
    parser.add_argument("--enkai-bin", default=None)
    return parser.parse_args()


def reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        payload = b"ok"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format: str, *_args: object) -> None:
        return


def run_python(requests: int) -> int:
    port = reserve_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.05)

    try:
        for _ in range(requests):
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2.0) as response:
                if response.status != 200:
                    return 1
                body = response.read()
                if body != b"ok":
                    return 1
    finally:
        server.shutdown()
        thread.join(timeout=2.0)
        server.server_close()

    return 0


def run_enkai(requests: int, enkai_bin: str) -> int:
    port = reserve_port()
    script = f'''policy default ::
    allow net
::

fn handler(req: Request) -> Response ::
    return http.ok("ok")
::

http.serve("127.0.0.1", {port}, handler)
task.sleep(80)

let i := 0
let status := 0
while i < {requests} ::
    let resp := http.get("http://127.0.0.1:{port}/")
    if resp.status != 200 ::
        status := 1
    ::
    i := i + 1
::

status
'''

    with tempfile.TemporaryDirectory(prefix="enkai_http_bench_") as tmp:
        script_path = Path(tmp) / "http_bench.enk"
        script_path.write_text(script, encoding="utf-8")

        env = os.environ.copy()
        env.setdefault("ENKAI_STD", str(ROOT / "std"))

        proc = subprocess.run(
            [enkai_bin, "run", str(script_path)],
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            if proc.stderr:
                print(proc.stderr.strip())
            return proc.returncode
    return 0


def main() -> int:
    args = parse_args()
    if args.requests <= 0:
        return 1

    if args.impl == "python":
        return run_python(args.requests)

    enkai_bin = args.enkai_bin
    if not enkai_bin:
        candidate = ROOT / ("target/release/enkai.exe" if os.name == "nt" else "target/release/enkai")
        enkai_bin = str(candidate)
    return run_enkai(args.requests, enkai_bin)


if __name__ == "__main__":
    raise SystemExit(main())
