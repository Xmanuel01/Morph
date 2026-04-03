#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def fail(message: str) -> None:
    raise SystemExit(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--server-log", required=True)
    parser.add_argument("--conversation-state", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    summary = read_json(Path(args.summary))
    probe = read_json(Path(args.probe))
    server_log = Path(args.server_log)
    conversation_state = read_json(Path(args.conversation_state))

    if summary.get("schema_version") != 1:
        fail("grpc summary schema_version must be 1")
    if summary.get("serve_exit_code") != 0:
        fail("gRPC serve process did not exit cleanly")
    if probe.get("health", {}).get("status") != "ok":
        fail("gRPC health response must be ok")
    if probe.get("ready", {}).get("status") != "ready":
        fail("gRPC ready response must be ready")
    if probe.get("chat", {}).get("api_version") != "v1":
        fail("gRPC chat api_version mismatch")
    if probe.get("chat", {}).get("reply") != "hello from enkai backend":
        fail("gRPC chat reply mismatch")

    events = probe.get("stream", [])
    if not isinstance(events, list) or len(events) < 4:
        fail("gRPC stream must contain at least four events")
    if events[-1].get("event") != "done":
        fail("gRPC stream final event must be done")
    if conversation_state.get("source") != "grpc-stream":
        fail("conversation state source must be grpc-stream")
    if conversation_state.get("id") != events[-1].get("conversation_id"):
        fail("conversation state id must match gRPC done event")

    log_text = server_log.read_text(encoding="utf-8")
    if '"protocol":"grpc"' not in log_text and '"protocol": "grpc"' not in log_text:
        fail("gRPC server log must record grpc protocol events")

    payload = {
        "schema_version": 1,
        "status": "ok",
        "conversation_id": conversation_state.get("id"),
        "stream_events": len(events),
        "probe": args.probe,
    }
    write_json(Path(args.output), payload)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as err:
        print(f"error: {err}", file=sys.stderr)
        raise SystemExit(1)
