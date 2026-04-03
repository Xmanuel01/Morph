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


def load_jsonl(path: Path):
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--run-01", required=True)
    parser.add_argument("--run-02", required=True)
    parser.add_argument("--run-03", required=True)
    parser.add_argument("--dead-letter", required=True)
    parser.add_argument("--pending", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    summary = read_json(Path(args.summary))
    run_01 = read_json(Path(args.run_01))
    run_02 = read_json(Path(args.run_02))
    run_03 = read_json(Path(args.run_03))
    dead_letter = load_jsonl(Path(args.dead_letter))
    pending = load_jsonl(Path(args.pending))

    if summary.get("schema_version") != 1:
        fail("worker queue summary schema_version must be 1")
    if run_01.get("status") != "acked":
        fail("worker run 01 must ack the first message")
    if run_02.get("status") != "requeued":
        fail("worker run 02 must requeue the retryable message")
    if run_03.get("status") != "dead_lettered":
        fail("worker run 03 must dead-letter the exhausted message")
    if len(dead_letter) != 1:
        fail("worker dead-letter queue must contain exactly one message")
    if pending:
        fail("worker pending queue must be empty after the smoke sequence")

    payload = {
        "schema_version": 1,
        "status": "ok",
        "dead_lettered": len(dead_letter),
        "acked_id": run_01.get("last_message_id"),
        "requeued_id": run_02.get("last_message_id"),
        "dead_letter_id": run_03.get("last_message_id"),
    }
    write_json(Path(args.output), payload)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as err:
        print(f"error: {err}", file=sys.stderr)
        raise SystemExit(1)
