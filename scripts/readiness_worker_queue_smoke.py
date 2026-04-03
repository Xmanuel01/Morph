#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd, cwd):
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def write_json(path: Path, payload):
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
    if not enkai_bin.exists():
        raise SystemExit(f"enkai binary not found: {enkai_bin}")

    state_dir = workspace / "artifacts" / "worker_queue"
    if state_dir.exists():
        shutil.rmtree(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    handler = state_dir / "handler.enk"
    handler.write_text(
        "import std::env\npolicy default ::\n    allow env\n::\nfn main() ::\n    let id := env.get(\"ENKAI_WORKER_ID\")?\n    if id == \"dead\" ::\n        return 2\n    ::\n    return 0\n::\n\nmain()\n",
        encoding="utf-8",
    )

    run([
        str(enkai_bin), "worker", "enqueue", "--queue", "default", "--dir", str(state_dir),
        "--payload", '{"kind":"ok"}', "--id", "ok", "--json",
        "--output", str(state_dir / "enqueue_ok.json")
    ], workspace)
    run([
        str(enkai_bin), "worker", "enqueue", "--queue", "default", "--dir", str(state_dir),
        "--payload", '{"kind":"dead"}', "--id", "dead", "--max-attempts", "2", "--json",
        "--output", str(state_dir / "enqueue_dead.json")
    ], workspace)
    run([
        str(enkai_bin), "worker", "run", "--queue", "default", "--dir", str(state_dir),
        "--handler", str(handler), "--once", "--json", "--output", str(state_dir / "run_01.json")
    ], workspace)
    run([
        str(enkai_bin), "worker", "run", "--queue", "default", "--dir", str(state_dir),
        "--handler", str(handler), "--once", "--json", "--output", str(state_dir / "run_02.json")
    ], workspace)
    run([
        str(enkai_bin), "worker", "run", "--queue", "default", "--dir", str(state_dir),
        "--handler", str(handler), "--once", "--json", "--output", str(state_dir / "run_03.json")
    ], workspace)

    payload = {
        "schema_version": 1,
        "queue": "default",
        "runs": [
            str(state_dir / "run_01.json"),
            str(state_dir / "run_02.json"),
            str(state_dir / "run_03.json"),
        ],
        "dead_letter_path": str(state_dir / "queues" / "default" / "dead_letter.jsonl"),
        "pending_path": str(state_dir / "queues" / "default" / "pending.jsonl"),
    }
    write_json(workspace / args.output, payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
