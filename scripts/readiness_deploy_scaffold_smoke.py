#!/usr/bin/env python3
import argparse
import json
import os
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
    parser.add_argument(
        "--mode",
        choices=["backend", "fullstack"],
        required=True,
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    enkai_bin = Path(args.enkai_bin)
    output = Path(args.output)

    if not enkai_bin.exists():
        raise SystemExit(f"enkai binary not found: {enkai_bin}")

    with tempfile.TemporaryDirectory(prefix=f"enkai_readiness_{args.mode}_") as tmp:
        root = Path(tmp)
        target = root / args.mode
        api_version = "v1"

        if args.mode == "backend":
            run(
                [str(enkai_bin), "new", "service", str(target), "--api-version", api_version],
                workspace,
            )
            report_path = workspace / "artifacts" / "readiness" / "deploy_backend.json"
            run(
                [
                    str(enkai_bin),
                    "deploy",
                    "validate",
                    str(target),
                    "--profile",
                    "backend",
                    "--strict",
                    "--json",
                    "--output",
                    str(report_path),
                ],
                workspace,
            )
            payload = {
                "schema_version": 1,
                "mode": "backend",
                "target_dir": str(target),
                "report": str(report_path),
            }
        else:
            run(
                [
                    str(enkai_bin),
                    "new",
                    "fullstack-chat",
                    str(target),
                    "--api-version",
                    api_version,
                    "--backend-url",
                    "http://127.0.0.1:8080",
                ],
                workspace,
            )
            report_path = workspace / "artifacts" / "readiness" / "deploy_fullstack.json"
            run(
                [
                    str(enkai_bin),
                    "deploy",
                    "validate",
                    str(target),
                    "--profile",
                    "fullstack",
                    "--strict",
                    "--json",
                    "--output",
                    str(report_path),
                ],
                workspace,
            )
            payload = {
                "schema_version": 1,
                "mode": "fullstack",
                "target_dir": str(target),
                "report": str(report_path),
            }

        write_json(workspace / output, payload)

    return 0


if __name__ == "__main__":
    sys.exit(main())
