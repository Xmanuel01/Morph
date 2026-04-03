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
    output = Path(args.output)
    if not enkai_bin.exists():
        raise SystemExit(f"enkai binary not found: {enkai_bin}")

    artifacts_dir = workspace / "artifacts" / "mobile"
    report_path = workspace / "artifacts" / "readiness" / "deploy_mobile.json"

    with tempfile.TemporaryDirectory(prefix="enkai_readiness_mobile_") as tmp:
        root = Path(tmp)
        target = root / "mobile-chat"
        run(
            [
                str(enkai_bin),
                "new",
                "mobile-chat",
                str(target),
                "--api-version",
                "v1",
                "--backend-url",
                "http://127.0.0.1:8080",
            ],
            workspace,
        )
        run(
            [
                str(enkai_bin),
                "deploy",
                "validate",
                str(target),
                "--profile",
                "mobile",
                "--strict",
                "--json",
                "--output",
                str(report_path),
            ],
            workspace,
        )

        artifacts_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target / "contracts" / "sdk_api.snapshot.json", artifacts_dir / "sdk_api.snapshot.json")
        shutil.copy2(target / "app.json", artifacts_dir / "app.json")
        shutil.copy2(target / "package.json", artifacts_dir / "package.json")

        payload = {
            "schema_version": 1,
            "mode": "mobile",
            "report": str(report_path),
            "sdk_snapshot": str(artifacts_dir / "sdk_api.snapshot.json"),
            "app_json": str(artifacts_dir / "app.json"),
            "package_json": str(artifacts_dir / "package.json"),
        }
        write_json(workspace / output, payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
