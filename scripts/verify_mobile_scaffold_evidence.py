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
    parser.add_argument("--report", required=True)
    parser.add_argument("--sdk-snapshot", required=True)
    parser.add_argument("--app-json", required=True)
    parser.add_argument("--package-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    summary = read_json(Path(args.summary))
    report = read_json(Path(args.report))
    sdk_snapshot = read_json(Path(args.sdk_snapshot))
    app_json = read_json(Path(args.app_json))
    package_json = read_json(Path(args.package_json))

    if summary.get("schema_version") != 1:
        fail("mobile summary schema_version must be 1")
    if report.get("profile") != "mobile":
        fail("mobile deploy report profile mismatch")
    if not bool(report.get("success", False)):
        fail("mobile deploy report is not successful")
    if sdk_snapshot.get("target") != "mobile":
        fail("mobile sdk snapshot target must be mobile")
    if "expo" not in json.dumps(package_json):
        fail("mobile package.json must include expo")
    if app_json.get("expo", {}).get("name") in (None, ""):
        fail("mobile app.json missing expo.name")

    payload = {
        "schema_version": 1,
        "status": "ok",
        "profile": report.get("profile"),
        "target": sdk_snapshot.get("target"),
        "report": args.report,
    }
    write_json(Path(args.output), payload)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as err:
        print(f"error: {err}", file=sys.stderr)
        raise SystemExit(1)
