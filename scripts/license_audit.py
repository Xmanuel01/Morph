#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import subprocess
import sys


def run_cargo_metadata(root: pathlib.Path) -> dict:
    proc = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--locked"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout)


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[1]
    meta = run_cargo_metadata(root)

    workspace_members = set(meta.get("workspace_members", []))
    packages = meta.get("packages", [])

    missing_license: list[str] = []
    for pkg in packages:
        pkg_id = pkg.get("id", "")
        if pkg_id in workspace_members:
            continue
        if pkg.get("source") is None:
            continue
        license_id = (pkg.get("license") or "").strip()
        license_file = (pkg.get("license_file") or "").strip()
        if not license_id and not license_file:
            missing_license.append(f"{pkg['name']} {pkg['version']}")

    if missing_license:
        print("license audit failed: dependencies missing license metadata:", file=sys.stderr)
        for item in missing_license:
            print(f"- {item}", file=sys.stderr)
        return 1

    summary = {
        "status": "ok",
        "checked_packages": len(packages),
        "workspace_members": len(workspace_members),
        "external_packages": len(packages) - len(workspace_members),
        "missing_license": 0,
    }
    print(json.dumps(summary, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
