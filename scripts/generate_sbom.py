#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import subprocess


def run_cargo_metadata(root: pathlib.Path) -> dict:
    proc = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--locked"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate lightweight SBOM JSON from cargo metadata.")
    parser.add_argument("--output", required=True, help="Output path for generated SBOM JSON")
    parser.add_argument(
        "--include-workspace",
        action="store_true",
        help="Include workspace members in components list (default false)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    out_path = pathlib.Path(args.output)
    if not out_path.is_absolute():
        out_path = (root / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = run_cargo_metadata(root)
    workspace_members = set(metadata.get("workspace_members", []))
    packages = metadata.get("packages", [])

    components = []
    for pkg in sorted(packages, key=lambda item: (item["name"], item["version"])):
        pkg_id = pkg.get("id", "")
        if not args.include_workspace and pkg_id in workspace_members:
            continue
        components.append(
            {
                "id": pkg_id,
                "name": pkg["name"],
                "version": pkg["version"],
                "source": pkg.get("source"),
                "license": pkg.get("license"),
                "license_file": pkg.get("license_file"),
                "repository": pkg.get("repository"),
            }
        )

    dependency_edges = []
    resolve = metadata.get("resolve") or {}
    for node in resolve.get("nodes", []):
        dependency_edges.append(
            {
                "package": node.get("id"),
                "dependencies": node.get("dependencies", []),
            }
        )

    document = {
        "schema": "enkai-sbom-v1",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "workspace_root": metadata.get("workspace_root"),
        "components": components,
        "dependency_edges": dependency_edges,
    }

    out_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(out_path.relative_to(root)),
                "components": len(components),
                "dependency_edges": len(dependency_edges),
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
