#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    summary_path = Path(args.summary)
    output_path = Path(args.output)
    payload = read_json(summary_path)

    issues: list[str] = []
    if payload.get("status") != "ok":
        issues.append("summary status must be ok")

    for key in ("cache_registry_json", "cache_audit_log", "remote_manifest", "remote_signature"):
        path = payload.get(key)
        if not path or not Path(path).is_file():
            issues.append(f"missing artifact for {key}")

    cache_registry_json = payload.get("cache_registry_json")
    if cache_registry_json and Path(cache_registry_json).is_file():
        registry = read_json(Path(cache_registry_json))
        model = payload.get("model")
        version = payload.get("version")
        version_entry = (
            registry.get("models", {})
            .get(model, {})
            .get("versions", {})
            .get(version, {})
        )
        if version_entry.get("artifact_kind") != "simulation":
            issues.append("cache registry artifact_kind must be simulation")
        if not version_entry.get("lineage_manifest_path"):
            issues.append("cache registry must preserve lineage manifest path")

    cache_audit_log = payload.get("cache_audit_log")
    if cache_audit_log and Path(cache_audit_log).is_file():
        text = Path(cache_audit_log).read_text(encoding="utf-8")
        if "\"status\":\"fallback_local\"" not in text:
            issues.append("cache audit log missing fallback_local status")
        if "\"operation\":\"pull_remote\"" not in text:
            issues.append("cache audit log missing pull_remote operation")

    result = {
        "schema_version": 1,
        "status": "ok" if not issues else "failed",
        "passed": not issues,
        "issues": issues,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
