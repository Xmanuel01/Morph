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

    expected = {
        "llm-chat": "checkpoint",
        "adam0-sim": "simulation",
        "adam0-env": "environment",
        "adam0-native": "native-extension",
    }
    for registry_name in ("local", "remote", "cache"):
        registry = payload.get("registries", {}).get(registry_name, {})
        models = registry.get("models", {})
        for name, kind in expected.items():
            if name not in models:
                issues.append(f"{registry_name} registry missing {name}")
                continue
            version = models[name].get("versions", {}).get("v2.9.6")
            if not isinstance(version, dict):
                issues.append(f"{registry_name} registry missing {name} v2.9.6")
                continue
            if version.get("artifact_kind") != kind:
                issues.append(
                    f"{registry_name} registry {name} artifact_kind mismatch: expected {kind}, found {version.get('artifact_kind')}"
                )
            if kind == "simulation":
                if not version.get("artifact_manifest_path"):
                    issues.append("simulation artifact missing artifact_manifest_path")
                if not version.get("lineage_manifest_path"):
                    issues.append("simulation artifact missing lineage_manifest_path")

    artifacts = payload.get("artifacts", {})
    for key in ("sim_run", "sim_lineage", "sim_snapshot_manifest", "environment_manifest", "native_manifest"):
        path = artifacts.get(key)
        if not path or not Path(path).is_file():
            issues.append(f"missing artifact file for {key}")

    if Path(artifacts["sim_lineage"]).is_file():
        lineage = read_json(Path(artifacts["sim_lineage"]))
        if lineage.get("manifest_kind") != "simulation_lineage_v1":
            issues.append("simulation lineage manifest kind mismatch")
        if not lineage.get("environment_hash"):
            issues.append("simulation lineage missing environment_hash")
    if Path(artifacts["sim_snapshot_manifest"]).is_file():
        manifest = read_json(Path(artifacts["sim_snapshot_manifest"]))
        if manifest.get("manifest_kind") != "world_snapshot_v1":
            issues.append("simulation snapshot manifest kind mismatch")
        if not manifest.get("snapshot_hash"):
            issues.append("simulation snapshot manifest missing snapshot_hash")

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
