#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
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

    root = workspace / "artifacts" / "registry_degraded"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    local_registry = root / "local"
    remote_registry = root / "remote"
    cache_registry = root / "cache"
    staging = root / "staging"
    staging.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["ENKAI_MODEL_SIGNING_KEY"] = "enkai-v2.9.1-registry-degraded-key"

    artifact = staging / "simulation_snapshot.json"
    write_json(
        artifact,
        {
            "schema_version": 1,
            "world": "adam0",
            "seed": 33,
            "steps": 4,
            "agents": 100,
        },
    )
    manifest = staging / "simulation_snapshot.manifest.json"
    write_json(
        manifest,
        {
            "manifest_kind": "world_snapshot_v1",
            "snapshot_hash": "demo",
            "artifact_kind": "simulation",
        },
    )
    lineage = staging / "simulation_lineage.json"
    write_json(
        lineage,
        {
            "manifest_kind": "simulation_lineage_v1",
            "run_id": "registry-degraded-smoke",
            "environment_hash": "demo",
        },
    )

    model_name = "adam0-degraded"
    version = "v2.9.1"
    run(
        [
            str(enkai_bin),
            "model",
            "register",
            str(local_registry),
            model_name,
            version,
            str(artifact),
            "--artifact-kind",
            "simulation",
            "--artifact-manifest",
            str(manifest),
            "--lineage-manifest",
            str(lineage),
            "--activate",
        ],
        workspace,
        env,
    )
    run(
        [
            str(enkai_bin),
            "model",
            "push",
            str(local_registry),
            model_name,
            version,
            "--registry",
            str(remote_registry),
            "--sign",
        ],
        workspace,
        env,
    )
    run(
        [
            str(enkai_bin),
            "model",
            "pull",
            str(cache_registry),
            model_name,
            version,
            "--registry",
            str(remote_registry),
            "--verify-signature",
        ],
        workspace,
        env,
    )

    remote_offline = root / "remote_offline"
    if remote_registry.exists():
        remote_registry.rename(remote_offline)
    missing_remote = root / "remote"
    run(
        [
            str(enkai_bin),
            "model",
            "pull",
            str(cache_registry),
            model_name,
            version,
            "--registry",
            str(missing_remote),
            "--verify-signature",
            "--fallback-local",
        ],
        workspace,
        env,
    )

    summary = {
        "schema_version": 1,
        "status": "ok",
        "model": model_name,
        "version": version,
        "cache_registry": str(cache_registry),
        "remote_offline_registry": str(remote_offline),
        "cache_registry_json": str(cache_registry / "registry.json"),
        "cache_audit_log": str(cache_registry / "audit.log.jsonl"),
        "remote_manifest": str(remote_offline / model_name / version / "remote.manifest.json"),
        "remote_signature": str(remote_offline / model_name / version / "remote.manifest.sig"),
    }
    write_json(workspace / args.output, summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
