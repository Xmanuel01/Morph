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


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enkai-bin", required=True)
    parser.add_argument("--workspace", default=".")
    parser.add_argument(
        "--output",
        required=True,
        help="Summary JSON output under artifacts/readiness/",
    )
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    enkai_bin = Path(args.enkai_bin)
    if not enkai_bin.exists():
        raise SystemExit(f"enkai binary not found: {enkai_bin}")

    root = workspace / "artifacts" / "registry"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    local_registry = root / "local"
    remote_registry = root / "remote"
    cache_registry = root / "cache"
    staging = root / "staging"
    staging.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["ENKAI_MODEL_SIGNING_KEY"] = "enkai-v2.9.3-registry-key"
    env["ENKAI_SIM_ACCEL"] = "1"
    env["ENKAI_SIM_RUN_ID"] = "adam0-registry-run"
    env["ENKAI_SIM_PARENT_RUN_ID"] = "adam0-parent"

    script = staging / "registry_sim.enk"
    script.write_text(
        """import json
import std::sim
fn main() ::
    let world := sim.make_seeded(32, 19)
    sim.schedule(world, 1.0, json.parse("{\\\"agent\\\":1,\\\"event\\\":\\\"boot\\\"}"))
    sim.schedule(world, 2.0, json.parse("{\\\"agent\\\":2,\\\"event\\\":\\\"sense\\\"}"))
    sim.run(world, 2)
    return sim.snapshot(world)
::
main()
""",
        encoding="utf-8",
    )

    sim_run = root / "sim_run.json"
    sim_lineage = root / "sim_lineage.json"
    sim_snapshot_manifest = root / "sim_snapshot.manifest.json"
    run(
        [
            str(enkai_bin),
            "sim",
            "run",
            "--output",
            str(sim_run),
            "--lineage-output",
            str(sim_lineage),
            "--snapshot-manifest-output",
            str(sim_snapshot_manifest),
            str(script),
        ],
        workspace,
        env,
    )
    run_payload = read_json(sim_run)
    snapshot = root / "sim_snapshot.json"
    write_json(snapshot, run_payload["result"])

    llm_checkpoint = staging / "llm_checkpoint"
    llm_checkpoint.mkdir(parents=True, exist_ok=True)
    write_json(
        llm_checkpoint / "meta.json",
        {
            "format_version": 1,
            "config_hash": sha256_text("llm-config"),
            "weights": [0.1, 0.2, 0.3],
        },
    )

    environment_asset = staging / "environment_asset.json"
    write_json(
        environment_asset,
        {
            "world": "grid",
            "size": [16, 16],
            "reward": "sparse",
        },
    )
    environment_manifest = staging / "environment_asset.manifest.json"
    write_json(
        environment_manifest,
        {
            "schema_version": 1,
            "artifact_kind": "environment",
            "artifact_path": str(environment_asset),
            "artifact_hash": sha256_text(environment_asset.read_text(encoding="utf-8")),
        },
    )

    native_bundle = staging / "native_extension.bundle.json"
    write_json(
        native_bundle,
        {
            "schema_version": 1,
            "bundle": "snn-native",
            "exports": ["enkai_abi_version", "enkai_symbol_table"],
        },
    )
    native_manifest = staging / "native_extension.manifest.json"
    write_json(
        native_manifest,
        {
            "schema_version": 1,
            "artifact_kind": "native-extension",
            "artifact_path": str(native_bundle),
            "artifact_hash": sha256_text(native_bundle.read_text(encoding="utf-8")),
        },
    )

    commands = [
        [
            "model",
            "register",
            str(local_registry),
            "llm-chat",
            "v2.9.3",
            str(llm_checkpoint),
            "--activate",
        ],
        [
            "model",
            "register",
            str(local_registry),
            "adam0-sim",
            "v2.9.3",
            str(snapshot),
            "--artifact-kind",
            "simulation",
            "--artifact-manifest",
            str(sim_snapshot_manifest),
            "--lineage-manifest",
            str(sim_lineage),
            "--activate",
        ],
        [
            "model",
            "register",
            str(local_registry),
            "adam0-env",
            "v2.9.3",
            str(environment_asset),
            "--artifact-kind",
            "environment",
            "--artifact-manifest",
            str(environment_manifest),
            "--activate",
        ],
        [
            "model",
            "register",
            str(local_registry),
            "adam0-native",
            "v2.9.3",
            str(native_bundle),
            "--artifact-kind",
            "native-extension",
            "--artifact-manifest",
            str(native_manifest),
            "--activate",
        ],
    ]
    for command in commands:
        run([str(enkai_bin), *command], workspace, env)

    for name in ("llm-chat", "adam0-sim", "adam0-env", "adam0-native"):
        run(
            [
                str(enkai_bin),
                "model",
                "push",
                str(local_registry),
                name,
                "v2.9.3",
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
                "verify-signature",
                str(local_registry),
                name,
                "v2.9.3",
                "--registry",
                str(remote_registry),
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
                name,
                "v2.9.3",
                "--registry",
                str(remote_registry),
                "--verify-signature",
            ],
            workspace,
            env,
        )

    summary = {
        "schema_version": 1,
        "status": "ok",
        "release_line": "v2.9.3",
        "local_registry": str(local_registry),
        "remote_registry": str(remote_registry),
        "cache_registry": str(cache_registry),
        "artifacts": {
            "sim_run": str(sim_run),
            "sim_lineage": str(sim_lineage),
            "sim_snapshot_manifest": str(sim_snapshot_manifest),
            "environment_manifest": str(environment_manifest),
            "native_manifest": str(native_manifest),
        },
        "artifact_kinds": ["checkpoint", "simulation", "environment", "native-extension"],
        "registries": {
            "local": read_json(local_registry / "registry.json"),
            "remote": read_json(remote_registry / "registry.json"),
            "cache": read_json(cache_registry / "registry.json"),
        },
    }
    write_json(workspace / args.output, summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
