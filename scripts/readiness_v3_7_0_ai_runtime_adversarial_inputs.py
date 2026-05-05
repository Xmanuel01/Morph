#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_config(path: Path, payload: dict[str, Any]) -> None:
    escaped = json.dumps(payload).replace('\\', '\\\\').replace('"', '\\"')
    source = f'fn main() ::\n    return json.parse("{escaped}")\n::\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def run_command(command: list[str], cwd: Path, env: dict[str, str]) -> dict[str, Any]:
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
    return {
        "command": command,
        "exit_code": result.returncode,
        "passed": result.returncode == 0,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def expect_error(check: dict[str, Any], snippets: list[str]) -> dict[str, Any]:
    stderr = check.get("stderr_tail", "")
    return {
        "passed": (not check["passed"]) and any(snippet in stderr for snippet in snippets),
        "stderr_tail": stderr,
        "exit_code": check["exit_code"],
        "expected_snippets": snippets,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate adversarial/corruption AI runtime coverage evidence.")
    parser.add_argument("--enkai-bin", required=True)
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--suite", default="bench/suites/v3_7_0_ai_runtime_foundation.json")
    parser.add_argument("--output", default="artifacts/readiness/v3_7_0_ai_runtime_adversarial_inputs.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = Path(args.workspace).resolve()
    enkai_bin = Path(args.enkai_bin).resolve()
    suite = read_json((workspace / args.suite).resolve())
    output_path = (workspace / args.output).resolve()

    work_root = workspace / "artifacts" / "v3_7_0_ai_runtime_adversarial_inputs"
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.setdefault("ENKAI_STD", str((workspace / "std").resolve()))
    env["ENKAI_ENABLE_DIST"] = "1"

    dataset_lines = suite["dataset_lines"] * max(1, int(suite.get("dataset_repeat", 1)))
    dataset_path = work_root / "data.txt"
    dataset_path.write_text("\n".join(dataset_lines) + "\n", encoding="utf-8")

    base_config = dict(suite["base_config"])
    base_config["dataset_path"] = str(dataset_path)
    base_config["checkpoint_dir"] = str(work_root / "base_ckpt")
    base_config["tokenizer_train"] = {
        "path": str(dataset_path),
        "vocab_size": int(base_config["tokenizer_train"]["vocab_size"]),
    }

    valid_cfg_path = work_root / "valid_train.enk"
    write_config(valid_cfg_path, base_config)
    valid_train = run_command([str(enkai_bin), "train", str(valid_cfg_path)], workspace, env)
    base_report = read_json(Path(base_config["checkpoint_dir"]) / "ai_runtime_report.json") if valid_train["passed"] else None

    malformed_json_path = work_root / "malformed_json.enk"
    malformed_json_path.write_text('fn main() ::\n    return json.parse("{broken")\n::\n', encoding="utf-8")
    malformed_json = run_command([str(enkai_bin), "train", str(malformed_json_path)], workspace, env)

    env_multinode_cfg = dict(base_config)
    env_multinode_cfg["world_size"] = 2
    env_multinode_cfg["dist"] = {
        "topology": "multi-node",
        "rendezvous": "env://",
        "retry_budget": 1,
        "device_map": [0, 1],
        "preview_mode": "networked-sync-preview",
    }
    env_multinode_cfg["checkpoint_dir"] = str(work_root / "env_multinode_ckpt")
    env_multinode_path = work_root / "env_multinode.enk"
    write_config(env_multinode_path, env_multinode_cfg)
    env_multinode = run_command([str(enkai_bin), "train", str(env_multinode_path)], workspace, env)

    bad_rendezvous_cfg = dict(env_multinode_cfg)
    bad_rendezvous_cfg["dist"] = dict(env_multinode_cfg["dist"])
    bad_rendezvous_cfg["dist"]["rendezvous"] = "tcp://127.0.0.1:notaport"
    bad_rendezvous_cfg["checkpoint_dir"] = str(work_root / "bad_rendezvous_ckpt")
    bad_rendezvous_path = work_root / "bad_rendezvous.enk"
    write_config(bad_rendezvous_path, bad_rendezvous_cfg)
    bad_rendezvous = run_command([str(enkai_bin), "train", str(bad_rendezvous_path)], workspace, env)

    bad_preview_cfg = dict(base_config)
    bad_preview_cfg["world_size"] = 2
    bad_preview_cfg["dist"] = {
        "topology": "single-node",
        "rendezvous": "env://",
        "retry_budget": 1,
        "device_map": [0, 1],
        "preview_mode": "chaos-preview",
    }
    bad_preview_cfg["checkpoint_dir"] = str(work_root / "bad_preview_ckpt")
    bad_preview_path = work_root / "bad_preview.enk"
    write_config(bad_preview_path, bad_preview_cfg)
    bad_preview = run_command([str(enkai_bin), "train", str(bad_preview_path)], workspace, env)

    adversarial_checkpoint = None
    if valid_train["passed"] and base_report:
        latest = Path(base_report["latest_checkpoint_path"])
        (latest / "weights.bin").write_bytes(b"not-a-valid-weight-buffer")
        adversarial_checkpoint = run_command([str(enkai_bin), "eval", str(valid_cfg_path)], workspace, env)
    else:
        adversarial_checkpoint = {"passed": False, "exit_code": -1, "stderr_tail": "base train failed"}

    results = {
        "malformed_json": expect_error(malformed_json, ["Train error:", "parse"]),
        "env_multinode": expect_error(env_multinode, ["dist.rendezvous cannot be env:// for multi-node topology"]),
        "bad_rendezvous": expect_error(bad_rendezvous, ["tcp://", "invalid digit", "E_DIST_RENDEZVOUS_URI"]),
        "bad_preview": expect_error(bad_preview, ["dist.preview_mode must be", "chaos-preview"]),
        "corrupted_weights": expect_error(adversarial_checkpoint, ["E_CHECKPOINT_CORRUPT", "unexpected weights", "Checkpoint weights length"]),
    }

    all_passed = valid_train["passed"] and all(item["passed"] for item in results.values())
    output = {
        "schema_version": 1,
        "verified_contract_version": "v3.7.0",
        "all_passed": all_passed,
        "base_train_passed": valid_train["passed"],
        "cases": results,
        "artifacts": {
            "workspace": str(work_root),
            "dataset_path": str(dataset_path),
        },
    }
    write_json(output_path, output)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
