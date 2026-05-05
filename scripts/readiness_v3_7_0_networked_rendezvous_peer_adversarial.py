#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import struct
import subprocess
import time
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_config(path: Path, payload: dict[str, Any]) -> None:
    escaped = json.dumps(payload).replace("\\", "\\\\").replace('"', '\\"')
    source = f'fn main() ::\n    return json.parse("{escaped}")\n::\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def usize_bytes(value: int) -> bytes:
    return int(value).to_bytes(8, "little", signed=False)


def group_id(config: dict[str, Any]) -> str:
    model = config.get("model", {})
    digest = hashlib.sha256()
    for text in (
        "train",
        str(config["backend"]),
        "enkai_accel",
        str(config.get("suite_id", "")),
        str(config["dataset_path"]),
    ):
        digest.update(text.encode("utf-8"))
    for value in (
        config["seq_len"],
        config["batch_size"],
        config["hidden_size"],
        config["vocab_size"],
        config["world_size"],
    ):
        digest.update(usize_bytes(int(value)))
    digest.update(str(model.get("preset", "enkai_accel_v1")).encode("utf-8"))
    digest.update(usize_bytes(int(model.get("n_layers", 1))))
    digest.update(usize_bytes(int(model.get("n_heads", 1))))
    digest.update(struct.pack("<f", float(model.get("ff_mult", 1.0))))
    digest.update(str(model.get("activation", "gelu")).lower().encode("utf-8"))
    digest.update(str(model.get("norm", "rmsnorm")).lower().encode("utf-8"))
    digest.update(bytes([1 if bool(model.get("tie_embeddings", False)) else 0]))
    dist = config["dist"]
    digest.update(str(dist["preview_mode"]).lower().encode("utf-8"))
    digest.update(str(dist["topology"]).lower().encode("utf-8"))
    digest.update(str(dist["rendezvous"]).encode("utf-8"))
    return digest.hexdigest()


def phase_port(base_port: int, step: int, phase: str) -> int:
    phase_index = {"pre-sync": 0, "post-sync": 1}.get(phase, 2)
    return base_port + step * 4 + phase_index


def send_peer_payload(host: str, port: int, payload: Any) -> str:
    deadline = time.time() + 15
    last_error = ""
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0) as sock:
                if isinstance(payload, str):
                    wire = payload
                else:
                    wire = json.dumps(payload)
                sock.sendall((wire + "\n").encode("utf-8"))
                try:
                    sock.settimeout(1.0)
                    return sock.recv(4096).decode("utf-8", errors="replace")
                except socket.timeout:
                    return ""
        except OSError as err:
            last_error = str(err)
            time.sleep(0.05)
    raise RuntimeError(f"failed to connect malicious peer to {host}:{port}: {last_error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exercise adversarial peer behavior on the networked rendezvous surface.")
    parser.add_argument("--enkai-bin", required=True)
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--suite", default="bench/suites/v3_7_0_ai_runtime_foundation.json")
    parser.add_argument("--output", default="artifacts/readiness/v3_7_0_networked_rendezvous_peer_adversarial.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.workspace).resolve()
    enkai_bin = Path(args.enkai_bin).resolve()
    suite = read_json((repo_root / args.suite).resolve())
    spec = suite["networked_peer_adversarial"]
    output_path = (repo_root / args.output).resolve()
    work_root = repo_root / "artifacts" / "v3_7_0_networked_rendezvous_peer_adversarial"
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    dataset_path = work_root / "data.txt"
    dataset_path.write_text("\n".join(suite["dataset_lines"]) + "\n", encoding="utf-8")
    env = dict(os.environ)
    env.setdefault("ENKAI_STD", str((repo_root / "std").resolve()))
    if spec.get("require_dist_opt_in", False):
        env["ENKAI_ENABLE_DIST"] = "1"

    case_results = []
    base_config = dict(suite["base_config"])
    base_config.update(
        {
            "dataset_path": str(dataset_path),
            "tokenizer_train": {
                "path": str(dataset_path),
                "vocab_size": int(base_config["tokenizer_train"]["vocab_size"]),
            },
            "seq_len": 8,
            "batch_size": 2,
            "max_steps": 1,
            "save_every": 1,
            "eval_steps": 1,
            "hidden_size": 24,
            "vocab_size": 96,
            "model": {
                "vocab_size": 96,
                "hidden_size": 24,
                "device": "cpu",
            },
        }
    )

    for index, case in enumerate(spec["cases"]):
        case_root = work_root / case["name"]
        base_port = int(spec["rendezvous_base_port"]) + index * 20
        rendezvous = f"tcp://127.0.0.1:{base_port}"
        cfg = dict(base_config)
        cfg["world_size"] = int(case["world_size"])
        cfg["rank"] = 0
        cfg["checkpoint_dir"] = str(case_root / "rank0_ckpt")
        cfg["dist"] = {
            "topology": "multi-node",
            "rendezvous": rendezvous,
            "retry_budget": 1,
            "device_map": list(range(int(case["world_size"]))),
            "preview_mode": "networked-sync-preview",
        }
        cfg_path = case_root / "rank0.enk"
        write_config(cfg_path, cfg)
        computed_group = group_id(cfg)
        proc = subprocess.Popen(
            [str(enkai_bin), "train", str(cfg_path)],
            cwd=repo_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        peer_errors = []
        try:
            for payload in case["peer_payloads"]:
                if isinstance(payload, dict):
                    payload = dict(payload)
                    payload["schema_version"] = 1
                    payload["step"] = 1
                    payload["phase"] = "pre-sync"
                    if payload.get("group_id") == "$computed":
                        payload["group_id"] = computed_group
                peer_errors.append(send_peer_payload("127.0.0.1", phase_port(base_port, 1, "pre-sync"), payload))
            stdout, stderr = proc.communicate(timeout=20)
        except Exception as err:
            proc.kill()
            stdout, stderr = proc.communicate()
            stderr = stderr + f"\npeer harness error: {err}"

        combined = stdout + "\n" + stderr + "\n" + "\n".join(peer_errors)
        expected = case["expected_error"]
        passed = proc.returncode != 0 and expected in combined
        case_results.append(
            {
                "name": case["name"],
                "passed": passed,
                "exit_code": proc.returncode,
                "expected_error": expected,
                "observed_expected_error": expected in combined,
                "computed_group_id": computed_group,
                "stderr_tail": stderr[-4000:],
                "peer_output_tail": "\n".join(peer_errors)[-1000:],
            }
        )

    summary = {
        "schema_version": 1,
        "verified_contract_version": "v3.7.0",
        "all_passed": all(case["passed"] for case in case_results),
        "surface": "networked_rendezvous_peer_behavior",
        "cases": case_results,
        "artifacts": {
            "workspace": str(work_root),
            "dataset_path": str(dataset_path),
        },
    }
    write_json(output_path, summary)
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
