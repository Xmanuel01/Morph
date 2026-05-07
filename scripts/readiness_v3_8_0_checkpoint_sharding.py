#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
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
    escaped = json.dumps(payload).replace('\\', '\\\\').replace('"', '\\"')
    source = f'fn main() ::\n    return json.parse("{escaped}")\n::\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def hash_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(path.rglob("*")):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        if child.is_file():
            digest.update(child.read_bytes())
    return digest.hexdigest()


def directory_size(path: Path) -> int:
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def semantic_meta_hash(path: Path) -> str:
    meta = read_json(path)
    canonical = {
        "format_version": meta["format_version"],
        "step": meta["step"],
        "tokens": meta["tokens"],
        "loss": meta["loss"],
        "model_sig": meta["model_sig"],
        "dtype": meta["dtype"],
        "device": meta["device"],
        "grad_accum_steps": meta["grad_accum_steps"],
        "grad_clip_norm": meta["grad_clip_norm"],
        "amp": meta["amp"],
    }
    digest = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the v3.8.0 checkpoint sharding/restore proof.")
    parser.add_argument("--enkai-bin", required=True)
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--suite", default="bench/suites/v3_8_0_worker_checkpoint.json")
    parser.add_argument("--worker-report", default="artifacts/readiness/v3_8_0_worker_lifecycle.json")
    parser.add_argument("--output", default="artifacts/readiness/v3_8_0_checkpoint_sharding.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.workspace).resolve()
    enkai_bin = Path(args.enkai_bin).resolve()
    suite = read_json((repo_root / args.suite).resolve())
    worker_report = read_json((repo_root / args.worker_report).resolve())
    output_path = (repo_root / args.output).resolve()
    spec = suite["checkpoint_sharding"]
    world_size = int(spec["world_size"])
    work_root = repo_root / "artifacts" / "v3_8_0_checkpoint_sharding"
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    if not worker_report.get("all_passed"):
        summary = {"schema_version": 1, "verified_contract_version": "v3.8.0", "all_passed": False, "failures": ["worker lifecycle report is not green"]}
        write_json(output_path, summary)
        return 1

    recovery_reports = sorted(worker_report["supervised_recovery"]["rank_reports"], key=lambda item: item["rank"])
    shard_root = work_root / "shards"
    manifest_entries = []
    copy_start = time.perf_counter()
    for rank_report in recovery_reports:
        rank = int(rank_report["rank"])
        source = Path(rank_report["latest_checkpoint_path"])
        dest = shard_root / f"rank{rank}" / source.name
        shutil.copytree(source, dest)
        entry = {
            "rank": rank,
            "source_checkpoint": str(source),
            "shard_checkpoint": str(dest),
            "checkpoint_hash": hash_tree(dest),
            "weights_hash": hash_file(dest / "weights.bin"),
            "optimizer_hash": hash_file(dest / "optimizer.bin"),
            "semantic_meta_hash": semantic_meta_hash(dest / "meta.json"),
            "bytes": directory_size(dest),
        }
        manifest_entries.append(entry)
    copy_elapsed_s = max(time.perf_counter() - copy_start, 1e-6)
    total_shard_bytes = sum(int(entry["bytes"]) for entry in manifest_entries)
    shard_copy_bytes_per_sec = total_shard_bytes / copy_elapsed_s
    shard_manifest = {
        "schema_version": 1,
        "world_size": world_size,
        "shards": manifest_entries,
        "total_shard_bytes": total_shard_bytes,
        "shard_copy_bytes_per_sec": round(shard_copy_bytes_per_sec, 3),
    }
    shard_manifest_path = work_root / "shard_manifest.json"
    write_json(shard_manifest_path, shard_manifest)

    identical_semantics = (
        len(manifest_entries) == world_size
        and len({entry["weights_hash"] for entry in manifest_entries}) == 1
        and len({entry["optimizer_hash"] for entry in manifest_entries}) == 1
        and len({entry["semantic_meta_hash"] for entry in manifest_entries}) == 1
    )

    merged_root = work_root / "merged_checkpoint"
    first_shard = Path(manifest_entries[0]["shard_checkpoint"])
    merged_step = merged_root / first_shard.name
    shutil.copytree(first_shard, merged_step)
    merge_manifest_path = merged_root / "merge_manifest.json"
    write_json(merge_manifest_path, {
        "schema_version": 1,
        "world_size": world_size,
        "source_shard_manifest": str(shard_manifest_path),
        "merged_checkpoint": str(merged_step),
        "merged_checkpoint_hash": hash_tree(merged_step),
        "identical_source_semantics": identical_semantics,
    })

    base_config = dict(suite["base_config"])
    dataset_path = Path(worker_report["artifacts"]["dataset_path"])
    base_config["dataset_path"] = str(dataset_path)
    base_config["tokenizer_train"] = {
        "path": str(dataset_path),
        "vocab_size": int(base_config["tokenizer_train"]["vocab_size"]),
    }
    base_config["world_size"] = world_size
    base_config["rank"] = 0
    base_config["checkpoint_dir"] = str(merged_root)
    base_config["dist"] = {
        "topology": suite["worker_lifecycle"]["topology"],
        "rendezvous": "tcp://127.0.0.1:43901",
        "retry_budget": int(suite["worker_lifecycle"].get("retry_budget", 1)),
        "device_map": list(range(world_size)),
        "preview_mode": suite["worker_lifecycle"]["execution_mode"],
    }
    eval_cfg = work_root / "merged_eval.enk"
    write_config(eval_cfg, base_config)
    env = dict(os.environ)
    env.setdefault("ENKAI_STD", str((repo_root / "std").resolve()))
    env["ENKAI_ENABLE_DIST"] = "1"
    replay = subprocess.run([str(enkai_bin), "eval", str(eval_cfg)], cwd=repo_root, env=env, capture_output=True, text=True)
    replay_report = None
    if (merged_root / "ai_runtime_report.json").is_file():
        replay_report = read_json(merged_root / "ai_runtime_report.json")
    merged_replay = {
        "passed": replay.returncode == 0 and replay_report is not None and int(replay_report.get("eval_batches", 0)) >= int(spec["throughput_gates"]["min_merged_replay_eval_batches"]),
        "exit_code": replay.returncode,
        "stderr_tail": replay.stderr[-4000:],
        "report": replay_report,
    }

    corrupt_root = work_root / "corrupt_checkpoint"
    corrupt_step = corrupt_root / first_shard.name
    shutil.copytree(first_shard, corrupt_step)
    weights_path = corrupt_step / "weights.bin"
    original = weights_path.read_bytes()
    weights_path.write_bytes(original[:-4] if len(original) >= 4 else b"")
    corrupt_config = dict(base_config)
    corrupt_config["checkpoint_dir"] = str(corrupt_root)
    corrupt_cfg = work_root / "corrupt_eval.enk"
    write_config(corrupt_cfg, corrupt_config)
    corrupt = subprocess.run([str(enkai_bin), "eval", str(corrupt_cfg)], cwd=repo_root, env=env, capture_output=True, text=True)
    corrupt_text = corrupt.stderr + corrupt.stdout
    corruption_expected = corrupt.returncode != 0 and any(token in corrupt_text for token in spec["expected_corruption_errors"])
    corruption_case = {
        "passed": corruption_expected,
        "exit_code": corrupt.returncode,
        "expected_errors": spec["expected_corruption_errors"],
        "stderr_tail": corrupt.stderr[-4000:],
        "stdout_tail": corrupt.stdout[-4000:],
    }

    gates = spec["throughput_gates"]
    gates_report = {
        "shard_copy_bytes_per_sec": round(shard_copy_bytes_per_sec, 3),
        "shard_copy_gate_passed": shard_copy_bytes_per_sec >= float(gates["min_shard_copy_bytes_per_sec"]),
        "merged_replay_eval_batches": 0 if replay_report is None else int(replay_report.get("eval_batches", 0)),
        "merged_replay_eval_gate_passed": merged_replay["passed"],
    }
    all_passed = bool(
        identical_semantics
        and merged_replay["passed"]
        and corruption_case["passed"]
        and gates_report["shard_copy_gate_passed"]
    )
    summary = {
        "schema_version": 1,
        "verified_contract_version": "v3.8.0",
        "all_passed": all_passed,
        "suite": suite["suite"],
        "world_size": world_size,
        "identical_checkpoint_semantics": identical_semantics,
        "shard_manifest": str(shard_manifest_path),
        "merge_manifest": str(merge_manifest_path),
        "merged_replay": merged_replay,
        "corruption_case": corruption_case,
        "throughput_gates": gates_report,
        "artifacts": {"workspace": str(work_root)},
    }
    write_json(output_path, summary)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
