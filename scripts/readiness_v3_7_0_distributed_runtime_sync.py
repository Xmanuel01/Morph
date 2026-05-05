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
    escaped = json.dumps(payload).replace("\\", "\\\\").replace('"', '\\"')
    source = f'fn main() ::\n    return json.parse("{escaped}")\n::\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def hash_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(path.rglob("*")):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        if child.is_file():
            digest.update(child.read_bytes())
    return digest.hexdigest()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def semantic_meta_hash(meta_path: Path) -> str:
    meta = read_json(meta_path)
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
    digest = hashlib.sha256()
    digest.update(json.dumps(canonical, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def apply_shape_overrides(base_config: dict[str, Any], shape: dict[str, Any], checkpoint_dir: Path) -> dict[str, Any]:
    cfg = dict(base_config)
    for key, value in shape.get("config_overrides", {}).items():
        cfg[key] = value
    model = dict(base_config.get("model", {}))
    model.update(shape.get("model", {}))
    cfg["model"] = model
    if "hidden_size" in model:
        cfg["hidden_size"] = int(model["hidden_size"])
    if "vocab_size" in model:
        cfg["vocab_size"] = int(model["vocab_size"])
    cfg["checkpoint_dir"] = str(checkpoint_dir)
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute the first synchronized-gradient v3.7.0 distributed preview.")
    parser.add_argument("--enkai-bin", required=True)
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--suite", default="bench/suites/v3_7_0_ai_runtime_foundation.json")
    parser.add_argument("--output", default="artifacts/readiness/v3_7_0_distributed_runtime_sync.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.workspace).resolve()
    enkai_bin = Path(args.enkai_bin).resolve()
    suite = read_json((repo_root / args.suite).resolve())
    output_path = (repo_root / args.output).resolve()

    dist = suite["distributed_sync_preview"]
    world_size = int(dist["world_size"])
    work_root = repo_root / "artifacts" / "v3_7_0_distributed_runtime_sync"
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    dataset_lines = suite["dataset_lines"] * max(1, int(suite.get("dataset_repeat", 1)))
    dataset_path = work_root / "data.txt"
    dataset_path.write_text("\n".join(dataset_lines) + "\n", encoding="utf-8")

    env = dict(os.environ)
    env.setdefault("ENKAI_STD", str((repo_root / "std").resolve()))
    if dist.get("require_dist_opt_in", False):
        env["ENKAI_ENABLE_DIST"] = "1"

    base_config = dict(suite["base_config"])
    base_config["dataset_path"] = str(dataset_path)
    base_config["tokenizer_train"] = {
        "path": str(dataset_path),
        "vocab_size": int(base_config["tokenizer_train"]["vocab_size"]),
    }
    base_config["world_size"] = world_size
    base_config["dist"] = {
        "topology": dist["topology"],
        "rendezvous": "env://",
        "retry_budget": 1,
        "device_map": list(range(world_size)),
        "preview_mode": dist["execution_mode"],
    }
    def run_sync_case(case_name: str, case_config: dict[str, Any]) -> dict[str, Any]:
        case_root = work_root / case_name
        rank_cfgs: list[tuple[int, Path, dict[str, Any]]] = []
        for rank in range(world_size):
            cfg = dict(case_config)
            cfg["rank"] = rank
            cfg["checkpoint_dir"] = str(case_root / f"rank{rank}_ckpt")
            cfg_path = case_root / f"rank{rank}.enk"
            write_config(cfg_path, cfg)
            rank_cfgs.append((rank, cfg_path, cfg))

        train_procs: list[tuple[int, subprocess.Popen[str], dict[str, Any]]] = []
        for rank, cfg_path, cfg in rank_cfgs:
            proc = subprocess.Popen(
                [str(enkai_bin), "train", str(cfg_path)],
                cwd=repo_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            train_procs.append((rank, proc, cfg))

        rank_reports: list[dict[str, Any]] = []
        case_passed = True
        checkpoint_hashes: list[str] = []
        weight_hashes: list[str] = []
        optimizer_hashes: list[str] = []
        semantic_meta_hashes: list[str] = []
        latest_checkpoint_paths: list[Path] = []
        train_reports: list[dict[str, Any]] = []
        eval_reports: list[dict[str, Any]] = []
        min_rank_tokens = int(dist.get("min_rank_tokens", 1))
        for rank, proc, cfg in train_procs:
            stdout, stderr = proc.communicate()
            train_passed = proc.returncode == 0
            train_report = None
            checkpoint_hash = None
            if train_passed:
                train_report = read_json(Path(cfg["checkpoint_dir"]) / "ai_runtime_report.json")
                train_reports.append(train_report)
                latest = Path(train_report["latest_checkpoint_path"])
                latest_checkpoint_paths.append(latest)
                checkpoint_hash = hash_tree(latest)
                checkpoint_hashes.append(checkpoint_hash)
                weight_hashes.append(hash_file(latest / "weights.bin"))
                optimizer_hashes.append(hash_file(latest / "optimizer.bin"))
                semantic_meta_hashes.append(semantic_meta_hash(latest / "meta.json"))

            eval_passed = False
            eval_report = None
            eval_stderr = ""
            if train_passed:
                eval_proc = subprocess.run(
                    [str(enkai_bin), "eval", str(case_root / f"rank{rank}.enk")],
                    cwd=repo_root,
                    env=env,
                    capture_output=True,
                    text=True,
                )
                eval_passed = eval_proc.returncode == 0
                eval_stderr = eval_proc.stderr[-4000:]
                if eval_passed:
                    eval_report = read_json(Path(cfg["checkpoint_dir"]) / "ai_runtime_report.json")
                    eval_reports.append(eval_report)

            rank_passed = bool(
                train_passed
                and eval_passed
                and train_report is not None
                and eval_report is not None
                and train_report.get("distributed_preview_mode") == "synchronized-grad-preview"
                and train_report.get("world_size") == world_size
                and train_report.get("rank") == rank
                and int(train_report.get("tokens", 0)) >= min_rank_tokens
            )
            case_passed = case_passed and rank_passed
            rank_reports.append(
                {
                    "rank": rank,
                    "passed": rank_passed,
                    "train_exit_code": proc.returncode,
                    "eval_exit_code": 0 if eval_passed else 1,
                    "train_tokens": None if train_report is None else train_report.get("tokens"),
                    "train_elapsed_ms": None if train_report is None else train_report.get("elapsed_ms"),
                    "train_worker_count": None if train_report is None else train_report.get("worker_count"),
                    "eval_loss": None if eval_report is None else eval_report.get("loss"),
                    "eval_elapsed_ms": None if eval_report is None else eval_report.get("elapsed_ms"),
                    "world_size": None if train_report is None else train_report.get("world_size"),
                    "executed_backend": None if train_report is None else train_report.get("executed_backend"),
                    "distributed_preview_mode": None if train_report is None else train_report.get("distributed_preview_mode"),
                    "checkpoint_hash": checkpoint_hash,
                    "weights_hash": None if train_report is None else hash_file(Path(train_report["latest_checkpoint_path"]) / "weights.bin"),
                    "optimizer_hash": None if train_report is None else hash_file(Path(train_report["latest_checkpoint_path"]) / "optimizer.bin"),
                    "semantic_meta_hash": None if train_report is None else semantic_meta_hash(Path(train_report["latest_checkpoint_path"]) / "meta.json"),
                    "checkpoint_dir": cfg["checkpoint_dir"],
                    "train_stderr_tail": stderr[-4000:],
                    "eval_stderr_tail": eval_stderr,
                }
            )

        identical_checkpoints = (
            len(set(weight_hashes)) == 1
            and len(set(optimizer_hashes)) == 1
            and len(set(semantic_meta_hashes)) == 1
            and len(weight_hashes) == world_size
        )
        case_passed = case_passed and identical_checkpoints

        merged_replay = {"passed": False}
        checkpoint_merge_bytes_per_sec = 0.0
        if identical_checkpoints and latest_checkpoint_paths:
            merged_ckpt_root = case_root / "merged_ckpt"
            merged_latest = merged_ckpt_root / latest_checkpoint_paths[0].name
            merge_start = time.perf_counter()
            shutil.copytree(latest_checkpoint_paths[0], merged_latest)
            merge_elapsed_s = max(time.perf_counter() - merge_start, 1e-6)
            merged_checkpoint_hash = hash_tree(merged_latest)
            merged_checkpoint_bytes = sum(child.stat().st_size for child in merged_latest.rglob("*") if child.is_file())
            checkpoint_merge_bytes_per_sec = merged_checkpoint_bytes / merge_elapsed_s
            merge_manifest = {
                "schema_version": 1,
                "world_size": world_size,
                "source_checkpoint_hashes": checkpoint_hashes,
                "source_weights_hashes": weight_hashes,
                "source_optimizer_hashes": optimizer_hashes,
                "source_semantic_meta_hashes": semantic_meta_hashes,
                "merged_checkpoint_hash": merged_checkpoint_hash,
                "source_ranks": list(range(world_size)),
            }
            merge_manifest_path = case_root / "merged_ckpt" / "merge_manifest.json"
            write_json(merge_manifest_path, merge_manifest)

            merged_config = dict(rank_cfgs[0][2])
            merged_config["checkpoint_dir"] = str(merged_ckpt_root)
            merged_cfg_path = case_root / "merged_eval.enk"
            write_config(merged_cfg_path, merged_config)
            replay_proc = subprocess.run(
                [str(enkai_bin), "eval", str(merged_cfg_path)],
                cwd=repo_root,
                env=env,
                capture_output=True,
                text=True,
            )
            if replay_proc.returncode == 0:
                replay_report = read_json(merged_ckpt_root / "ai_runtime_report.json")
                merged_replay = {
                    "passed": True,
                    "merged_checkpoint_dir": str(merged_ckpt_root),
                    "merge_manifest": str(merge_manifest_path),
                    "merged_checkpoint_hash": merged_checkpoint_hash,
                    "eval_loss": replay_report.get("loss"),
                    "stderr_tail": replay_proc.stderr[-4000:],
                }
            else:
                merged_replay = {
                    "passed": False,
                    "stderr_tail": replay_proc.stderr[-4000:],
                }
            case_passed = case_passed and merged_replay["passed"]

        combined_train_tokens = sum(int(report.get("tokens", 0)) for report in train_reports)
        max_train_elapsed_ms = max((int(report.get("elapsed_ms", 0)) for report in train_reports), default=1)
        combined_train_tokens_per_sec = combined_train_tokens / max(max_train_elapsed_ms / 1000.0, 1e-6)
        eval_tokens = sum(
            int(case_config["batch_size"]) * int(case_config["seq_len"]) * int(report.get("eval_batches", 0))
            for report in eval_reports
        )
        max_eval_elapsed_ms = max((int(report.get("elapsed_ms", 0)) for report in eval_reports), default=1)
        combined_eval_tokens_per_sec = eval_tokens / max(max_eval_elapsed_ms / 1000.0, 1e-6)
        gates = dist.get("throughput_gates", {})
        throughput = {
            "combined_train_tokens": combined_train_tokens,
            "combined_train_tokens_per_sec": round(combined_train_tokens_per_sec, 3),
            "combined_eval_tokens": eval_tokens,
            "combined_eval_tokens_per_sec": round(combined_eval_tokens_per_sec, 3),
            "checkpoint_merge_bytes_per_sec": round(checkpoint_merge_bytes_per_sec, 3),
            "combined_train_gate_passed": combined_train_tokens_per_sec >= float(gates.get("min_combined_train_tokens_per_sec", 0.0)),
            "combined_eval_gate_passed": combined_eval_tokens_per_sec >= float(gates.get("min_combined_eval_tokens_per_sec", 0.0)),
            "checkpoint_merge_gate_passed": checkpoint_merge_bytes_per_sec >= float(gates.get("min_checkpoint_merge_bytes_per_sec", 0.0)),
        }
        case_passed = case_passed and all((
            throughput["combined_train_gate_passed"],
            throughput["combined_eval_gate_passed"],
            throughput["checkpoint_merge_gate_passed"],
        ))

        return {
            "name": case_name,
            "passed": case_passed,
            "hidden_size": case_config.get("hidden_size"),
            "layers": case_config.get("model", {}).get("n_layers"),
            "heads": case_config.get("model", {}).get("n_heads"),
            "identical_rank_checkpoints": identical_checkpoints,
            "rank_reports": rank_reports,
            "merged_replay": merged_replay,
            "throughput": throughput,
            "artifacts": {
                "workspace": str(case_root),
                "checkpoint_semantic_hashes": semantic_meta_hashes,
            },
        }

    sync_shapes = suite.get("distributed_sync_shape_frontier") or [
        {"name": "sync_base", "model": base_config["model"], "config_overrides": {}}
    ]
    case_results = []
    all_passed = True
    for shape in sync_shapes:
        case_cfg = apply_shape_overrides(base_config, shape, work_root / shape["name"] / "rank0_ckpt")
        case_result = run_sync_case(shape["name"], case_cfg)
        case_results.append(case_result)
        all_passed = all_passed and case_result["passed"]

    summary = {
        "schema_version": 1,
        "verified_contract_version": "v3.7.0",
        "all_passed": all_passed,
        "execution_mode": dist["execution_mode"],
        "topology": dist["topology"],
        "world_size": world_size,
        "synchronized_gradients": True,
        "shape_envelope": {
            "all_shapes_passed": all(result["passed"] for result in case_results),
            "cases": case_results,
        },
        "distributed_throughput": {
            "all_gates_passed": all(
                result["throughput"]["combined_train_gate_passed"]
                and result["throughput"]["combined_eval_gate_passed"]
                and result["throughput"]["checkpoint_merge_gate_passed"]
                for result in case_results
            ),
            "cases": [
                {
                    "name": result["name"],
                    **result["throughput"],
                }
                for result in case_results
            ],
        },
        "artifacts": {
            "workspace": str(work_root),
            "dataset_path": str(dataset_path),
        },
    }
    write_json(output_path, summary)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
