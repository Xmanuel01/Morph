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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute the first networked multi-process rendezvous v3.7.0 preview.")
    parser.add_argument("--enkai-bin", required=True)
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--suite", default="bench/suites/v3_7_0_ai_runtime_foundation.json")
    parser.add_argument("--output", default="artifacts/readiness/v3_7_0_networked_rendezvous_exec.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.workspace).resolve()
    enkai_bin = Path(args.enkai_bin).resolve()
    suite = read_json((repo_root / args.suite).resolve())
    output_path = (repo_root / args.output).resolve()

    spec = suite["networked_rendezvous_exec"]
    world_size = int(spec["world_size"])
    work_root = repo_root / "artifacts" / "v3_7_0_networked_rendezvous_exec"
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    dataset_lines = suite["dataset_lines"] * max(1, int(suite.get("dataset_repeat", 1)))
    dataset_path = work_root / "data.txt"
    dataset_path.write_text("\n".join(dataset_lines) + "\n", encoding="utf-8")

    base_env = dict(os.environ)
    base_env.setdefault("ENKAI_STD", str((repo_root / "std").resolve()))
    if spec.get("require_dist_opt_in", False):
        base_env["ENKAI_ENABLE_DIST"] = "1"

    base_config = dict(suite["base_config"])
    base_config["dataset_path"] = str(dataset_path)
    base_config["tokenizer_train"] = {
        "path": str(dataset_path),
        "vocab_size": int(base_config["tokenizer_train"]["vocab_size"]),
    }
    base_config["world_size"] = world_size
    base_config["dist"] = {
        "topology": spec["topology"],
        "rendezvous": spec["rendezvous"],
        "retry_budget": int(spec.get("retry_budget", 1)),
        "device_map": list(range(world_size)),
        "preview_mode": spec["execution_mode"],
    }

    fault = suite.get("networked_rendezvous_fault_injection", {})

    def run_case(case_name: str, fault_enabled: bool) -> dict[str, Any]:
        case_root = work_root / case_name
        rank_cfgs: list[tuple[int, Path, dict[str, Any]]] = []
        for rank in range(world_size):
            cfg = dict(base_config)
            cfg["rank"] = rank
            cfg["checkpoint_dir"] = str(case_root / f"rank{rank}_ckpt")
            cfg_path = case_root / f"rank{rank}.enk"
            write_config(cfg_path, cfg)
            rank_cfgs.append((rank, cfg_path, cfg))

        env = dict(base_env)
        if fault_enabled:
            env["ENKAI_DIST_FAULT_MODE"] = fault["fault_mode"]
            env["ENKAI_DIST_FAULT_RANK"] = str(fault.get("target_rank", 0))
            env["ENKAI_DIST_FAULT_STEP"] = str(fault.get("target_step", 1))
            env["ENKAI_DIST_FAULT_PHASE"] = fault.get("target_phase", "pre-sync")
            env["ENKAI_DIST_FAULT_DELAY_MS"] = str(fault.get("delay_ms", 250))

        launch_cfgs = rank_cfgs
        if fault_enabled:
            launch_cfgs = sorted(rank_cfgs, key=lambda item: (item[0] == 0, item[0]))

        train_procs: list[tuple[int, subprocess.Popen[str], dict[str, Any]]] = []
        for rank, cfg_path, cfg in launch_cfgs:
            proc = subprocess.Popen(
                [str(enkai_bin), "train", str(cfg_path)],
                cwd=repo_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            train_procs.append((rank, proc, cfg))

        all_passed = True
        rank_reports = []
        retries = 0
        fault_observed = False
        weight_hashes = []
        optimizer_hashes = []
        semantic_hashes = []
        latest_checkpoint_paths = []
        min_rank_tokens = int(spec.get("min_rank_tokens", 1))
        train_results: dict[int, dict[str, Any]] = {}
        for rank, proc, cfg in train_procs:
            stdout, stderr = proc.communicate()
            train_report = None
            if proc.returncode == 0:
                train_report = read_json(Path(cfg["checkpoint_dir"]) / "ai_runtime_report.json")
            train_results[rank] = {
                "cfg": cfg,
                "train_exit": proc.returncode,
                "train_stderr": stderr[-4000:],
                "train_report": train_report,
            }
            latest = None if train_report is None else Path(train_report["latest_checkpoint_path"])
            if latest is not None:
                latest_checkpoint_paths.append(latest)
                weight_hashes.append(hash_file(latest / "weights.bin"))
                optimizer_hashes.append(hash_file(latest / "optimizer.bin"))
                semantic_hashes.append(semantic_meta_hash(latest / "meta.json"))
                retries += int(train_report.get("dist_retry_count", 0))
                fault_observed = fault_observed or bool(train_report.get("fault_injection_observed", False))

        eval_procs: list[tuple[int, subprocess.Popen[str]]] = []
        for rank, cfg_path, cfg in rank_cfgs:
            if train_results[rank]["train_report"] is None:
                continue
            proc = subprocess.Popen(
                [str(enkai_bin), "eval", str(cfg_path)],
                cwd=repo_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            eval_procs.append((rank, proc))

        eval_results: dict[int, dict[str, Any]] = {}
        for rank, proc in eval_procs:
            stdout, stderr = proc.communicate()
            eval_report = None
            if proc.returncode == 0:
                eval_report = read_json(Path(train_results[rank]["cfg"]["checkpoint_dir"]) / "ai_runtime_report.json")
            eval_results[rank] = {
                "eval_exit": proc.returncode,
                "eval_stderr": stderr[-4000:],
                "eval_report": eval_report,
            }

        for rank, cfg_path, cfg in rank_cfgs:
            train_result = train_results[rank]
            train_report = train_result["train_report"]
            eval_result = eval_results.get(rank, {"eval_exit": 1, "eval_stderr": "", "eval_report": None})
            eval_report = eval_result["eval_report"]
            passed = bool(
                train_result["train_exit"] == 0
                and eval_result["eval_exit"] == 0
                and train_report is not None
                and eval_report is not None
                and train_report.get("networked_rendezvous") is True
                and train_report.get("distributed_preview_mode") == "networked-sync-preview"
                and train_report.get("world_size") == world_size
                and train_report.get("networked_gradient_exchange") is True
                and int(train_report.get("networked_gradient_bytes", 0)) > 0
                and eval_report.get("networked_rendezvous") is True
                and eval_report.get("distributed_preview_mode") == "networked-sync-preview"
                and int(train_report.get("tokens", 0)) >= min_rank_tokens
            )
            all_passed = all_passed and passed
            rank_reports.append({
                "rank": rank,
                "passed": passed,
                "train_exit_code": train_result["train_exit"],
                "eval_exit_code": eval_result["eval_exit"],
                "train_tokens": None if train_report is None else train_report.get("tokens"),
                "train_elapsed_ms": None if train_report is None else train_report.get("elapsed_ms"),
                "dist_retry_count": None if train_report is None else train_report.get("dist_retry_count"),
                "fault_injection_observed": None if train_report is None else train_report.get("fault_injection_observed"),
                "networked_gradient_exchange": None if train_report is None else train_report.get("networked_gradient_exchange"),
                "networked_gradient_bytes": None if train_report is None else train_report.get("networked_gradient_bytes"),
                "checkpoint_bytes": None if train_report is None else train_report.get("checkpoint_bytes"),
                "eval_loss": None if eval_report is None else eval_report.get("loss"),
                "eval_elapsed_ms": None if eval_report is None else eval_report.get("elapsed_ms"),
                "eval_batches": None if eval_report is None else eval_report.get("eval_batches"),
                "train_stderr_tail": train_result["train_stderr"],
                "eval_stderr_tail": eval_result["eval_stderr"],
            })
        identical_semantics = len(set(weight_hashes)) == 1 and len(set(optimizer_hashes)) == 1 and len(set(semantic_hashes)) == 1 and len(weight_hashes) == world_size
        all_passed = all_passed and identical_semantics
        merged_replay = {"passed": False}
        checkpoint_merge_bytes_per_sec = 0.0
        if identical_semantics and latest_checkpoint_paths:
            merged_ckpt_root = case_root / "merged_ckpt"
            merged_latest = merged_ckpt_root / latest_checkpoint_paths[0].name
            merge_start = time.perf_counter()
            if merged_ckpt_root.exists():
                shutil.rmtree(merged_ckpt_root)
            shutil.copytree(latest_checkpoint_paths[0], merged_latest)
            merge_elapsed_s = max(time.perf_counter() - merge_start, 1e-6)
            merged_checkpoint_hash = hash_tree(merged_latest)
            merged_checkpoint_bytes = sum(child.stat().st_size for child in merged_latest.rglob("*") if child.is_file())
            checkpoint_merge_bytes_per_sec = merged_checkpoint_bytes / merge_elapsed_s
            merge_manifest_path = merged_ckpt_root / "merge_manifest.json"
            write_json(merge_manifest_path, {
                "schema_version": 1,
                "world_size": world_size,
                "source_weights_hashes": weight_hashes,
                "source_optimizer_hashes": optimizer_hashes,
                "source_semantic_meta_hashes": semantic_hashes,
                "merged_checkpoint_hash": merged_checkpoint_hash,
            })
            merged_config = dict(rank_cfgs[0][2])
            merged_config["checkpoint_dir"] = str(merged_ckpt_root)
            merged_cfg_path = case_root / "merged_eval.enk"
            write_config(merged_cfg_path, merged_config)
            replay_proc = subprocess.run(
                [str(enkai_bin), "eval", str(merged_cfg_path)],
                cwd=repo_root,
                env=base_env,
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
            all_passed = all_passed and merged_replay["passed"]

        train_reports = [
            result["train_report"]
            for result in train_results.values()
            if result["train_report"] is not None
        ]
        eval_reports = [
            result["eval_report"]
            for result in eval_results.values()
            if result["eval_report"] is not None
        ]
        combined_train_tokens = sum(int(report.get("tokens", 0)) for report in train_reports)
        max_train_elapsed_ms = max((int(report.get("elapsed_ms", 0)) for report in train_reports), default=1)
        combined_train_tokens_per_sec = combined_train_tokens / max(max_train_elapsed_ms / 1000.0, 1e-6)
        combined_eval_tokens = sum(
            int(base_config["batch_size"]) * int(base_config["seq_len"]) * int(report.get("eval_batches", 0))
            for report in eval_reports
        )
        max_eval_elapsed_ms = max((int(report.get("elapsed_ms", 0)) for report in eval_reports), default=1)
        combined_eval_tokens_per_sec = combined_eval_tokens / max(max_eval_elapsed_ms / 1000.0, 1e-6)
        total_gradient_bytes = sum(int(report.get("networked_gradient_bytes", 0)) for report in train_reports)
        gates = spec.get("throughput_gates", {})
        throughput = {
            "combined_train_tokens_per_sec": round(combined_train_tokens_per_sec, 3),
            "combined_eval_tokens_per_sec": round(combined_eval_tokens_per_sec, 3),
            "checkpoint_merge_bytes_per_sec": round(checkpoint_merge_bytes_per_sec, 3),
            "networked_gradient_bytes": total_gradient_bytes,
            "combined_train_gate_passed": combined_train_tokens_per_sec >= float(gates.get("min_combined_train_tokens_per_sec", 0.0)),
            "combined_eval_gate_passed": combined_eval_tokens_per_sec >= float(gates.get("min_combined_eval_tokens_per_sec", 0.0)),
            "checkpoint_merge_gate_passed": checkpoint_merge_bytes_per_sec >= float(gates.get("min_checkpoint_merge_bytes_per_sec", 0.0)),
            "networked_gradient_bytes_gate_passed": total_gradient_bytes >= int(gates.get("min_networked_gradient_bytes", 0)),
        }
        all_passed = all_passed and all((
            throughput["combined_train_gate_passed"],
            throughput["combined_eval_gate_passed"],
            throughput["checkpoint_merge_gate_passed"],
            throughput["networked_gradient_bytes_gate_passed"],
        ))
        if fault_enabled:
            all_passed = all_passed and retries >= int(fault.get("expected_min_retry_count", 1)) and fault_observed
        return {
            "name": case_name,
            "passed": all_passed,
            "fault_enabled": fault_enabled,
            "total_retry_count": retries,
            "fault_injection_observed": fault_observed,
            "identical_checkpoint_semantics": identical_semantics,
            "merged_replay": merged_replay,
            "throughput": throughput,
            "rank_reports": rank_reports,
        }

    baseline = run_case("baseline", False)
    injected = run_case("fault_injected", True)
    summary = {
        "schema_version": 1,
        "verified_contract_version": "v3.7.0",
        "all_passed": bool(baseline["passed"] and injected["passed"]),
        "execution_mode": spec["execution_mode"],
        "topology": spec["topology"],
        "world_size": world_size,
        "baseline": baseline,
        "fault_injection": injected,
        "artifacts": {
            "workspace": str(work_root),
            "dataset_path": str(dataset_path),
        },
    }
    write_json(output_path, summary)
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
