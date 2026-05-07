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
    parser = argparse.ArgumentParser(description="Run the v3.8.0 worker lifecycle proof.")
    parser.add_argument("--enkai-bin", required=True)
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--suite", default="bench/suites/v3_8_0_worker_checkpoint.json")
    parser.add_argument("--output", default="artifacts/readiness/v3_8_0_worker_lifecycle.json")
    return parser.parse_args()


def wait_all(procs: list[tuple[int, subprocess.Popen[str], dict[str, Any]]], timeout_s: float) -> dict[int, dict[str, Any]]:
    deadline = time.monotonic() + timeout_s
    results: dict[int, dict[str, Any]] = {}
    while procs and time.monotonic() < deadline:
        remaining: list[tuple[int, subprocess.Popen[str], dict[str, Any]]] = []
        for rank, proc, cfg in procs:
            if proc.poll() is None:
                remaining.append((rank, proc, cfg))
                continue
            stdout, stderr = proc.communicate()
            results[rank] = {
                "cfg": cfg,
                "exit_code": proc.returncode,
                "stdout_tail": stdout[-4000:],
                "stderr_tail": stderr[-4000:],
                "timed_out": False,
            }
        procs = remaining
        if procs:
            time.sleep(0.05)
    for rank, proc, cfg in procs:
        timed_out = proc.poll() is None
        if timed_out:
            proc.kill()
        stdout, stderr = proc.communicate()
        results[rank] = {
            "cfg": cfg,
            "exit_code": proc.returncode,
            "stdout_tail": stdout[-4000:],
            "stderr_tail": stderr[-4000:],
            "timed_out": timed_out,
        }
    return results


def main() -> int:
    args = parse_args()
    repo_root = Path(args.workspace).resolve()
    enkai_bin = Path(args.enkai_bin).resolve()
    suite = read_json((repo_root / args.suite).resolve())
    output_path = (repo_root / args.output).resolve()
    spec = suite["worker_lifecycle"]
    world_size = int(spec["world_size"])
    work_root = repo_root / "artifacts" / "v3_8_0_worker_lifecycle"
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

    def case_rendezvous(offset: int) -> str:
        host_port = spec["rendezvous"].rsplit(":", 1)
        return f"{host_port[0]}:{int(host_port[1]) + offset}"

    def launch_case(case_name: str, max_steps: int, port_offset: int, fault: bool = False) -> dict[str, Any]:
        case_root = work_root / case_name
        rank_cfgs: list[tuple[int, Path, dict[str, Any]]] = []
        for rank in range(world_size):
            cfg = dict(base_config)
            cfg["rank"] = rank
            cfg["max_steps"] = max_steps
            cfg["checkpoint_dir"] = str(work_root / "rank_checkpoints" / f"rank{rank}")
            cfg["dist"] = {
                "topology": spec["topology"],
                "rendezvous": case_rendezvous(port_offset),
                "retry_budget": int(spec.get("retry_budget", 1)),
                "device_map": list(range(world_size)),
                "preview_mode": spec["execution_mode"],
            }
            cfg_path = case_root / f"rank{rank}.enk"
            write_config(cfg_path, cfg)
            rank_cfgs.append((rank, cfg_path, cfg))

        env = dict(base_env)
        if fault:
            fault_spec = spec["fault"]
            env["ENKAI_DIST_GRAD_FAULT_MODE"] = fault_spec["mode"]
            env["ENKAI_DIST_GRAD_FAULT_RANK"] = str(fault_spec["target_rank"])
            env["ENKAI_DIST_GRAD_FAULT_STEP"] = str(fault_spec["target_step"])
            env["ENKAI_DIST_GRAD_TIMEOUT_MS"] = str(fault_spec.get("timeout_ms", 1000))

        procs: list[tuple[int, subprocess.Popen[str], dict[str, Any]]] = []
        for rank, cfg_path, cfg in sorted(rank_cfgs, key=lambda item: (item[0] == 0, item[0])):
            proc = subprocess.Popen(
                [str(enkai_bin), "train", str(cfg_path)],
                cwd=repo_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            procs.append((rank, proc, cfg))
        raw_results = wait_all(procs, 90.0)
        rank_reports = []
        weight_hashes = []
        optimizer_hashes = []
        semantic_hashes = []
        for rank in range(world_size):
            result = raw_results[rank]
            report = None
            report_path = Path(result["cfg"]["checkpoint_dir"]) / "ai_runtime_report.json"
            if report_path.is_file():
                report = read_json(report_path)
            latest = None if report is None else report.get("latest_checkpoint_path")
            if latest and Path(latest).is_dir():
                latest_path = Path(latest)
                weight_hashes.append(hash_file(latest_path / "weights.bin"))
                optimizer_hashes.append(hash_file(latest_path / "optimizer.bin"))
                semantic_hashes.append(semantic_meta_hash(latest_path / "meta.json"))
            passed = bool(
                result["exit_code"] == 0
                and report is not None
                and report.get("step") == max_steps
                and report.get("world_size") == world_size
                and report.get("rank") == rank
                and report.get("networked_rendezvous") is True
                and report.get("networked_gradient_exchange") is True
                and int(report.get("networked_gradient_bytes", 0)) > 0
                and int(report.get("tokens", 0)) >= int(spec.get("min_rank_tokens", 1))
            )
            rank_reports.append({
                "rank": rank,
                "passed": passed,
                "exit_code": result["exit_code"],
                "timed_out": result["timed_out"],
                "stderr_tail": result["stderr_tail"],
                "stdout_tail": result["stdout_tail"],
                "checkpoint_dir": result["cfg"]["checkpoint_dir"],
                "latest_checkpoint_path": latest,
                "report": report,
            })
        identical_semantics = (
            len(weight_hashes) == world_size
            and len(set(weight_hashes)) == 1
            and len(set(optimizer_hashes)) == 1
            and len(set(semantic_hashes)) == 1
        )
        expected_ok = not fault
        if expected_ok:
            passed = all(item["passed"] for item in rank_reports) and identical_semantics
        else:
            exits = [item["exit_code"] for item in rank_reports]
            stderr_text = "\n".join(item["stderr_tail"] + item["stdout_tail"] for item in rank_reports)
            passed = any(code != 0 for code in exits) and spec["fault"]["expected_error"] in stderr_text
        return {
            "name": case_name,
            "passed": passed,
            "fault_enabled": fault,
            "identical_checkpoint_semantics": identical_semantics,
            "rank_reports": rank_reports,
        }

    baseline = launch_case("baseline", int(spec["baseline_max_steps"]), 0, False)
    crash = launch_case("faulted_recovery_attempt", int(spec["recovery_max_steps"]), 80, True)
    recovery = launch_case("supervised_recovery", int(spec["recovery_max_steps"]), 160, False)

    eval_results = []
    for rank_report in recovery["rank_reports"]:
        if not rank_report["passed"]:
            continue
        rank = int(rank_report["rank"])
        cfg = dict(base_config)
        cfg["rank"] = rank
        cfg["max_steps"] = int(spec["recovery_max_steps"])
        cfg["checkpoint_dir"] = rank_report["checkpoint_dir"]
        cfg["dist"] = {
            "topology": spec["topology"],
            "rendezvous": case_rendezvous(240),
            "retry_budget": int(spec.get("retry_budget", 1)),
            "device_map": list(range(world_size)),
            "preview_mode": spec["execution_mode"],
        }
        cfg_path = work_root / "eval" / f"rank{rank}.enk"
        write_config(cfg_path, cfg)
        proc = subprocess.run([str(enkai_bin), "eval", str(cfg_path)], cwd=repo_root, env=base_env, capture_output=True, text=True)
        report = None
        report_path = Path(cfg["checkpoint_dir"]) / "ai_runtime_report.json"
        if report_path.is_file():
            report = read_json(report_path)
        eval_results.append({
            "rank": rank,
            "passed": proc.returncode == 0 and report is not None and int(report.get("eval_batches", 0)) > 0,
            "exit_code": proc.returncode,
            "stderr_tail": proc.stderr[-4000:],
            "report": report,
        })

    all_passed = bool(
        baseline["passed"]
        and crash["passed"]
        and recovery["passed"]
        and len(eval_results) == world_size
        and all(item["passed"] for item in eval_results)
    )
    summary = {
        "schema_version": 1,
        "verified_contract_version": "v3.8.0",
        "all_passed": all_passed,
        "suite": suite["suite"],
        "execution_mode": spec["execution_mode"],
        "topology": spec["topology"],
        "world_size": world_size,
        "baseline": baseline,
        "faulted_recovery_attempt": crash,
        "supervised_recovery": recovery,
        "post_recovery_eval": eval_results,
        "health_checks": {
            "baseline_all_workers_green": baseline["passed"],
            "fault_detected_by_supervisor": crash["passed"],
            "recovery_all_workers_green": recovery["passed"],
            "post_recovery_eval_green": len(eval_results) == world_size and all(item["passed"] for item in eval_results),
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
