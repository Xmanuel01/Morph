#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def nvidia_smi_inventory() -> list[dict[str, Any]]:
    query = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(query, text=True, stderr=subprocess.STDOUT)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    gpus: list[dict[str, Any]] = []
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        index, uuid, name, driver, memory_total_mib = parts[:5]
        gpus.append(
            {
                "index": int(index) if index.isdigit() else index,
                "uuid": uuid,
                "name": name,
                "driver_version": driver,
                "memory_total_mib": memory_total_mib,
            }
        )
    return gpus


def parse_timestamp(raw: Any) -> datetime | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def path_exists(path_text: Any) -> bool:
    return isinstance(path_text, str) and Path(path_text).exists()


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify real-hardware multi-node orchestration closure.")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--contract", default="enkai/contracts/v4_0_0_multi_node_orchestration_closure.json")
    parser.add_argument("--input", default="artifacts/gpu/multi_gpu_evidence.json")
    parser.add_argument("--output", default="artifacts/readiness/v4_0_0_multi_node_orchestration_closure.json")
    args = parser.parse_args()

    root = Path(args.workspace).resolve()
    contract = read_json(root / args.contract)
    evidence_path = root / args.input
    failures: list[str] = []
    evidence: dict[str, Any] = {}

    if not evidence_path.exists():
        failures.append(f"missing evidence artifact: {args.input}")
    else:
        evidence = read_json(evidence_path)

    live_gpus = nvidia_smi_inventory()
    live_uuids = {str(gpu.get("uuid")) for gpu in live_gpus if gpu.get("uuid")}
    req = contract["two_rank_requirements"]

    if len(live_gpus) < int(req["requires_cuda_devices"]):
        failures.append(f"live GPU count below requirement: found {len(live_gpus)}, need {req['requires_cuda_devices']}")

    if evidence:
        if evidence.get("status") != req["required_status"]:
            failures.append(f"evidence status expected {req['required_status']}, found {evidence.get('status')}")
        if evidence.get("schema_version") != 1:
            failures.append("evidence schema_version must be 1")
        if evidence.get("gate") != "multi_gpu_parity":
            failures.append("evidence gate must be multi_gpu_parity")
        if int(evidence.get("world_size", 0)) != int(req["world_size"]):
            failures.append("evidence world_size mismatch")
        if evidence.get("status") in {"BLOCKED", "SKIPPED"} or evidence.get("reason"):
            failures.append("blocked/skipped evidence cannot close production multi-node orchestration")

        timestamp = parse_timestamp(evidence.get("timestamp_utc"))
        if timestamp is None:
            failures.append("evidence timestamp_utc missing or invalid")
        else:
            age_hours = (datetime.now(timezone.utc) - timestamp).total_seconds() / 3600.0
            if age_hours < 0:
                failures.append("evidence timestamp is in the future")
            if age_hours > float(req["maximum_evidence_age_hours"]):
                failures.append(f"evidence is stale: {age_hours:.2f}h > {req['maximum_evidence_age_hours']}h")

        checks = evidence.get("checks", {}) if isinstance(evidence.get("checks"), dict) else {}
        for check in req["required_checks"]:
            if checks.get(check) is not True:
                failures.append(f"required distributed check failed: {check}")

        ranks = evidence.get("ranks", []) if isinstance(evidence.get("ranks"), list) else []
        if len(ranks) != int(req["world_size"]):
            failures.append("rank report count mismatch")
        else:
            rank_ids = sorted(rank.get("rank") for rank in ranks if isinstance(rank, dict))
            if rank_ids != [0, 1]:
                failures.append(f"rank IDs expected [0, 1], found {rank_ids}")

        gpu_uuids = evidence.get("gpu_uuids", []) if isinstance(evidence.get("gpu_uuids"), list) else []
        gpu_uuids = [str(uuid) for uuid in gpu_uuids if uuid]
        if req.get("requires_distinct_gpu_uuids"):
            if len(set(gpu_uuids)) < int(req["requires_cuda_devices"]):
                failures.append("evidence must include at least two distinct GPU UUIDs")
            elif live_uuids and not set(gpu_uuids).issubset(live_uuids):
                failures.append("evidence GPU UUIDs do not match live nvidia-smi inventory")

        artifacts = evidence.get("artifacts", {}) if isinstance(evidence.get("artifacts"), dict) else {}
        for key in req["required_artifacts"]:
            if not path_exists(artifacts.get(key)):
                failures.append(f"missing archived artifact path: {key}")

        exits = evidence.get("rank_process_exit_codes", {})
        if not isinstance(exits, dict):
            failures.append("rank_process_exit_codes missing")
        else:
            for rank in ["0", "1"]:
                value = exits.get(rank, exits.get(int(rank)))
                if value != 0:
                    failures.append(f"rank {rank} exit code expected 0, found {value}")

    gates = {
        "real_two_plus_gpu_hardware": len(live_gpus) >= int(req["requires_cuda_devices"]),
        "two_rank_data_parallel_execution": bool(evidence and evidence.get("status") == "PASS" and int(evidence.get("world_size", 0)) == 2),
        "distributed_gradient_parity": bool(evidence and evidence.get("checks", {}).get("grad_parity") is True),
        "distributed_loss_parity": bool(evidence and evidence.get("checks", {}).get("loss_parity") is True),
        "rank_artifact_archival": bool(evidence and all(path_exists((evidence.get("artifacts") or {}).get(key)) for key in req["required_artifacts"])),
        "rank_exit_code_archival": bool(evidence and isinstance(evidence.get("rank_process_exit_codes"), dict)),
        "fresh_evidence_window": not any("stale" in failure or "future" in failure for failure in failures),
        "no_blocked_or_synthetic_evidence": bool(evidence and evidence.get("status") == "PASS" and not evidence.get("reason")),
    }

    result = {
        "schema_version": 1,
        "contract_version": contract["contract_version"],
        "scope": contract["scope"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": str((root / args.contract).resolve()),
        "input": str(evidence_path.resolve()),
        "live_gpus": live_gpus,
        "gates": gates,
        "production_claims": {
            "multi_node_orchestration_proven": not failures,
            "claim_without_hardware_evidence": False,
            "synthetic_evidence_allowed": False,
        },
        "all_passed": not failures,
        "failures": failures,
    }
    write_json(root / args.output, result)
    print(json.dumps({"all_passed": result["all_passed"], "failures": failures, "output": args.output}, indent=2))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
