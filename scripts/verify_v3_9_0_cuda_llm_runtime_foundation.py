#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify v3.9.0 CUDA-first LLM runtime foundation evidence.")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--contract", default="enkai/contracts/v3_9_0_cuda_llm_runtime_foundation.json")
    parser.add_argument("--input", default="artifacts/readiness/v3_9_0_cuda_llm_runtime_foundation.json")
    parser.add_argument("--output", default="artifacts/readiness/v3_9_0_cuda_llm_runtime_foundation_verify.json")
    args = parser.parse_args()

    root = Path(args.workspace).resolve()
    contract = read_json(root / args.contract)
    evidence = read_json(root / args.input)
    failures: list[str] = []

    if evidence.get("schema_version") != 1:
        failures.append("evidence schema_version must be 1")
    if evidence.get("contract_version") != contract.get("contract_version"):
        failures.append("contract version mismatch")
    if evidence.get("scope") != contract.get("scope"):
        failures.append("scope mismatch")
    if evidence.get("all_passed") is not True:
        failures.append("evidence all_passed expected True")

    backend = evidence.get("backend_catalog_source_checks", {})
    for key in [
        "catalog_abi_present",
        "cuda_backend_kind_present",
        "rocm_reserved_error_present",
        "metal_reserved_error_present",
        "cuda_unavailable_error_present",
        "enkai_cuda_harness_present",
        "first_party_cuda_source_present",
        "fusion_kernel_source_present",
        "cuda_bounded_llm_kernel_set_source_complete",
        "kv_cache_attention_source_present",
        "cuda_feature_requires_nvcc",
        "cuda_kernel_manifest_abi_present",
        "mixed_precision_policy_abi_present",
        "gpu_memory_planner_abi_present",
        "gpu_memory_deterministic_errors_present",
        "kv_cache_policy_abi_present",
        "large_checkpoint_policy_abi_present",
        "distributed_training_policy_abi_present",
        "kv_checkpoint_dist_errors_present",
        "amp_deterministic_errors_present",
        "tokenizer_provenance_present",
        "dataset_cursor_replay_present",
        "dataset_pipeline_manifest_present",
        "rocm_metal_feature_surfaces_present",
        "rocm_real_backend_source_present",
        "metal_real_backend_source_present",
        "stable_package_model_api_lock_present",
        "distributed_gpu_execution_proof_surface_present",
    ]:
        if backend.get(key) is not True:
            failures.append(f"backend source check failed: {key}")

    pytorch = evidence.get("pytorch_reference", {})
    if pytorch.get("available") is not True:
        failures.append("PyTorch reference unavailable")
    if pytorch.get("cuda_available") is not True:
        failures.append("PyTorch CUDA unavailable")
    if pytorch.get("passed") is not True:
        failures.append("PyTorch reference benchmark failed")
    metrics = pytorch.get("metrics", {})
    for metric in contract["required_metrics"]:
        if not isinstance(metrics.get(metric), (int, float)) or metrics.get(metric) <= 0:
            failures.append(f"required PyTorch metric missing or non-positive: {metric}")
    if (
        isinstance(metrics.get("loss_initial"), (int, float))
        and isinstance(metrics.get("loss_final"), (int, float))
        and metrics["loss_final"] > metrics["loss_initial"] * 1.25
    ):
        failures.append("PyTorch loss_final regressed more than allowed bounded threshold")

    enkai = evidence.get("enkai_cuda_backend", {})
    if enkai.get("passed") is not True:
        failures.append("Enkai CUDA benchmark failed")
    enkai_metrics = enkai.get("metrics", {})
    if enkai_metrics.get("skipped") is True:
        failures.append(f"Enkai CUDA benchmark skipped: {enkai_metrics.get('reason')}")
    for metric in contract["required_metrics"]:
        if not isinstance(enkai_metrics.get(metric), (int, float)) or enkai_metrics.get(metric) <= 0:
            failures.append(f"required Enkai CUDA metric missing or non-positive: {metric}")
    if (
        isinstance(enkai_metrics.get("loss_initial"), (int, float))
        and isinstance(enkai_metrics.get("loss_final"), (int, float))
        and enkai_metrics["loss_final"] > enkai_metrics["loss_initial"] * 1.25
    ):
        failures.append("Enkai CUDA loss_final regressed more than allowed bounded threshold")

    comparisons = evidence.get("comparisons", {})
    for ratio in [
        "enkai_vs_pytorch_train_tokens_per_sec_ratio",
        "enkai_vs_pytorch_eval_tokens_per_sec_ratio",
        "enkai_vs_pytorch_peak_memory_bytes_ratio",
        "enkai_vs_pytorch_checkpoint_write_bytes_per_sec_ratio",
        "enkai_vs_pytorch_checkpoint_resume_ms_ratio",
    ]:
        if not isinstance(comparisons.get(ratio), (int, float)) or comparisons.get(ratio) <= 0:
            failures.append(f"missing positive comparison ratio: {ratio}")

    targets = contract.get("performance_targets", {})
    for ratio, minimum in targets.get("throughput_min_ratios", {}).items():
        value = comparisons.get(ratio)
        if not isinstance(value, (int, float)) or value < float(minimum):
            failures.append(
                f"performance target failed: {ratio} expected >= {minimum}, got {value}"
            )
    for ratio, maximum in targets.get("resource_max_ratios", {}).items():
        value = comparisons.get(ratio)
        if not isinstance(value, (int, float)) or value > float(maximum):
            failures.append(
                f"resource target failed: {ratio} expected <= {maximum}, got {value}"
            )
    loss_delta = comparisons.get("loss_delta_abs")
    loss_limit = targets.get("quality_limits", {}).get("loss_delta_abs_max")
    if loss_limit is not None and (not isinstance(loss_delta, (int, float)) or loss_delta > float(loss_limit)):
        failures.append(f"loss parity target failed: loss_delta_abs expected <= {loss_limit}, got {loss_delta}")

    claims = evidence.get("production_claims", {})
    if claims.get("rocm_production_supported") is not False:
        failures.append("ROCm must not be claimed production-supported in this tranche")
    if claims.get("metal_production_supported") is not False:
        failures.append("Metal must not be claimed production-supported in this tranche")
    if claims.get("full_pytorch_parity_claimed") is not False:
        failures.append("full PyTorch parity must not be claimed in this tranche")
    if claims.get("enkai_cuda_reference_archived") is not True:
        failures.append("Enkai CUDA reference evidence must be archived")
    if claims.get("bounded_50_percent_performance_target_passed") is not True:
        failures.append("bounded 50% performance target claim must be true after green performance gate")

    for rel, snippets in contract.get("required_text_snippets", {}).items():
        text = (root / rel).read_text(encoding="utf-8-sig")
        for snippet in snippets:
            if snippet not in text:
                failures.append(f"missing snippet in {rel}: {snippet}")

    result = {
        "schema_version": 1,
        "contract": str((root / args.contract).resolve()),
        "input": str((root / args.input).resolve()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "all_passed": not failures,
        "failures": failures,
    }
    write_json(root / args.output, result)
    print(json.dumps(result, indent=2))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
