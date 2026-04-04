#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import re
import sys
from dataclasses import dataclass


def parse_version(root: pathlib.Path) -> str:
    cargo = (root / "enkai" / "Cargo.toml").read_text(encoding="utf-8")
    for line in cargo.splitlines():
        line = line.strip()
        if line.startswith("version"):
            parts = line.split("=", 1)
            if len(parts) == 2:
                return parts[1].strip().strip('"')
    raise RuntimeError("failed to parse version from enkai/Cargo.toml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate capability-complete report from archived release evidence."
    )
    parser.add_argument("--version", help="Release version (defaults to enkai/Cargo.toml)")
    parser.add_argument(
        "--manifest",
        help="Path to release evidence manifest (defaults to artifacts/release/v<version>/manifest.json)",
    )
    parser.add_argument("--output-json", help="Output JSON report path")
    parser.add_argument("--output-md", help="Output markdown report path")
    parser.add_argument("--require-gpu", action="store_true", help="Require GPU evidence checks")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when required capability checks fail",
    )
    return parser.parse_args()


@dataclass
class CheckResult:
    id: str
    description: str
    required: bool
    passed: bool
    details: str


def normalize(path: str) -> str:
    return path.replace("\\", "/")


def first_match(paths: list[str], predicate) -> str | None:
    for path in paths:
        if predicate(path):
            return path
    return None


def has_exact(paths: list[str], suffix: str) -> bool:
    needle = normalize(suffix)
    for path in paths:
        if path.endswith(needle):
            return True
    return False


def validate_benchmark_report(path: pathlib.Path) -> tuple[bool, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as err:  # pragma: no cover - defensive parse guard
        return False, f"{path.name}: invalid JSON ({err})"

    if not isinstance(payload, dict):
        return False, f"{path.name}: report root must be object"
    if int(payload.get("schema_version", 0)) < 2:
        return False, f"{path.name}: schema_version must be >= 2"

    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return False, f"{path.name}: summary object missing"
    if not bool(summary.get("pass", False)):
        return False, f"{path.name}: summary.pass is false"
    if int(summary.get("case_fail_count", 0)) != 0:
        return False, f"{path.name}: case_fail_count is non-zero"
    if not bool(summary.get("enforce_class_targets", False)):
        return False, f"{path.name}: enforce_class_targets must be true"
    if float(summary.get("target_speedup_pct", 0.0)) < 15.0:
        return False, f"{path.name}: target_speedup_pct must be >= 15"
    if float(summary.get("target_memory_pct", 0.0)) < 5.0:
        return False, f"{path.name}: target_memory_pct must be >= 5"

    class_summaries = summary.get("class_summaries")
    if not isinstance(class_summaries, dict):
        return False, f"{path.name}: class_summaries missing"
    required_classes = {
        "vm_compute",
        "native_bridge",
        "cli_workflows",
        "ai_data_workflows",
    }
    missing_classes = sorted(required_classes.difference(class_summaries.keys()))
    if missing_classes:
        return False, f"{path.name}: missing class summaries for {', '.join(missing_classes)}"

    class_gate_failures = summary.get("class_gate_failures")
    if isinstance(class_gate_failures, list):
        if class_gate_failures:
            return False, f"{path.name}: class_gate_failures is non-empty"
    elif isinstance(class_gate_failures, dict):
        if class_gate_failures:
            return False, f"{path.name}: class_gate_failures is non-empty"
    else:
        return False, f"{path.name}: class_gate_failures must be list or object"

    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        return False, f"{path.name}: cases list missing or empty"
    for case in cases:
        if not isinstance(case, dict):
            return False, f"{path.name}: case entries must be objects"
        case_id = str(case.get("id", "<unknown>"))
        if not bool(case.get("pass", False)):
            return False, f"{path.name}: case {case_id} did not pass"
        delta = case.get("delta")
        if not isinstance(delta, dict):
            return False, f"{path.name}: case {case_id} missing delta"
        speedup = float(delta.get("speedup_pct", 0.0))
        memory = float(delta.get("memory_reduction_pct", 0.0))
        if speedup < 15.0:
            return False, f"{path.name}: case {case_id} speedup {speedup:.2f}% < 15.00%"
        if memory < 5.0:
            return False, f"{path.name}: case {case_id} memory reduction {memory:.2f}% < 5.00%"

    return True, path.name


def validate_blocker_report(path: pathlib.Path) -> tuple[bool, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as err:  # pragma: no cover - defensive parse guard
        return False, f"{path.name}: invalid JSON ({err})"

    if not isinstance(payload, dict):
        return False, f"{path.name}: report root must be object"
    if int(payload.get("schema_version", 0)) < 1:
        return False, f"{path.name}: schema_version must be >= 1"
    if payload.get("profile") != "full_platform":
        return False, f"{path.name}: profile must be full_platform"
    if not bool(payload.get("all_passed", False)):
        return False, f"{path.name}: all_passed is false"
    if bool(payload.get("skip_release_evidence", True)):
        return False, f"{path.name}: skip_release_evidence must be false for archived strict evidence"
    for field in (
        "missing_checks",
        "failed_checks",
        "skipped_required_checks",
        "missing_artifacts",
        "missing_gpu_artifacts",
    ):
        value = payload.get(field)
        if not isinstance(value, list):
            return False, f"{path.name}: {field} must be a list"
        if value:
            return False, f"{path.name}: {field} is non-empty"

    return True, path.name


def validate_validation_report(path: pathlib.Path, expected_validation: str) -> tuple[bool, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as err:  # pragma: no cover - defensive parse guard
        return False, f"{path.name}: invalid JSON ({err})"

    if not isinstance(payload, dict):
        return False, f"{path.name}: report root must be object"
    if int(payload.get("schema_version", 0)) < 1:
        return False, f"{path.name}: schema_version must be >= 1"
    if payload.get("validation") != expected_validation:
        return False, f"{path.name}: validation must be {expected_validation}"
    if not bool(payload.get("passed", False)):
        return False, f"{path.name}: passed is false"
    return True, path.name


def select_benchmark_report_paths(root: pathlib.Path, copied_paths: list[str]) -> list[pathlib.Path]:
    bench_regex = re.compile(r"/dist/benchmark_official_[^/]+\.json$")
    benchmark_paths: list[pathlib.Path] = []
    for path in copied_paths:
        if bench_regex.search(path):
            benchmark_paths.append(root / path)
    benchmark_paths.sort(
        key=lambda p: (0 if "benchmark_official_v2_3_0_matrix_" in p.name else 1, p.name)
    )
    return benchmark_paths


def build_checks(
    root: pathlib.Path,
    version: str,
    copied_paths: list[str],
    require_gpu: bool,
) -> list[CheckResult]:
    checks: list[CheckResult] = []
    archive_prefix = f"/dist/enkai-{version}-"
    sbom_prefix = f"/dist/sbom-{version}-"

    archive = first_match(
        copied_paths,
        lambda path: archive_prefix in path and (path.endswith(".zip") or path.endswith(".tar.gz")),
    )
    checks.append(
        CheckResult(
            id="release_archive",
            description="Release archive is present",
            required=True,
            passed=archive is not None,
            details=archive or f"missing dist/enkai-{version}-<os>-<arch>.zip|tar.gz",
        )
    )

    checksum = first_match(
        copied_paths, lambda path: archive_prefix in path and path.endswith(".sha256")
    )
    checks.append(
        CheckResult(
            id="release_checksum",
            description="Release checksum is present",
            required=True,
            passed=checksum is not None,
            details=checksum
            or f"missing dist/enkai-{version}-<os>-<arch>.<ext>.sha256",
        )
    )

    sbom = first_match(copied_paths, lambda path: sbom_prefix in path and path.endswith(".json"))
    checks.append(
        CheckResult(
            id="sbom",
            description="SBOM artifact is present",
            required=True,
            passed=sbom is not None,
            details=sbom or f"missing dist/sbom-{version}-<os>-<arch>.json",
        )
    )

    benchmark_paths = select_benchmark_report_paths(root, copied_paths)
    bench = str(benchmark_paths[0].relative_to(root)) if benchmark_paths else None
    benchmark_valid = False
    benchmark_detail = "missing dist/benchmark_official_<suite>_<platform>.json"
    if benchmark_paths:
        errors: list[str] = []
        for report_path in benchmark_paths:
            valid, detail = validate_benchmark_report(report_path)
            if valid:
                benchmark_valid = True
                benchmark_detail = str(report_path.relative_to(root))
                break
            errors.append(detail)
        if not benchmark_valid:
            benchmark_detail = "; ".join(errors)
    checks.append(
        CheckResult(
            id="benchmark_target",
            description="Official benchmark target evidence is present",
            required=True,
            passed=benchmark_valid,
            details=benchmark_detail if bench is not None else benchmark_detail,
        )
    )

    has_linux_bench = any("_linux.json" in path.name for path in benchmark_paths)
    has_windows_bench = any("_windows.json" in path.name for path in benchmark_paths)
    checks.append(
        CheckResult(
            id="benchmark_cross_platform_visibility",
            description="Benchmark evidence includes both Linux and Windows artifacts",
            required=False,
            passed=has_linux_bench and has_windows_bench,
            details=(
                "linux+windows benchmark artifacts present"
                if has_linux_bench and has_windows_bench
                else (
                    f"linux_present={has_linux_bench}, windows_present={has_windows_bench}"
                )
            ),
        )
    )

    for name in (
        "litec_selfhost_ci_report.json",
        "litec_replace_check_report.json",
        "litec_mainline_ci_report.json",
        "litec_release_ci_report.json",
    ):
        present = has_exact(copied_paths, f"/selfhost/{name}")
        checks.append(
            CheckResult(
                id=f"selfhost_{name.replace('.', '_')}",
                description=f"Self-host triage artifact `{name}` is present",
                required=True,
                passed=present,
                details=name if present else f"missing selfhost/{name}",
            )
        )

    for name in (
        "backend_api_v1.snapshot.json",
        "sdk_api_v1.snapshot.json",
        "grpc_api_v1.snapshot.json",
        "worker_queue_v1.snapshot.json",
        "db_engines_v1.snapshot.json",
        "conversation_state_v1.schema.json",
    ):
        present = has_exact(copied_paths, f"/contracts/{name}")
        checks.append(
            CheckResult(
                id=f"contract_{name.replace('.', '_')}",
                description=f"Contract snapshot `{name}` is present",
                required=True,
                passed=present,
                details=name if present else f"missing contracts/{name}",
            )
        )

    readiness_present = has_exact(copied_paths, "/readiness/production.json")
    checks.append(
        CheckResult(
            id="readiness_production_report",
            description="Production readiness JSON report is present",
            required=False,
            passed=readiness_present,
            details=(
                "readiness/production.json"
                if readiness_present
                else "missing readiness/production.json"
            ),
        )
    )
    full_platform_readiness_present = has_exact(copied_paths, "/readiness/full_platform.json")
    checks.append(
        CheckResult(
            id="readiness_full_platform_report",
            description="Full-platform readiness JSON report is present",
            required=True,
            passed=full_platform_readiness_present,
            details=(
                "readiness/full_platform.json"
                if full_platform_readiness_present
                else "missing readiness/full_platform.json"
            ),
        )
    )
    blocker_report_path = first_match(
        copied_paths, lambda path: path.endswith("/readiness/full_platform_blockers.json")
    )
    blocker_report_valid = False
    blocker_report_detail = "missing readiness/full_platform_blockers.json"
    if blocker_report_path is not None:
        blocker_path = root / blocker_report_path
        blocker_report_valid, blocker_report_detail = validate_blocker_report(blocker_path)
        if blocker_report_valid:
            blocker_report_detail = blocker_report_path
    checks.append(
        CheckResult(
            id="readiness_full_platform_blocker_report",
            description="Full-platform blocker verification report is present and passing",
            required=True,
            passed=blocker_report_valid,
            details=blocker_report_detail,
        )
    )
    sim_smoke = has_exact(copied_paths, "/readiness/sim_smoke.json")
    checks.append(
        CheckResult(
            id="readiness_sim_smoke_report",
            description="Simulation readiness smoke summary is present",
            required=True,
            passed=sim_smoke,
            details=(
                "readiness/sim_smoke.json"
                if sim_smoke
                else "missing readiness/sim_smoke.json"
            ),
        )
    )
    sim_verify = has_exact(copied_paths, "/readiness/sim_evidence_verify.json")
    checks.append(
        CheckResult(
            id="readiness_sim_evidence_verify_report",
            description="Simulation evidence verification report is present",
            required=True,
            passed=sim_verify,
            details=(
                "readiness/sim_evidence_verify.json"
                if sim_verify
                else "missing readiness/sim_evidence_verify.json"
            ),
        )
    )
    sim_native_smoke = has_exact(copied_paths, "/readiness/sim_native_smoke.json")
    checks.append(
        CheckResult(
            id="readiness_sim_native_smoke_report",
            description="Simulation native FFI smoke summary is present",
            required=True,
            passed=sim_native_smoke,
            details=(
                "readiness/sim_native_smoke.json"
                if sim_native_smoke
                else "missing readiness/sim_native_smoke.json"
            ),
        )
    )
    sim_native_verify = has_exact(copied_paths, "/readiness/sim_native_evidence_verify.json")
    checks.append(
        CheckResult(
            id="readiness_sim_native_evidence_verify_report",
            description="Simulation native FFI evidence verification report is present",
            required=True,
            passed=sim_native_verify,
            details=(
                "readiness/sim_native_evidence_verify.json"
                if sim_native_verify
                else "missing readiness/sim_native_evidence_verify.json"
            ),
        )
    )
    sim_stdlib_smoke = has_exact(copied_paths, "/readiness/sim_stdlib_smoke.json")
    checks.append(
        CheckResult(
            id="readiness_sim_stdlib_smoke_report",
            description="Simulation stdlib smoke summary is present",
            required=True,
            passed=sim_stdlib_smoke,
            details=(
                "readiness/sim_stdlib_smoke.json"
                if sim_stdlib_smoke
                else "missing readiness/sim_stdlib_smoke.json"
            ),
        )
    )
    sim_stdlib_verify = has_exact(copied_paths, "/readiness/sim_stdlib_evidence_verify.json")
    checks.append(
        CheckResult(
            id="readiness_sim_stdlib_evidence_verify_report",
            description="Simulation stdlib evidence verification report is present",
            required=True,
            passed=sim_stdlib_verify,
            details=(
                "readiness/sim_stdlib_evidence_verify.json"
                if sim_stdlib_verify
                else "missing readiness/sim_stdlib_evidence_verify.json"
            ),
        )
    )
    adam0_smoke = has_exact(copied_paths, "/readiness/adam0_100_smoke.json")
    checks.append(
        CheckResult(
            id="readiness_adam0_100_smoke_report",
            description="Adam-0 100-agent simulation smoke summary is present",
            required=True,
            passed=adam0_smoke,
            details=(
                "readiness/adam0_100_smoke.json"
                if adam0_smoke
                else "missing readiness/adam0_100_smoke.json"
            ),
        )
    )
    adam0_verify = has_exact(copied_paths, "/readiness/adam0_100_evidence_verify.json")
    checks.append(
        CheckResult(
            id="readiness_adam0_100_evidence_verify_report",
            description="Adam-0 100-agent simulation evidence verification report is present",
            required=True,
            passed=adam0_verify,
            details=(
                "readiness/adam0_100_evidence_verify.json"
                if adam0_verify
                else "missing readiness/adam0_100_evidence_verify.json"
            ),
        )
    )
    adam0_reference_suite = has_exact(copied_paths, "/readiness/adam0_reference_suite.json")
    checks.append(
        CheckResult(
            id="readiness_adam0_reference_suite_report",
            description="Adam-0 reference suite summary is present",
            required=True,
            passed=adam0_reference_suite,
            details=(
                "readiness/adam0_reference_suite.json"
                if adam0_reference_suite
                else "missing readiness/adam0_reference_suite.json"
            ),
        )
    )
    adam0_reference_verify = has_exact(
        copied_paths, "/readiness/adam0_reference_suite_verify.json"
    )
    checks.append(
        CheckResult(
            id="readiness_adam0_reference_suite_verify_report",
            description="Adam-0 reference suite evidence verification report is present",
            required=True,
            passed=adam0_reference_verify,
            details=(
                "readiness/adam0_reference_suite_verify.json"
                if adam0_reference_verify
                else "missing readiness/adam0_reference_suite_verify.json"
            ),
        )
    )
    registry_convergence = has_exact(
        copied_paths, "/readiness/model_registry_convergence.json"
    )
    checks.append(
        CheckResult(
            id="readiness_model_registry_convergence_report",
            description="Model/simulation registry convergence summary is present",
            required=True,
            passed=registry_convergence,
            details=(
                "readiness/model_registry_convergence.json"
                if registry_convergence
                else "missing readiness/model_registry_convergence.json"
            ),
        )
    )
    registry_convergence_verify = has_exact(
        copied_paths, "/readiness/model_registry_convergence_verify.json"
    )
    checks.append(
        CheckResult(
            id="readiness_model_registry_convergence_verify_report",
            description="Model/simulation registry convergence verification report is present",
            required=True,
            passed=registry_convergence_verify,
            details=(
                "readiness/model_registry_convergence_verify.json"
                if registry_convergence_verify
                else "missing readiness/model_registry_convergence_verify.json"
            ),
        )
    )
    cluster_scale_smoke = has_exact(copied_paths, "/readiness/cluster_scale_smoke.json")
    checks.append(
        CheckResult(
            id="readiness_cluster_scale_smoke_report",
            description="Cluster scale smoke summary is present",
            required=True,
            passed=cluster_scale_smoke,
            details=(
                "readiness/cluster_scale_smoke.json"
                if cluster_scale_smoke
                else "missing readiness/cluster_scale_smoke.json"
            ),
        )
    )
    cluster_scale_verify = has_exact(
        copied_paths, "/readiness/cluster_scale_evidence_verify.json"
    )
    checks.append(
        CheckResult(
            id="readiness_cluster_scale_evidence_verify_report",
            description="Cluster scale evidence verification report is present",
            required=True,
            passed=cluster_scale_verify,
            details=(
                "readiness/cluster_scale_evidence_verify.json"
                if cluster_scale_verify
                else "missing readiness/cluster_scale_evidence_verify.json"
            ),
        )
    )
    registry_degraded_smoke = has_exact(copied_paths, "/readiness/registry_degraded_smoke.json")
    checks.append(
        CheckResult(
            id="readiness_registry_degraded_smoke_report",
            description="Registry degraded-mode smoke summary is present",
            required=True,
            passed=registry_degraded_smoke,
            details=(
                "readiness/registry_degraded_smoke.json"
                if registry_degraded_smoke
                else "missing readiness/registry_degraded_smoke.json"
            ),
        )
    )
    registry_degraded_verify = has_exact(
        copied_paths, "/readiness/registry_degraded_evidence_verify.json"
    )
    checks.append(
        CheckResult(
            id="readiness_registry_degraded_verify_report",
            description="Registry degraded-mode evidence verification report is present",
            required=True,
            passed=registry_degraded_verify,
            details=(
                "readiness/registry_degraded_evidence_verify.json"
                if registry_degraded_verify
                else "missing readiness/registry_degraded_evidence_verify.json"
            ),
        )
    )
    validation_specs = [
        ("ffi_correctness.json", "ffi_correctness", "Validation FFI correctness report is present and passing"),
        ("determinism_event_queue.json", "determinism", "Validation event-queue determinism report is present and passing"),
        ("determinism_sim_replay.json", "determinism", "Validation simulation replay determinism report is present and passing"),
        ("determinism_sim_coroutines.json", "determinism", "Validation simulation coroutine determinism report is present and passing"),
        ("determinism_adam0_reference_100.json", "determinism", "Validation Adam-0 100-agent determinism report is present and passing"),
        ("pool_safety.json", "pool_safety", "Validation pool safety report is present and passing"),
        ("adam0_fake10.json", "adam0_cpu", "Validation fake Adam-0 CPU report is present and passing"),
        ("adam0_ref100.json", "adam0_cpu", "Validation Adam-0 100-agent CPU report is present and passing"),
        ("adam0_stress1000.json", "adam0_cpu", "Validation Adam-0 1000-agent CPU stress report is present and passing"),
        ("adam0_target10000.json", "adam0_cpu", "Validation Adam-0 10000-agent CPU target report is present and passing"),
        ("perf_ffi_noop.json", "perf_baseline", "Validation FFI noop performance baseline report is present and passing"),
        ("perf_sparse_dot.json", "perf_baseline", "Validation sparse-dot performance baseline report is present and passing"),
        ("perf_adam0_reference_100.json", "perf_baseline", "Validation Adam-0 100-agent performance baseline report is present and passing"),
        ("perf_adam0_reference_1000.json", "perf_baseline", "Validation Adam-0 1000-agent performance baseline report is present and passing"),
        ("perf_adam0_reference_10000.json", "perf_baseline", "Validation Adam-0 10000-agent performance baseline report is present and passing"),
    ]
    for filename, expected_validation, description in validation_specs:
        matched = first_match(copied_paths, lambda path, name=filename: path.endswith(f"/validation/{name}"))
        passed = False
        details = f"missing validation/{filename}"
        if matched is not None:
            report_path = root / matched
            passed, details = validate_validation_report(report_path, expected_validation)
            if passed:
                details = matched
        checks.append(
            CheckResult(
                id=f"validation_{filename.replace('.', '_')}",
                description=description,
                required=True,
                passed=passed,
                details=details,
            )
        )
    validation_cross_platform = (
        has_exact(copied_paths, "/validation/adam0_ref100.json")
        and has_exact(copied_paths, "/validation/adam0_stress1000.json")
        and has_exact(copied_paths, "/validation/adam0_target10000.json")
        and has_exact(copied_paths, "/validation/determinism_adam0_reference_100.json")
        and has_exact(copied_paths, "/validation/perf_adam0_reference_100.json")
        and has_exact(copied_paths, "/validation/perf_adam0_reference_1000.json")
        and has_exact(copied_paths, "/validation/perf_adam0_reference_10000.json")
    )
    checks.append(
        CheckResult(
            id="validation_cpu_suite_visibility",
            description="CPU validation suite artifacts are archived",
            required=True,
            passed=validation_cross_platform,
            details=(
                "validation/adam0_ref100.json + validation/adam0_stress1000.json + validation/adam0_target10000.json + validation/determinism_adam0_reference_100.json + validation/perf_adam0_reference_100.json + validation/perf_adam0_reference_1000.json + validation/perf_adam0_reference_10000.json"
                if validation_cross_platform
                else "missing validation CPU reference artifacts"
            ),
        )
    )
    deploy_mobile = has_exact(copied_paths, "/readiness/deploy_mobile_smoke.json")
    checks.append(
        CheckResult(
            id="readiness_deploy_mobile_smoke_report",
            description="Mobile scaffold deploy validation summary is present",
            required=True,
            passed=deploy_mobile,
            details=(
                "readiness/deploy_mobile_smoke.json"
                if deploy_mobile
                else "missing readiness/deploy_mobile_smoke.json"
            ),
        )
    )
    deploy_mobile_verify = has_exact(
        copied_paths, "/readiness/deploy_mobile_evidence_verify.json"
    )
    checks.append(
        CheckResult(
            id="readiness_deploy_mobile_evidence_verify_report",
            description="Mobile scaffold evidence verification report is present",
            required=True,
            passed=deploy_mobile_verify,
            details=(
                "readiness/deploy_mobile_evidence_verify.json"
                if deploy_mobile_verify
                else "missing readiness/deploy_mobile_evidence_verify.json"
            ),
        )
    )
    grpc_smoke = has_exact(copied_paths, "/readiness/grpc_smoke.json")
    checks.append(
        CheckResult(
            id="readiness_grpc_smoke_report",
            description="gRPC runtime smoke summary is present",
            required=True,
            passed=grpc_smoke,
            details=(
                "readiness/grpc_smoke.json"
                if grpc_smoke
                else "missing readiness/grpc_smoke.json"
            ),
        )
    )
    grpc_verify = has_exact(copied_paths, "/readiness/grpc_evidence_verify.json")
    checks.append(
        CheckResult(
            id="readiness_grpc_evidence_verify_report",
            description="gRPC runtime evidence verification report is present",
            required=True,
            passed=grpc_verify,
            details=(
                "readiness/grpc_evidence_verify.json"
                if grpc_verify
                else "missing readiness/grpc_evidence_verify.json"
            ),
        )
    )
    worker_queue_smoke = has_exact(copied_paths, "/readiness/worker_queue_smoke.json")
    checks.append(
        CheckResult(
            id="readiness_worker_queue_smoke_report",
            description="Worker queue smoke summary is present",
            required=True,
            passed=worker_queue_smoke,
            details=(
                "readiness/worker_queue_smoke.json"
                if worker_queue_smoke
                else "missing readiness/worker_queue_smoke.json"
            ),
        )
    )
    worker_queue_verify = has_exact(
        copied_paths, "/readiness/worker_queue_evidence_verify.json"
    )
    checks.append(
        CheckResult(
            id="readiness_worker_queue_evidence_verify_report",
            description="Worker queue evidence verification report is present",
            required=True,
            passed=worker_queue_verify,
            details=(
                "readiness/worker_queue_evidence_verify.json"
                if worker_queue_verify
                else "missing readiness/worker_queue_evidence_verify.json"
            ),
        )
    )
    snn_agent_smoke = has_exact(copied_paths, "/readiness/snn_agent_kernel_smoke.json")
    checks.append(
        CheckResult(
            id="readiness_snn_agent_kernel_smoke_report",
            description="SNN agent kernel smoke summary is present",
            required=True,
            passed=snn_agent_smoke,
            details=(
                "readiness/snn_agent_kernel_smoke.json"
                if snn_agent_smoke
                else "missing readiness/snn_agent_kernel_smoke.json"
            ),
        )
    )
    snn_agent_verify = has_exact(copied_paths, "/readiness/snn_agent_kernel_evidence_verify.json")
    checks.append(
        CheckResult(
            id="readiness_snn_agent_kernel_evidence_verify_report",
            description="SNN agent kernel evidence verification report is present",
            required=True,
            passed=snn_agent_verify,
            details=(
                "readiness/snn_agent_kernel_evidence_verify.json"
                if snn_agent_verify
                else "missing readiness/snn_agent_kernel_evidence_verify.json"
            ),
        )
    )
    for name in ("smoke_run.json", "smoke_profile.json", "smoke_replay.json"):
        present = has_exact(copied_paths, f"/sim/{name}")
        checks.append(
            CheckResult(
                id=f"simulation_{name.replace('.', '_')}",
                description=f"Simulation evidence `{name}` is present",
                required=True,
                passed=present,
                details=name if present else f"missing sim/{name}",
            )
        )
    for name in ("native_smoke_run.json", "native_smoke_profile.json"):
        present = has_exact(copied_paths, f"/sim/{name}")
        checks.append(
            CheckResult(
                id=f"simulation_{name.replace('.', '_')}",
                description=f"Simulation native evidence `{name}` is present",
                required=True,
                passed=present,
                details=name if present else f"missing sim/{name}",
            )
        )
    for name in ("stdlib_smoke_run.json", "stdlib_smoke_profile.json"):
        present = has_exact(copied_paths, f"/sim/{name}")
        checks.append(
            CheckResult(
                id=f"simulation_{name.replace('.', '_')}",
                description=f"Simulation stdlib evidence `{name}` is present",
                required=True,
                passed=present,
                details=name if present else f"missing sim/{name}",
            )
        )
    for name in ("adam0_100_run.json", "adam0_100_profile.json"):
        present = has_exact(copied_paths, f"/sim/{name}")
        checks.append(
            CheckResult(
                id=f"simulation_{name.replace('.', '_')}",
                description=f"Adam-0 100-agent simulation evidence `{name}` is present",
                required=True,
                passed=present,
                details=name if present else f"missing sim/{name}",
            )
        )
    for name in (
        "adam0_baseline_100_run.json",
        "adam0_baseline_100_profile.json",
        "adam0_baseline_100_snapshot.json",
        "adam0_baseline_100_replay.json",
        "adam0_stress_1000_run.json",
        "adam0_stress_1000_profile.json",
        "adam0_stress_1000_snapshot.json",
        "adam0_stress_1000_replay.json",
        "adam0_target_10000_run.json",
        "adam0_target_10000_profile.json",
        "adam0_target_10000_snapshot.json",
        "adam0_target_10000_replay.json",
    ):
        present = has_exact(copied_paths, f"/sim/{name}")
        checks.append(
            CheckResult(
                id=f"simulation_{name.replace('.', '_')}",
                description=f"Adam-0 reference suite evidence `{name}` is present",
                required=True,
                passed=present,
                details=name if present else f"missing sim/{name}",
            )
        )
    for name in ("snn_agent_kernel_run.json", "snn_agent_kernel_profile.json"):
        present = has_exact(copied_paths, f"/sim/{name}")
        checks.append(
            CheckResult(
                id=f"simulation_{name.replace('.', '_')}",
                description=f"SNN agent kernel evidence `{name}` is present",
                required=True,
                passed=present,
                details=name if present else f"missing sim/{name}",
            )
        )
    for name in (
        "validate.json",
        "plan.json",
        "run.json",
        "recovery/rank0/window_0000.run.json",
        "recovery/rank0/window_0000.snapshot.json",
        "recovery/rank1/window_0000.run.json",
        "recovery/rank1/window_0000.snapshot.json",
    ):
        present = has_exact(copied_paths, f"/cluster_scale/{name}")
        checks.append(
            CheckResult(
                id=f"cluster_scale_{name.replace('/', '_').replace('.', '_')}",
                description=f"Cluster scale evidence `{name}` is present",
                required=True,
                passed=present,
                details=name if present else f"missing cluster_scale/{name}",
            )
        )
    for name in (
        "sim_lineage.json",
        "sim_snapshot.manifest.json",
        "local/registry.json",
        "remote/registry.json",
        "cache/registry.json",
        f"remote/adam0-sim/v{version}/remote.manifest.json",
        f"remote/adam0-sim/v{version}/remote.manifest.sig",
    ):
        present = has_exact(copied_paths, f"/registry/{name}")
        checks.append(
            CheckResult(
                id=f"registry_{name.replace('/', '_').replace('.', '_')}",
                description=f"Registry convergence evidence `{name}` is present",
                required=True,
                passed=present,
                details=name if present else f"missing registry/{name}",
            )
        )
    for name in (
        "cache/registry.json",
        "cache/audit.log.jsonl",
        f"remote_offline/adam0-degraded/v{version}/remote.manifest.json",
        f"remote_offline/adam0-degraded/v{version}/remote.manifest.sig",
    ):
        present = has_exact(copied_paths, f"/registry_degraded/{name}")
        checks.append(
            CheckResult(
                id=f"registry_degraded_{name.replace('/', '_').replace('.', '_')}",
                description=f"Registry degraded evidence `{name}` is present",
                required=True,
                passed=present,
                details=name if present else f"missing registry_degraded/{name}",
            )
        )

    for name in (
        "sdk_api.snapshot.json",
        "app.json",
        "package.json",
    ):
        present = has_exact(copied_paths, f"/mobile/{name}")
        checks.append(
            CheckResult(
                id=f"mobile_{name.replace('/', '_').replace('.', '_')}",
                description=f"Mobile scaffold evidence `{name}` is present",
                required=True,
                passed=present,
                details=name if present else f"missing mobile/{name}",
            )
        )

    for name in (
        "probe.json",
        "server.jsonl",
        "conversation_state.json",
        "conversation_state.backup.json",
    ):
        present = has_exact(copied_paths, f"/grpc/{name}")
        checks.append(
            CheckResult(
                id=f"grpc_{name.replace('/', '_').replace('.', '_')}",
                description=f"gRPC runtime evidence `{name}` is present",
                required=True,
                passed=present,
                details=name if present else f"missing grpc/{name}",
            )
        )

    for name in (
        "run_01.json",
        "run_02.json",
        "run_03.json",
        "queues/default/dead_letter.jsonl",
        "queues/default/pending.jsonl",
    ):
        present = has_exact(copied_paths, f"/worker_queue/{name}")
        checks.append(
            CheckResult(
                id=f"worker_queue_{name.replace('/', '_').replace('.', '_')}",
                description=f"Worker queue evidence `{name}` is present",
                required=True,
                passed=present,
                details=name if present else f"missing worker_queue/{name}",
            )
        )

    for name in (
        "single_gpu.log",
        "single_gpu_evidence.json",
        "multi_gpu.log",
        "multi_gpu_evidence.json",
        "soak_4gpu.log",
        "soak_4gpu_evidence.json",
    ):
        present = has_exact(copied_paths, f"/gpu/{name}")
        checks.append(
            CheckResult(
                id=f"gpu_{name.replace('.', '_')}",
                description=f"GPU evidence `{name}` is present",
                required=require_gpu,
                passed=present,
                details=name if present else f"missing gpu/{name}",
            )
        )

    return checks


def to_markdown(report: dict[str, object]) -> str:
    rows = [
        "| Check | Required | Status | Details |",
        "| --- | --- | --- | --- |",
    ]
    for check in report["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        required = "yes" if check["required"] else "no"
        rows.append(
            f"| `{check['id']}` | {required} | {status} | {check['details']} |"
        )

    summary = report["summary"]
    lines = [
        f"# Capability-Complete Report ({report['version']})",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Manifest: `{report['manifest']}`",
        f"- Strict mode: `{report['strict']}`",
        f"- GPU required: `{report['require_gpu']}`",
        f"- Required checks passed: `{summary['passed_required']}/{summary['required_checks']}`",
        f"- Status: `{summary['status']}`",
        "",
        "## Checks",
        "",
        *rows,
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    version = args.version or parse_version(root)

    manifest_path = pathlib.Path(args.manifest).resolve() if args.manifest else (
        root / "artifacts" / "release" / f"v{version}" / "manifest.json"
    )
    if not manifest_path.is_file():
        raise RuntimeError(f"manifest not found: {manifest_path}")

    out_json = pathlib.Path(args.output_json).resolve() if args.output_json else (
        manifest_path.parent / "capability_complete.json"
    )
    out_md = pathlib.Path(args.output_md).resolve() if args.output_md else (
        manifest_path.parent / "capability_complete.md"
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    copied_paths = [normalize(entry.get("copied_to", "")) for entry in manifest.get("files", [])]

    checks = build_checks(root, version, copied_paths, args.require_gpu)
    required_checks = [check for check in checks if check.required]
    passed_required = sum(1 for check in required_checks if check.passed)
    failed_required = [check for check in required_checks if not check.passed]
    status = "pass" if not failed_required else "fail"

    report = {
        "schema": "enkai-capability-report-v1",
        "version": f"v{version}",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "manifest": str(manifest_path.relative_to(root)),
        "strict": args.strict,
        "require_gpu": args.require_gpu,
        "summary": {
            "required_checks": len(required_checks),
            "passed_required": passed_required,
            "status": status,
        },
        "checks": [
            {
                "id": check.id,
                "description": check.description,
                "required": check.required,
                "passed": check.passed,
                "details": check.details,
            }
            for check in checks
        ],
    }

    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text(to_markdown(report) + "\n", encoding="utf-8")

    payload = {
        "status": "ok" if (not args.strict or not failed_required) else "failed",
        "version": f"v{version}",
        "manifest": str(manifest_path.relative_to(root)),
        "output_json": str(out_json.relative_to(root)),
        "output_md": str(out_md.relative_to(root)),
        "required_checks": len(required_checks),
        "passed_required": passed_required,
        "require_gpu": args.require_gpu,
        "strict": args.strict,
    }
    print(json.dumps(payload, separators=(",", ":")))

    if args.strict and failed_required:
        failed_ids = ", ".join(check.id for check in failed_required)
        raise RuntimeError(f"capability report strict checks failed: {failed_ids}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as err:  # pragma: no cover - script entrypoint
        print(f"error: {err}", file=sys.stderr)
        raise SystemExit(1)
