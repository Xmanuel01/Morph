#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import re
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def cargo_version() -> str:
    cargo = read("enkai/Cargo.toml")
    m = re.search(r'^version\s*=\s*"([^"]+)"', cargo, flags=re.MULTILINE)
    if not m:
        raise RuntimeError("failed to parse version from enkai/Cargo.toml")
    return m.group(1)


def main() -> int:
    version = cargo_version()
    expected_tag = f"v{version}"
    failures: list[str] = []

    main_rs = read("enkai/src/main.rs")
    if 'const LANG_VERSION: &' in main_rs and 'env!("ENKAI_LANG_VERSION")' not in main_rs:
        failures.append("enkai/src/main.rs still hardcodes LANG_VERSION")
    if '"readiness" => readiness::readiness_command' not in main_rs:
        failures.append("enkai/src/main.rs missing readiness command wiring")
    if '"deploy" => deploy::deploy_command' not in main_rs:
        failures.append("enkai/src/main.rs missing deploy command wiring")

    readme = read("README.md")
    if f"Status ({expected_tag})" not in readme:
        failures.append(f"README.md missing Status ({expected_tag})")
    if "Distributed stubs:" in readme:
        failures.append("README.md contains outdated distributed stubs claim")

    docs_readme = read("docs/README.md")
    if expected_tag not in docs_readme:
        failures.append(f"docs/README.md missing release tag {expected_tag}")
    if "VALIDATION.md" not in docs_readme:
        failures.append("docs/README.md missing VALIDATION.md reference")

    spec = read("docs/Enkai.spec")
    if f"v0.1 -> {expected_tag}" not in spec:
        failures.append("docs/Enkai.spec title is out of sync with crate version")
    if f"Known Limits in {expected_tag}" not in spec:
        failures.append("docs/Enkai.spec known limits header is out of sync")
    if "compile to stub functions" in spec:
        failures.append("docs/Enkai.spec still claims tool declarations compile to stubs")

    validation = read("VALIDATION.md")
    if "Validation Matrix" not in validation:
        failures.append("VALIDATION.md title should use release-line validation matrix wording")
    for token in (
        "enkai validate ffi-correctness",
        "enkai validate ffi-safety",
        "artifacts/validation/ffi_correctness.json",
        "artifacts/validation/ffi_safety.json",
        "artifacts/validation/determinism_event_queue.json",
        "artifacts/validation/pool_safety.json",
        "artifacts/validation/adam0_ref100.json",
    ):
        if token not in validation:
            failures.append(f"VALIDATION.md missing validation reference: {token}")

    if "enkai validate ffi-correctness" not in readme:
        failures.append("README.md missing validate ffi-correctness reference")
    if "enkai validate ffi-safety" not in readme:
        failures.append("README.md missing validate ffi-safety reference")
    if "artifacts/validation/" not in readme:
        failures.append("README.md missing validation artifact reference")

    release_checklist = read("docs/RELEASE_CHECKLIST.md")
    if "scripts/release_pipeline.ps1" not in release_checklist:
        failures.append("docs/RELEASE_CHECKLIST.md missing scripts/release_pipeline.ps1")
    if "scripts/release_pipeline.sh" not in release_checklist:
        failures.append("docs/RELEASE_CHECKLIST.md missing scripts/release_pipeline.sh")
    if "scripts/package_release.py" not in release_checklist:
        failures.append("docs/RELEASE_CHECKLIST.md missing scripts/package_release.py")
    if "scripts/verify_release_artifact.py" not in release_checklist:
        failures.append("docs/RELEASE_CHECKLIST.md missing scripts/verify_release_artifact.py")
    if "v1.9 consolidated pipeline" in release_checklist:
        failures.append("docs/RELEASE_CHECKLIST.md still references v1.9-specific pipeline wording")
    if "scripts/rc_pipeline.ps1" not in release_checklist:
        failures.append("docs/RELEASE_CHECKLIST.md missing scripts/rc_pipeline.ps1")
    if "scripts/rc_pipeline.sh" not in release_checklist:
        failures.append("docs/RELEASE_CHECKLIST.md missing scripts/rc_pipeline.sh")
    if "scripts/collect_release_evidence.py" not in release_checklist:
        failures.append("docs/RELEASE_CHECKLIST.md missing scripts/collect_release_evidence.py")
    if "scripts/generate_capability_report.py" not in release_checklist:
        failures.append("docs/RELEASE_CHECKLIST.md missing scripts/generate_capability_report.py")
    if "scripts/generate_release_dashboard.py" not in release_checklist:
        failures.append("docs/RELEASE_CHECKLIST.md missing scripts/generate_release_dashboard.py")
    if "enkai readiness check --profile production" not in release_checklist:
        failures.append("docs/RELEASE_CHECKLIST.md missing readiness command gate")
    if "enkai readiness check --profile full_platform" not in release_checklist:
        failures.append("docs/RELEASE_CHECKLIST.md missing full_platform readiness gate")
    if "enkai readiness verify-blockers --profile full_platform" not in release_checklist:
        failures.append("docs/RELEASE_CHECKLIST.md missing blocker verification gate")
    if "enkai litec release-ci" not in release_checklist:
        failures.append("docs/RELEASE_CHECKLIST.md missing litec release-ci gate")
    if "official_v2_3_0_matrix" not in release_checklist:
        failures.append("docs/RELEASE_CHECKLIST.md missing official_v2_3_0_matrix benchmark gate")
    if "bench/machines/linux_ref.json" not in release_checklist:
        failures.append("docs/RELEASE_CHECKLIST.md missing linux_ref benchmark gate")
    if "bench/machines/windows_ref.json" not in release_checklist:
        failures.append("docs/RELEASE_CHECKLIST.md missing windows_ref benchmark gate")
    if "--enforce-class-targets --class-targets bench/suites/official_v2_3_0_targets.json" not in release_checklist:
        failures.append("docs/RELEASE_CHECKLIST.md missing class-target benchmark enforcement gate")
    if "--fairness-check-only" not in release_checklist:
        failures.append("docs/RELEASE_CHECKLIST.md missing fairness-only benchmark precheck gate")

    version_token = version.replace(".", "_")
    for wrapper in (
        ROOT / f"scripts/v{version_token}_rc_pipeline.ps1",
        ROOT / f"scripts/v{version_token}_rc_pipeline.sh",
    ):
        if not wrapper.is_file():
            failures.append(f"missing RC wrapper for current version: {wrapper.relative_to(ROOT)}")

    frontend_docs = read("docs/27_frontend_stack.md")
    if "backend_api.snapshot.json" not in frontend_docs:
        failures.append("docs/27_frontend_stack.md missing backend snapshot reference")
    if "sdk_api.snapshot.json" not in frontend_docs:
        failures.append("docs/27_frontend_stack.md missing SDK snapshot reference")

    capability_doc = ROOT / "docs/36_capability_complete_report.md"
    if not capability_doc.is_file():
        failures.append("missing docs/36_capability_complete_report.md")
    else:
        capability_text = capability_doc.read_text(encoding="utf-8")
        if "scripts/collect_release_evidence.py" not in capability_text:
            failures.append(
                "docs/36_capability_complete_report.md missing collect_release_evidence reference"
            )
        if "scripts/generate_capability_report.py" not in capability_text:
            failures.append(
                "docs/36_capability_complete_report.md missing generate_capability_report reference"
            )
        if "scripts/generate_release_dashboard.py" not in capability_text:
            failures.append(
                "docs/36_capability_complete_report.md missing generate_release_dashboard reference"
            )
        if "readiness/production.json" not in capability_text:
            failures.append(
                "docs/36_capability_complete_report.md missing readiness evidence reference"
            )
        if "readiness/full_platform.json" not in capability_text:
            failures.append(
                "docs/36_capability_complete_report.md missing full_platform readiness evidence reference"
            )
        if "readiness/full_platform_blockers.json" not in capability_text:
            failures.append(
                "docs/36_capability_complete_report.md missing blocker readiness evidence reference"
            )
        if "readiness/sim_smoke.json" not in capability_text:
            failures.append(
                "docs/36_capability_complete_report.md missing simulation readiness evidence reference"
            )
        if "validation/ffi_correctness.json" not in capability_text:
            failures.append(
                "docs/36_capability_complete_report.md missing validation archive reference"
            )
        if "validation/ffi_safety.json" not in capability_text:
            failures.append(
                "docs/36_capability_complete_report.md missing validation FFI safety archive reference"
            )
        if "validation/perf_adam0_reference_100.json" not in capability_text:
            failures.append(
                "docs/36_capability_complete_report.md missing validation perf archive reference"
            )
        if "validation/perf_adam0_reference_1000.json" not in capability_text:
            failures.append(
                "docs/36_capability_complete_report.md missing Adam-0 1000-agent perf archive reference"
            )
        if "validation/perf_adam0_reference_10000.json" not in capability_text:
            failures.append(
                "docs/36_capability_complete_report.md missing Adam-0 10000-agent perf archive reference"
            )
        if "readiness/sim_evidence_verify.json" not in capability_text:
            failures.append(
                "docs/36_capability_complete_report.md missing simulation evidence verification reference"
            )
        if "readiness/sim_native_smoke.json" not in capability_text:
            failures.append(
                "docs/36_capability_complete_report.md missing simulation native smoke reference"
            )
        if "readiness/sim_native_evidence_verify.json" not in capability_text:
            failures.append(
                "docs/36_capability_complete_report.md missing simulation native evidence verification reference"
            )
        if "readiness/sim_stdlib_smoke.json" not in capability_text:
            failures.append(
                "docs/36_capability_complete_report.md missing simulation stdlib smoke reference"
            )
        if "readiness/sim_stdlib_evidence_verify.json" not in capability_text:
            failures.append(
                "docs/36_capability_complete_report.md missing simulation stdlib evidence verification reference"
            )
        if "readiness/adam0_reference_suite.json" not in capability_text:
            failures.append(
                "docs/36_capability_complete_report.md missing Adam-0 reference suite summary reference"
            )
        if "readiness/adam0_reference_suite_verify.json" not in capability_text:
            failures.append(
                "docs/36_capability_complete_report.md missing Adam-0 reference suite verification reference"
            )
        if "readiness/model_registry_convergence.json" not in capability_text:
            failures.append(
                "docs/36_capability_complete_report.md missing registry convergence summary reference"
            )
        if "readiness/model_registry_convergence_verify.json" not in capability_text:
            failures.append(
                "docs/36_capability_complete_report.md missing registry convergence verification reference"
            )
        if "readiness/snn_agent_kernel_smoke.json" not in capability_text:
            failures.append(
                "docs/36_capability_complete_report.md missing SNN agent kernel smoke reference"
            )
        if "readiness/snn_agent_kernel_evidence_verify.json" not in capability_text:
            failures.append(
                "docs/36_capability_complete_report.md missing SNN agent kernel evidence verification reference"
            )
        if "readiness/runtime_safety.json" not in capability_text:
            failures.append(
                "docs/36_capability_complete_report.md missing runtime safety summary reference"
            )
        if "readiness/runtime_safety_verify.json" not in capability_text:
            failures.append(
                "docs/36_capability_complete_report.md missing runtime safety verification reference"
            )
        if "sim/smoke_run.json" not in capability_text:
            failures.append(
                "docs/36_capability_complete_report.md missing simulation evidence archive reference"
            )
        if "sim/native_smoke_run.json" not in capability_text:
            failures.append(
                "docs/36_capability_complete_report.md missing simulation native evidence archive reference"
            )
        if "sim/stdlib_smoke_run.json" not in capability_text:
            failures.append(
                "docs/36_capability_complete_report.md missing simulation stdlib evidence archive reference"
            )
        if "sim/adam0_target_10000_run.json" not in capability_text:
            failures.append(
                "docs/36_capability_complete_report.md missing Adam-0 reference suite archive reference"
            )
        if "sim/snn_agent_kernel_run.json" not in capability_text:
            failures.append(
                "docs/36_capability_complete_report.md missing SNN agent kernel evidence archive reference"
            )
        if "registry/sim_lineage.json" not in capability_text:
            failures.append(
                "docs/36_capability_complete_report.md missing registry simulation lineage archive reference"
            )
        if "registry/remote/adam0-sim/v<version>/remote.manifest.json" not in capability_text:
            failures.append(
                "docs/36_capability_complete_report.md missing registry remote manifest archive reference"
            )
        if "release_dashboard.json" not in capability_text:
            failures.append(
                "docs/36_capability_complete_report.md missing release dashboard JSON reference"
            )
        if "release_dashboard.md" not in capability_text:
            failures.append(
                "docs/36_capability_complete_report.md missing release dashboard markdown reference"
            )

    readiness_doc = ROOT / "docs/37_readiness_matrix.md"
    if not readiness_doc.is_file():
        failures.append("missing docs/37_readiness_matrix.md")
    else:
        readiness_text = readiness_doc.read_text(encoding="utf-8")
        if "enkai readiness verify-blockers --profile full_platform" not in readiness_text:
            failures.append("docs/37_readiness_matrix.md missing blocker verification reference")
        if "artifacts/readiness/sim_smoke.json" not in readiness_text:
            failures.append("docs/37_readiness_matrix.md missing simulation smoke artifact reference")
        if "artifacts/readiness/sim_evidence_verify.json" not in readiness_text:
            failures.append("docs/37_readiness_matrix.md missing simulation verification artifact reference")
        if "artifacts/readiness/sim_native_smoke.json" not in readiness_text:
            failures.append("docs/37_readiness_matrix.md missing simulation native smoke artifact reference")
        if "artifacts/readiness/sim_native_evidence_verify.json" not in readiness_text:
            failures.append("docs/37_readiness_matrix.md missing simulation native verification artifact reference")
        if "artifacts/readiness/sim_stdlib_smoke.json" not in readiness_text:
            failures.append("docs/37_readiness_matrix.md missing simulation stdlib smoke artifact reference")
        if "artifacts/readiness/sim_stdlib_evidence_verify.json" not in readiness_text:
            failures.append("docs/37_readiness_matrix.md missing simulation stdlib verification artifact reference")
        if "artifacts/readiness/adam0_reference_suite.json" not in readiness_text:
            failures.append("docs/37_readiness_matrix.md missing Adam-0 reference suite artifact reference")
        if "artifacts/readiness/adam0_reference_suite_verify.json" not in readiness_text:
            failures.append("docs/37_readiness_matrix.md missing Adam-0 reference suite verification artifact reference")
        if "artifacts/readiness/model_registry_convergence.json" not in readiness_text:
            failures.append("docs/37_readiness_matrix.md missing registry convergence artifact reference")
        if "artifacts/readiness/model_registry_convergence_verify.json" not in readiness_text:
            failures.append("docs/37_readiness_matrix.md missing registry convergence verification artifact reference")
        if "artifacts/readiness/snn_agent_kernel_smoke.json" not in readiness_text:
            failures.append("docs/37_readiness_matrix.md missing SNN agent kernel smoke artifact reference")
        if "artifacts/readiness/snn_agent_kernel_evidence_verify.json" not in readiness_text:
            failures.append("docs/37_readiness_matrix.md missing SNN agent kernel verification artifact reference")
        if "artifacts/validation/ffi_correctness.json" not in readiness_text:
            failures.append("docs/37_readiness_matrix.md missing validation ffi correctness artifact reference")
        if "artifacts/validation/ffi_safety.json" not in readiness_text:
            failures.append("docs/37_readiness_matrix.md missing validation ffi safety artifact reference")
        if "artifacts/validation/adam0_ref100.json" not in readiness_text:
            failures.append("docs/37_readiness_matrix.md missing validation Adam-0 ref100 artifact reference")
        if "artifacts/validation/perf_adam0_reference_100.json" not in readiness_text:
            failures.append("docs/37_readiness_matrix.md missing validation Adam-0 perf artifact reference")
        if "artifacts/validation/perf_adam0_reference_1000.json" not in readiness_text:
            failures.append("docs/37_readiness_matrix.md missing Adam-0 1000-agent perf artifact reference")
        if "artifacts/validation/perf_adam0_reference_10000.json" not in readiness_text:
            failures.append("docs/37_readiness_matrix.md missing Adam-0 10000-agent perf artifact reference")
        if "artifacts/readiness/runtime_safety.json" not in readiness_text:
            failures.append("docs/37_readiness_matrix.md missing runtime safety artifact reference")
        if "artifacts/readiness/runtime_safety_verify.json" not in readiness_text:
            failures.append("docs/37_readiness_matrix.md missing runtime safety verification artifact reference")
    adam0_doc = ROOT / "docs/39_adam0_reference_stack.md"
    if not adam0_doc.is_file():
        failures.append("missing docs/39_adam0_reference_stack.md")
    else:
        adam0_text = adam0_doc.read_text(encoding="utf-8")
        if "examples/adam0_reference/main.enk" not in adam0_text:
            failures.append("docs/39_adam0_reference_stack.md missing Adam-0 reference script reference")
        if "bench/suites/adam0_reference_v2_9_4.json" not in adam0_text:
            failures.append("docs/39_adam0_reference_stack.md missing Adam-0 reference suite reference")
        if "artifacts/readiness/adam0_reference_suite.json" not in adam0_text:
            failures.append("docs/39_adam0_reference_stack.md missing Adam-0 readiness suite artifact reference")
        if "artifacts/sim/adam0_target_10000_run.json" not in adam0_text:
            failures.append("docs/39_adam0_reference_stack.md missing Adam-0 target run artifact reference")
    registry_doc = ROOT / "docs/40_registry_convergence.md"
    if not registry_doc.is_file():
        failures.append("missing docs/40_registry_convergence.md")
    else:
        registry_text = registry_doc.read_text(encoding="utf-8")
        if "enkai model verify-signature" not in registry_text:
            failures.append("docs/40_registry_convergence.md missing verify-signature command reference")
        if "artifacts/readiness/model_registry_convergence.json" not in registry_text:
            failures.append("docs/40_registry_convergence.md missing registry convergence readiness artifact reference")
        if "artifacts/registry/sim_lineage.json" not in registry_text:
            failures.append("docs/40_registry_convergence.md missing simulation lineage artifact reference")
    release_dashboard_doc = ROOT / "docs/48_release_dashboard.md"
    if not release_dashboard_doc.is_file():
        failures.append("missing docs/48_release_dashboard.md")
    else:
        dashboard_text = release_dashboard_doc.read_text(encoding="utf-8")
        for token in (
            "release_dashboard.json",
            "release_dashboard.md",
            "scripts/generate_release_dashboard.py",
            "GPU/operator evidence",
        ):
            if token not in dashboard_text:
                failures.append(
                    f"docs/48_release_dashboard.md missing required reference: {token}"
                )
    benchmark_doc = ROOT / "docs/33_benchmark_suite.md"
    if not benchmark_doc.is_file():
        failures.append("missing docs/33_benchmark_suite.md")
    else:
        benchmark_text = benchmark_doc.read_text(encoding="utf-8")
        if "official_v2_3_0_matrix" not in benchmark_text:
            failures.append("docs/33_benchmark_suite.md missing official_v2_3_0_matrix reference")
        if "workload_equivalence_v1.json" not in benchmark_text:
            failures.append("docs/33_benchmark_suite.md missing workload equivalence contract reference")
    bench_readme = ROOT / "bench/README.md"
    if not bench_readme.is_file():
        failures.append("missing bench/README.md")
    else:
        bench_readme_text = bench_readme.read_text(encoding="utf-8")
        if "official_v2_3_0_matrix" not in bench_readme_text:
            failures.append("bench/README.md missing official_v2_3_0_matrix reference")
        if "--enforce-class-targets" not in bench_readme_text:
            failures.append("bench/README.md missing class-target enforcement reference")
        if "workload_equivalence_v1.json" not in bench_readme_text:
            failures.append("bench/README.md missing workload equivalence contract reference")

    required_snapshots = [
        ROOT / "enkai/contracts/backend_api_v1.snapshot.json",
        ROOT / "enkai/contracts/sdk_api_v1.snapshot.json",
        ROOT / "enkai/contracts/conversation_state_v1.schema.json",
        ROOT / "enkai/contracts/readiness_production_v2_3_0.json",
        ROOT / "enkai/contracts/readiness_full_platform_v2_5_0.json",
        ROOT / "enkai/contracts/full_platform_release_blockers_v2_5_0.json",
    ]
    for snapshot in required_snapshots:
        if not snapshot.is_file():
            failures.append(f"missing contract snapshot file: {snapshot.relative_to(ROOT)}")

    required_bench_assets = [
        ROOT / "bench/suites/official_v2_3_0_matrix.json",
        ROOT / "bench/suites/official_v2_3_0_vm_compute.json",
        ROOT / "bench/suites/official_v2_3_0_native_bridge.json",
        ROOT / "bench/suites/official_v2_3_0_cli_workflows.json",
        ROOT / "bench/suites/official_v2_3_0_ai_data_workflows.json",
        ROOT / "bench/suites/official_v2_3_0_targets.json",
        ROOT / "bench/contracts/workload_equivalence_v1.json",
        ROOT / "bench/baselines/v2_2_0/pre_recovery_baseline.json",
    ]
    for asset in required_bench_assets:
        if not asset.is_file():
            failures.append(f"missing benchmark asset: {asset.relative_to(ROOT)}")

    rc_notes = ROOT / "docs/31_v2_rc_notes.md"
    if not rc_notes.is_file():
        failures.append("missing docs/31_v2_rc_notes.md")
    else:
        rc_text = rc_notes.read_text(encoding="utf-8")
        if "v2." not in rc_text:
            failures.append("docs/31_v2_rc_notes.md missing v2.x references")
        if "strict compatibility" not in rc_text.lower():
            failures.append("docs/31_v2_rc_notes.md missing strict compatibility language")

    migration_guide = ROOT / "docs/32_v2_migration_guide.md"
    if not migration_guide.is_file():
        failures.append("missing docs/32_v2_migration_guide.md")
    else:
        migration_text = migration_guide.read_text(encoding="utf-8")
        for token in (
            "enkai migrate config-v1",
            "enkai migrate checkpoint-meta-v1",
            "enkai doctor",
        ):
            if token not in migration_text:
                failures.append(
                    f"docs/32_v2_migration_guide.md missing required command reference: {token}"
                )

    if failures:
        print("docs consistency check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(f"docs consistency check passed for {expected_tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
