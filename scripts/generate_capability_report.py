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
