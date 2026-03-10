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


def build_checks(version: str, copied_paths: list[str], require_gpu: bool) -> list[CheckResult]:
    checks: list[CheckResult] = []

    archive = first_match(
        copied_paths,
        lambda path: "/dist/enkai-" in path and (path.endswith(".zip") or path.endswith(".tar.gz")),
    )
    checks.append(
        CheckResult(
            id="release_archive",
            description="Release archive is present",
            required=True,
            passed=archive is not None,
            details=archive or "missing dist/enkai-<version>-<os>-<arch>.zip|tar.gz",
        )
    )

    checksum = first_match(
        copied_paths, lambda path: "/dist/enkai-" in path and path.endswith(".sha256")
    )
    checks.append(
        CheckResult(
            id="release_checksum",
            description="Release checksum is present",
            required=True,
            passed=checksum is not None,
            details=checksum or "missing dist/enkai-<version>-<os>-<arch>.<ext>.sha256",
        )
    )

    sbom = first_match(copied_paths, lambda path: "/dist/sbom-" in path and path.endswith(".json"))
    checks.append(
        CheckResult(
            id="sbom",
            description="SBOM artifact is present",
            required=True,
            passed=sbom is not None,
            details=sbom or "missing dist/sbom-<version>-<os>-<arch>.json",
        )
    )

    bench_regex = re.compile(r"/dist/benchmark_official_[^/]+\.json$")
    bench = first_match(copied_paths, lambda path: bool(bench_regex.search(path)))
    checks.append(
        CheckResult(
            id="benchmark_target",
            description="Official benchmark target evidence is present",
            required=True,
            passed=bench is not None,
            details=bench or "missing dist/benchmark_official_<suite>_<platform>.json",
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
            required=True,
            passed=readiness_present,
            details=(
                "readiness/production.json"
                if readiness_present
                else "missing readiness/production.json"
            ),
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

    checks = build_checks(version, copied_paths, args.require_gpu)
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
