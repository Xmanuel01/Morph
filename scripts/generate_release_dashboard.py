#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys


def parse_version(root: pathlib.Path) -> str:
    cargo = (root / "enkai" / "Cargo.toml").read_text(encoding="utf-8")
    for line in cargo.splitlines():
        line = line.strip()
        if line.startswith("version"):
            _, value = line.split("=", 1)
            return value.strip().strip('"')
    raise RuntimeError("failed to parse version from enkai/Cargo.toml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a release dashboard from archived proof artifacts."
    )
    parser.add_argument("--version", help="Release version (defaults to enkai/Cargo.toml)")
    parser.add_argument(
        "--capability-report",
        help="Path to capability_complete.json (defaults to artifacts/release/v<version>/capability_complete.json)",
    )
    parser.add_argument(
        "--blocker-report",
        help="Path to archived full_platform blocker report (defaults to artifacts/release/v<version>/readiness/full_platform_blockers.json)",
    )
    parser.add_argument(
        "--strict-selfhost-inventory",
        help="Path to archived strict self-host dependency inventory (defaults to artifacts/release/v<version>/readiness/strict_selfhost_dependency_inventory.json)",
    )
    parser.add_argument("--output-json", help="Output dashboard JSON path")
    parser.add_argument("--output-md", help="Output dashboard markdown path")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when CPU-complete release proof is not satisfied",
    )
    return parser.parse_args()


def read_json(path: pathlib.Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def to_markdown(dashboard: dict[str, object]) -> str:
    summary = dashboard["summary"]
    hardware = dashboard["hardware_envelope"]
    rows = [
        "| Group | Status | Details |",
        "| --- | --- | --- |",
    ]
    for group in dashboard["proof_groups"]:
        rows.append(
            f"| `{group['id']}` | {'PASS' if group['passed'] else 'FAIL'} | {group['details']} |"
        )

    lines = [
        f"# Release Dashboard ({dashboard['version']})",
        "",
        f"- Generated: `{dashboard['generated_at_utc']}`",
        f"- CPU complete: `{summary['cpu_complete']}`",
        f"- Strict self-host CPU complete: `{summary['strict_selfhost_cpu_complete']}`",
        f"- Strict self-host GPU pending: `{summary['strict_selfhost_gpu_pending']}`",
        f"- GPU ready for operator sign-off: `{summary['gpu_ready_for_operator_signoff']}`",
        f"- Final release sign-off complete: `{summary['final_signoff_complete']}`",
        "",
        "## Hardware Envelope",
        "",
        f"- Local CPU profiles: `{', '.join(hardware['local_cpu_profiles'])}`",
        f"- Reference CPU profiles: `{', '.join(hardware['reference_cpu_profiles'])}`",
        f"- GPU evidence required for final sign-off: `{', '.join(hardware['gpu_required_artifacts'])}`",
        "",
        "## Remaining Rust Dependencies",
        "",
    ]
    remaining = dashboard["strict_selfhost"]["remaining_rust_dependencies"]
    if remaining:
        for item in remaining:
            lines.append(f"- `{item}`")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Proof Groups",
            "",
            *rows,
            "",
            "## Unverified Areas",
            "",
        ]
    )
    if dashboard["unverified_areas"]:
        for item in dashboard["unverified_areas"]:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Required Operator GPU Steps",
            "",
        ]
    )
    for item in dashboard["required_operator_gpu_steps"]:
        lines.append(f"- `{item}`")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    version = args.version or parse_version(root)

    capability_path = (
        pathlib.Path(args.capability_report).resolve()
        if args.capability_report
        else root / "artifacts" / "release" / f"v{version}" / "capability_complete.json"
    )
    blocker_path = (
        pathlib.Path(args.blocker_report).resolve()
        if args.blocker_report
        else root
        / "artifacts"
        / "release"
        / f"v{version}"
        / "readiness"
        / "full_platform_blockers.json"
    )
    strict_selfhost_inventory_path = (
        pathlib.Path(args.strict_selfhost_inventory).resolve()
        if args.strict_selfhost_inventory
        else root
        / "artifacts"
        / "release"
        / f"v{version}"
        / "readiness"
        / "strict_selfhost_dependency_inventory.json"
    )
    if not capability_path.is_file():
        raise RuntimeError(f"capability report not found: {capability_path}")
    if not blocker_path.is_file():
        raise RuntimeError(f"blocker report not found: {blocker_path}")
    if not strict_selfhost_inventory_path.is_file():
        raise RuntimeError(
            f"strict self-host inventory not found: {strict_selfhost_inventory_path}"
        )

    out_json = (
        pathlib.Path(args.output_json).resolve()
        if args.output_json
        else capability_path.parent / "release_dashboard.json"
    )
    out_md = (
        pathlib.Path(args.output_md).resolve()
        if args.output_md
        else capability_path.parent / "release_dashboard.md"
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    capability = read_json(capability_path)
    blocker = read_json(blocker_path)
    strict_selfhost_inventory = read_json(strict_selfhost_inventory_path)
    check_map = {check["id"]: check for check in capability.get("checks", []) if isinstance(check, dict)}

    proof_group_ids = [
        "proof_cpu_correctness_suite",
        "proof_determinism_suite",
        "proof_runtime_safety_suite",
        "proof_adam0_cpu_suite",
        "claim_non_hardware_release_complete",
    ]
    proof_groups = []
    unverified_areas: list[str] = []
    for group_id in proof_group_ids:
        check = check_map.get(group_id)
        if check is None:
            proof_groups.append({"id": group_id, "passed": False, "details": "missing capability check"})
            unverified_areas.append(f"missing capability proof group `{group_id}`")
            continue
        passed = bool(check.get("passed", False))
        details = str(check.get("details", ""))
        proof_groups.append({"id": group_id, "passed": passed, "details": details})
        if not passed:
            unverified_areas.append(f"{group_id}: {details}")

    gpu_required_artifacts = [
        "artifacts/gpu/single_gpu_evidence.json",
        "artifacts/gpu/multi_gpu_evidence.json",
        "artifacts/gpu/soak_4gpu_evidence.json",
    ]
    gpu_capability_ids = {
        "artifacts/gpu/single_gpu_evidence.json": "gpu_single_gpu_evidence_json",
        "artifacts/gpu/multi_gpu_evidence.json": "gpu_multi_gpu_evidence_json",
        "artifacts/gpu/soak_4gpu_evidence.json": "gpu_soak_4gpu_evidence_json",
    }
    missing_gpu = []
    blocker_missing_gpu = blocker.get("missing_gpu_artifacts", [])
    if isinstance(blocker_missing_gpu, list):
        missing_gpu.extend(str(item) for item in blocker_missing_gpu)
    for artifact in gpu_required_artifacts:
        check = check_map.get(gpu_capability_ids[artifact])
        if check is not None and not bool(check.get("passed", False)) and artifact not in missing_gpu:
            missing_gpu.append(artifact)
    gpu_ready = len(missing_gpu) == 0
    if not gpu_ready:
        unverified_areas.append(
            "GPU/operator evidence is still missing for final hardware sign-off: "
            + ", ".join(missing_gpu)
        )

    cpu_complete = all(group["passed"] for group in proof_groups)
    strict_selfhost_summary = strict_selfhost_inventory.get("summary", {})
    strict_selfhost_cpu_complete = bool(
        isinstance(strict_selfhost_summary, dict)
        and strict_selfhost_summary.get("strict_selfhost_cpu_complete", False)
    )
    remaining_rust_dependencies = (
        strict_selfhost_summary.get("remaining_rust_dependencies", [])
        if isinstance(strict_selfhost_summary, dict)
        else []
    )
    if not isinstance(remaining_rust_dependencies, list):
        remaining_rust_dependencies = []
    strict_selfhost_gpu_pending = strict_selfhost_cpu_complete and not gpu_ready
    final_signoff_complete = cpu_complete and gpu_ready

    dashboard = {
        "schema": "enkai-release-dashboard-v1",
        "version": f"v{version}",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "capability_report": str(capability_path.relative_to(root)),
        "blocker_report": str(blocker_path.relative_to(root)),
        "summary": {
            "cpu_complete": cpu_complete,
            "strict_selfhost_cpu_complete": strict_selfhost_cpu_complete,
            "strict_selfhost_gpu_pending": strict_selfhost_gpu_pending,
            "gpu_ready_for_operator_signoff": gpu_ready,
            "final_signoff_complete": final_signoff_complete,
        },
        "strict_selfhost": {
            "inventory": str(strict_selfhost_inventory_path.relative_to(root)),
            "remaining_rust_dependencies": remaining_rust_dependencies,
        },
        "hardware_envelope": {
            "local_cpu_profiles": [
                "bench/machines/windows_local.json",
                "bench/machines/linux_local.json",
            ],
            "reference_cpu_profiles": [
                "bench/machines/windows_ref.json",
                "bench/machines/linux_ref.json",
            ],
            "gpu_required_artifacts": gpu_required_artifacts,
        },
        "proof_groups": proof_groups,
        "unverified_areas": unverified_areas,
        "required_operator_gpu_steps": [
            "powershell -ExecutionPolicy Bypass -File scripts/gpu_preflight.ps1 -Profile full -Output artifacts/gpu/preflight.json",
            "powershell -ExecutionPolicy Bypass -File scripts/soak_single_gpu.ps1",
            "powershell -ExecutionPolicy Bypass -File scripts/multi_gpu_harness.ps1",
            "powershell -ExecutionPolicy Bypass -File scripts/soak_4gpu.ps1",
            "powershell -ExecutionPolicy Bypass -File scripts/verify_gpu_gates.ps1 -LogDir artifacts/gpu",
            "powershell -ExecutionPolicy Bypass -File scripts/v3_0_0_rc_pipeline.ps1",
        ],
    }

    out_json.write_text(json.dumps(dashboard, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text(to_markdown(dashboard) + "\n", encoding="utf-8")

    payload = {
        "status": "ok" if (not args.strict or cpu_complete) else "failed",
        "version": f"v{version}",
        "output_json": str(out_json.relative_to(root)),
        "output_md": str(out_md.relative_to(root)),
        "cpu_complete": cpu_complete,
        "gpu_ready_for_operator_signoff": gpu_ready,
        "final_signoff_complete": final_signoff_complete,
    }
    print(json.dumps(payload, separators=(",", ":")))

    if args.strict and not cpu_complete:
        raise RuntimeError("release dashboard strict checks failed: CPU proof is incomplete")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as err:  # pragma: no cover - script entrypoint
        print(f"error: {err}", file=sys.stderr)
        raise SystemExit(1)
