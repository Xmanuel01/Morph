#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import re
import shutil
import sys


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    parser = argparse.ArgumentParser(description="Collect and archive release evidence bundle.")
    parser.add_argument("--version", help="Release version (defaults to enkai/Cargo.toml)")
    parser.add_argument("--gpu-log-dir", default="artifacts/gpu", help="GPU evidence source directory")
    parser.add_argument("--dist-dir", default="dist", help="Built artifacts source directory")
    parser.add_argument(
        "--selfhost-dir",
        default="artifacts/selfhost",
        help="Self-host triage source directory",
    )
    parser.add_argument(
        "--contracts-dir",
        default="enkai/contracts",
        help="Contract snapshot source directory",
    )
    parser.add_argument(
        "--readiness-dir",
        default="artifacts/readiness",
        help="Readiness report source directory",
    )
    parser.add_argument(
        "--sim-dir",
        default="artifacts/sim",
        help="Simulation evidence source directory",
    )
    parser.add_argument(
        "--registry-dir",
        default="artifacts/registry",
        help="Registry convergence evidence source directory",
    )
    parser.add_argument(
        "--grpc-dir",
        default="artifacts/grpc",
        help="gRPC evidence source directory",
    )
    parser.add_argument(
        "--validation-dir",
        default="artifacts/validation",
        help="Validation evidence source directory",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/release",
        help="Output directory for evidence bundle",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail if required GPU evidence files are missing",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if required non-GPU release evidence is missing",
    )
    return parser.parse_args()


def ensure_required_gpu_files(gpu_dir: pathlib.Path) -> list[pathlib.Path]:
    required = [
        "single_gpu.log",
        "single_gpu_evidence.json",
        "multi_gpu.log",
        "multi_gpu_evidence.json",
        "soak_4gpu.log",
        "soak_4gpu_evidence.json",
    ]
    missing = [name for name in required if not (gpu_dir / name).is_file()]
    if missing:
        raise RuntimeError(
            "missing required GPU evidence files: " + ", ".join(sorted(missing))
        )
    return [gpu_dir / name for name in required]


def copy_files(
    root: pathlib.Path,
    source_base: pathlib.Path,
    destination_base: pathlib.Path,
    category: str,
    files: list[pathlib.Path],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source in sorted(files):
        if not source.is_file():
            continue
        rel = source.relative_to(source_base)
        target = destination_base / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        rows.append(
            {
                "category": category,
                "source": str(source.relative_to(root)),
                "copied_to": str(target.relative_to(root)),
                "sha256": sha256_file(source),
                "bytes": source.stat().st_size,
            }
        )
    return rows


def ensure_required_files(directory: pathlib.Path, required: list[str], label: str) -> list[pathlib.Path]:
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise RuntimeError(f"missing required {label} files: {', '.join(sorted(missing))}")
    return [directory / name for name in required]


def include_dist_file(path: pathlib.Path, version: str) -> bool:
    name = path.name
    if name.startswith("enkai-"):
        return name.startswith(f"enkai-{version}-")
    if name.startswith("sbom-"):
        return name.startswith(f"sbom-{version}-")
    if name.startswith("benchmark_official_") and name.endswith(".json"):
        return True
    return False


def validate_dist_artifacts_for_version(dist_files: list[pathlib.Path], version: str) -> None:
    names = [file.name for file in dist_files if file.is_file()]
    has_archive = any(
        name.startswith(f"enkai-{version}-")
        and (name.endswith(".zip") or name.endswith(".tar.gz"))
        for name in names
    )
    has_checksum = any(
        name.startswith(f"enkai-{version}-") and name.endswith(".sha256") for name in names
    )
    has_sbom = any(name.startswith(f"sbom-{version}-") and name.endswith(".json") for name in names)
    has_bench = any(re.match(r"benchmark_official_.*\.json$", name) for name in names)

    missing: list[str] = []
    if not has_archive:
        missing.append(
            f"release archive (enkai-{version}-<os>-<arch>.zip|tar.gz)"
        )
    if not has_checksum:
        missing.append(f"release checksum for enkai-{version}-* (.sha256)")
    if not has_sbom:
        missing.append(f"SBOM (sbom-{version}-<os>-<arch>.json)")
    if not has_bench:
        missing.append("benchmark evidence (benchmark_official_*.json)")
    if missing:
        raise RuntimeError("missing required dist evidence: " + ", ".join(missing))


def main() -> int:
    args = parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    version = args.version or parse_version(root)

    gpu_dir = (root / args.gpu_log_dir).resolve()
    dist_dir = (root / args.dist_dir).resolve()
    selfhost_dir = (root / args.selfhost_dir).resolve()
    contracts_dir = (root / args.contracts_dir).resolve()
    readiness_dir = (root / args.readiness_dir).resolve()
    sim_dir = (root / args.sim_dir).resolve()
    registry_dir = (root / args.registry_dir).resolve()
    grpc_dir = (root / args.grpc_dir).resolve()
    validation_dir = (root / args.validation_dir).resolve()
    out_root = (root / args.out_dir / f"v{version}").resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "schema": "enkai-release-evidence-v2",
        "version": f"v{version}",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "gpu_required": args.require_gpu,
        "strict": args.strict,
        "files": [],
    }

    file_rows: list[dict[str, object]] = []
    if gpu_dir.is_dir():
        if args.require_gpu:
            gpu_files = ensure_required_gpu_files(gpu_dir)
        else:
            gpu_files = sorted(path for path in gpu_dir.glob("*") if path.is_file())
        file_rows.extend(copy_files(root, gpu_dir, out_root / "gpu", "gpu", gpu_files))
    elif args.require_gpu:
        raise RuntimeError(f"GPU evidence directory not found: {gpu_dir}")

    dist_files: list[pathlib.Path] = []
    if dist_dir.is_dir():
        dist_files = sorted(
            path
            for path in dist_dir.glob("*")
            if path.is_file() and include_dist_file(path, version)
        )
        if args.strict:
            validate_dist_artifacts_for_version(dist_files, version)
        file_rows.extend(copy_files(root, dist_dir, out_root / "dist", "dist", dist_files))
    elif args.strict:
        raise RuntimeError(f"dist directory not found for strict evidence mode: {dist_dir}")

    required_selfhost = [
        "litec_selfhost_ci_report.json",
        "litec_replace_check_report.json",
        "litec_mainline_ci_report.json",
        "litec_release_ci_report.json",
    ]
    if selfhost_dir.is_dir():
        if args.strict:
            selfhost_files = ensure_required_files(selfhost_dir, required_selfhost, "self-host triage")
        else:
            selfhost_files = sorted(path for path in selfhost_dir.glob("*") if path.is_file())
        file_rows.extend(
            copy_files(root, selfhost_dir, out_root / "selfhost", "selfhost", selfhost_files)
        )
    elif args.strict:
        raise RuntimeError(f"self-host triage directory not found for strict evidence mode: {selfhost_dir}")

    required_contracts = [
        "backend_api_v1.snapshot.json",
        "sdk_api_v1.snapshot.json",
        "grpc_api_v1.snapshot.json",
        "worker_queue_v1.snapshot.json",
        "db_engines_v1.snapshot.json",
        "conversation_state_v1.schema.json",
    ]
    if contracts_dir.is_dir():
        if args.strict:
            contract_files = ensure_required_files(contracts_dir, required_contracts, "contract snapshot")
        else:
            contract_files = sorted(path for path in contracts_dir.glob("*.json") if path.is_file())
        file_rows.extend(
            copy_files(root, contracts_dir, out_root / "contracts", "contracts", contract_files)
        )
    elif args.strict:
        raise RuntimeError(
            f"contract snapshot directory not found for strict evidence mode: {contracts_dir}"
        )

    required_readiness = [
        "full_platform.json",
        "full_platform_blockers.json",
        "grpc_smoke.json",
        "grpc_evidence_verify.json",
        "sim_smoke.json",
        "sim_evidence_verify.json",
        "sim_native_smoke.json",
        "sim_native_evidence_verify.json",
        "sim_stdlib_smoke.json",
        "sim_stdlib_evidence_verify.json",
        "adam0_100_smoke.json",
        "adam0_100_evidence_verify.json",
        "adam0_reference_suite.json",
        "adam0_reference_suite_verify.json",
        "snn_agent_kernel_smoke.json",
        "snn_agent_kernel_evidence_verify.json",
        "model_registry_convergence.json",
        "model_registry_convergence_verify.json",
        "cluster_scale_smoke.json",
        "cluster_scale_evidence_verify.json",
        "registry_degraded_smoke.json",
        "registry_degraded_evidence_verify.json",
        "deploy_mobile.json",
        "deploy_mobile_smoke.json",
        "deploy_mobile_evidence_verify.json",
        "worker_queue_smoke.json",
        "worker_queue_evidence_verify.json",
    ]
    if readiness_dir.is_dir():
        if args.strict:
            readiness_files = ensure_required_files(
                readiness_dir, required_readiness, "readiness report"
            )
        else:
            readiness_files = sorted(path for path in readiness_dir.glob("*.json") if path.is_file())
        file_rows.extend(
            copy_files(root, readiness_dir, out_root / "readiness", "readiness", readiness_files)
        )
    elif args.strict:
        raise RuntimeError(
            f"readiness directory not found for strict evidence mode: {readiness_dir}"
        )

    required_validation = [
        "ffi_correctness.json",
        "determinism_event_queue.json",
        "determinism_sim_replay.json",
        "determinism_sim_coroutines.json",
        "determinism_adam0_reference_100.json",
        "pool_safety.json",
        "adam0_fake10.json",
        "adam0_ref100.json",
        "adam0_stress1000.json",
        "adam0_target10000.json",
        "perf_ffi_noop.json",
        "perf_sparse_dot.json",
        "perf_adam0_reference_100.json",
        "perf_adam0_reference_1000.json",
        "perf_adam0_reference_10000.json",
    ]
    if validation_dir.is_dir():
        if args.strict:
            validation_files = ensure_required_files(
                validation_dir, required_validation, "validation evidence"
            )
        else:
            validation_files = sorted(
                path for path in validation_dir.glob("*.json") if path.is_file()
            )
        file_rows.extend(
            copy_files(
                root,
                validation_dir,
                out_root / "validation",
                "validation",
                validation_files,
            )
        )
    elif args.strict:
        raise RuntimeError(
            f"validation directory not found for strict evidence mode: {validation_dir}"
        )

    required_sim = [
        "smoke_run.json",
        "smoke_profile.json",
        "smoke_replay.json",
        "native_smoke_run.json",
        "native_smoke_profile.json",
        "stdlib_smoke_run.json",
        "stdlib_smoke_profile.json",
        "adam0_100_run.json",
        "adam0_100_profile.json",
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
        "snn_agent_kernel_run.json",
        "snn_agent_kernel_profile.json",
    ]
    if sim_dir.is_dir():
        if args.strict:
            sim_files = ensure_required_files(sim_dir, required_sim, "simulation evidence")
        else:
            sim_files = sorted(path for path in sim_dir.glob("*.json") if path.is_file())
        file_rows.extend(copy_files(root, sim_dir, out_root / "sim", "sim", sim_files))
    elif args.strict:
        raise RuntimeError(
            f"simulation evidence directory not found for strict evidence mode: {sim_dir}"
        )

    required_grpc = [
        "probe.json",
        "server.jsonl",
        "conversation_state.json",
        "conversation_state.backup.json",
    ]
    if grpc_dir.is_dir():
        if args.strict:
            grpc_files = ensure_required_files(grpc_dir, required_grpc, "grpc evidence")
        else:
            grpc_files = sorted(path for path in grpc_dir.rglob("*") if path.is_file())
        file_rows.extend(copy_files(root, grpc_dir, out_root / "grpc", "grpc", grpc_files))
    elif args.strict:
        raise RuntimeError(
            f"grpc evidence directory not found for strict evidence mode: {grpc_dir}"
        )

    required_registry = [
        "sim_lineage.json",
        "sim_snapshot.manifest.json",
        "local/registry.json",
        "remote/registry.json",
        "cache/registry.json",
        f"remote/adam0-sim/v{version}/remote.manifest.json",
        f"remote/adam0-sim/v{version}/remote.manifest.sig",
    ]
    if registry_dir.is_dir():
        if args.strict:
            registry_files = ensure_required_files(
                registry_dir, required_registry, "registry convergence evidence"
            )
        else:
            registry_files = sorted(path for path in registry_dir.rglob("*") if path.is_file())
        file_rows.extend(copy_files(root, registry_dir, out_root / "registry", "registry", registry_files))
    elif args.strict:
        raise RuntimeError(
            f"registry convergence evidence directory not found for strict evidence mode: {registry_dir}"
        )

    cluster_scale_dir = root / "artifacts" / "cluster_scale"
    required_cluster_scale = [
        "validate.json",
        "plan.json",
        "run.json",
        "recovery/rank0/window_0000.run.json",
        "recovery/rank0/window_0000.snapshot.json",
        "recovery/rank1/window_0000.run.json",
        "recovery/rank1/window_0000.snapshot.json",
    ]
    if cluster_scale_dir.is_dir():
        if args.strict:
            cluster_scale_files = ensure_required_files(
                cluster_scale_dir, required_cluster_scale, "cluster scale evidence"
            )
        else:
            cluster_scale_files = sorted(path for path in cluster_scale_dir.rglob("*") if path.is_file())
        file_rows.extend(
            copy_files(root, cluster_scale_dir, out_root / "cluster_scale", "cluster_scale", cluster_scale_files)
        )
    elif args.strict:
        raise RuntimeError(
            f"cluster scale evidence directory not found for strict evidence mode: {cluster_scale_dir}"
        )

    registry_degraded_dir = root / "artifacts" / "registry_degraded"
    required_registry_degraded = [
        "cache/registry.json",
        "cache/audit.log.jsonl",
        f"remote_offline/adam0-degraded/v{version}/remote.manifest.json",
        f"remote_offline/adam0-degraded/v{version}/remote.manifest.sig",
    ]
    if registry_degraded_dir.is_dir():
        if args.strict:
            registry_degraded_files = ensure_required_files(
                registry_degraded_dir, required_registry_degraded, "registry degraded evidence"
            )
        else:
            registry_degraded_files = sorted(
                path for path in registry_degraded_dir.rglob("*") if path.is_file()
            )
        file_rows.extend(
            copy_files(
                root,
                registry_degraded_dir,
                out_root / "registry_degraded",
                "registry_degraded",
                registry_degraded_files,
            )
        )
    elif args.strict:
        raise RuntimeError(
            f"registry degraded evidence directory not found for strict evidence mode: {registry_degraded_dir}"
        )

    mobile_dir = root / "artifacts" / "mobile"
    required_mobile = [
        "sdk_api.snapshot.json",
        "app.json",
        "package.json",
    ]
    if mobile_dir.is_dir():
        if args.strict:
            mobile_files = ensure_required_files(
                mobile_dir, required_mobile, "mobile scaffold evidence"
            )
        else:
            mobile_files = sorted(path for path in mobile_dir.rglob("*") if path.is_file())
        file_rows.extend(
            copy_files(root, mobile_dir, out_root / "mobile", "mobile", mobile_files)
        )
    elif args.strict:
        raise RuntimeError(
            f"mobile scaffold evidence directory not found for strict evidence mode: {mobile_dir}"
        )

    worker_queue_dir = root / "artifacts" / "worker_queue"
    required_worker_queue = [
        "run_01.json",
        "run_02.json",
        "run_03.json",
        "queues/default/dead_letter.jsonl",
        "queues/default/pending.jsonl",
    ]
    if worker_queue_dir.is_dir():
        if args.strict:
            worker_queue_files = ensure_required_files(
                worker_queue_dir, required_worker_queue, "worker queue evidence"
            )
        else:
            worker_queue_files = sorted(
                path for path in worker_queue_dir.rglob("*") if path.is_file()
            )
        file_rows.extend(
            copy_files(
                root,
                worker_queue_dir,
                out_root / "worker_queue",
                "worker_queue",
                worker_queue_files,
            )
        )
    elif args.strict:
        raise RuntimeError(
            f"worker queue evidence directory not found for strict evidence mode: {worker_queue_dir}"
        )

    manifest["files"] = file_rows
    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    payload = {
        "status": "ok",
        "version": f"v{version}",
        "out_dir": str(out_root.relative_to(root)),
        "files": len(file_rows),
        "manifest": str(manifest_path.relative_to(root)),
        "gpu_required": args.require_gpu,
    }
    print(json.dumps(payload, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as err:  # pragma: no cover - script entrypoint
        print(f"error: {err}", file=sys.stderr)
        raise SystemExit(1)
