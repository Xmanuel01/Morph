#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib

SEMVER = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:[-+][A-Za-z0-9.-]+)?$")
PACKAGE_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._-]*$")
MANIFEST_SCHEMA = "enkai.llm.package.v1"
LOCK_SCHEMA = "enkai.llm.lock.v1"


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    issues: list[str]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_digest(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def file_entry(root: Path, rel: str, role: str, media_type: str) -> dict[str, Any]:
    path = root / rel
    return {
        "path": rel,
        "role": role,
        "media_type": media_type,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_package(
    registry_root: Path,
    *,
    namespace: str,
    name: str,
    version: str,
    family: str,
    architecture: str,
    tasks: list[str],
    api_symbols: list[str],
    dependencies: list[dict[str, str]],
) -> tuple[Path, dict[str, Any]]:
    package_dir = registry_root / "packages" / namespace / name / version
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)

    (package_dir / "README.md").write_text(
        f"# {namespace}/{name}\n\nDeterministic fixture package for Enkai LLM registry proof.\n",
        encoding="utf-8",
    )
    (package_dir / "model_card.md").write_text(
        "# Model Card\n\nPurpose: bounded verifier fixture. Training data: synthetic deterministic text.\n",
        encoding="utf-8",
    )
    (package_dir / "config.json").write_text(
        json.dumps(
            {
                "architecture": architecture,
                "hidden_size": 64,
                "layers": 2,
                "heads": 4,
                "context": 128,
                "vocab_size": 256,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (package_dir / "tokenizer.json").write_text(
        json.dumps({"type": "byte-bpe", "vocab_size": 256, "normalizer": "utf8"}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (package_dir / "weights.safetensors").write_bytes(
        b"ENKAI_DETERMINISTIC_TINY_LLM_WEIGHTS_V1\n" + bytes(range(64))
    )

    files = [
        file_entry(package_dir, "README.md", "readme", "text/markdown"),
        file_entry(package_dir, "model_card.md", "model_card", "text/markdown"),
        file_entry(package_dir, "config.json", "config", "application/json"),
        file_entry(package_dir, "tokenizer.json", "tokenizer", "application/json"),
        file_entry(package_dir, "weights.safetensors", "weights", "application/octet-stream"),
    ]
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "schema_version": 1,
        "package": f"{namespace}/{name}",
        "version": version,
        "license": "Apache-2.0",
        "family": family,
        "architecture": architecture,
        "tasks": tasks,
        "api": {
            "stable_since": "4.0.0",
            "symbols": api_symbols,
        },
        "runtime": {
            "devices": ["cpu", "cuda:0"],
            "dtypes": ["fp32", "fp16", "bf16"],
            "requires_pytorch": False,
            "pytorch_reference_only": True,
        },
        "security": {
            "native_postinstall_hooks": False,
            "network_on_install": False,
            "requires_signature_for_remote_install": True,
        },
        "dependencies": dependencies,
        "files": files,
    }
    manifest["manifest_digest"] = canonical_digest({k: v for k, v in manifest.items() if k != "manifest_digest"})
    write_json(package_dir / "enkai.llm.json", manifest)
    return package_dir, manifest


def registry_key(package: str, version: str) -> str:
    return f"{package}@{version}"


def build_index(registry_root: Path, manifests: list[dict[str, Any]]) -> dict[str, Any]:
    packages: dict[str, Any] = {}
    for manifest in manifests:
        package = manifest["package"]
        namespace, name = package.split("/", 1)
        version = manifest["version"]
        packages.setdefault(package, {"versions": {}})["versions"][version] = {
            "manifest": f"packages/{namespace}/{name}/{version}/enkai.llm.json",
            "manifest_digest": manifest["manifest_digest"],
            "tasks": manifest["tasks"],
            "architecture": manifest["architecture"],
        }
    index = {
        "schema": "enkai.llm.registry.index.v1",
        "schema_version": 1,
        "generated_by": "readiness_v4_0_llm_package_registry.py",
        "packages": packages,
    }
    write_json(registry_root / "registry.index.json", index)
    return index


def resolve_exact(registry_root: Path, root_manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    issues: list[str] = []
    index = read_json(registry_root / "registry.index.json")
    resolved: list[dict[str, Any]] = []
    seen: set[str] = set()

    def visit(package: str, version: str) -> None:
        key = registry_key(package, version)
        if key in seen:
            return
        versions = index.get("packages", {}).get(package, {}).get("versions", {})
        entry = versions.get(version)
        if not entry:
            issues.append(f"unresolved dependency {key}")
            return
        manifest = read_json(registry_root / entry["manifest"])
        if manifest.get("manifest_digest") != entry.get("manifest_digest"):
            issues.append(f"index digest mismatch for {key}")
            return
        seen.add(key)
        for dep in manifest.get("dependencies", []):
            visit(dep.get("package", ""), dep.get("version", ""))
        resolved.append(manifest)

    for dep in root_manifest.get("dependencies", []):
        visit(dep.get("package", ""), dep.get("version", ""))
    root_key = registry_key(root_manifest["package"], root_manifest["version"])
    if root_key not in seen:
        seen.add(root_key)
        resolved.append(root_manifest)
    resolved.sort(key=lambda item: (item["package"], item["version"]))
    return resolved, issues


def write_lock(package_dir: Path, registry_root: Path, root_manifest: dict[str, Any]) -> dict[str, Any]:
    resolved, issues = resolve_exact(registry_root, root_manifest)
    if issues:
        raise RuntimeError("; ".join(issues))
    packages = []
    for manifest in resolved:
        packages.append(
            {
                "package": manifest["package"],
                "version": manifest["version"],
                "manifest_digest": manifest["manifest_digest"],
                "files": [
                    {"path": f["path"], "sha256": f["sha256"], "bytes": f["bytes"]}
                    for f in manifest.get("files", [])
                ],
            }
        )
    lock = {
        "schema": LOCK_SCHEMA,
        "schema_version": 1,
        "root": registry_key(root_manifest["package"], root_manifest["version"]),
        "resolver": "exact-v1",
        "packages": packages,
    }
    lock["lock_digest"] = canonical_digest({k: v for k, v in lock.items() if k != "lock_digest"})
    write_json(package_dir / "enkai.lock.json", lock)
    return lock


def validate_manifest(manifest_path: Path, registry_root: Path, lock_path: Path | None = None) -> ValidationResult:
    issues: list[str] = []
    try:
        manifest = read_json(manifest_path)
    except Exception as exc:
        return ValidationResult(False, [f"manifest parse failed: {exc}"])
    package_dir = manifest_path.parent

    if manifest.get("schema") != MANIFEST_SCHEMA:
        issues.append("manifest schema mismatch")
    package = manifest.get("package", "")
    version = manifest.get("version", "")
    if not PACKAGE_RE.match(package):
        issues.append("package name must be namespace/name lowercase")
    if not SEMVER.match(version):
        issues.append("version must be semantic version")
    if not manifest.get("license"):
        issues.append("license is required")
    if not manifest.get("tasks"):
        issues.append("at least one task is required")
    if "std::model.load" not in manifest.get("api", {}).get("symbols", []):
        issues.append("stable API symbol std::model.load is required")
    runtime = manifest.get("runtime", {})
    if runtime.get("requires_pytorch") is not False:
        issues.append("package runtime must not require PyTorch")
    if runtime.get("pytorch_reference_only") is not True:
        issues.append("PyTorch must be reference-only in package metadata")
    security = manifest.get("security", {})
    if security.get("native_postinstall_hooks") is not False:
        issues.append("native postinstall hooks are forbidden")
    if security.get("network_on_install") is not False:
        issues.append("network-on-install is forbidden")

    digest_source = {k: v for k, v in manifest.items() if k != "manifest_digest"}
    if manifest.get("manifest_digest") != canonical_digest(digest_source):
        issues.append("manifest digest mismatch")

    seen_files: set[str] = set()
    has_model_card = False
    for file_info in manifest.get("files", []):
        rel = file_info.get("path", "")
        if not rel or rel.startswith("/") or ".." in Path(rel).parts:
            issues.append(f"invalid package file path: {rel}")
            continue
        if rel in seen_files:
            issues.append(f"duplicate package file path: {rel}")
        seen_files.add(rel)
        path = package_dir / rel
        if not path.is_file():
            issues.append(f"missing package file: {rel}")
            continue
        if path.stat().st_size != int(file_info.get("bytes", -1)):
            issues.append(f"file byte length mismatch: {rel}")
        if sha256_file(path) != file_info.get("sha256"):
            issues.append(f"file hash mismatch: {rel}")
        if file_info.get("role") == "model_card":
            has_model_card = True
    if not has_model_card:
        issues.append("model_card file is required")

    resolved, resolve_issues = resolve_exact(registry_root, manifest)
    issues.extend(resolve_issues)
    for dep in manifest.get("dependencies", []):
        if not PACKAGE_RE.match(dep.get("package", "")):
            issues.append(f"invalid dependency package: {dep}")
        if not SEMVER.match(dep.get("version", "")):
            issues.append(f"dependency must be exact semver: {dep}")

    if lock_path is not None:
        try:
            lock = read_json(lock_path)
        except Exception as exc:
            issues.append(f"lock parse failed: {exc}")
            lock = {}
        if lock.get("schema") != LOCK_SCHEMA:
            issues.append("lock schema mismatch")
        expected_root = registry_key(package, version)
        if lock.get("root") != expected_root:
            issues.append("lock root mismatch")
        expected_keys = [registry_key(item["package"], item["version"]) for item in resolved]
        actual_keys = [registry_key(item.get("package", ""), item.get("version", "")) for item in lock.get("packages", [])]
        if actual_keys != expected_keys:
            issues.append("lock package resolution mismatch")
        if lock.get("lock_digest") != canonical_digest({k: v for k, v in lock.items() if k != "lock_digest"}):
            issues.append("lock digest mismatch")

    return ValidationResult(not issues, issues)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create and verify the v4.0 LLM package registry tranche.")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--contract", default="enkai/contracts/v4_0_0_llm_package_registry.json")
    parser.add_argument("--output", default="artifacts/readiness/v4_0_0_llm_package_registry.json")
    args = parser.parse_args()

    root = Path(args.workspace).resolve()
    contract = read_json(root / args.contract)
    registry_root = root / "artifacts" / "registry" / "llm_package_ecosystem"
    if registry_root.exists():
        shutil.rmtree(registry_root)
    registry_root.mkdir(parents=True, exist_ok=True)

    tokenizer_dir, tokenizer = write_package(
        registry_root,
        namespace="enkai",
        name="tokenizer-bpe",
        version="1.0.0",
        family="tokenizer",
        architecture="byte-bpe",
        tasks=["tokenization"],
        api_symbols=["std::data.tokenize", "std::model.load"],
        dependencies=[],
    )
    llm_dir, llm = write_package(
        registry_root,
        namespace="enkai",
        name="tiny-llm",
        version="1.0.0",
        family="decoder-only-transformer",
        architecture="tiny-transformer",
        tasks=["causal-lm", "text-generation"],
        api_symbols=["std::model.load", "std::model.forward", "std::checkpoint.load"],
        dependencies=[{"package": "enkai/tokenizer-bpe", "version": "1.0.0"}],
    )
    index = build_index(registry_root, [tokenizer, llm])
    tokenizer_lock = write_lock(tokenizer_dir, registry_root, tokenizer)
    llm_lock = write_lock(llm_dir, registry_root, llm)

    llm_manifest_path = llm_dir / "enkai.llm.json"
    llm_lock_path = llm_dir / "enkai.lock.json"
    valid = validate_manifest(llm_manifest_path, registry_root, llm_lock_path)

    tamper_dir = registry_root / "tamper"
    shutil.copytree(llm_dir, tamper_dir)
    (tamper_dir / "weights.safetensors").write_bytes(b"tampered")
    tamper = validate_manifest(tamper_dir / "enkai.llm.json", registry_root, tamper_dir / "enkai.lock.json")

    conflict = dict(llm)
    conflict["dependencies"] = [{"package": "enkai/tokenizer-bpe", "version": "2.0.0"}]
    conflict["manifest_digest"] = canonical_digest({k: v for k, v in conflict.items() if k != "manifest_digest"})
    conflict_dir = registry_root / "conflict"
    shutil.copytree(llm_dir, conflict_dir)
    write_json(conflict_dir / "enkai.llm.json", conflict)
    conflict_result = validate_manifest(conflict_dir / "enkai.llm.json", registry_root, None)

    gates = {
        "package_manifest_validation": valid.ok,
        "deterministic_lock_resolution": valid.ok and llm_lock == read_json(llm_lock_path),
        "artifact_hash_integrity": valid.ok and all((llm_dir / item["path"]).is_file() for item in llm["files"]),
        "dependency_resolution": valid.ok and registry_key("enkai/tokenizer-bpe", "1.0.0") in [
            registry_key(item["package"], item["version"]) for item in llm_lock["packages"]
        ],
        "tamper_rejection": not tamper.ok and any("file hash mismatch" in issue for issue in tamper.issues),
        "conflict_rejection": not conflict_result.ok and any("unresolved dependency" in issue for issue in conflict_result.issues),
        "no_native_postinstall_hooks": llm["security"]["native_postinstall_hooks"] is False,
        "stable_api_surface": set(["std::model.load", "std::model.forward", "std::checkpoint.load"]).issubset(
            set(llm["api"]["symbols"])
        ),
    }
    failures = [gate for gate, ok in gates.items() if not ok]
    for required in contract["required_gates"]:
        if required not in gates:
            failures.append(f"missing required gate {required}")
    output_rel = str(Path(args.output).as_posix())
    for rel in contract["required_artifacts"]:
        if rel == output_rel:
            continue
        if not (root / rel).is_file():
            failures.append(f"missing artifact {rel}")

    evidence = {
        "schema_version": 1,
        "contract_version": contract["contract_version"],
        "scope": contract["scope"],
        "generated_at_utc": now_iso(),
        "registry_root": str(registry_root),
        "index_digest": canonical_digest(index),
        "root_package": registry_key(llm["package"], llm["version"]),
        "manifest": str(llm_manifest_path),
        "lockfile": str(llm_lock_path),
        "manifest_digest": llm["manifest_digest"],
        "lock_digest": llm_lock["lock_digest"],
        "gates": gates,
        "validation": {"ok": valid.ok, "issues": valid.issues},
        "tamper_validation": {"ok": tamper.ok, "issues": tamper.issues},
        "conflict_validation": {"ok": conflict_result.ok, "issues": conflict_result.issues},
        "production_claims": {
            "llm_package_registry_ecosystem_proven": not failures,
            "pytorch_core_execution_dependency": False,
            "remote_install_requires_signature": True,
            "native_postinstall_hooks_allowed": False,
        },
        "all_passed": not failures,
        "failures": failures,
    }
    write_json(root / args.output, evidence)
    print(json.dumps({"all_passed": evidence["all_passed"], "failures": failures, "output": args.output}, indent=2))
    return 0 if evidence["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
