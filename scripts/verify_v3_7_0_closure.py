#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify full v3.7.0 closure evidence.")
    parser.add_argument("--contract", default="enkai/contracts/v3_7_0_closure.json")
    parser.add_argument("--output", default="artifacts/readiness/v3_7_0_closure.json")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def file_contains_version(path: Path, version: str) -> bool:
    text = path.read_text(encoding="utf-8-sig")
    return version in text


def cargo_version(path: Path) -> str | None:
    text = path.read_text(encoding="utf-8-sig")
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', text)
    return None if match is None else match.group(1)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    contract_path = (repo_root / args.contract).resolve()
    contract = read_json(contract_path)
    version = contract["version"]
    failures: list[str] = []
    artifact_results: dict[str, dict[str, Any]] = {}

    for rel_path in contract["version_files"]:
        path = (repo_root / rel_path).resolve()
        if path.name == "Cargo.toml":
            observed = cargo_version(path)
            if observed != version:
                failures.append(f"{rel_path} version expected {version}, got {observed}")
        elif not file_contains_version(path, version):
            failures.append(f"{rel_path} does not contain {version}")

    for name, rel_path in contract["required_artifacts"].items():
        path = (repo_root / rel_path).resolve()
        if not path.exists():
            failures.append(f"missing artifact {name}: {rel_path}")
            artifact_results[name] = {"exists": False, "all_passed": False}
            continue
        artifact = read_json(path)
        passed = artifact.get("all_passed") is True
        artifact_results[name] = {"exists": True, "all_passed": passed, "artifact": rel_path}
        if not passed:
            failures.append(f"artifact {name} all_passed expected True")

    for rel_path, snippets in contract.get("required_text_snippets", {}).items():
        text = (repo_root / rel_path).read_text(encoding="utf-8-sig")
        for snippet in snippets:
            if snippet not in text:
                failures.append(f"missing snippet in {rel_path}: {snippet}")

    output = {
        "schema_version": 1,
        "contract": str(contract_path),
        "version": version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "all_passed": not failures,
        "artifact_results": artifact_results,
        "failures": failures,
    }
    output_path = (repo_root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return 0 if output["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
