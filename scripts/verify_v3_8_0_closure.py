#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify final v3.8.0 closure.")
    parser.add_argument("--contract", default="enkai/contracts/v3_8_0_closure.json")
    parser.add_argument("--output", default="artifacts/readiness/v3_8_0_closure.json")
    return parser.parse_args()


def cargo_version(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    if not match:
        raise RuntimeError(f"missing version in {path}")
    return match.group(1)


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    contract_path = (root / args.contract).resolve()
    contract = read_json(contract_path)
    failures: list[str] = []
    version = contract["required_version"]
    for rel in [
        "enkai/Cargo.toml",
        "enkaic/Cargo.toml",
        "enkairt/Cargo.toml",
        "enkai_native/Cargo.toml",
        "enkai_tensor/Cargo.toml",
    ]:
        actual = cargo_version(root / rel)
        if actual != version:
            failures.append(f"{rel} expected version {version}, got {actual}")
    if f'version = "{version}"' not in (root / "enkai.toml").read_text(encoding="utf-8-sig"):
        failures.append("enkai.toml version mismatch")
    report_statuses = []
    for rel in contract["required_reports"]:
        path = root / rel
        if not path.is_file():
            failures.append(f"missing required report: {rel}")
            continue
        payload = read_json(path)
        passed = payload.get("all_passed") is True
        report_statuses.append({"path": rel, "all_passed": passed})
        if not passed:
            failures.append(f"{rel} all_passed expected True")
    for rel, snippets in contract["required_text_snippets"].items():
        text = (root / rel).read_text(encoding="utf-8-sig")
        for snippet in snippets:
            if snippet not in text:
                failures.append(f"missing snippet in {rel}: {snippet}")
    output = {
        "schema_version": 1,
        "contract": str(contract_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "verified_contract_version": "v3.8.0",
        "all_passed": not failures,
        "required_reports": report_statuses,
        "failures": failures,
    }
    output_path = (root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return 0 if output["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
