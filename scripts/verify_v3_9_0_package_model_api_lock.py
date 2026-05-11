#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify v3.9.0 package/model API lock.")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--contract", default="enkai/contracts/v3_9_0_package_model_api_lock.json")
    parser.add_argument("--output", default="artifacts/readiness/v3_9_0_package_model_api_lock_verify.json")
    args = parser.parse_args()
    root = Path(args.workspace).resolve()
    contract = read_json(root / args.contract)
    failures: list[str] = []

    module_files = {
        "std::tensor": root / "std/tensor.enk",
        "std::nn": root / "std/nn.enk",
        "std::optim": root / "std/optim.enk",
        "std::data": root / "std/data.enk",
        "std::checkpoint": root / "std/checkpoint.enk",
        "std::model": root / "std/model.enk",
    }
    for module, spec in contract["locked_modules"].items():
        path = module_files.get(module)
        if path is None or not path.exists():
            failures.append(f"missing module file for {module}")
            continue
        text = path.read_text(encoding="utf-8-sig")
        for symbol in spec.get("required_symbols", []):
            if symbol not in text:
                failures.append(f"{module} missing locked symbol snippet: {symbol}")

    tensor_doc = (root / "docs/tensor_api.md").read_text(encoding="utf-8-sig")
    for snippet in ["rocm-kernels", "metal-kernels", "hardware-backed verifier evidence"]:
        if snippet not in tensor_doc:
            failures.append(f"docs/tensor_api.md missing snippet: {snippet}")

    result = {
        "schema_version": 1,
        "contract_version": contract.get("contract_version"),
        "scope": contract.get("scope"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "all_passed": not failures,
        "failures": failures,
    }
    write_json(root / args.output, result)
    print(json.dumps(result, indent=2))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
