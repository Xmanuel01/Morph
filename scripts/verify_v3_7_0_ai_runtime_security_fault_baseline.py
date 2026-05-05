#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the bounded AI runtime security and fault baseline.")
    parser.add_argument("--contract", default="enkai/contracts/v3_7_0_ai_runtime_security_fault_baseline.json")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    contract_path = (repo_root / args.contract).resolve()
    contract = read_json(contract_path)
    foundation = read_json((repo_root / contract["required_reports"]["foundation"]).resolve())
    networked = read_json((repo_root / contract["required_reports"]["networked_exec"]).resolve())
    adversarial = read_json((repo_root / contract["required_reports"]["adversarial_inputs"]).resolve())
    peer_adversarial = read_json((repo_root / contract["required_reports"]["networked_peer_adversarial"]).resolve())
    gradient_adversarial = read_json((repo_root / contract["required_reports"]["networked_gradient_adversarial"]).resolve())
    inventory = read_json((repo_root / contract["required_reports"]["strict_selfhost_inventory"]).resolve())
    failures: list[str] = []

    mem = foundation.get("memory_safety", {})
    sec = foundation.get("security_compliance", {})
    if not foundation.get("all_passed"):
        failures.append("foundation report is not green")
    for key in ["allocator_behavior_measured", "peak_memory_accounted"]:
        if mem.get(key) is not True:
            failures.append(f"memory_safety.{key} expected True")
    for key in ["invalid_backend", "oom_budget", "corrupted_checkpoint", "invalid_runtime_state"]:
        if mem.get(key, {}).get("passed") is not True:
            failures.append(f"memory_safety.{key}.passed expected True")
    for key in ["deterministic_validation_outputs", "backend_selection_archived", "fallback_behavior_archived", "no_hidden_rust_requirement"]:
        if sec.get(key) is not True:
            failures.append(f"security_compliance.{key} expected True")
    if "none added" not in str(sec.get("unsafe_escape_hatches", "")):
        failures.append("unsafe_escape_hatches does not confirm no new escape hatches")

    if not networked.get("all_passed"):
        failures.append("networked_exec report is not green")
    baseline = networked.get("baseline", {})
    fault = networked.get("fault_injection", {})
    if baseline.get("passed") is not True:
        failures.append("networked baseline did not pass")
    if fault.get("passed") is not True:
        failures.append("networked fault injection did not pass")
    if int(fault.get("total_retry_count", 0)) < 1:
        failures.append("networked fault injection did not exercise retry path")
    if fault.get("fault_injection_observed") is not True:
        failures.append("networked fault injection was not observed")
    if fault.get("identical_checkpoint_semantics") is not True:
        failures.append("networked fault injection did not preserve checkpoint semantics")
    if adversarial.get("all_passed") is not True:
        failures.append("adversarial input/corruption report is not green")
    if peer_adversarial.get("all_passed") is not True:
        failures.append("networked rendezvous peer-adversarial report is not green")
    if gradient_adversarial.get("all_passed") is not True:
        failures.append("networked gradient adversarial report is not green")

    policy = inventory.get("policy", {})
    summary = inventory.get("summary", {})
    if policy.get("rust_free_shipped_path") is not True:
        failures.append("strict_selfhost rust_free_shipped_path expected True")
    if summary.get("remaining_rust_dependencies") not in ([], None):
        failures.append("strict_selfhost remaining_rust_dependencies expected empty")

    for rel_path, snippets in contract.get("required_text_snippets", {}).items():
        text = (repo_root / rel_path).read_text(encoding="utf-8-sig")
        for snippet in snippets:
            if snippet not in text:
                failures.append(f"missing snippet in {rel_path}: {snippet}")

    output = {
        "schema_version": 1,
        "contract": str(contract_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "all_passed": not failures,
        "security_summary": {
            "memory_safety_checks": {k: mem.get(k, {}).get("passed") if isinstance(mem.get(k), dict) else mem.get(k) for k in ["allocator_behavior_measured", "peak_memory_accounted", "invalid_backend", "oom_budget", "corrupted_checkpoint", "invalid_runtime_state"]},
            "compliance_checks": {k: sec.get(k) for k in ["deterministic_validation_outputs", "backend_selection_archived", "fallback_behavior_archived", "no_hidden_rust_requirement"]},
            "unsafe_escape_hatches": sec.get("unsafe_escape_hatches"),
            "networked_retry_count": fault.get("total_retry_count", 0),
            "networked_fault_observed": fault.get("fault_injection_observed", False),
            "networked_checkpoint_semantics": fault.get("identical_checkpoint_semantics", False),
            "adversarial_inputs_green": adversarial.get("all_passed", False),
            "networked_peer_adversarial_green": peer_adversarial.get("all_passed", False),
            "networked_gradient_adversarial_green": gradient_adversarial.get("all_passed", False),
            "rust_free_shipped_path": policy.get("rust_free_shipped_path", False),
        },
        "failures": failures,
    }

    output_path = (repo_root / (args.output or contract["output_report"])).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return 0 if output["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
