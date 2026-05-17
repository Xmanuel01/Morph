#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(command: list[str], cwd: Path, timeout: int = 300) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        return {
            "command": command,
            "exit_code": proc.returncode,
            "passed": proc.returncode == 0,
            "stdout_tail": stdout[-12000:],
            "stderr_tail": stderr[-12000:],
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return {
            "command": command,
            "exit_code": None,
            "passed": False,
            "stdout_tail": stdout[-12000:],
            "stderr_tail": stderr[-12000:],
            "error": "timeout",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify native SNN batched kernel closure.")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--contract", default="enkai/contracts/v4_0_0_native_snn_batched_kernels.json")
    parser.add_argument("--output", default="artifacts/readiness/v4_0_0_native_snn_batched_kernels.json")
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()

    root = Path(args.workspace).resolve()
    contract = read_json(root / args.contract)
    native_path = root / "enkai_native" / "src" / "lib.rs"
    vm_path = root / "enkairt" / "src" / "vm.rs"
    std_path = root / "std" / "snn.enk"
    vm_tests_path = root / "enkairt" / "tests" / "interpreter.rs"

    native = native_path.read_text(encoding="utf-8")
    vm = vm_path.read_text(encoding="utf-8")
    std = std_path.read_text(encoding="utf-8")
    vm_tests = vm_tests_path.read_text(encoding="utf-8")
    combined = "\n".join([native, vm, std, vm_tests])

    failures: list[str] = []
    missing_markers = [m for m in contract["required_source_markers"] if m not in combined]
    missing_tests = [m for m in contract["required_tests"] if m not in combined]
    if missing_markers:
        failures.append(f"missing source markers: {missing_markers}")
    if missing_tests:
        failures.append(f"missing tests: {missing_tests}")

    batch_body = re.search(
        r"pub unsafe extern \"C\" fn sim_snn_step_batch\(.*?\) -> FfiSlice \{(?P<body>.*?)\n\}",
        native,
        re.S,
    )
    vm_batch_body = re.search(r"fn snn_step_batch\(&mut self,.*?\) -> Result<Value, RuntimeError> \{(?P<body>.*?)\n    \}", vm, re.S)
    step_kernel_body = re.search(r"fn step_kernel\(&mut self, inputs: \&\[f64\]\) -> Vec<u8> \{(?P<body>.*?)\n    \}", native, re.S)

    gates = {
        "native_batch_kernel_exported": "pub unsafe extern \"C\" fn sim_snn_step_batch" in native,
        "std_step_batch_exported": "pub fn step_batch(network: SnnNetwork, inputs)" in std and "snn.step_batch(network, inputs)" in std,
        "vm_single_native_call_per_batch": bool(
            vm_batch_body
            and "bindings.snn_step_batch.call(&[handle, encoded])" in vm_batch_body.group("body")
            and "self.note_snn_native_call();" in vm_batch_body.group("body")
        ),
        "fail_closed_invalid_batch_inputs": bool(
            batch_body
            and "return null_slice();" in batch_body.group("body")
            and "inputs.is_empty()" in batch_body.group("body")
            and "inputs.len() % net.neuron_count != 0" in batch_body.group("body")
            and "!value.is_finite()" in batch_body.group("body")
        ),
        "deterministic_native_vm_equivalence": "snn_batch_native_and_vm_paths_match_for_seeded_scenarios" in vm_tests,
        "adjacency_backed_recurrent_kernel": bool(
            step_kernel_body
            and "outgoing_edges" in native
            and "self.outgoing_edges[from]" in step_kernel_body.group("body")
            and "edges.sort_by_key" in native
        ),
        "policy_export_machine_readable": "sim_snn_runtime_policy_json" in native and "single_ffi_call_per_batch" in native,
    }

    for gate in contract["required_gates"]:
        if gates.get(gate) is not True:
            failures.append(f"gate failed: {gate}")

    commands: list[dict[str, Any]] = []
    if args.run_tests:
        test_commands = [
            ["cargo", "test", "-p", "enkai_native", "sim_snn"],
            [
                "cargo",
                "test",
                "-p",
                "enkairt",
                "snn_batched_kernel_matches_scalar_frontier_and_rejects_invalid_policy",
            ],
            [
                "cargo",
                "test",
                "-p",
                "enkairt",
                "snn_batch_native_and_vm_paths_match_for_seeded_scenarios",
            ],
            ["cargo", "test", "-p", "enkairt", "snn_runtime_and_agent_environment_kernel_work"],
        ]
        for command in test_commands:
            result = run(command, root, timeout=600)
            commands.append(result)
            if not result["passed"]:
                failures.append(f"command failed: {' '.join(command)}")

    payload = {
        "schema_version": 1,
        "contract_version": contract["contract_version"],
        "scope": contract["scope"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gates": gates,
        "commands": commands,
        "sources": [str(native_path), str(vm_path), str(std_path), str(vm_tests_path)],
        "production_claims": {
            "native_batched_snn_kernel_proven": not failures,
            "single_native_ffi_call_per_batch": gates["vm_single_native_call_per_batch"],
            "invalid_inputs_fail_closed": gates["fail_closed_invalid_batch_inputs"],
            "native_vm_determinism_proven": gates["deterministic_native_vm_equivalence"],
            "full_neuromorphic_gpu_platform_claim": False,
        },
        "all_passed": not failures,
        "failures": failures,
    }
    write_json(root / args.output, payload)
    print(json.dumps({"all_passed": payload["all_passed"], "failures": failures, "output": args.output}, indent=2))
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
