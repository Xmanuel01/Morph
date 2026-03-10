#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EQ_CONTRACT = ROOT / "bench" / "contracts" / "workload_equivalence_v1.json"


@dataclass
class Sample:
    wall_ms: float
    peak_rss_mb: float


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    sample: Sample
    timed_out: bool


def _windows_rss_bytes(pid: int) -> int | None:
    try:
        import ctypes
        from ctypes import wintypes

        PROCESS_QUERY_INFORMATION = 0x0400
        PROCESS_VM_READ = 0x0010

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)

        open_process = kernel32.OpenProcess
        open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        open_process.restype = wintypes.HANDLE

        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [wintypes.HANDLE]
        close_handle.restype = wintypes.BOOL

        get_process_memory_info = psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        get_process_memory_info.restype = wintypes.BOOL

        handle = open_process(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
        if not handle:
            return None
        try:
            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            if not get_process_memory_info(handle, ctypes.byref(counters), counters.cb):
                return None
            return int(counters.WorkingSetSize)
        finally:
            close_handle(handle)
    except Exception:
        return None


def _linux_rss_bytes(pid: int) -> int | None:
    status = Path(f"/proc/{pid}/status")
    if not status.is_file():
        return None
    try:
        text = status.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) * 1024
    return None


def _fallback_rss_bytes(pid: int) -> int | None:
    try:
        output = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True)
    except Exception:
        return None
    value = output.strip()
    if not value.isdigit():
        return None
    return int(value) * 1024


def process_rss_bytes(pid: int) -> int | None:
    system = platform.system().lower()
    if system.startswith("win"):
        value = _windows_rss_bytes(pid)
        if value is not None:
            return value
    elif system == "linux":
        value = _linux_rss_bytes(pid)
        if value is not None:
            return value
    return _fallback_rss_bytes(pid)


def run_command(
    command: list[str],
    cwd: Path,
    timeout_sec: float | None = None,
    env_overrides: dict[str, str] | None = None,
) -> RunResult:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    peak_bytes = 0
    timed_out = False
    while process.poll() is None:
        if timeout_sec is not None and (time.perf_counter() - started) > timeout_sec:
            timed_out = True
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
            break
        rss = process_rss_bytes(process.pid)
        if rss is not None and rss > peak_bytes:
            peak_bytes = rss
        time.sleep(0.01)
    stdout, stderr = process.communicate()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    rss = process_rss_bytes(process.pid)
    if rss is not None and rss > peak_bytes:
        peak_bytes = rss
    return RunResult(
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
        sample=Sample(
            wall_ms=elapsed_ms,
            peak_rss_mb=peak_bytes / (1024.0 * 1024.0),
        ),
        timed_out=timed_out,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Enkai benchmark suite")
    parser.add_argument("--suite", default="core")
    parser.add_argument("--baseline", choices=["python", "none"], default="python")
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--machine-profile")
    parser.add_argument("--enkai-bin")
    parser.add_argument("--python-exe")
    parser.add_argument("--target-speedup", type=float, default=5.0)
    parser.add_argument("--target-memory", type=float, default=5.0)
    parser.add_argument("--enforce-target", action="store_true")
    parser.add_argument(
        "--enforce-all-cases",
        action="store_true",
        help="Require every case to meet speedup/memory targets (default: suite medians only)",
    )
    parser.add_argument("--enforce-class-targets", action="store_true")
    parser.add_argument("--class-targets")
    parser.add_argument("--fairness-check-only", action="store_true")
    parser.add_argument("--equivalence-contract")
    parser.add_argument("--profile-case")
    parser.add_argument("--profile-output")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def expand_command(tokens: list[str], enkai_bin: str, python_exe: str) -> list[str]:
    out: list[str] = []
    for token in tokens:
        out.append(token.replace("${ENKAI_BIN}", enkai_bin).replace("${PYTHON_EXE}", python_exe))
    return out


def ensure_positive(name: str, value: int) -> None:
    if value <= 0:
        raise SystemExit(f"{name} must be > 0")


def resolve_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve_suite_path(value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() and candidate.is_file():
        return candidate
    if candidate.suffix == ".json" or "/" in value or "\\" in value:
        local = ROOT / candidate
        if local.is_file():
            return local
    suite = ROOT / "bench" / "suites" / f"{value}.json"
    if suite.is_file():
        return suite
    raise SystemExit(f"suite not found: {suite}")


def load_suite_recursive(path: Path, visited: set[Path]) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    canonical = path.resolve()
    if canonical in visited:
        raise SystemExit(f"suite include cycle detected: {path}")
    visited.add(canonical)

    suite = load_json(path)
    suite_name = str(suite.get("suite", path.stem))
    default_class = suite.get("class")

    cases: list[dict[str, Any]] = []
    for case in suite.get("cases", []):
        cloned = copy.deepcopy(case)
        if default_class and "class" not in cloned:
            cloned["class"] = default_class
        cloned.setdefault("_suite_component", suite_name)
        cases.append(cloned)

    components = [suite_name]
    for include in suite.get("includes", []):
        include_path = resolve_suite_path(str(include))
        _, include_cases, include_components = load_suite_recursive(include_path, visited)
        cases.extend(include_cases)
        components.extend(include_components)

    visited.remove(canonical)
    return suite, cases, components


def ensure_unique_case_ids(cases: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for case in cases:
        case_id = str(case.get("id", "")).strip()
        if not case_id:
            raise SystemExit("suite case is missing id")
        if case_id in seen:
            raise SystemExit(f"duplicate benchmark case id: {case_id}")
        seen.add(case_id)


def parse_expected_python_version(machine_profile: dict[str, Any] | None) -> str | None:
    if not machine_profile:
        return None
    raw = machine_profile.get("python")
    if raw is None:
        return None
    value = str(raw).strip()
    if not value:
        return None
    if not re.fullmatch(r"\d+\.\d+", value):
        raise SystemExit(
            "machine profile python version must be exact major.minor (example: 3.11)"
        )
    return value


def detect_python_version_major_minor(python_exe: str) -> str:
    output = subprocess.check_output(
        [python_exe, "-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"],
        text=True,
    )
    return output.strip()


def load_equivalence_contract(path: Path | None) -> dict[str, Any]:
    if path is None:
        path = DEFAULT_EQ_CONTRACT
    if not path.is_file():
        raise SystemExit(f"workload equivalence contract not found: {path}")
    return load_json(path)


def validate_case_equivalence(case: dict[str, Any], contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    case_id = str(case.get("id", "<unknown>"))
    metadata = case.get("equivalence")
    if not isinstance(metadata, dict):
        return [f"{case_id}: missing equivalence metadata object"]

    required_fields = contract.get("required_fields", [])
    for field in required_fields:
        if field not in metadata:
            errors.append(f"{case_id}: equivalence.{field} is required")

    for field in contract.get("positive_integer_fields", []):
        value = metadata.get(field)
        if not isinstance(value, int) or value <= 0:
            errors.append(f"{case_id}: equivalence.{field} must be a positive integer")

    for left, right in contract.get("equal_pairs", []):
        if metadata.get(left) != metadata.get(right):
            errors.append(
                f"{case_id}: equivalence mismatch ({left}={metadata.get(left)!r}, {right}={metadata.get(right)!r})"
            )

    allowed_values = contract.get("allowed_values", {})
    for field, options in allowed_values.items():
        if field not in metadata:
            continue
        value = metadata[field]
        if value not in options:
            errors.append(
                f"{case_id}: equivalence.{field}={value!r} is not in allowed set {options!r}"
            )

    return errors


def load_class_targets(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"class targets file not found: {path}")
    return load_json(path)


def class_thresholds(targets: dict[str, Any], class_name: str) -> dict[str, float]:
    merged: dict[str, float] = {}
    default = targets.get("default", {})
    specific = targets.get("classes", {}).get(class_name, {})
    for source in (default, specific):
        for key, value in source.items():
            merged[key] = float(value)
    return merged


def enforce_class_targets(
    case_reports: list[dict[str, Any]],
    class_summaries: dict[str, Any],
    targets: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    for class_name, summary in class_summaries.items():
        thresholds = class_thresholds(targets, class_name)
        if not thresholds:
            continue

        median_speedup_target = thresholds.get("median_speedup_pct")
        if median_speedup_target is not None and summary["median_speedup_pct"] < median_speedup_target:
            failures.append(
                f"class {class_name}: median speedup {summary['median_speedup_pct']:.2f}% < target {median_speedup_target:.2f}%"
            )

        median_memory_target = thresholds.get("median_memory_reduction_pct")
        if median_memory_target is not None and summary["median_memory_reduction_pct"] < median_memory_target:
            failures.append(
                f"class {class_name}: median memory reduction {summary['median_memory_reduction_pct']:.2f}% < target {median_memory_target:.2f}%"
            )

        case_speedup_target = thresholds.get("case_speedup_pct")
        case_memory_target = thresholds.get("case_memory_reduction_pct")

        for case in case_reports:
            if case.get("class") != class_name:
                continue
            delta = case.get("delta") or {}
            speedup = float(delta.get("speedup_pct", 0.0))
            memory = float(delta.get("memory_reduction_pct", 0.0))
            if case_speedup_target is not None and speedup < case_speedup_target:
                failures.append(
                    f"case {case['id']} ({class_name}): speedup {speedup:.2f}% < target {case_speedup_target:.2f}%"
                )
            if case_memory_target is not None and memory < case_memory_target:
                failures.append(
                    f"case {case['id']} ({class_name}): memory reduction {memory:.2f}% < target {case_memory_target:.2f}%"
                )
    return failures


def main() -> int:
    args = parse_args()
    ensure_positive("--iterations", args.iterations)
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")

    suite_path = resolve_suite_path(args.suite)
    suite_root, cases, suite_components = load_suite_recursive(suite_path, set())
    ensure_unique_case_ids(cases)

    machine_profile = None
    if args.machine_profile:
        machine_path = Path(args.machine_profile)
        if not machine_path.is_absolute():
            machine_path = ROOT / machine_path
        machine_profile = load_json(machine_path)

    enkai_bin = args.enkai_bin or str(
        (ROOT / ("target/release/enkai.exe" if os.name == "nt" else "target/release/enkai")).resolve()
    )
    python_exe = args.python_exe or sys.executable

    expected_python = parse_expected_python_version(machine_profile)
    if expected_python:
        actual_python = detect_python_version_major_minor(python_exe)
        if actual_python != expected_python:
            raise SystemExit(
                f"python major.minor mismatch: profile requires {expected_python}, runner is {actual_python}"
            )

    contract_path = None
    if args.equivalence_contract:
        contract_path = Path(args.equivalence_contract)
        if not contract_path.is_absolute():
            contract_path = ROOT / contract_path
    eq_contract = load_equivalence_contract(contract_path)
    fairness_errors: list[str] = []
    for case in cases:
        fairness_errors.extend(validate_case_equivalence(case, eq_contract))
    if fairness_errors:
        details = "\n".join(f"- {item}" for item in fairness_errors)
        raise SystemExit(f"benchmark fairness check failed:\n{details}")

    started_utc = datetime.now(timezone.utc).isoformat()

    if args.fairness_check_only:
        report = {
            "schema_version": 2,
            "suite": args.suite,
            "suite_file": resolve_relative(suite_path),
            "suite_components": sorted(set(suite_components)),
            "baseline": args.baseline,
            "started_utc": started_utc,
            "host": {
                "platform": platform.platform(),
                "python": sys.version.split()[0],
            },
            "machine_profile": machine_profile,
            "fairness": {
                "contract": resolve_relative(contract_path or DEFAULT_EQ_CONTRACT),
                "pass": True,
                "case_count": len(cases),
            },
            "cases": [
                {
                    "id": case["id"],
                    "class": case.get("class", "uncategorized"),
                    "equivalence": case.get("equivalence"),
                    "component": case.get("_suite_component"),
                }
                for case in cases
            ],
            "summary": {
                "pass": True,
                "fairness_check_only": True,
            },
        }
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"fairness check passed for suite '{args.suite}'")
        print(f"output: {output_path}")
        return 0

    profile_case = args.profile_case.strip() if args.profile_case else None
    if profile_case:
        filtered = [case for case in cases if case.get("id") == profile_case]
        if not filtered:
            raise SystemExit(f"profile case not found in suite: {profile_case}")
        cases = filtered

    profile_output_path: Path | None = None
    if args.profile_output:
        profile_output_path = Path(args.profile_output)
        if not profile_output_path.is_absolute():
            profile_output_path = ROOT / profile_output_path

    case_reports: list[dict[str, Any]] = []
    speedups: list[float] = []
    memory_reductions: list[float] = []

    for case in cases:
        case_id = case["id"]
        case_class = str(case.get("class", "uncategorized"))
        enkai_cmd = expand_command(case["enkai_command"], enkai_bin, python_exe)
        python_cmd = expand_command(case["python_command"], enkai_bin, python_exe)
        case_timeout = float(case.get("timeout_sec", 0))
        timeout_sec = case_timeout if case_timeout > 0.0 else None
        case_env = case.get("env", {})
        if not isinstance(case_env, dict):
            raise SystemExit(f"case {case_id} env must be an object")
        common_env = {
            "PYTHONHASHSEED": "0",
            "ENKAI_STD": str(ROOT / "std"),
            **{str(key): str(value) for key, value in case_env.items()},
        }
        enkai_env = dict(common_env)
        if profile_case and profile_case == case_id:
            profile_path = profile_output_path or (ROOT / "bench" / "results" / "profiles" / f"{case_id}.json")
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            enkai_env["ENKAI_BENCH_PROFILE_OUT"] = str(profile_path)
            enkai_env["ENKAI_BENCH_PROFILE_CASE"] = case_id

        for _ in range(args.warmup):
            warmup_result = run_command(
                enkai_cmd,
                ROOT,
                timeout_sec=timeout_sec,
                env_overrides=enkai_env,
            )
            if warmup_result.returncode != 0:
                raise SystemExit(f"warmup failed for {case_id} (enkai): {warmup_result.stderr.strip()}")
            if args.baseline == "python":
                warmup_py = run_command(
                    python_cmd,
                    ROOT,
                    timeout_sec=timeout_sec,
                    env_overrides=common_env,
                )
                if warmup_py.returncode != 0:
                    raise SystemExit(f"warmup failed for {case_id} (python): {warmup_py.stderr.strip()}")

        enkai_samples: list[Sample] = []
        py_samples: list[Sample] = []

        for _ in range(args.iterations):
            result = run_command(
                enkai_cmd,
                ROOT,
                timeout_sec=timeout_sec,
                env_overrides=enkai_env,
            )
            if result.timed_out:
                raise SystemExit(f"case {case_id} timed out for enkai after {timeout_sec}s")
            if result.returncode != 0:
                raise SystemExit(
                    f"case {case_id} failed for enkai (exit={result.returncode}): {result.stderr.strip()}"
                )
            enkai_samples.append(result.sample)

            if args.baseline == "python":
                py_result = run_command(
                    python_cmd,
                    ROOT,
                    timeout_sec=timeout_sec,
                    env_overrides=common_env,
                )
                if py_result.timed_out:
                    raise SystemExit(f"case {case_id} timed out for python after {timeout_sec}s")
                if py_result.returncode != 0:
                    raise SystemExit(
                        f"case {case_id} failed for python (exit={py_result.returncode}): {py_result.stderr.strip()}"
                    )
                py_samples.append(py_result.sample)

        enkai_times = [sample.wall_ms for sample in enkai_samples]
        enkai_mem = [sample.peak_rss_mb for sample in enkai_samples]
        case_report: dict[str, Any] = {
            "id": case_id,
            "class": case_class,
            "component": case.get("_suite_component"),
            "description": case.get("description", ""),
            "equivalence": case.get("equivalence"),
            "enkai": {
                "samples": [sample.__dict__ for sample in enkai_samples],
                "median_wall_ms": median(enkai_times),
                "median_peak_rss_mb": median(enkai_mem),
                "command": enkai_cmd,
            },
        }

        if profile_case and profile_case == case_id:
            profile_file = profile_output_path or (ROOT / "bench" / "results" / "profiles" / f"{case_id}.json")
            case_report["profile_output"] = resolve_relative(profile_file)

        pass_case = True
        if args.baseline == "python":
            py_times = [sample.wall_ms for sample in py_samples]
            py_mem = [sample.peak_rss_mb for sample in py_samples]
            py_median_time = median(py_times)
            py_median_mem = median(py_mem)
            enkai_median_time = case_report["enkai"]["median_wall_ms"]
            enkai_median_mem = case_report["enkai"]["median_peak_rss_mb"]

            speedup_pct = 0.0
            memory_reduction_pct = 0.0
            if py_median_time > 0:
                speedup_pct = ((py_median_time - enkai_median_time) / py_median_time) * 100.0
            if py_median_mem > 0:
                memory_reduction_pct = ((py_median_mem - enkai_median_mem) / py_median_mem) * 100.0

            pass_case = speedup_pct >= args.target_speedup and memory_reduction_pct >= args.target_memory
            speedups.append(speedup_pct)
            memory_reductions.append(memory_reduction_pct)

            case_report["python"] = {
                "samples": [sample.__dict__ for sample in py_samples],
                "median_wall_ms": py_median_time,
                "median_peak_rss_mb": py_median_mem,
                "command": python_cmd,
            }
            case_report["delta"] = {
                "speedup_pct": speedup_pct,
                "memory_reduction_pct": memory_reduction_pct,
            }
        case_report["pass"] = pass_case
        case_reports.append(case_report)

    class_summaries: dict[str, Any] = {}
    if args.baseline == "python":
        grouped: dict[str, dict[str, list[float]]] = {}
        for case in case_reports:
            class_name = str(case.get("class", "uncategorized"))
            grouped.setdefault(class_name, {"speedups": [], "memory": []})
            delta = case.get("delta") or {}
            grouped[class_name]["speedups"].append(float(delta.get("speedup_pct", 0.0)))
            grouped[class_name]["memory"].append(float(delta.get("memory_reduction_pct", 0.0)))

        for class_name, values in grouped.items():
            class_summaries[class_name] = {
                "median_speedup_pct": median(values["speedups"]),
                "median_memory_reduction_pct": median(values["memory"]),
                "case_count": len(values["speedups"]),
            }

    class_failures: list[str] = []
    class_targets_payload: dict[str, Any] | None = None
    if args.enforce_class_targets:
        if args.baseline != "python":
            raise SystemExit("--enforce-class-targets requires --baseline python")
        targets_path = Path(args.class_targets) if args.class_targets else None
        if targets_path is None:
            raise SystemExit("--enforce-class-targets requires --class-targets <file>")
        if not targets_path.is_absolute():
            targets_path = ROOT / targets_path
        class_targets_payload = load_class_targets(targets_path)
        class_failures = enforce_class_targets(case_reports, class_summaries, class_targets_payload)

    median_speedup = median(speedups) if speedups else 0.0
    median_memory_reduction = median(memory_reductions) if memory_reductions else 0.0
    case_pass_count = sum(1 for case in case_reports if case.get("pass", False))
    case_fail_count = len(case_reports) - case_pass_count
    summary_pass = median_speedup >= args.target_speedup and median_memory_reduction >= args.target_memory
    if args.enforce_all_cases and case_fail_count > 0:
        summary_pass = False
    if args.enforce_class_targets and class_failures:
        summary_pass = False

    summary = {
        "median_speedup_pct": median_speedup,
        "median_memory_reduction_pct": median_memory_reduction,
        "target_speedup_pct": args.target_speedup,
        "target_memory_pct": args.target_memory,
        "case_pass_count": case_pass_count,
        "case_fail_count": case_fail_count,
        "enforce_all_cases": args.enforce_all_cases,
        "enforce_class_targets": args.enforce_class_targets,
        "class_summaries": class_summaries,
        "class_gate_failures": class_failures,
        "pass": summary_pass,
    }

    report = {
        "schema_version": 2,
        "suite": args.suite,
        "suite_file": resolve_relative(suite_path),
        "suite_components": sorted(set(suite_components)),
        "baseline": args.baseline,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "started_utc": started_utc,
        "host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        "machine_profile": machine_profile,
        "fairness": {
            "contract": resolve_relative(contract_path or DEFAULT_EQ_CONTRACT),
            "pass": True,
        },
        "class_targets": class_targets_payload,
        "cases": case_reports,
        "summary": summary,
    }

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"bench suite '{args.suite}' completed")
    print(f"output: {output_path}")
    if args.baseline == "python":
        print(f"median speedup: {summary['median_speedup_pct']:.2f}%")
        print(f"median memory reduction: {summary['median_memory_reduction_pct']:.2f}%")
        if class_summaries:
            print("class medians:")
            for name in sorted(class_summaries):
                row = class_summaries[name]
                print(
                    f"- {name}: speedup {row['median_speedup_pct']:.2f}% | memory {row['median_memory_reduction_pct']:.2f}%"
                )
    print(f"pass: {summary['pass']}")

    if args.enforce_target and not summary["pass"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
