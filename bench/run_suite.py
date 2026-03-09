#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


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
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def main() -> int:
    args = parse_args()
    ensure_positive("--iterations", args.iterations)
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")

    suite_path = ROOT / "bench" / "suites" / f"{args.suite}.json"
    if not suite_path.is_file():
        raise SystemExit(f"suite not found: {suite_path}")
    suite = load_json(suite_path)

    machine_profile = None
    if args.machine_profile:
        machine_path = Path(args.machine_profile)
        if not machine_path.is_absolute():
            machine_path = ROOT / machine_path
        machine_profile = load_json(machine_path)

    enkai_bin = args.enkai_bin or str((ROOT / ("target/release/enkai.exe" if os.name == "nt" else "target/release/enkai")).resolve())
    python_exe = args.python_exe or sys.executable

    started_utc = datetime.now(timezone.utc).isoformat()

    case_reports: list[dict[str, Any]] = []
    speedups: list[float] = []
    memory_reductions: list[float] = []

    for case in suite.get("cases", []):
        case_id = case["id"]
        enkai_cmd = expand_command(case["enkai_command"], enkai_bin, python_exe)
        python_cmd = expand_command(case["python_command"], enkai_bin, python_exe)
        case_timeout = float(case.get("timeout_sec", 0))
        timeout_sec = case_timeout if case_timeout > 0.0 else None
        case_env = case.get("env", {})
        if not isinstance(case_env, dict):
            raise SystemExit(f"case {case_id} env must be an object")
        env_overrides = {
            "PYTHONHASHSEED": "0",
            "ENKAI_STD": str(ROOT / "std"),
            **{str(key): str(value) for key, value in case_env.items()},
        }

        for _ in range(args.warmup):
            warmup_result = run_command(
                enkai_cmd,
                ROOT,
                timeout_sec=timeout_sec,
                env_overrides=env_overrides,
            )
            if warmup_result.returncode != 0:
                raise SystemExit(f"warmup failed for {case_id} (enkai): {warmup_result.stderr.strip()}")
            if args.baseline == "python":
                warmup_py = run_command(
                    python_cmd,
                    ROOT,
                    timeout_sec=timeout_sec,
                    env_overrides=env_overrides,
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
                env_overrides=env_overrides,
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
                    env_overrides=env_overrides,
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
            "description": case.get("description", ""),
            "enkai": {
                "samples": [sample.__dict__ for sample in enkai_samples],
                "median_wall_ms": median(enkai_times),
                "median_peak_rss_mb": median(enkai_mem),
                "command": enkai_cmd,
            },
        }

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

    median_speedup = median(speedups) if speedups else 0.0
    median_memory_reduction = median(memory_reductions) if memory_reductions else 0.0
    case_pass_count = sum(1 for case in case_reports if case.get("pass", False))
    case_fail_count = len(case_reports) - case_pass_count
    summary_pass = median_speedup >= args.target_speedup and median_memory_reduction >= args.target_memory
    if args.enforce_all_cases and case_fail_count > 0:
        summary_pass = False

    summary = {
        "median_speedup_pct": median_speedup,
        "median_memory_reduction_pct": median_memory_reduction,
        "target_speedup_pct": args.target_speedup,
        "target_memory_pct": args.target_memory,
        "case_pass_count": case_pass_count,
        "case_fail_count": case_fail_count,
        "enforce_all_cases": args.enforce_all_cases,
        "pass": summary_pass,
    }

    report = {
        "schema_version": 1,
        "suite": args.suite,
        "suite_file": str(suite_path.relative_to(ROOT)),
        "baseline": args.baseline,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "started_utc": started_utc,
        "host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        "machine_profile": machine_profile,
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
    print(f"pass: {summary['pass']}")

    if args.enforce_target and not summary["pass"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
