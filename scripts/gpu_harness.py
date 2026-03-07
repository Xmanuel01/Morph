#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import random
import shutil
import socket
import subprocess
import sys
import time
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def gpu_count() -> int:
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True, stderr=subprocess.STDOUT)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return 0
    return sum(1 for line in out.splitlines() if line.strip())


def choose_master_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def resolve_enkai() -> pathlib.Path:
    env_path = os.environ.get("ENKAI_EXE")
    if env_path:
        path = pathlib.Path(env_path).expanduser().resolve()
        if path.is_file():
            return path
    which = shutil.which("enkai")
    if which:
        return pathlib.Path(which).resolve()
    local_candidates = [
        ROOT / "target" / "release" / ("enkai.exe" if os.name == "nt" else "enkai"),
        ROOT / "target" / "debug" / ("enkai.exe" if os.name == "nt" else "enkai"),
    ]
    for candidate in local_candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError("could not resolve enkai executable (set ENKAI_EXE or build target/debug)")


def resolve_tensor_lib() -> pathlib.Path | None:
    env_path = os.environ.get("ENKAI_TENSOR_PATH")
    if env_path:
        path = pathlib.Path(env_path).expanduser().resolve()
        if path.is_file():
            return path
    names = []
    if os.name == "nt":
        names.extend(["enkai_tensor.dll"])
    elif sys.platform == "darwin":
        names.extend(["libenkai_tensor.dylib"])
    else:
        names.extend(["libenkai_tensor.so"])
    candidates = []
    for name in names:
        candidates.extend(
            [
                ROOT / "target" / "release" / name,
                ROOT / "target" / "debug" / name,
                ROOT / "target" / "release" / "deps" / name,
                ROOT / "target" / "debug" / "deps" / name,
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def write_enkai_config(path: pathlib.Path, cfg: dict[str, Any]) -> None:
    payload = json.dumps(cfg, separators=(",", ":")).replace("\\", "\\\\").replace('"', '\\"')
    source = f'fn main() ::\n    return json.parse("{payload}")\n::\n'
    path.write_text(source, encoding="utf-8")


def ensure_parent(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def run_checked(
    cmd: list[str],
    *,
    env: dict[str, str],
    cwd: pathlib.Path,
    stdout_path: pathlib.Path,
    stderr_path: pathlib.Path,
    timeout_sec: int,
) -> tuple[int, float]:
    ensure_parent(stdout_path)
    ensure_parent(stderr_path)
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open(
        "w", encoding="utf-8"
    ) as err:
        start = time.time()
        proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=out, stderr=err)
        try:
            code = proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise RuntimeError(
                f"timeout after {timeout_sec}s running: {' '.join(cmd)} "
                f"(stdout={stdout_path}, stderr={stderr_path})"
            ) from None
    return code, time.time() - start


def run_multi_rank(
    rank_commands: list[tuple[int, list[str], dict[str, str], pathlib.Path, pathlib.Path]],
    *,
    cwd: pathlib.Path,
    timeout_sec: int,
) -> tuple[dict[int, int], float]:
    procs: dict[int, subprocess.Popen[str]] = {}
    start = time.time()
    handles: dict[int, tuple[Any, Any]] = {}
    try:
        for rank, cmd, env, stdout_path, stderr_path in rank_commands:
            ensure_parent(stdout_path)
            ensure_parent(stderr_path)
            out = stdout_path.open("w", encoding="utf-8")
            err = stderr_path.open("w", encoding="utf-8")
            handles[rank] = (out, err)
            procs[rank] = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=out, stderr=err)

        exit_codes: dict[int, int] = {}
        deadline = time.time() + timeout_sec
        while procs:
            finished: list[int] = []
            for rank, proc in list(procs.items()):
                code = proc.poll()
                if code is not None:
                    exit_codes[rank] = int(code)
                    finished.append(rank)
            for rank in finished:
                procs.pop(rank, None)
            if not procs:
                break
            if time.time() > deadline:
                for proc in procs.values():
                    proc.kill()
                raise RuntimeError(f"timeout after {timeout_sec}s waiting for distributed ranks")
            time.sleep(1.0)
        return exit_codes, time.time() - start
    finally:
        for proc in procs.values():
            proc.kill()
        for out, err in handles.values():
            out.close()
            err.close()


def load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_last_jsonl(path: pathlib.Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"log file missing: {path}")
    last: dict[str, Any] | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        last = json.loads(line)
    if last is None:
        raise RuntimeError(f"log file is empty: {path}")
    return last


def copy_if_exists(src: pathlib.Path, dst: pathlib.Path) -> None:
    if src.is_file():
        ensure_parent(dst)
        shutil.copy2(src, dst)


def write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_log(path: pathlib.Path, lines: list[str]) -> None:
    ensure_parent(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def migrate_base_config(enkai: pathlib.Path, base_config: pathlib.Path, out_json: pathlib.Path) -> dict[str, Any]:
    ensure_parent(out_json)
    cmd = [str(enkai), "migrate", "config-v1", str(base_config), str(out_json)]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"failed to migrate config-v1 ({base_config}): {proc.stderr.strip() or proc.stdout.strip()}"
        )
    return load_json(out_json)


def build_dataset(path: pathlib.Path, *, count: int = 200) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for i in range(1, count + 1):
            f.write(f"sample {i} alpha beta gamma\n")


def patch_train_config(
    base: dict[str, Any],
    *,
    dataset_path: pathlib.Path,
    checkpoint_dir: pathlib.Path,
    world_size: int,
    rank: int,
    steps: int,
    backend: str,
    quick_model: bool,
) -> dict[str, Any]:
    cfg = json.loads(json.dumps(base))
    cfg["config_version"] = 1
    cfg["backend"] = backend
    cfg["dataset_path"] = str(dataset_path)
    cfg["checkpoint_dir"] = str(checkpoint_dir)
    cfg["max_steps"] = int(steps)
    cfg["save_every"] = max(1, min(int(steps), int(cfg.get("save_every", steps))))
    cfg["log_every"] = 1
    cfg["eval_steps"] = max(1, int(cfg.get("eval_steps", 1)))
    cfg["drop_remainder"] = False
    cfg["world_size"] = int(world_size)
    cfg["rank"] = int(rank)
    cfg["grad_accum_steps"] = 1
    cfg["grad_clip_norm"] = float(cfg.get("grad_clip_norm", 1.0))
    cfg["prefetch_batches"] = int(cfg.get("prefetch_batches", 1))
    cfg.pop("eval_dataset_path", None)

    model = cfg.setdefault("model", {})
    if not isinstance(model, dict):
        model = {}
        cfg["model"] = model
    if quick_model:
        model["vocab_size"] = 512
        model["hidden_size"] = 64
        model["d_model"] = 64
        model["n_layers"] = 2
        model["n_heads"] = 4
        cfg["seq_len"] = 16
        cfg["batch_size"] = 2
    model["device"] = "cpu" if backend == "cpu" else f"cuda:{rank}"
    model["dtype"] = model.get("dtype", "fp32")
    cfg["vocab_size"] = int(model.get("vocab_size", cfg.get("vocab_size", 512)))
    cfg["hidden_size"] = int(model.get("hidden_size", cfg.get("hidden_size", 64)))
    cfg["seq_len"] = int(cfg.get("seq_len", 16))
    cfg["batch_size"] = int(cfg.get("batch_size", 2))
    cfg["lr"] = float(cfg.get("lr", 3e-4))

    tokenizer_train = cfg.get("tokenizer_train")
    if not isinstance(tokenizer_train, dict):
        tokenizer_train = {}
        cfg["tokenizer_train"] = tokenizer_train
    tokenizer_train["path"] = str(dataset_path)
    tokenizer_train["vocab_size"] = int(tokenizer_train.get("vocab_size", cfg["vocab_size"]))
    tokenizer_train["save_path"] = str(checkpoint_dir / "tokenizer.json")
    cfg.pop("tokenizer_path", None)

    return cfg


def find_last_grad(entry: dict[str, Any]) -> float:
    raw = entry.get("grad_norm")
    if raw is None:
        return float(entry.get("loss", 0.0))
    return float(raw)


def run_multi() -> int:
    if not env_flag("ENKAI_RUN_MULTI_GPU_TESTS"):
        print("SKIPPED: ENKAI_RUN_MULTI_GPU_TESTS not set to 1")
        return 0
    if not env_flag("ENKAI_SINGLE_GPU_GREEN"):
        print("SKIPPED: single-GPU gate not marked green (set ENKAI_SINGLE_GPU_GREEN=1 after soak pass)")
        return 0
    if not env_flag("ENKAI_ENABLE_DIST"):
        print("SKIPPED: ENKAI_ENABLE_DIST not set to 1")
        return 0
    if shutil.which("nvidia-smi") is None:
        print("SKIPPED: nvidia-smi not available")
        return 0
    count = gpu_count()
    if count < 2:
        print("SKIPPED: fewer than 2 GPUs detected")
        return 0

    enkai = resolve_enkai()
    tensor = resolve_tensor_lib()
    base_config = pathlib.Path(os.environ.get("ENKAI_DP_BASE_CONFIG", "configs/enkai_50m.enk"))
    if not base_config.is_absolute():
        base_config = (ROOT / base_config).resolve()
    if not base_config.is_file():
        print(f"FAIL: base config not found: {base_config}")
        return 1

    work_dir = pathlib.Path(os.environ.get("ENKAI_DP_WORKDIR", "tmp/dp_harness"))
    if not work_dir.is_absolute():
        work_dir = (ROOT / work_dir).resolve()
    artifacts_dir = pathlib.Path(os.environ.get("ENKAI_GPU_ARTIFACT_DIR", "artifacts/gpu"))
    if not artifacts_dir.is_absolute():
        artifacts_dir = (ROOT / artifacts_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    steps = int(os.environ.get("ENKAI_DP_STEPS", "20"))
    timeout_sec = int(os.environ.get("ENKAI_DP_TIMEOUT_SEC", "1800"))
    tol_loss = float(os.environ.get("ENKAI_DP_LOSS_TOL", "0.05"))
    tol_grad = float(os.environ.get("ENKAI_DP_GRAD_TOL", "0.0001"))
    backend = os.environ.get("ENKAI_DP_BACKEND", "native")
    quick_model = env_flag("ENKAI_DP_QUICK_MODEL", default=True)

    dataset = work_dir / "deterministic.txt"
    build_dataset(dataset)
    base_json = migrate_base_config(enkai, base_config, work_dir / "base_config.json")

    baseline_cfg = patch_train_config(
        base_json,
        dataset_path=dataset,
        checkpoint_dir=work_dir / "baseline_ckpt",
        world_size=1,
        rank=0,
        steps=steps,
        backend=backend,
        quick_model=quick_model,
    )
    rank0_cfg = patch_train_config(
        base_json,
        dataset_path=dataset,
        checkpoint_dir=work_dir / "rank0_ckpt",
        world_size=2,
        rank=0,
        steps=steps,
        backend=backend,
        quick_model=quick_model,
    )
    rank1_cfg = patch_train_config(
        base_json,
        dataset_path=dataset,
        checkpoint_dir=work_dir / "rank1_ckpt",
        world_size=2,
        rank=1,
        steps=steps,
        backend=backend,
        quick_model=quick_model,
    )

    baseline_path = work_dir / "baseline.enk"
    rank0_path = work_dir / "rank0.enk"
    rank1_path = work_dir / "rank1.enk"
    write_enkai_config(baseline_path, baseline_cfg)
    write_enkai_config(rank0_path, rank0_cfg)
    write_enkai_config(rank1_path, rank1_cfg)

    common_env = os.environ.copy()
    common_env["ENKAI_ENABLE_DIST"] = "1"
    if tensor is not None:
        common_env["ENKAI_TENSOR_PATH"] = str(tensor)

    base_stdout = artifacts_dir / "multi_gpu_baseline.stdout.log"
    base_stderr = artifacts_dir / "multi_gpu_baseline.stderr.log"
    code, baseline_runtime = run_checked(
        [str(enkai), "train", str(baseline_path)],
        env=common_env,
        cwd=ROOT,
        stdout_path=base_stdout,
        stderr_path=base_stderr,
        timeout_sec=timeout_sec,
    )
    if code != 0:
        print(f"FAIL: baseline command failed (exit={code})")
        return 1

    master_port = choose_master_port()
    dist_common = common_env.copy()
    dist_common.update(
        {
            "MASTER_ADDR": os.environ.get("ENKAI_MASTER_ADDR", "127.0.0.1"),
            "MASTER_PORT": str(master_port),
            "WORLD_SIZE": "2",
        }
    )
    rank_cmds = []
    for rank, cfg_path in [(0, rank0_path), (1, rank1_path)]:
        env = dist_common.copy()
        env["RANK"] = str(rank)
        env["LOCAL_RANK"] = str(rank)
        stdout_path = artifacts_dir / f"multi_gpu_rank{rank}.stdout.log"
        stderr_path = artifacts_dir / f"multi_gpu_rank{rank}.stderr.log"
        rank_cmds.append((rank, [str(enkai), "train", str(cfg_path)], env, stdout_path, stderr_path))

    exits, distributed_runtime = run_multi_rank(rank_cmds, cwd=ROOT, timeout_sec=timeout_sec)
    if exits.get(0, 1) != 0 or exits.get(1, 1) != 0:
        print(f"FAIL: distributed launcher failed (rank exits={exits})")
        return 1

    baseline_log = work_dir / "baseline_ckpt" / "train_log.jsonl"
    rank0_log = work_dir / "rank0_ckpt" / "train_log.jsonl"
    rank1_log = work_dir / "rank1_ckpt" / "train_log.jsonl"
    baseline_last = read_last_jsonl(baseline_log)
    rank0_last = read_last_jsonl(rank0_log)
    rank1_last = read_last_jsonl(rank1_log)

    copy_if_exists(baseline_log, work_dir / "baseline.jsonl")
    copy_if_exists(rank0_log, work_dir / "rank0.jsonl")
    copy_if_exists(rank1_log, work_dir / "rank1.jsonl")
    copy_if_exists(work_dir / "baseline.jsonl", artifacts_dir / "baseline.jsonl")
    copy_if_exists(work_dir / "rank0.jsonl", artifacts_dir / "rank0.jsonl")
    copy_if_exists(work_dir / "rank1.jsonl", artifacts_dir / "rank1.jsonl")

    base_loss = float(baseline_last.get("loss", 0.0))
    r0_loss = float(rank0_last.get("loss", 0.0))
    r1_loss = float(rank1_last.get("loss", 0.0))
    g0 = find_last_grad(rank0_last)
    g1 = find_last_grad(rank1_last)

    (work_dir / "rank0_grads.json").write_text(json.dumps([g0]), encoding="utf-8")
    (work_dir / "rank1_grads.json").write_text(json.dumps([g1]), encoding="utf-8")
    copy_if_exists(work_dir / "rank0_grads.json", artifacts_dir / "rank0_grads.json")
    copy_if_exists(work_dir / "rank1_grads.json", artifacts_dir / "rank1_grads.json")

    loss_ok = abs(r0_loss - base_loss) <= tol_loss and abs(r1_loss - base_loss) <= tol_loss
    grad_ok = abs(g0 - g1) <= tol_grad
    passed = loss_ok and grad_ok

    report = {
        "schema_version": 1,
        "gate": "multi_gpu_parity",
        "timestamp_utc": now_iso(),
        "status": "PASS" if passed else "FAIL",
        "world_size": 2,
        "backend": backend,
        "tolerances": {"loss": tol_loss, "grad": tol_grad},
        "checks": {"loss_parity": loss_ok, "grad_parity": grad_ok},
        "baseline": {
            "loss": base_loss,
            "grad_norm": baseline_last.get("grad_norm"),
            "step": baseline_last.get("step"),
            "log": str((artifacts_dir / "baseline.jsonl").resolve()),
            "runtime_sec": baseline_runtime,
        },
        "ranks": [
            {
                "rank": 0,
                "loss": r0_loss,
                "grad_norm": rank0_last.get("grad_norm"),
                "step": rank0_last.get("step"),
                "log": str((artifacts_dir / "rank0.jsonl").resolve()),
            },
            {
                "rank": 1,
                "loss": r1_loss,
                "grad_norm": rank1_last.get("grad_norm"),
                "step": rank1_last.get("step"),
                "log": str((artifacts_dir / "rank1.jsonl").resolve()),
            },
        ],
        "artifacts": {
            "rank0_grads": str((artifacts_dir / "rank0_grads.json").resolve()),
            "rank1_grads": str((artifacts_dir / "rank1_grads.json").resolve()),
            "stdout_rank0": str((artifacts_dir / "multi_gpu_rank0.stdout.log").resolve()),
            "stderr_rank0": str((artifacts_dir / "multi_gpu_rank0.stderr.log").resolve()),
            "stdout_rank1": str((artifacts_dir / "multi_gpu_rank1.stdout.log").resolve()),
            "stderr_rank1": str((artifacts_dir / "multi_gpu_rank1.stderr.log").resolve()),
        },
        "distributed_runtime_sec": distributed_runtime,
    }
    evidence_path = artifacts_dir / "multi_gpu_evidence.json"
    write_json(evidence_path, report)

    lines = [
        f"timestamp_utc: {report['timestamp_utc']}",
        f"status: {report['status']}",
        f"loss_parity: {loss_ok}",
        f"grad_parity: {grad_ok}",
        f"baseline_loss: {base_loss}",
        f"rank0_loss: {r0_loss}",
        f"rank1_loss: {r1_loss}",
        f"rank0_grad: {g0}",
        f"rank1_grad: {g1}",
        f"evidence_json: {evidence_path}",
    ]
    if passed:
        lines.append("PASS: 2-GPU DP correctness validated")
    else:
        lines.append("FAIL: 2-GPU DP correctness validation failed")
    multi_log = artifacts_dir / "multi_gpu.log"
    write_log(multi_log, lines)
    print("\n".join(lines))
    return 0 if passed else 1


def run_soak4() -> int:
    if not env_flag("ENKAI_RUN_MULTI_GPU_TESTS"):
        print("SKIPPED: ENKAI_RUN_MULTI_GPU_TESTS not set to 1")
        return 0
    if not env_flag("ENKAI_SINGLE_GPU_GREEN"):
        print("SKIPPED: single-GPU gate not marked green (set ENKAI_SINGLE_GPU_GREEN=1 after soak pass)")
        return 0
    if not env_flag("ENKAI_ENABLE_DIST"):
        print("SKIPPED: ENKAI_ENABLE_DIST not set to 1")
        return 0
    if shutil.which("nvidia-smi") is None:
        print("SKIPPED: nvidia-smi not available")
        return 0
    count = gpu_count()
    if count < 4:
        print("SKIPPED: fewer than 4 GPUs detected")
        return 0

    artifacts_dir = pathlib.Path(os.environ.get("ENKAI_GPU_ARTIFACT_DIR", "artifacts/gpu"))
    if not artifacts_dir.is_absolute():
        artifacts_dir = (ROOT / artifacts_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    min_hours = float(os.environ.get("ENKAI_4GPU_MIN_HOURS", "3"))
    timeout_sec = int(os.environ.get("ENKAI_4GPU_TIMEOUT_SEC", str(max(3600, int(min_hours * 7200)))))
    launcher = os.environ.get("ENKAI_4GPU_LAUNCH_CMD", "").strip()
    start = time.time()
    status = "FAIL"
    failure_reason = ""

    if launcher:
        proc = subprocess.run(launcher, cwd=str(ROOT), shell=True, env=os.environ.copy(), text=True)
        if proc.returncode != 0:
            failure_reason = f"launcher exited non-zero ({proc.returncode})"
    else:
        enkai = resolve_enkai()
        tensor = resolve_tensor_lib()
        base_config = pathlib.Path(os.environ.get("ENKAI_4GPU_BASE_CONFIG", "configs/enkai_50m.enk"))
        if not base_config.is_absolute():
            base_config = (ROOT / base_config).resolve()
        if not base_config.is_file():
            failure_reason = f"base config not found: {base_config}"
        else:
            work_dir = pathlib.Path(os.environ.get("ENKAI_4GPU_WORKDIR", "tmp/soak4"))
            if not work_dir.is_absolute():
                work_dir = (ROOT / work_dir).resolve()
            work_dir.mkdir(parents=True, exist_ok=True)
            dataset = work_dir / "deterministic.txt"
            build_dataset(dataset, count=400)
            base_json = migrate_base_config(enkai, base_config, work_dir / "base_config.json")
            steps = int(os.environ.get("ENKAI_4GPU_STEPS", "200"))
            backend = os.environ.get("ENKAI_4GPU_BACKEND", "native")
            quick_model = env_flag("ENKAI_4GPU_QUICK_MODEL", default=False)
            rank_cfgs: list[pathlib.Path] = []
            for rank in range(4):
                cfg = patch_train_config(
                    base_json,
                    dataset_path=dataset,
                    checkpoint_dir=work_dir / f"rank{rank}_ckpt",
                    world_size=4,
                    rank=rank,
                    steps=steps,
                    backend=backend,
                    quick_model=quick_model,
                )
                cfg_path = work_dir / f"rank{rank}.enk"
                write_enkai_config(cfg_path, cfg)
                rank_cfgs.append(cfg_path)

            master_port = choose_master_port()
            common_env = os.environ.copy()
            common_env["ENKAI_ENABLE_DIST"] = "1"
            common_env["MASTER_ADDR"] = os.environ.get("ENKAI_MASTER_ADDR", "127.0.0.1")
            common_env["MASTER_PORT"] = str(master_port)
            common_env["WORLD_SIZE"] = "4"
            if tensor is not None:
                common_env["ENKAI_TENSOR_PATH"] = str(tensor)

            rank_cmds = []
            for rank, cfg in enumerate(rank_cfgs):
                env = common_env.copy()
                env["RANK"] = str(rank)
                env["LOCAL_RANK"] = str(rank)
                rank_cmds.append(
                    (
                        rank,
                        [str(enkai), "train", str(cfg)],
                        env,
                        artifacts_dir / f"soak4_rank{rank}.stdout.log",
                        artifacts_dir / f"soak4_rank{rank}.stderr.log",
                    )
                )
            try:
                exits, _runtime = run_multi_rank(rank_cmds, cwd=ROOT, timeout_sec=timeout_sec)
                if any(exits.get(rank, 1) != 0 for rank in range(4)):
                    failure_reason = f"distributed ranks failed: {exits}"
            except RuntimeError as err:
                failure_reason = str(err)

    runtime_hours = (time.time() - start) / 3600.0
    if not failure_reason and runtime_hours >= min_hours:
        status = "PASS"
    elif not failure_reason:
        failure_reason = f"runtime too short ({runtime_hours:.2f}h < {min_hours:.2f}h)"

    report = {
        "schema_version": 1,
        "gate": "soak_4gpu",
        "timestamp_utc": now_iso(),
        "status": status,
        "runtime_hours": round(runtime_hours, 4),
        "min_hours": min_hours,
        "launcher_mode": "custom" if launcher else "first_party",
        "failure_reason": failure_reason or None,
    }
    evidence_path = artifacts_dir / "soak_4gpu_evidence.json"
    write_json(evidence_path, report)

    lines = [
        f"timestamp_utc: {report['timestamp_utc']}",
        f"status: {status}",
        f"runtime_hours: {runtime_hours:.2f}",
        f"min_hours: {min_hours:.2f}",
        f"launcher_mode: {report['launcher_mode']}",
        f"evidence_json: {evidence_path}",
    ]
    if failure_reason:
        lines.append(f"failure_reason: {failure_reason}")
    if status == "PASS":
        lines.append(f"PASS: 4-GPU soak completed ({runtime_hours:.2f}h)")
    else:
        lines.append("FAIL: 4-GPU soak did not complete")
    soak_log = artifacts_dir / "soak_4gpu.log"
    write_log(soak_log, lines)
    print("\n".join(lines))
    return 0 if status == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Enkai GPU harness utilities")
    parser.add_argument("mode", choices=["multi", "soak4"], help="Harness mode to execute")
    args = parser.parse_args()
    if args.mode == "multi":
        return run_multi()
    return run_soak4()


if __name__ == "__main__":
    sys.exit(main())
