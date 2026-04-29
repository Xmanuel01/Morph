#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify manifest-backed model/registry dispatch.")
    parser.add_argument("--enkai-bin", required=True)
    parser.add_argument("--contract", default="enkai/contracts/selfhost_model_registry_ops_v3_3_0.json")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)


def write_train_config(path: Path, payload: dict) -> None:
    escaped = json.dumps(payload).replace('\\', '\\\\').replace('"', '\\"')
    source = f'fn main() ::\n    return json.parse("{escaped}")\n::\n'
    path.write_text(source, encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    enkai_bin = Path(args.enkai_bin).resolve()
    contract_path = (root / args.contract).resolve() if not Path(args.contract).is_absolute() else Path(args.contract).resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    output_path = Path(args.output).resolve()

    checks = []
    all_passed = True
    with tempfile.TemporaryDirectory(prefix="enkai_model_registry_") as tmp:
        tmpdir = Path(tmp)
        registry = tmpdir / "registry"
        checkpoint = tmpdir / "checkpoint"
        checkpoint.mkdir(parents=True)
        manifest_dir = tmpdir / "manifests"
        manifest_dir.mkdir()

        register_manifest = manifest_dir / "model_register.json"
        register_proc = run([
            str(enkai_bin), "model", "manifest", "register",
            str(registry), "chat", "v1.0.0", str(checkpoint), "--activate",
            "--manifest-output", str(register_manifest),
        ], root)
        register_exec = run([str(enkai_bin), "model", "exec", "--manifest", str(register_manifest)], root)
        register_ok = register_proc.returncode == 0 and register_exec.returncode == 0 and register_manifest.is_file()
        checks.append({
            "id": "register",
            "manifest_returncode": register_proc.returncode,
            "exec_returncode": register_exec.returncode,
            "passed": register_ok,
        })
        all_passed &= register_ok

        load_manifest = manifest_dir / "model_load.json"
        load_proc = run([
            str(enkai_bin), "model", "manifest", "load",
            str(registry), "chat", "v1.0.0",
            "--manifest-output", str(load_manifest),
        ], root)
        load_exec = run([str(enkai_bin), "model", "exec", "--manifest", str(load_manifest)], root)
        load_ok = load_proc.returncode == 0 and load_exec.returncode == 0 and load_manifest.is_file()
        checks.append({
            "id": "load",
            "manifest_returncode": load_proc.returncode,
            "exec_returncode": load_exec.returncode,
            "passed": load_ok,
        })
        all_passed &= load_ok

        loaded_manifest = manifest_dir / "model_loaded.json"
        loaded_proc = run([
            str(enkai_bin), "model", "manifest", "loaded",
            str(registry), "chat", "--json",
            "--manifest-output", str(loaded_manifest),
        ], root)
        loaded_exec = run([str(enkai_bin), "model", "exec", "--manifest", str(loaded_manifest)], root)
        loaded_json_ok = False
        if loaded_exec.returncode == 0:
            try:
                payload = json.loads(loaded_exec.stdout)
                loaded_json_ok = "v1.0.0" in payload
            except Exception:
                loaded_json_ok = False
        loaded_ok = loaded_proc.returncode == 0 and loaded_exec.returncode == 0 and loaded_manifest.is_file() and loaded_json_ok
        checks.append({
            "id": "loaded",
            "manifest_returncode": loaded_proc.returncode,
            "exec_returncode": loaded_exec.returncode,
            "loaded_json_ok": loaded_json_ok,
            "passed": loaded_ok,
        })
        all_passed &= loaded_ok

        data = tmpdir / "data.txt"
        data.write_text("alpha beta gamma\ndelta epsilon\n", encoding="utf-8")
        ckpt = tmpdir / "train_ckpt"
        train_config = tmpdir / "train_config.enk"
        write_train_config(train_config, {
            "config_version": 1,
            "backend": "cpu",
            "vocab_size": 8,
            "hidden_size": 4,
            "seq_len": 4,
            "batch_size": 2,
            "lr": 0.1,
            "dataset_path": str(data),
            "checkpoint_dir": str(ckpt),
            "max_steps": 1,
            "save_every": 1,
            "log_every": 1,
            "drop_remainder": False,
            "tokenizer_train": {"path": str(data), "vocab_size": 8},
        })
        train_manifest = manifest_dir / "train_manifest.json"
        train_manifest_proc = run([
            str(enkai_bin), "train-manifest", "train", str(train_config),
            "--manifest-output", str(train_manifest),
        ], root)
        train_manifest_ok = False
        if train_manifest_proc.returncode == 0 and train_manifest.is_file():
            payload = json.loads(train_manifest.read_text(encoding="utf-8"))
            train_manifest_ok = payload.get("evaluated_config", {}).get("backend") == "cpu"
        train_exec = run([str(enkai_bin), "train-exec", "--manifest", str(train_manifest)], root)
        train_exec_ok = train_exec.returncode == 0 and any(ckpt.glob("step_00000001*"))
        checks.append({
            "id": "train",
            "manifest_returncode": train_manifest_proc.returncode,
            "exec_returncode": train_exec.returncode,
            "train_manifest_ok": train_manifest_ok,
            "train_exec_ok": train_exec_ok,
            "passed": train_manifest_ok and train_exec_ok,
        })
        all_passed &= train_manifest_ok and train_exec_ok

        native_ckpt = tmpdir / "train_native_ckpt"
        native_train_config = tmpdir / "train_native_config.enk"
        write_train_config(native_train_config, {
            "config_version": 1,
            "backend": "native",
            "vocab_size": 8,
            "hidden_size": 4,
            "seq_len": 4,
            "batch_size": 2,
            "lr": 0.1,
            "dataset_path": str(data),
            "checkpoint_dir": str(native_ckpt),
            "max_steps": 1,
            "save_every": 1,
            "log_every": 1,
            "drop_remainder": False,
            "tokenizer_train": {"path": str(data), "vocab_size": 8},
        })
        native_manifest = manifest_dir / "train_native_manifest.json"
        native_manifest_proc = run([
            str(enkai_bin), "train-manifest", "train", str(native_train_config),
            "--manifest-output", str(native_manifest),
        ], root)
        native_manifest_ok = False
        if native_manifest_proc.returncode == 0 and native_manifest.is_file():
            payload = json.loads(native_manifest.read_text(encoding="utf-8"))
            native_manifest_ok = payload.get("evaluated_config", {}).get("backend") == "native"
        native_exec = run([str(enkai_bin), "train-exec", "--manifest", str(native_manifest)], root)
        native_exec_ok = native_exec.returncode == 0 and any(native_ckpt.glob("step_00000001*"))
        native_stderr = (native_exec.stderr or "").strip()
        native_stdout = (native_exec.stdout or "").strip()
        fallback_note_ok = (
            contract.get("require_native_fallback_note", False)
            and "[train-runtime] native backend unavailable, falling back to cpu execution:" in native_stderr
        )
        native_passed = native_manifest_ok and native_exec_ok
        checks.append({
            "id": "train-native",
            "manifest_returncode": native_manifest_proc.returncode,
            "exec_returncode": native_exec.returncode,
            "native_manifest_ok": native_manifest_ok,
            "native_exec_ok": native_exec_ok,
            "fallback_note_ok": fallback_note_ok,
            "native_stdout": native_stdout,
            "native_stderr": native_stderr,
            "passed": native_passed,
        })
        all_passed &= native_passed

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": str(contract_path),
        "all_passed": all_passed,
        "checks": checks,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok" if all_passed else "failed", "output": str(output_path), "all_passed": all_passed}, separators=(",", ":")))
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
