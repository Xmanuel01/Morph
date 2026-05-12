#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_command(command: list[str], cwd: Path, env: dict[str, str] | None = None, timeout: int = 300) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
        return {
            "command": command,
            "exit_code": result.returncode,
            "passed": result.returncode == 0,
            "elapsed_ms": max(1, int((time.perf_counter() - started) * 1000)),
            "stdout_tail": result.stdout[-8000:],
            "stderr_tail": result.stderr[-8000:],
        }
    except Exception as exc:
        return {
            "command": command,
            "exit_code": 1,
            "passed": False,
            "elapsed_ms": max(1, int((time.perf_counter() - started) * 1000)),
            "stdout_tail": "",
            "stderr_tail": repr(exc),
        }


def source_backend_catalog_checks(root: Path) -> dict[str, Any]:
    lib = (root / "enkai_tensor" / "src" / "lib.rs").read_text(encoding="utf-8-sig")
    backend = (root / "enkai_tensor" / "src" / "backend.rs").read_text(encoding="utf-8-sig")
    test = (root / "enkai_tensor" / "tests" / "cuda_llm_foundation.rs").read_text(encoding="utf-8-sig")
    cuda_source = (root / "enkai_tensor" / "cuda" / "enkai_kernels.cu").read_text(encoding="utf-8-sig")
    rocm_source = (root / "enkai_tensor" / "rocm" / "enkai_kernels.hip.cpp").read_text(encoding="utf-8-sig")
    metal_source = (root / "enkai_tensor" / "metal" / "enkai_kernels.metal").read_text(encoding="utf-8-sig")
    cuda_manifest = (root / "enkai_tensor" / "src" / "cuda_kernels.rs").read_text(encoding="utf-8-sig")
    accel_manifest = (root / "enkai_tensor" / "src" / "accelerator_kernels.rs").read_text(encoding="utf-8-sig")
    build_rs = (root / "enkai_tensor" / "build.rs").read_text(encoding="utf-8-sig")
    tensor_cargo = (root / "enkai_tensor" / "Cargo.toml").read_text(encoding="utf-8-sig")
    tokenizer = (root / "enkairt" / "src" / "tokenizer.rs").read_text(encoding="utf-8-sig")
    dataset = (root / "enkairt" / "src" / "dataset.rs").read_text(encoding="utf-8-sig")
    vm = (root / "enkairt" / "src" / "vm.rs").read_text(encoding="utf-8-sig")
    api_lock = (root / "enkai" / "contracts" / "v3_9_0_package_model_api_lock.json").read_text(encoding="utf-8-sig")
    dist_contract = root / "enkai" / "contracts" / "v3_9_0_distributed_gpu_execution.json"
    dist_readiness = root / "scripts" / "readiness_v3_9_0_distributed_gpu_execution.py"
    dist_verify = root / "scripts" / "verify_v3_9_0_distributed_gpu_execution.py"
    gpu_proof_ps = (root / "scripts" / "run_v3_9_0_gpu_proof.ps1").read_text(encoding="utf-8-sig")
    gpu_proof_sh = (root / "scripts" / "run_v3_9_0_gpu_proof.sh").read_text(encoding="utf-8-sig")
    checks = {
        "catalog_abi_present": "enkai_backend_catalog" in lib,
        "cuda_backend_kind_present": "Cuda" in backend,
        "rocm_reserved_error_present": "rocm" in lib and "E_BACKEND_FEATURE_MISSING" in lib and "E_BACKEND_UNAVAILABLE" in lib,
        "metal_reserved_error_present": "metal" in lib and "E_BACKEND_FEATURE_MISSING" in lib and "E_BACKEND_UNAVAILABLE" in lib,
        "cuda_unavailable_error_present": "E_BACKEND_UNAVAILABLE" in lib,
        "enkai_cuda_harness_present": "ENKAI_CUDA_LLM_METRICS" in test,
        "first_party_cuda_source_present": "enkai_cuda_bias_gelu_f32" in cuda_source and "enkai_cuda_softmax_f32" in cuda_source,
        "fusion_kernel_source_present": all(symbol in cuda_source for symbol in [
            "enkai_cuda_matmul_bias_f32",
            "enkai_cuda_masked_softmax_f32",
            "enkai_cuda_adamw_update_f32",
        ]),
        "cuda_bounded_llm_kernel_set_source_complete": all(symbol in cuda_source and symbol in cuda_manifest for symbol in [
            "enkai_cuda_vec_add_f16",
            "enkai_cuda_vec_add_bf16",
            "enkai_cuda_matmul_bias_f16",
            "enkai_cuda_matmul_bias_bf16",
            "enkai_cuda_cross_entropy_forward_f32",
            "enkai_cuda_cross_entropy_backward_f32",
            "enkai_cuda_embedding_forward_f32",
            "enkai_cuda_embedding_backward_f32",
            "enkai_cuda_causal_attention_backward_f32",
            "enkai_cuda_clip_grad_norm_f32",
        ]) and "missing_for_full_llm_training" not in cuda_manifest,
        "kv_cache_attention_source_present": all(symbol in cuda_source and symbol in cuda_manifest for symbol in [
            "enkai_cuda_causal_attention_prefill_f32",
            "enkai_cuda_kv_cache_decode_f32",
        ]),
        "cuda_feature_requires_nvcc": "CARGO_FEATURE_CUDA_KERNELS" in build_rs and "nvcc" in build_rs,
        "cuda_kernel_manifest_abi_present": "enkai_cuda_kernel_manifest" in lib and "kernel_manifest" in cuda_manifest,
        "mixed_precision_policy_abi_present": "enkai_mixed_precision_policy" in lib and "mixed_precision_policy" in cuda_manifest,
        "gpu_memory_planner_abi_present": "enkai_gpu_memory_planner_policy" in lib and "enkai_gpu_memory_plan" in lib and "memory_plan_from_json" in cuda_manifest,
        "gpu_memory_deterministic_errors_present": all(code in cuda_manifest for code in [
            "E_GPU_PLAN_INVALID_SPEC",
            "E_GPU_PLAN_BUDGET_EXCEEDED",
            "E_GPU_PLAN_UNSUPPORTED_DTYPE",
        ]),
        "kv_cache_policy_abi_present": "enkai_kv_cache_attention_policy" in lib and "enkai_kv_cache_plan" in lib and "kv_cache_plan_from_json" in cuda_manifest,
        "large_checkpoint_policy_abi_present": "enkai_large_checkpoint_format_policy" in lib and "enkai_large_checkpoint_plan" in lib and "large_checkpoint_plan_from_json" in cuda_manifest,
        "distributed_training_policy_abi_present": "enkai_distributed_training_policy" in lib and "enkai_distributed_training_plan" in lib and "distributed_training_plan_from_json" in cuda_manifest,
        "kv_checkpoint_dist_errors_present": all(code in cuda_manifest for code in [
            "E_KV_BUDGET_EXCEEDED",
            "E_CKPT_HASH_MISMATCH",
            "E_DIST_BUCKET_TOO_SMALL",
        ]),
        "amp_deterministic_errors_present": all(code in lib or code in cuda_manifest for code in [
            "E_AMP_INVALID_SCALE",
            "E_AMP_INVALID_GROWTH_FACTOR",
            "E_AMP_INVALID_BACKOFF_FACTOR",
            "E_AMP_INVALID_GROWTH_INTERVAL",
            "E_AMP_NONFINITE_GRADIENT",
        ]),
        "tokenizer_provenance_present": "vocab_sha1" in tokenizer and "Tokenizer fingerprint mismatch" in tokenizer and "tokenizer.provenance" in vm,
        "dataset_cursor_replay_present": "DatasetCursor" in dataset and "restore_cursor" in dataset and "dataset.cursor" in vm,
        "dataset_pipeline_manifest_present": "dataset_pipeline_manifest" in dataset and "dataset.manifest" in vm and "tokenizer_sha1" in dataset,
        "rocm_metal_feature_surfaces_present": "rocm-kernels" in tensor_cargo and "metal-kernels" in tensor_cargo and "enkai_accelerator_backend_policy" in lib,
        "rocm_real_backend_source_present": all(symbol in rocm_source and symbol in accel_manifest for symbol in [
            "enkai_rocm_vec_add_f32",
            "enkai_rocm_matmul_bias_f32",
            "enkai_rocm_cross_entropy_backward_f32",
            "enkai_rocm_embedding_backward_f32",
            "enkai_rocm_clip_grad_norm_f32",
        ]) and "hipLaunchKernelGGL" in rocm_source and "hipGetDeviceCount" in accel_manifest and "hipcc" in build_rs,
        "metal_real_backend_source_present": all(symbol in metal_source and symbol in accel_manifest for symbol in [
            "enkai_metal_vec_add_f32",
            "enkai_metal_matmul_bias_f32",
            "enkai_metal_cross_entropy_backward_f32",
            "enkai_metal_embedding_backward_f32",
            "enkai_metal_clip_grad_norm_f32",
        ]) and "atomic_fetch_add_explicit" in metal_source and "xcrun" in build_rs and "Metal.framework" in accel_manifest,
        "stable_package_model_api_lock_present": "std::tensor" in api_lock and "std::model" in api_lock and (root / "scripts" / "verify_v3_9_0_package_model_api_lock.py").exists(),
        "distributed_gpu_execution_proof_surface_present": dist_contract.exists()
            and dist_readiness.exists()
            and dist_verify.exists()
            and "RunDistributedGpuProof" in gpu_proof_ps
            and "RUN_DISTRIBUTED_GPU_PROOF" in gpu_proof_sh,
    }
    checks["passed"] = all(checks.values())
    return checks


def python_cmd(value: str | None) -> list[str]:
    if value:
        return shlex.split(value, posix=False)
    env_value = os.environ.get("ENKAI_PYTORCH_PYTHON")
    if env_value:
        return shlex.split(env_value, posix=False)
    if os.name == "nt":
        return ["py", "-3.11"]
    return [sys.executable]


def run_python_json(py_cmd: list[str], code: str, cwd: Path, timeout: int = 300) -> dict[str, Any]:
    command = py_cmd + ["-"]
    result = run_command(command, cwd, env=os.environ.copy(), timeout=timeout)
    if code:
        started = time.perf_counter()
        try:
            proc = subprocess.run(command, cwd=cwd, input=code, capture_output=True, text=True, timeout=timeout)
            result = {
                "command": command,
                "exit_code": proc.returncode,
                "passed": proc.returncode == 0,
                "elapsed_ms": max(1, int((time.perf_counter() - started) * 1000)),
                "stdout_tail": proc.stdout[-16000:],
                "stderr_tail": proc.stderr[-16000:],
            }
        except Exception as exc:
            result = {
                "command": command,
                "exit_code": 1,
                "passed": False,
                "elapsed_ms": max(1, int((time.perf_counter() - started) * 1000)),
                "stdout_tail": "",
                "stderr_tail": repr(exc),
            }
    try:
        payload = json.loads(result["stdout_tail"].strip().splitlines()[-1])
    except Exception as exc:
        payload = {"passed": False, "available": False, "cuda_available": False, "error": f"json parse failed: {exc}", "raw": result}
    payload["process"] = result
    return payload


def resolve_python(py_cmd: list[str], cwd: Path) -> dict[str, Any]:
    code = "import json, sys; print(json.dumps({'executable': sys.executable, 'version': sys.version}))"
    return run_python_json(py_cmd, code, cwd, timeout=30)


def torch_lib_dir(py_cmd: list[str], cwd: Path) -> dict[str, Any]:
    code = "import json, os, torch; print(json.dumps({'torch_version': torch.__version__, 'lib': os.path.join(os.path.dirname(torch.__file__), 'lib'), 'cuda_available': torch.cuda.is_available(), 'cuda_count': torch.cuda.device_count()}))"
    return run_python_json(py_cmd, code, cwd, timeout=30)


def run_pytorch_reference(py_cmd: list[str], model_cfg: dict[str, int], work_dir: Path, cwd: Path) -> dict[str, Any]:
    code = r'''
import hashlib, json, math, os, shutil, time
from pathlib import Path
result = {"available": False, "cuda_available": False, "passed": False, "error": None}
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as exc:
    result["error"] = f"pytorch import failed: {exc}"
    print(json.dumps(result))
    raise SystemExit(0)
result["available"] = True
result["torch_version"] = getattr(torch, "__version__", "unknown")
result["cuda_available"] = bool(torch.cuda.is_available())
result["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
if not torch.cuda.is_available():
    result["error"] = "CUDA is not available to PyTorch"
    print(json.dumps(result))
    raise SystemExit(0)
model_cfg = json.loads(r''' + repr(json.dumps(model_cfg)) + r''')
work_dir = Path(r''' + repr(str(work_dir)) + r''')
work_dir.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda:0")
torch.cuda.set_device(device)
torch.manual_seed(1337)
torch.cuda.manual_seed_all(1337)
torch.use_deterministic_algorithms(True, warn_only=True)
vocab = int(model_cfg["vocab_size"]); hidden = int(model_cfg["hidden_size"])
layers = int(model_cfg["layers"]); heads = int(model_cfg["heads"])
batch = int(model_cfg["batch_size"]); seq = int(model_cfg["seq_len"])
train_steps = int(model_cfg["train_steps"]); eval_steps = int(model_cfg["eval_steps"])
class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, hidden * 3)
        self.proj = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden * 4)
        self.fc2 = nn.Linear(hidden * 4, hidden)
    def forward(self, x, causal_mask):
        bsz, seq_len, _ = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(bsz, seq_len, 3, heads, hidden // heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) * ((hidden // heads) ** -0.5)
        att = att.masked_fill(~causal_mask, float("-inf")).softmax(dim=-1)
        ctx = (att @ v).transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        x = x + self.proj(ctx)
        h = self.ln2(x)
        x = x + self.fc2(F.gelu(self.fc1(h), approximate="none"))
        return x
class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.pos = nn.Parameter(torch.zeros(seq, hidden))
        self.blocks = nn.ModuleList([TinyBlock() for _ in range(layers)])
        self.ln = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab, bias=True)
    def forward(self, tokens):
        x = self.embed(tokens) + self.pos[:tokens.shape[1]].unsqueeze(0)
        causal_mask = torch.tril(torch.ones(tokens.shape[1], tokens.shape[1], device=tokens.device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
        for block in self.blocks:
            x = block(x, causal_mask)
        return self.head(self.ln(x))
model = TinyDecoder().to(device)
with torch.no_grad():
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    nn.init.normal_(model.pos, mean=0.0, std=0.02)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False, fused=False)
data = torch.randint(0, vocab, (batch, seq), device=device)
target = torch.roll(data, shifts=-1, dims=1)
torch.cuda.synchronize(device)
torch.cuda.reset_peak_memory_stats(device)
losses = []
train_start = time.perf_counter()
for _ in range(train_steps):
    opt.zero_grad(set_to_none=True)
    logits = model(data)
    if not torch.isfinite(logits).all().item():
        raise RuntimeError("pytorch reference logits contain NaN/Inf")
    loss = F.cross_entropy(logits.reshape(-1, vocab), target.reshape(-1))
    if not torch.isfinite(loss).all().item():
        raise RuntimeError("pytorch reference loss is NaN/Inf")
    loss.backward()
    grad_l2 = 0.0
    param_l2 = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            raise RuntimeError("pytorch reference missing gradient")
        if not torch.isfinite(parameter.grad).all().item():
            raise RuntimeError("pytorch reference gradient contains NaN/Inf")
        if not torch.isfinite(parameter).all().item():
            raise RuntimeError("pytorch reference parameter contains NaN/Inf")
        grad_l2 += float(parameter.grad.detach().pow(2).sum().cpu())
        param_l2 += float(parameter.detach().pow(2).sum().cpu())
    if not math.isfinite(grad_l2) or not math.isfinite(param_l2):
        raise RuntimeError("pytorch reference norm accounting produced NaN/Inf")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    losses.append(float(loss.detach().cpu()))
torch.cuda.synchronize(device)
train_elapsed = max(time.perf_counter() - train_start, 1e-9)
eval_start = time.perf_counter(); checksum = 0.0
with torch.no_grad():
    for _ in range(eval_steps):
        eval_logits = model(data)
        if not torch.isfinite(eval_logits).all().item():
            raise RuntimeError("pytorch reference eval logits contain NaN/Inf")
        eval_norm = float(eval_logits.detach().pow(2).sum().cpu())
        if not math.isfinite(eval_norm):
            raise RuntimeError("pytorch reference eval norm accounting produced NaN/Inf")
        checksum += float(eval_logits.sum().detach().cpu())
torch.cuda.synchronize(device)
eval_elapsed = max(time.perf_counter() - eval_start, 1e-9)
checkpoint_dir = work_dir / "pytorch_reference_checkpoint"
tmp_dir = checkpoint_dir / ".tmp"
final_dir = checkpoint_dir / "latest"
if tmp_dir.exists():
    shutil.rmtree(tmp_dir)
if final_dir.exists():
    shutil.rmtree(final_dir)
tmp_dir.mkdir(parents=True, exist_ok=True)
model_path = tmp_dir / "params.pt"
optim_path = tmp_dir / "optim_rank0.pt"
meta_path = tmp_dir / "meta.json"
cursor_path = tmp_dir / "data_cursor.json"
rng_path = tmp_dir / "rng_state.pt"
cuda_rng_path = tmp_dir / "cuda_rng_state.pt"
integrity = tmp_dir / "integrity.json"
tensor_index_path = tmp_dir / "tensor_index.json"
manifest_path = tmp_dir / "manifest.json"
provenance_path = tmp_dir / "provenance.json"
ckpt_started = time.perf_counter()
torch.save(model.state_dict(), model_path)
torch.save(opt.state_dict(), optim_path)
torch.save(torch.get_rng_state(), rng_path)
torch.save(torch.cuda.get_rng_state_all(), cuda_rng_path)
tensor_index = []
for name, tensor in model.state_dict().items():
    cpu_tensor = tensor.detach().cpu().contiguous()
    tensor_index.append({
        "name": name,
        "shape": list(cpu_tensor.shape),
        "dtype": str(cpu_tensor.dtype),
        "sha256": hashlib.sha256(cpu_tensor.numpy().tobytes()).hexdigest(),
    })
for state_key, state in opt.state_dict().get("state", {}).items():
    for slot_name, value in state.items():
        if torch.is_tensor(value):
            cpu_tensor = value.detach().cpu().contiguous()
            tensor_index.append({
                "name": f"optimizer.{state_key}.{slot_name}",
                "shape": list(cpu_tensor.shape),
                "dtype": str(cpu_tensor.dtype),
                "sha256": hashlib.sha256(cpu_tensor.numpy().tobytes()).hexdigest(),
            })
tensor_index_path.write_text(json.dumps({"version": 1, "entries": tensor_index}, sort_keys=True), encoding="utf-8")
manifest_path.write_text(json.dumps({
    "format_version": 1,
    "checkpoint_kind": "pytorch_deterministic_reference",
    "files": {
        "params": model_path.name,
        "optimizer": optim_path.name,
        "rng": rng_path.name,
        "cuda_rng": cuda_rng_path.name,
        "data_cursor": cursor_path.name,
        "tensor_index": tensor_index_path.name,
    },
    "compatibility": {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "optimizer_mode": "adamw_foreach_false_fused_false",
        "requires_tensor_hash_validation": True,
    },
}, sort_keys=True), encoding="utf-8")
provenance_path.write_text(json.dumps({
    "schema_version": 1,
    "created_by": "v3_9_0_cuda_llm_runtime_foundation",
    "reference_backend": "pytorch_cuda",
    "safety_checks": [
        "loss_finite",
        "logits_finite",
        "gradient_finite_per_parameter",
        "parameter_finite_per_parameter",
        "tensor_index_sha256",
        "file_integrity_sha256",
        "atomic_directory_rename"
    ],
    "hardware": {
        "device": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
    },
}, sort_keys=True), encoding="utf-8")
meta_path.write_text(json.dumps({"format_version": 1, "rng_state_file": rng_path.name, "cuda_rng_state_file": cuda_rng_path.name, "model_cfg": model_cfg, "optimizer_mode": "adamw_foreach_false_fused_false", "tensor_index_file": tensor_index_path.name}), encoding="utf-8")
cursor_path.write_text(json.dumps({"dataset": "synthetic_bounded_transformer", "batch": 0, "seq": int(seq), "replay_seed": 1337}), encoding="utf-8")
for path in [model_path, optim_path, rng_path, cuda_rng_path, meta_path, cursor_path, tensor_index_path, manifest_path, provenance_path]:
    with open(path, "rb") as f:
        os.fsync(f.fileno())
hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in [model_path, optim_path, rng_path, cuda_rng_path, meta_path, cursor_path, tensor_index_path, manifest_path, provenance_path]}
integrity.write_text(json.dumps({"version": 1, "files": hashes}, sort_keys=True), encoding="utf-8")
with open(integrity, "rb") as f:
    os.fsync(f.fileno())
dir_fd = os.open(tmp_dir, os.O_RDONLY)
try:
    os.fsync(dir_fd)
finally:
    os.close(dir_fd)
os.rename(tmp_dir, final_dir)
dir_fd = os.open(checkpoint_dir, os.O_RDONLY)
try:
    os.fsync(dir_fd)
finally:
    os.close(dir_fd)
ckpt_write_elapsed = max(time.perf_counter() - ckpt_started, 1e-9)
ckpt_files = [p for p in final_dir.iterdir() if p.is_file()]
ckpt_bytes = sum(p.stat().st_size for p in ckpt_files)
resume_started = time.perf_counter()
loaded_model = torch.load(final_dir / "params.pt", map_location=device)
loaded_optim = torch.load(final_dir / "optim_rank0.pt", map_location=device)
loaded_rng = torch.load(final_dir / "rng_state.pt", map_location="cpu")
loaded_cuda_rng = torch.load(final_dir / "cuda_rng_state.pt", map_location=device)
loaded_meta = json.loads((final_dir / "meta.json").read_text(encoding="utf-8"))
loaded_integrity = json.loads((final_dir / "integrity.json").read_text(encoding="utf-8"))
for name, expected_hash in loaded_integrity.get("files", {}).items():
    actual_hash = hashlib.sha256((final_dir / name).read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise RuntimeError(f"checkpoint hash mismatch: {name}")
resume_elapsed_ms = max(1, int((time.perf_counter() - resume_started) * 1000))
digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode("utf-8")).hexdigest()
result.update({"passed": True, "device": torch.cuda.get_device_name(0), "metrics": {"train_tokens_per_sec": (batch*seq*train_steps)/train_elapsed, "eval_tokens_per_sec": (batch*seq*eval_steps)/eval_elapsed, "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)), "checkpoint_write_bytes_per_sec": ckpt_bytes/ckpt_write_elapsed, "checkpoint_resume_ms": resume_elapsed_ms, "loss_initial": losses[0], "loss_final": losses[-1], "eval_checksum": checksum, "checkpoint_bytes": ckpt_bytes, "checkpoint_sha256": digest, "loaded_format_version": loaded_meta.get("format_version"), "loaded_model_tensors": len(loaded_model), "loaded_optimizer_entries": len(loaded_optim), "loaded_rng_bytes": int(loaded_rng.numel()), "loaded_cuda_rng_entries": len(loaded_cuda_rng)}})
print(json.dumps(result))
'''
    return run_python_json(py_cmd, code, cwd, timeout=600)


def parse_enkai_metrics(stdout: str) -> dict[str, Any]:
    for line in stdout.splitlines():
        if line.startswith("ENKAI_CUDA_LLM_METRICS="):
            return json.loads(line.split("=", 1)[1])
    return {"skipped": True, "reason": "metrics line missing"}


def run_enkai_cuda_reference(root: Path, py_info: dict[str, Any], torch_info: dict[str, Any]) -> dict[str, Any]:
    env = os.environ.copy()
    executable = py_info.get("executable")
    torch_lib = torch_info.get("lib")
    if executable:
        env["PYTHON_SYS_EXECUTABLE"] = executable
    env["LIBTORCH_USE_PYTORCH"] = "1"
    if torch_lib:
        env["PATH"] = f"{torch_lib}{os.pathsep}{env.get('PATH', '')}"
        env["LD_LIBRARY_PATH"] = f"{torch_lib}{os.pathsep}{env.get('LD_LIBRARY_PATH', '')}"
    if executable:
        conda_lib = str(Path(executable).resolve().parents[1] / "lib")
        env["LD_LIBRARY_PATH"] = f"{conda_lib}{os.pathsep}{env.get('LD_LIBRARY_PATH', '')}"
    command = ["cargo", "test", "--release", "-p", "enkai_tensor", "--features", "torch", "--test", "cuda_llm_foundation", "--", "--nocapture"]
    result = run_command(command, root, env=env, timeout=900)
    metrics = parse_enkai_metrics(result.get("stdout_tail", ""))
    return {
        "passed": bool(result["passed"] and not metrics.get("skipped") and metrics.get("train_tokens_per_sec", 0) > 0),
        "cargo": result,
        "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate v3.9.0 CUDA-first LLM runtime foundation evidence.")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--contract", default="enkai/contracts/v3_9_0_cuda_llm_runtime_foundation.json")
    parser.add_argument("--output", default="artifacts/readiness/v3_9_0_cuda_llm_runtime_foundation.json")
    parser.add_argument("--python", default=None, help="Python command for PyTorch, e.g. 'py -3.11'")
    args = parser.parse_args()

    root = Path(args.workspace).resolve()
    contract = read_json(root / args.contract)
    work_dir = root / "artifacts" / "v3_9_0_cuda_llm_runtime_foundation"
    work_dir.mkdir(parents=True, exist_ok=True)
    py_cmd = python_cmd(args.python)

    backend_checks = source_backend_catalog_checks(root)
    py_info = resolve_python(py_cmd, root)
    torch_info = torch_lib_dir(py_cmd, root)
    pytorch = run_pytorch_reference(py_cmd, contract["bounded_model"], work_dir, root)
    torch_ready = bool(torch_info.get("process", {}).get("passed") and torch_info.get("cuda_available"))
    enkai_cuda = run_enkai_cuda_reference(root, py_info, torch_info) if torch_ready else {"passed": False, "metrics": {"skipped": True, "reason": "torch unavailable"}, "cargo": None}

    failures: list[str] = []
    if not backend_checks["passed"]: failures.append("backend catalog source checks failed")
    if not pytorch.get("available"): failures.append("PyTorch reference baseline unavailable")
    if not pytorch.get("cuda_available"): failures.append("CUDA reference device unavailable")
    if not pytorch.get("passed"): failures.append("PyTorch CUDA bounded benchmark did not pass")
    if not enkai_cuda.get("passed"): failures.append("Enkai CUDA bounded benchmark did not pass")

    pytorch_metrics = pytorch.get("metrics", {}) if isinstance(pytorch.get("metrics"), dict) else {}
    enkai_metrics = enkai_cuda.get("metrics", {}) if isinstance(enkai_cuda.get("metrics"), dict) else {}
    for metric in contract["required_metrics"]:
        if pytorch_metrics.get(metric, 0) <= 0: failures.append(f"missing positive PyTorch metric: {metric}")
        if enkai_metrics.get(metric, 0) <= 0: failures.append(f"missing positive Enkai CUDA metric: {metric}")

    comparisons = {}
    if pytorch_metrics and enkai_metrics:
        for key in ["train_tokens_per_sec", "eval_tokens_per_sec", "peak_memory_bytes", "checkpoint_write_bytes_per_sec", "checkpoint_resume_ms"]:
            if pytorch_metrics.get(key, 0) > 0 and enkai_metrics.get(key, 0) > 0:
                comparisons[f"enkai_vs_pytorch_{key}_ratio"] = enkai_metrics[key] / pytorch_metrics[key]
        comparisons["loss_delta_abs"] = abs(float(enkai_metrics.get("loss_final", 0)) - float(pytorch_metrics.get("loss_final", 0)))

    performance_targets = contract.get("performance_targets", {})
    performance_gate = {"passed": True, "failures": []}
    for ratio, minimum in performance_targets.get("throughput_min_ratios", {}).items():
        value = comparisons.get(ratio)
        if not isinstance(value, (int, float)) or value < float(minimum):
            performance_gate["passed"] = False
            performance_gate["failures"].append({"metric": ratio, "expected": f">= {minimum}", "actual": value})
    for ratio, maximum in performance_targets.get("resource_max_ratios", {}).items():
        value = comparisons.get(ratio)
        if not isinstance(value, (int, float)) or value > float(maximum):
            performance_gate["passed"] = False
            performance_gate["failures"].append({"metric": ratio, "expected": f"<= {maximum}", "actual": value})
    loss_delta = comparisons.get("loss_delta_abs")
    loss_limit = performance_targets.get("quality_limits", {}).get("loss_delta_abs_max")
    if loss_limit is not None and (not isinstance(loss_delta, (int, float)) or loss_delta > float(loss_limit)):
        performance_gate["passed"] = False
        performance_gate["failures"].append({"metric": "loss_delta_abs", "expected": f"<= {loss_limit}", "actual": loss_delta})
    if not performance_gate["passed"]:
        failures.append("50% bounded performance/resource target gate did not pass")

    payload = {
        "schema_version": 1,
        "contract_version": contract["contract_version"],
        "scope": contract["scope"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {"platform": platform.platform(), "python_driver": platform.python_version(), "pytorch_python_command": py_cmd, "cwd": str(root)},
        "contract": str((root / args.contract).resolve()),
        "suite": "bench/suites/v3_9_0_cuda_llm_runtime_foundation.json",
        "backend_catalog_source_checks": backend_checks,
        "python_resolution": py_info,
        "torch_resolution": torch_info,
        "pytorch_reference": pytorch,
        "enkai_cuda_backend": enkai_cuda,
        "comparisons": comparisons,
        "performance_targets": performance_targets,
        "performance_gate": performance_gate,
        "production_claims": {"cuda_first_contract_frozen": backend_checks["passed"], "pytorch_reference_archived": bool(pytorch.get("passed")), "enkai_cuda_reference_archived": bool(enkai_cuda.get("passed")), "bounded_50_percent_performance_target_passed": bool(performance_gate["passed"]), "rocm_production_supported": False, "metal_production_supported": False, "full_pytorch_parity_claimed": False},
        "all_passed": not failures,
        "failures": failures,
    }
    write_json(root / args.output, payload)
    print(json.dumps({"all_passed": payload["all_passed"], "failures": failures, "output": args.output}, indent=2))
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
