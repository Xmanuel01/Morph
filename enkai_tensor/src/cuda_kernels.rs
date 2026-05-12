#![allow(dead_code)]

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelBuildStatus {
    Compiled,
    NotCompiled,
}

pub fn build_status() -> KernelBuildStatus {
    if cfg!(feature = "cuda-kernels") {
        KernelBuildStatus::Compiled
    } else {
        KernelBuildStatus::NotCompiled
    }
}

pub fn kernel_manifest() -> serde_json::Value {
    serde_json::json!({
        "schema_version": 1,
        "backend": "enkai_cuda_first_party",
        "build_status": if cfg!(feature = "cuda-kernels") { "compiled" } else { "not_compiled" },
        "source": "enkai_tensor/cuda/enkai_kernels.cu",
        "production_gate": {
            "requires_feature": "cuda-kernels",
            "requires_nvcc": true,
            "requires_cuda_hardware": true,
            "requires_pytorch_comparison": true,
            "claim_without_green_verifier": false
        },
        "kernels": [
            {"name": "vec_add", "symbol": "enkai_cuda_vec_add_f32", "dtype": "fp32", "class": "elementwise"},
            {"name": "vec_add", "symbol": "enkai_cuda_vec_add_f16", "dtype": "fp16", "class": "elementwise"},
            {"name": "vec_add", "symbol": "enkai_cuda_vec_add_bf16", "dtype": "bf16", "class": "elementwise"},
            {"name": "vec_mul", "symbol": "enkai_cuda_vec_mul_f32", "dtype": "fp32", "class": "elementwise"},
            {"name": "vec_mul", "symbol": "enkai_cuda_vec_mul_f16", "dtype": "fp16", "class": "elementwise"},
            {"name": "vec_mul", "symbol": "enkai_cuda_vec_mul_bf16", "dtype": "bf16", "class": "elementwise"},
            {"name": "vec_scale", "symbol": "enkai_cuda_vec_scale_f32", "dtype": "fp32", "class": "elementwise"},
            {"name": "vec_scale", "symbol": "enkai_cuda_vec_scale_f16", "dtype": "fp16", "class": "elementwise"},
            {"name": "vec_scale", "symbol": "enkai_cuda_vec_scale_bf16", "dtype": "bf16", "class": "elementwise"},
            {"name": "bias_gelu", "symbol": "enkai_cuda_bias_gelu_f32", "dtype": "fp32", "class": "fusion"},
            {"name": "bias_gelu", "symbol": "enkai_cuda_bias_gelu_f16", "dtype": "fp16", "class": "fusion"},
            {"name": "bias_gelu", "symbol": "enkai_cuda_bias_gelu_bf16", "dtype": "bf16", "class": "fusion"},
            {"name": "matmul_bias", "symbol": "enkai_cuda_matmul_bias_f32", "dtype": "fp32", "class": "fusion"},
            {"name": "matmul_bias", "symbol": "enkai_cuda_matmul_bias_cublas_f32", "dtype": "fp32", "class": "cublas_fusion"},
            {"name": "matmul_bias", "symbol": "enkai_cuda_matmul_bias_f16", "dtype": "fp16", "class": "fusion"},
            {"name": "matmul_bias", "symbol": "enkai_cuda_matmul_bias_bf16", "dtype": "bf16", "class": "fusion"},
            {"name": "layernorm", "symbol": "enkai_cuda_layernorm_f32", "dtype": "fp32", "class": "normalization"},
            {"name": "softmax", "symbol": "enkai_cuda_softmax_f32", "dtype": "fp32", "class": "attention_core"},
            {"name": "masked_softmax", "symbol": "enkai_cuda_masked_softmax_f32", "dtype": "fp32", "class": "attention_core"},
            {"name": "causal_attention_prefill", "symbol": "enkai_cuda_causal_attention_prefill_f32", "dtype": "fp32", "class": "attention_prefill"},
            {"name": "causal_attention_backward", "symbol": "enkai_cuda_causal_attention_backward_f32", "dtype": "fp32", "class": "attention_backward"},
            {"name": "causal_attention_backward_value", "symbol": "enkai_cuda_causal_attention_backward_value_f32", "dtype": "fp32", "class": "attention_backward"},
            {"name": "kv_cache_decode", "symbol": "enkai_cuda_kv_cache_decode_f32", "dtype": "fp32", "class": "attention_decode"},
            {"name": "cross_entropy_forward", "symbol": "enkai_cuda_cross_entropy_forward_f32", "dtype": "fp32", "class": "loss"},
            {"name": "cross_entropy_backward", "symbol": "enkai_cuda_cross_entropy_backward_f32", "dtype": "fp32", "class": "loss_backward"},
            {"name": "embedding_forward", "symbol": "enkai_cuda_embedding_forward_f32", "dtype": "fp32", "class": "embedding"},
            {"name": "embedding_backward", "symbol": "enkai_cuda_embedding_backward_f32", "dtype": "fp32", "class": "embedding_backward"},
            {"name": "adamw_update", "symbol": "enkai_cuda_adamw_update_f32", "dtype": "fp32", "class": "optimizer_fusion"},
            {"name": "clip_grad_norm", "symbol": "enkai_cuda_clip_grad_norm_f32", "dtype": "fp32", "class": "gradient_safety"}
        ],
        "bounded_llm_training_kernel_set": "source_complete_hardware_gated",
        "remaining_before_universal_pytorch_parity": [
            "Tensor Core tiled matmul kernels",
            "full q/k/v attention backward fusion",
            "multi-GPU collective CUDA/NCCL kernels",
            "hardware/PyTorch tolerance proof on a CUDA runner"
        ]
    })
}

fn get_u64(spec: &serde_json::Value, key: &str, default: u64, code: &str) -> Result<u64, String> {
    match spec.get(key) {
        Some(value) => value
            .as_u64()
            .ok_or_else(|| format!("{code}: {key} must be a non-negative integer")),
        None => Ok(default),
    }
}

fn dtype_bytes(dtype: &str, code: &str) -> Result<u64, String> {
    match dtype {
        "fp32" | "f32" => Ok(4),
        "fp16" | "f16" | "bf16" => Ok(2),
        _ => Err(format!("{code}: dtype must be fp32, fp16, or bf16")),
    }
}

fn round_up(bytes: u64, align: u64) -> u64 {
    if bytes == 0 {
        0
    } else {
        bytes.div_ceil(align) * align
    }
}

pub fn memory_planner_policy() -> serde_json::Value {
    serde_json::json!({
        "schema_version": 1,
        "policy": "enkai_gpu_memory_planner_v1",
        "status": "deterministic_foundation",
        "allocation_strategy": "static_lifetime_pool",
        "pool_bins_bytes": [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216],
        "alignment_bytes": 256,
        "deterministic_errors": [
            "E_GPU_PLAN_INVALID_SPEC",
            "E_GPU_PLAN_BUDGET_EXCEEDED",
            "E_GPU_PLAN_UNSUPPORTED_DTYPE"
        ],
        "tracks": [
            "parameter_bytes",
            "optimizer_bytes",
            "activation_bytes",
            "workspace_bytes",
            "kv_cache_bytes",
            "peak_estimated_bytes"
        ],
        "production_gate": {
            "requires_cuda_hardware": true,
            "requires_peak_memory_measurement": true,
            "requires_fragmentation_stress": true,
            "claim_without_green_verifier": false
        }
    })
}

pub fn memory_plan_from_json(spec_json: &str) -> Result<serde_json::Value, String> {
    let spec: serde_json::Value = serde_json::from_str(spec_json)
        .map_err(|_| "E_GPU_PLAN_INVALID_SPEC: spec must be JSON".to_string())?;
    let dtype = spec.get("dtype").and_then(|v| v.as_str()).unwrap_or("fp32");
    let dtype_bytes = dtype_bytes(dtype, "E_GPU_PLAN_UNSUPPORTED_DTYPE")?;
    let vocab = get_u64(&spec, "vocab_size", 128, "E_GPU_PLAN_INVALID_SPEC")?;
    let hidden = get_u64(&spec, "hidden_size", 64, "E_GPU_PLAN_INVALID_SPEC")?;
    let layers = get_u64(&spec, "layers", 2, "E_GPU_PLAN_INVALID_SPEC")?;
    let heads = get_u64(&spec, "heads", 4, "E_GPU_PLAN_INVALID_SPEC")?;
    let batch = get_u64(&spec, "batch_size", 2, "E_GPU_PLAN_INVALID_SPEC")?;
    let seq = get_u64(&spec, "seq_len", 32, "E_GPU_PLAN_INVALID_SPEC")?;
    let budget = get_u64(&spec, "budget_bytes", u64::MAX, "E_GPU_PLAN_INVALID_SPEC")?;
    if hidden == 0 || layers == 0 || heads == 0 || batch == 0 || seq == 0 || hidden % heads != 0 {
        return Err("E_GPU_PLAN_INVALID_SPEC: hidden/layers/heads/batch/seq must be positive and hidden must divide heads".to_string());
    }
    let align = 256u64;
    let param_elems = vocab * hidden
        + seq * hidden
        + layers * (12 * hidden * hidden + 10 * hidden)
        + hidden * vocab;
    let parameter_bytes = round_up(param_elems * dtype_bytes, align);
    let optimizer_bytes = round_up(parameter_bytes * 2, align);
    let activation_bytes = round_up(batch * seq * hidden * layers * 8 * dtype_bytes, align);
    let workspace_bytes = round_up(batch * heads * seq * seq * dtype_bytes, align);
    let kv_cache_bytes = round_up(batch * layers * seq * hidden * 2 * dtype_bytes, align);
    let peak_estimated_bytes = parameter_bytes
        .saturating_add(optimizer_bytes)
        .saturating_add(activation_bytes)
        .saturating_add(workspace_bytes)
        .saturating_add(kv_cache_bytes);
    if peak_estimated_bytes > budget {
        return Err(format!(
            "E_GPU_PLAN_BUDGET_EXCEEDED: estimated peak {peak_estimated_bytes} exceeds budget {budget}"
        ));
    }
    Ok(serde_json::json!({
        "schema_version": 1,
        "policy": "enkai_gpu_memory_planner_v1",
        "dtype": dtype,
        "dtype_bytes": dtype_bytes,
        "alignment_bytes": align,
        "parameter_bytes": parameter_bytes,
        "optimizer_bytes": optimizer_bytes,
        "activation_bytes": activation_bytes,
        "workspace_bytes": workspace_bytes,
        "kv_cache_bytes": kv_cache_bytes,
        "peak_estimated_bytes": peak_estimated_bytes,
        "pool_reuse_enabled": true,
        "fragmentation_budget_ratio": 0.08,
        "budget_bytes": if budget == u64::MAX { serde_json::Value::Null } else { serde_json::json!(budget) }
    }))
}

pub fn kv_cache_attention_policy() -> serde_json::Value {
    serde_json::json!({
        "schema_version": 1,
        "policy": "enkai_kv_cache_attention_v1",
        "status": "deterministic_foundation",
        "variants": ["causal_prefill", "decode_with_kv_cache"],
        "layout": "batch_layer_head_token_dim",
        "deterministic_errors": [
            "E_KV_INVALID_SPEC",
            "E_KV_UNSUPPORTED_DTYPE",
            "E_KV_BUDGET_EXCEEDED"
        ],
        "production_gate": {
            "requires_cuda_hardware": true,
            "requires_pytorch_attention_parity": true,
            "requires_long_context_benchmark": true,
            "requires_cache_corruption_faults": true,
            "claim_without_green_verifier": false
        }
    })
}

pub fn kv_cache_plan_from_json(spec_json: &str) -> Result<serde_json::Value, String> {
    let spec: serde_json::Value = serde_json::from_str(spec_json)
        .map_err(|_| "E_KV_INVALID_SPEC: spec must be JSON".to_string())?;
    let dtype = spec.get("dtype").and_then(|v| v.as_str()).unwrap_or("fp16");
    let dtype_bytes = dtype_bytes(dtype, "E_KV_UNSUPPORTED_DTYPE")?;
    let batch = get_u64(&spec, "batch_size", 1, "E_KV_INVALID_SPEC")?;
    let layers = get_u64(&spec, "layers", 2, "E_KV_INVALID_SPEC")?;
    let heads = get_u64(&spec, "heads", 4, "E_KV_INVALID_SPEC")?;
    let max_seq = get_u64(&spec, "max_seq_len", 1024, "E_KV_INVALID_SPEC")?;
    let head_dim = get_u64(&spec, "head_dim", 64, "E_KV_INVALID_SPEC")?;
    let budget = get_u64(&spec, "budget_bytes", u64::MAX, "E_KV_INVALID_SPEC")?;
    if batch == 0 || layers == 0 || heads == 0 || max_seq == 0 || head_dim == 0 {
        return Err(
            "E_KV_INVALID_SPEC: batch/layers/heads/max_seq/head_dim must be positive".to_string(),
        );
    }
    let cache_bytes = round_up(
        batch * layers * heads * max_seq * head_dim * 2 * dtype_bytes,
        256,
    );
    let prefill_attention_scores_bytes =
        round_up(batch * heads * max_seq * max_seq * dtype_bytes, 256);
    let decode_workspace_bytes = round_up(batch * heads * max_seq * dtype_bytes, 256);
    let peak_estimated_bytes = cache_bytes
        .saturating_add(prefill_attention_scores_bytes)
        .saturating_add(decode_workspace_bytes);
    if peak_estimated_bytes > budget {
        return Err(format!(
            "E_KV_BUDGET_EXCEEDED: estimated peak {peak_estimated_bytes} exceeds budget {budget}"
        ));
    }
    Ok(serde_json::json!({
        "schema_version": 1,
        "policy": "enkai_kv_cache_attention_v1",
        "dtype": dtype,
        "dtype_bytes": dtype_bytes,
        "layout": "batch_layer_head_token_dim",
        "kv_cache_bytes": cache_bytes,
        "prefill_attention_scores_bytes": prefill_attention_scores_bytes,
        "decode_workspace_bytes": decode_workspace_bytes,
        "peak_estimated_bytes": peak_estimated_bytes,
        "prefill_kernel": "enkai_cuda_causal_attention_prefill_f32",
        "decode_kernel": "enkai_cuda_kv_cache_decode_f32",
        "cache_corruption_checks": ["shape", "position", "dtype", "budget"]
    }))
}

pub fn large_checkpoint_format_policy() -> serde_json::Value {
    serde_json::json!({
        "schema_version": 1,
        "policy": "enkai_large_checkpoint_v1",
        "status": "deterministic_foundation",
        "files": ["manifest.json", "model-*.bin", "optimizer-*.bin", "rng.json", "data_cursor.json"],
        "features": ["tensor_shards", "optimizer_shards", "rng_state", "data_cursor", "hashes", "atomic_write", "resume_validation"],
        "deterministic_errors": [
            "E_CKPT_INVALID_SPEC",
            "E_CKPT_SHARD_TOO_SMALL",
            "E_CKPT_STALE_MANIFEST",
            "E_CKPT_HASH_MISMATCH"
        ],
        "production_gate": {
            "requires_atomic_filesystem_proof": true,
            "requires_corruption_faults": true,
            "requires_resume_replay": true,
            "claim_without_green_verifier": false
        }
    })
}

pub fn large_checkpoint_plan_from_json(spec_json: &str) -> Result<serde_json::Value, String> {
    let spec: serde_json::Value = serde_json::from_str(spec_json)
        .map_err(|_| "E_CKPT_INVALID_SPEC: spec must be JSON".to_string())?;
    let parameter_bytes = get_u64(&spec, "parameter_bytes", 0, "E_CKPT_INVALID_SPEC")?;
    let optimizer_bytes = get_u64(
        &spec,
        "optimizer_bytes",
        parameter_bytes * 2,
        "E_CKPT_INVALID_SPEC",
    )?;
    let shard_target = get_u64(
        &spec,
        "shard_target_bytes",
        64 * 1024 * 1024,
        "E_CKPT_INVALID_SPEC",
    )?;
    let world_size = get_u64(&spec, "world_size", 1, "E_CKPT_INVALID_SPEC")?;
    if parameter_bytes == 0 || shard_target < 4096 || world_size == 0 {
        return Err("E_CKPT_INVALID_SPEC: parameter_bytes/world_size must be positive and shard_target_bytes must be >= 4096".to_string());
    }
    let model_shards = parameter_bytes.div_ceil(shard_target);
    let optimizer_shards = optimizer_bytes.div_ceil(shard_target);
    Ok(serde_json::json!({
        "schema_version": 1,
        "policy": "enkai_large_checkpoint_v1",
        "world_size": world_size,
        "parameter_bytes": parameter_bytes,
        "optimizer_bytes": optimizer_bytes,
        "shard_target_bytes": shard_target,
        "model_shard_count": model_shards,
        "optimizer_shard_count": optimizer_shards,
        "manifest": {
            "format_version": 1,
            "hash": "sha256",
            "atomic_write": true,
            "includes_rng_state": true,
            "includes_data_cursor": true,
            "includes_optimizer": optimizer_bytes > 0,
            "requires_resume_validation": true
        }
    }))
}

pub fn distributed_training_policy() -> serde_json::Value {
    serde_json::json!({
        "schema_version": 1,
        "policy": "enkai_distributed_training_v1",
        "status": "deterministic_foundation",
        "collectives": ["barrier", "allreduce_mean"],
        "checkpoint": "sharded_merge_replay",
        "deterministic_errors": [
            "E_DIST_PLAN_INVALID_SPEC",
            "E_DIST_UNSUPPORTED_BACKEND",
            "E_DIST_BUCKET_TOO_SMALL",
            "E_DIST_WORLD_SIZE_UNSUPPORTED"
        ],
        "faults": [
            "rank_disconnect",
            "stale_gradient_payload",
            "duplicate_rank_payload",
            "wrong_tensor_length",
            "timeout"
        ],
        "production_gate": {
            "requires_multi_rank_hardware": true,
            "requires_gradient_parity": true,
            "requires_fault_injection": true,
            "requires_sharded_checkpoint_replay": true,
            "claim_without_green_verifier": false
        }
    })
}

pub fn distributed_training_plan_from_json(spec_json: &str) -> Result<serde_json::Value, String> {
    let spec: serde_json::Value = serde_json::from_str(spec_json)
        .map_err(|_| "E_DIST_PLAN_INVALID_SPEC: spec must be JSON".to_string())?;
    let backend = spec
        .get("backend")
        .and_then(|v| v.as_str())
        .unwrap_or("tcp");
    if !matches!(backend, "tcp" | "nccl") {
        return Err("E_DIST_UNSUPPORTED_BACKEND: backend must be tcp or nccl".to_string());
    }
    let world_size = get_u64(&spec, "world_size", 2, "E_DIST_PLAN_INVALID_SPEC")?;
    let gradient_bytes = get_u64(&spec, "gradient_bytes", 1, "E_DIST_PLAN_INVALID_SPEC")?;
    let bucket_bytes = get_u64(
        &spec,
        "bucket_bytes",
        4 * 1024 * 1024,
        "E_DIST_PLAN_INVALID_SPEC",
    )?;
    if world_size < 2 {
        return Err(
            "E_DIST_WORLD_SIZE_UNSUPPORTED: distributed training requires world_size >= 2"
                .to_string(),
        );
    }
    if bucket_bytes < 4096 {
        return Err("E_DIST_BUCKET_TOO_SMALL: bucket_bytes must be >= 4096".to_string());
    }
    let buckets = gradient_bytes.div_ceil(bucket_bytes).max(1);
    Ok(serde_json::json!({
        "schema_version": 1,
        "policy": "enkai_distributed_training_v1",
        "backend": backend,
        "world_size": world_size,
        "gradient_bytes": gradient_bytes,
        "bucket_bytes": bucket_bytes,
        "allreduce_bucket_count": buckets,
        "barrier_required": true,
        "checkpoint_merge_replay_required": true,
        "fault_injection_required": ["rank_disconnect", "stale_gradient_payload", "duplicate_rank_payload", "wrong_tensor_length", "timeout"]
    }))
}

pub fn accelerator_backend_policy() -> serde_json::Value {
    serde_json::json!({
        "schema_version": 1,
        "policy": "enkai_accelerator_backends_v1",
        "status": "stable_interface_hardware_gated",
        "backends": {
            "cuda": {
                "feature": "cuda-kernels",
                "status": if cfg!(feature = "cuda-kernels") { "feature_enabled" } else { "feature_missing" },
                "production_gate": ["nvcc", "cuda_hardware", "pytorch_parity", "green_verifier"]
            },
            "rocm": {
                "feature": "rocm-kernels",
                "status": if cfg!(feature = "rocm-kernels") { "feature_enabled_hardware_unverified" } else { "feature_missing" },
                "production_gate": ["hipcc", "rocm_hardware", "pytorch_rocm_parity", "green_verifier"]
            },
            "metal": {
                "feature": "metal-kernels",
                "status": if cfg!(feature = "metal-kernels") { "feature_enabled_hardware_unverified" } else { "feature_missing" },
                "production_gate": ["metal_toolchain", "apple_gpu_hardware", "pytorch_mps_parity", "green_verifier"]
            }
        },
        "deterministic_errors": [
            "E_BACKEND_FEATURE_MISSING",
            "E_BACKEND_UNAVAILABLE",
            "E_BACKEND_UNKNOWN"
        ],
        "claim_without_green_verifier": false
    })
}

pub fn accelerator_backend_plan_from_json(spec_json: &str) -> Result<serde_json::Value, String> {
    let spec: serde_json::Value = serde_json::from_str(spec_json)
        .map_err(|_| "E_ACCEL_PLAN_INVALID_SPEC: spec must be JSON".to_string())?;
    let backend = spec
        .get("backend")
        .and_then(|v| v.as_str())
        .unwrap_or("cuda");
    let feature_enabled = match backend {
        "cuda" => cfg!(feature = "cuda-kernels"),
        "rocm" => cfg!(feature = "rocm-kernels"),
        "metal" => cfg!(feature = "metal-kernels"),
        _ => {
            return Err(
                "E_ACCEL_PLAN_UNKNOWN_BACKEND: backend must be cuda, rocm, or metal".to_string(),
            )
        }
    };
    let required_feature = match backend {
        "cuda" => "cuda-kernels",
        "rocm" => "rocm-kernels",
        "metal" => "metal-kernels",
        _ => unreachable!(),
    };
    let parity_target = match backend {
        "cuda" => "pytorch_cuda",
        "rocm" => "pytorch_rocm",
        "metal" => "pytorch_mps",
        _ => unreachable!(),
    };
    Ok(serde_json::json!({
        "schema_version": 1,
        "policy": "enkai_accelerator_backends_v1",
        "backend": backend,
        "required_feature": required_feature,
        "feature_enabled": feature_enabled,
        "parity_target": parity_target,
        "hardware_proof_required": true,
        "production_supported": false,
        "reason": "backend is interface-locked but hardware verifier evidence is required before production support"
    }))
}

pub fn mixed_precision_policy() -> serde_json::Value {
    serde_json::json!({
        "schema_version": 1,
        "policy": "enkai_amp_v1",
        "status": "guarded_foundation",
        "supported_dtypes": ["fp32", "fp16", "bf16"],
        "accumulation_dtype": "fp32",
        "loss_scaling": {
            "mode": "dynamic",
            "initial_scale_min": 1.0,
            "initial_scale_max": 281474976710656.0,
            "growth_factor_min": 1.0,
            "backoff_factor_range": [0.0, 1.0],
            "growth_interval_min": 1,
            "non_finite_overflow_detection": true,
            "skip_optimizer_step_on_overflow": true
        },
        "deterministic_errors": [
            "E_AMP_INVALID_SCALE",
            "E_AMP_INVALID_GROWTH_FACTOR",
            "E_AMP_INVALID_BACKOFF_FACTOR",
            "E_AMP_INVALID_GROWTH_INTERVAL",
            "E_AMP_NONFINITE_GRADIENT"
        ],
        "production_gate": {
            "requires_cuda_hardware": true,
            "requires_gradient_parity_against_pytorch": true,
            "requires_overflow_fault_injection": true,
            "claim_without_green_verifier": false
        }
    })
}

#[cfg(feature = "cuda-kernels")]
#[allow(dead_code)]
extern "C" {
    pub fn enkai_cuda_vec_add_f32(
        a: *const f32,
        b: *const f32,
        out: *mut f32,
        n: i64,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    pub fn enkai_cuda_vec_mul_f32(
        a: *const f32,
        b: *const f32,
        out: *mut f32,
        n: i64,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    pub fn enkai_cuda_vec_scale_f32(
        x: *const f32,
        scale: f32,
        out: *mut f32,
        n: i64,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    pub fn enkai_cuda_vec_add_f16(
        a: *const std::ffi::c_void,
        b: *const std::ffi::c_void,
        out: *mut std::ffi::c_void,
        n: i64,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    pub fn enkai_cuda_vec_mul_f16(
        a: *const std::ffi::c_void,
        b: *const std::ffi::c_void,
        out: *mut std::ffi::c_void,
        n: i64,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    pub fn enkai_cuda_vec_scale_f16(
        x: *const std::ffi::c_void,
        scale: f32,
        out: *mut std::ffi::c_void,
        n: i64,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    pub fn enkai_cuda_vec_add_bf16(
        a: *const u16,
        b: *const u16,
        out: *mut u16,
        n: i64,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    pub fn enkai_cuda_vec_mul_bf16(
        a: *const u16,
        b: *const u16,
        out: *mut u16,
        n: i64,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    pub fn enkai_cuda_vec_scale_bf16(
        x: *const u16,
        scale: f32,
        out: *mut u16,
        n: i64,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    pub fn enkai_cuda_bias_gelu_f32(
        x: *const f32,
        bias: *const f32,
        out: *mut f32,
        rows: i64,
        cols: i64,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    pub fn enkai_cuda_bias_gelu_f16(
        x: *const std::ffi::c_void,
        bias: *const std::ffi::c_void,
        out: *mut std::ffi::c_void,
        rows: i64,
        cols: i64,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    pub fn enkai_cuda_bias_gelu_bf16(
        x: *const u16,
        bias: *const u16,
        out: *mut u16,
        rows: i64,
        cols: i64,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    pub fn enkai_cuda_matmul_bias_f32(
        a: *const f32,
        b: *const f32,
        bias: *const f32,
        out: *mut f32,
        m: i64,
        n: i64,
        k: i64,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    pub fn enkai_cuda_matmul_bias_cublas_f32(
        a: *const f32,
        b: *const f32,
        bias: *const f32,
        out: *mut f32,
        m: i64,
        n: i64,
        k: i64,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    pub fn enkai_cuda_matmul_bias_f16(
        a: *const std::ffi::c_void,
        b: *const std::ffi::c_void,
        bias: *const std::ffi::c_void,
        out: *mut std::ffi::c_void,
        m: i64,
        n: i64,
        k: i64,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    pub fn enkai_cuda_matmul_bias_bf16(
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        out: *mut u16,
        m: i64,
        n: i64,
        k: i64,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    pub fn enkai_cuda_layernorm_f32(
        x: *const f32,
        gamma: *const f32,
        beta: *const f32,
        out: *mut f32,
        rows: i64,
        cols: i64,
        eps: f32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    pub fn enkai_cuda_softmax_f32(
        x: *const f32,
        out: *mut f32,
        rows: i64,
        cols: i64,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    pub fn enkai_cuda_masked_softmax_f32(
        x: *const f32,
        mask: *const u8,
        out: *mut f32,
        rows: i64,
        cols: i64,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    pub fn enkai_cuda_causal_attention_prefill_f32(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        out: *mut f32,
        batch_heads: i64,
        seq: i64,
        head_dim: i64,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    pub fn enkai_cuda_kv_cache_decode_f32(
        q: *const f32,
        k_cache: *const f32,
        v_cache: *const f32,
        out: *mut f32,
        batch_heads: i64,
        cache_len: i64,
        head_dim: i64,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    pub fn enkai_cuda_causal_attention_backward_value_f32(
        q: *const f32,
        k: *const f32,
        grad_out: *const f32,
        grad_v: *mut f32,
        batch_heads: i64,
        seq: i64,
        head_dim: i64,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    pub fn enkai_cuda_causal_attention_backward_f32(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        grad_out: *const f32,
        grad_q: *mut f32,
        grad_k: *mut f32,
        grad_v: *mut f32,
        batch_heads: i64,
        seq: i64,
        head_dim: i64,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    pub fn enkai_cuda_cross_entropy_forward_f32(
        logits: *const f32,
        targets: *const i64,
        losses: *mut f32,
        rows: i64,
        cols: i64,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    pub fn enkai_cuda_cross_entropy_backward_f32(
        logits: *const f32,
        targets: *const i64,
        grad: *mut f32,
        rows: i64,
        cols: i64,
        scale: f32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    pub fn enkai_cuda_embedding_forward_f32(
        weights: *const f32,
        ids: *const i64,
        out: *mut f32,
        ids_len: i64,
        dim: i64,
        vocab: i64,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    pub fn enkai_cuda_embedding_backward_f32(
        grad_out: *const f32,
        ids: *const i64,
        grad_weights: *mut f32,
        ids_len: i64,
        dim: i64,
        vocab: i64,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    pub fn enkai_cuda_adamw_update_f32(
        param: *mut f32,
        grad: *const f32,
        m: *mut f32,
        v: *mut f32,
        n: i64,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        wd: f32,
        step: i64,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    pub fn enkai_cuda_clip_grad_norm_f32(
        grad: *mut f32,
        n: i64,
        max_norm: f32,
        eps: f32,
        out_norm: *mut f32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
}

#[cfg(feature = "cuda-kernels")]
#[allow(non_snake_case)]
mod runtime {
    use std::ffi::{c_int, c_void};
    use std::marker::PhantomData;
    use std::mem;
    use std::ptr;

    use super::*;

    const CUDA_SUCCESS: c_int = 0;
    const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
    const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;

    extern "C" {
        fn cudaGetDeviceCount(count: *mut c_int) -> c_int;
        fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> c_int;
        fn cudaFree(ptr: *mut c_void) -> c_int;
        fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: c_int) -> c_int;
        fn cudaDeviceSynchronize() -> c_int;
    }

    fn check(code: c_int, context: &str) -> Result<(), String> {
        if code == CUDA_SUCCESS {
            Ok(())
        } else {
            Err(format!(
                "E_CUDA_RUNTIME: {context} failed with cuda error {code}"
            ))
        }
    }

    pub fn device_count() -> Result<i32, String> {
        let mut count = 0;
        unsafe { check(cudaGetDeviceCount(&mut count), "cudaGetDeviceCount")? };
        Ok(count)
    }

    pub fn synchronize() -> Result<(), String> {
        unsafe { check(cudaDeviceSynchronize(), "cudaDeviceSynchronize") }
    }

    struct DeviceBuffer<T> {
        ptr: *mut T,
        len: usize,
        _marker: PhantomData<T>,
    }

    impl<T: Copy> DeviceBuffer<T> {
        fn uninit(len: usize) -> Result<Self, String> {
            let mut raw = ptr::null_mut();
            let bytes = len
                .checked_mul(mem::size_of::<T>())
                .ok_or_else(|| "E_CUDA_RUNTIME: device allocation size overflow".to_string())?;
            unsafe { check(cudaMalloc(&mut raw, bytes), "cudaMalloc")? };
            Ok(Self {
                ptr: raw.cast::<T>(),
                len,
                _marker: PhantomData,
            })
        }

        fn from_host(values: &[T]) -> Result<Self, String> {
            let buffer = Self::uninit(values.len())?;
            let bytes = values
                .len()
                .checked_mul(mem::size_of::<T>())
                .ok_or_else(|| "E_CUDA_RUNTIME: host copy size overflow".to_string())?;
            unsafe {
                check(
                    cudaMemcpy(
                        buffer.ptr.cast::<c_void>(),
                        values.as_ptr().cast::<c_void>(),
                        bytes,
                        CUDA_MEMCPY_HOST_TO_DEVICE,
                    ),
                    "cudaMemcpy host->device",
                )?;
            }
            Ok(buffer)
        }

        fn to_host(&self) -> Result<Vec<T>, String>
        where
            T: Default,
        {
            let mut out = vec![T::default(); self.len];
            let bytes = self
                .len
                .checked_mul(mem::size_of::<T>())
                .ok_or_else(|| "E_CUDA_RUNTIME: device copy size overflow".to_string())?;
            unsafe {
                check(
                    cudaMemcpy(
                        out.as_mut_ptr().cast::<c_void>(),
                        self.ptr.cast::<c_void>(),
                        bytes,
                        CUDA_MEMCPY_DEVICE_TO_HOST,
                    ),
                    "cudaMemcpy device->host",
                )?;
            }
            Ok(out)
        }
    }

    impl<T> Drop for DeviceBuffer<T> {
        fn drop(&mut self) {
            if !self.ptr.is_null() {
                unsafe {
                    let _ = cudaFree(self.ptr.cast::<c_void>());
                }
            }
        }
    }

    pub struct CudaF32Buffer {
        buffer: DeviceBuffer<f32>,
    }

    impl CudaF32Buffer {
        pub fn from_host(values: &[f32]) -> Result<Self, String> {
            Ok(Self {
                buffer: DeviceBuffer::from_host(values)?,
            })
        }

        pub fn zeros(len: usize) -> Result<Self, String> {
            Ok(Self {
                buffer: DeviceBuffer::uninit(len)?,
            })
        }

        pub fn len(&self) -> usize {
            self.buffer.len
        }

        pub fn to_host(&self) -> Result<Vec<f32>, String> {
            self.buffer.to_host()
        }

        pub fn as_ptr(&self) -> *const f32 {
            self.buffer.ptr
        }

        pub fn as_mut_ptr(&mut self) -> *mut f32 {
            self.buffer.ptr
        }
    }

    pub fn matmul_bias_f32_device(
        a: &CudaF32Buffer,
        b: &CudaF32Buffer,
        bias: Option<&CudaF32Buffer>,
        out: &mut CudaF32Buffer,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), String> {
        if a.len() != m * k || b.len() != k * n || out.len() != m * n {
            return Err("E_CUDA_SHAPE: resident matmul input shape mismatch".to_string());
        }
        if let Some(bias) = bias {
            if bias.len() != n {
                return Err("E_CUDA_SHAPE: resident matmul bias length mismatch".to_string());
            }
        }
        let bias_ptr = bias.map(|buf| buf.as_ptr()).unwrap_or(ptr::null());
        unsafe {
            let cublas_status = enkai_cuda_matmul_bias_cublas_f32(
                a.as_ptr(),
                b.as_ptr(),
                bias_ptr,
                out.as_mut_ptr(),
                m as i64,
                n as i64,
                k as i64,
                ptr::null_mut(),
            );
            if cublas_status != 0 {
                check(
                    enkai_cuda_matmul_bias_f32(
                        a.as_ptr(),
                        b.as_ptr(),
                        bias_ptr,
                        out.as_mut_ptr(),
                        m as i64,
                        n as i64,
                        k as i64,
                        ptr::null_mut(),
                    ),
                    "enkai_cuda_matmul_bias_f32 resident fallback",
                )?;
            }
        }
        Ok(())
    }

    pub fn vec_add_f32_host(a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
        if a.len() != b.len() {
            return Err("E_CUDA_SHAPE: vec_add input length mismatch".to_string());
        }
        let da = DeviceBuffer::from_host(a)?;
        let db = DeviceBuffer::from_host(b)?;
        let out = DeviceBuffer::<f32>::uninit(a.len())?;
        unsafe {
            check(
                enkai_cuda_vec_add_f32(da.ptr, db.ptr, out.ptr, a.len() as i64, ptr::null_mut()),
                "enkai_cuda_vec_add_f32",
            )?;
        }
        synchronize()?;
        out.to_host()
    }

    pub fn vec_mul_f32_host(a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
        if a.len() != b.len() {
            return Err("E_CUDA_SHAPE: vec_mul input length mismatch".to_string());
        }
        let da = DeviceBuffer::from_host(a)?;
        let db = DeviceBuffer::from_host(b)?;
        let out = DeviceBuffer::<f32>::uninit(a.len())?;
        unsafe {
            check(
                enkai_cuda_vec_mul_f32(da.ptr, db.ptr, out.ptr, a.len() as i64, ptr::null_mut()),
                "enkai_cuda_vec_mul_f32",
            )?;
        }
        synchronize()?;
        out.to_host()
    }

    pub fn matmul_bias_f32_host(
        a: &[f32],
        b: &[f32],
        bias: Option<&[f32]>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>, String> {
        if a.len() != m * k || b.len() != k * n {
            return Err("E_CUDA_SHAPE: matmul input shape mismatch".to_string());
        }
        if let Some(bias) = bias {
            if bias.len() != n {
                return Err("E_CUDA_SHAPE: matmul bias length mismatch".to_string());
            }
        }
        let da = DeviceBuffer::from_host(a)?;
        let db = DeviceBuffer::from_host(b)?;
        let dbias = match bias {
            Some(values) => Some(DeviceBuffer::from_host(values)?),
            None => None,
        };
        let out = DeviceBuffer::<f32>::uninit(m * n)?;
        let bias_ptr = dbias.as_ref().map(|buf| buf.ptr).unwrap_or(ptr::null_mut());
        unsafe {
            let cublas_status = enkai_cuda_matmul_bias_cublas_f32(
                da.ptr,
                db.ptr,
                bias_ptr,
                out.ptr,
                m as i64,
                n as i64,
                k as i64,
                ptr::null_mut(),
            );
            if cublas_status != 0 {
                check(
                    enkai_cuda_matmul_bias_f32(
                        da.ptr,
                        db.ptr,
                        bias_ptr,
                        out.ptr,
                        m as i64,
                        n as i64,
                        k as i64,
                        ptr::null_mut(),
                    ),
                    "enkai_cuda_matmul_bias_f32 fallback",
                )?;
            }
        }
        synchronize()?;
        out.to_host()
    }

    pub fn softmax_f32_host(x: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>, String> {
        if x.len() != rows * cols {
            return Err("E_CUDA_SHAPE: softmax input shape mismatch".to_string());
        }
        let dx = DeviceBuffer::from_host(x)?;
        let out = DeviceBuffer::<f32>::uninit(x.len())?;
        unsafe {
            check(
                enkai_cuda_softmax_f32(dx.ptr, out.ptr, rows as i64, cols as i64, ptr::null_mut()),
                "enkai_cuda_softmax_f32",
            )?;
        }
        synchronize()?;
        out.to_host()
    }

    pub fn cross_entropy_losses_f32_host(
        logits: &[f32],
        targets: &[i64],
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f32>, String> {
        if logits.len() != rows * cols || targets.len() != rows {
            return Err("E_CUDA_SHAPE: cross_entropy input shape mismatch".to_string());
        }
        let dlogits = DeviceBuffer::from_host(logits)?;
        let dtargets = DeviceBuffer::from_host(targets)?;
        let losses = DeviceBuffer::<f32>::uninit(rows)?;
        unsafe {
            check(
                enkai_cuda_cross_entropy_forward_f32(
                    dlogits.ptr,
                    dtargets.ptr,
                    losses.ptr,
                    rows as i64,
                    cols as i64,
                    ptr::null_mut(),
                ),
                "enkai_cuda_cross_entropy_forward_f32",
            )?;
        }
        synchronize()?;
        losses.to_host()
    }

    pub fn adamw_update_f32_host(
        param: &[f32],
        grad: &[f32],
        m: &[f32],
        v: &[f32],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        wd: f32,
        step: i64,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
        if param.len() != grad.len() || param.len() != m.len() || param.len() != v.len() {
            return Err("E_CUDA_SHAPE: adamw input length mismatch".to_string());
        }
        let dparam = DeviceBuffer::from_host(param)?;
        let dgrad = DeviceBuffer::from_host(grad)?;
        let dm = DeviceBuffer::from_host(m)?;
        let dv = DeviceBuffer::from_host(v)?;
        unsafe {
            check(
                enkai_cuda_adamw_update_f32(
                    dparam.ptr,
                    dgrad.ptr,
                    dm.ptr,
                    dv.ptr,
                    param.len() as i64,
                    lr,
                    beta1,
                    beta2,
                    eps,
                    wd,
                    step,
                    ptr::null_mut(),
                ),
                "enkai_cuda_adamw_update_f32",
            )?;
        }
        synchronize()?;
        Ok((dparam.to_host()?, dm.to_host()?, dv.to_host()?))
    }
}

#[cfg(feature = "cuda-kernels")]
pub use runtime::CudaF32Buffer;

#[cfg(feature = "cuda-kernels")]
pub fn cuda_device_count() -> Result<i32, String> {
    runtime::device_count()
}

#[cfg(not(feature = "cuda-kernels"))]
pub fn cuda_device_count() -> Result<i32, String> {
    Err("E_CUDA_UNAVAILABLE: build without cuda-kernels feature".to_string())
}

#[cfg(feature = "cuda-kernels")]
pub fn cuda_synchronize() -> Result<(), String> {
    runtime::synchronize()
}

#[cfg(not(feature = "cuda-kernels"))]
pub fn cuda_synchronize() -> Result<(), String> {
    Err("E_CUDA_UNAVAILABLE: build without cuda-kernels feature".to_string())
}

#[cfg(feature = "cuda-kernels")]
pub fn cuda_vec_add_f32_host(a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
    runtime::vec_add_f32_host(a, b)
}

#[cfg(not(feature = "cuda-kernels"))]
pub fn cuda_vec_add_f32_host(_a: &[f32], _b: &[f32]) -> Result<Vec<f32>, String> {
    Err("E_CUDA_UNAVAILABLE: build without cuda-kernels feature".to_string())
}

#[cfg(feature = "cuda-kernels")]
pub fn cuda_vec_mul_f32_host(a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
    runtime::vec_mul_f32_host(a, b)
}

#[cfg(not(feature = "cuda-kernels"))]
pub fn cuda_vec_mul_f32_host(_a: &[f32], _b: &[f32]) -> Result<Vec<f32>, String> {
    Err("E_CUDA_UNAVAILABLE: build without cuda-kernels feature".to_string())
}

#[cfg(feature = "cuda-kernels")]
pub fn cuda_matmul_bias_f32_host(
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    m: usize,
    n: usize,
    k: usize,
) -> Result<Vec<f32>, String> {
    runtime::matmul_bias_f32_host(a, b, bias, m, n, k)
}

#[cfg(not(feature = "cuda-kernels"))]
pub fn cuda_matmul_bias_f32_host(
    _a: &[f32],
    _b: &[f32],
    _bias: Option<&[f32]>,
    _m: usize,
    _n: usize,
    _k: usize,
) -> Result<Vec<f32>, String> {
    Err("E_CUDA_UNAVAILABLE: build without cuda-kernels feature".to_string())
}

#[cfg(feature = "cuda-kernels")]
pub fn cuda_softmax_f32_host(x: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>, String> {
    runtime::softmax_f32_host(x, rows, cols)
}

#[cfg(not(feature = "cuda-kernels"))]
pub fn cuda_softmax_f32_host(_x: &[f32], _rows: usize, _cols: usize) -> Result<Vec<f32>, String> {
    Err("E_CUDA_UNAVAILABLE: build without cuda-kernels feature".to_string())
}

#[cfg(feature = "cuda-kernels")]
pub fn cuda_cross_entropy_losses_f32_host(
    logits: &[f32],
    targets: &[i64],
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>, String> {
    runtime::cross_entropy_losses_f32_host(logits, targets, rows, cols)
}

#[cfg(not(feature = "cuda-kernels"))]
pub fn cuda_cross_entropy_losses_f32_host(
    _logits: &[f32],
    _targets: &[i64],
    _rows: usize,
    _cols: usize,
) -> Result<Vec<f32>, String> {
    Err("E_CUDA_UNAVAILABLE: build without cuda-kernels feature".to_string())
}

#[cfg(feature = "cuda-kernels")]
pub fn cuda_adamw_update_f32_host(
    param: &[f32],
    grad: &[f32],
    m: &[f32],
    v: &[f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    wd: f32,
    step: i64,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
    runtime::adamw_update_f32_host(param, grad, m, v, lr, beta1, beta2, eps, wd, step)
}

#[cfg(feature = "cuda-kernels")]
pub fn cuda_matmul_bias_f32_device(
    a: &CudaF32Buffer,
    b: &CudaF32Buffer,
    bias: Option<&CudaF32Buffer>,
    out: &mut CudaF32Buffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), String> {
    runtime::matmul_bias_f32_device(a, b, bias, out, m, n, k)
}

#[cfg(not(feature = "cuda-kernels"))]
pub fn cuda_adamw_update_f32_host(
    _param: &[f32],
    _grad: &[f32],
    _m: &[f32],
    _v: &[f32],
    _lr: f32,
    _beta1: f32,
    _beta2: f32,
    _eps: f32,
    _wd: f32,
    _step: i64,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
    Err("E_CUDA_UNAVAILABLE: build without cuda-kernels feature".to_string())
}
