#[test]
fn cuda_kernel_manifest_and_amp_policy_are_stable_without_cuda_feature() {
    let mut out: *mut i8 = std::ptr::null_mut();
    let mut len: usize = 0;
    let rc = unsafe { enkai_tensor::enkai_cuda_kernel_manifest(&mut out, &mut len) };
    assert_eq!(rc, 0);
    assert!(len > 0);
    let text = unsafe {
        let slice = std::slice::from_raw_parts(out as *const u8, len);
        std::str::from_utf8(slice).unwrap().to_string()
    };
    unsafe { enkai_tensor::enkai_free(out as *mut u8, len) };
    assert!(text.contains("enkai_cuda_first_party"));
    assert!(text.contains("bias_gelu"));
    assert!(text.contains("matmul_bias"));
    assert!(text.contains("enkai_cuda_vec_add_f16"));
    assert!(text.contains("enkai_cuda_vec_add_bf16"));
    assert!(text.contains("enkai_cuda_matmul_bias_f16"));
    assert!(text.contains("enkai_cuda_matmul_bias_bf16"));
    assert!(text.contains("masked_softmax"));
    assert!(text.contains("cross_entropy_forward"));
    assert!(text.contains("cross_entropy_backward"));
    assert!(text.contains("embedding_forward"));
    assert!(text.contains("embedding_backward"));
    assert!(text.contains("causal_attention_backward"));
    assert!(text.contains("adamw_update"));
    assert!(text.contains("clip_grad_norm"));
    assert!(text.contains("bounded_llm_training_kernel_set"));
    assert!(!text.contains("missing_for_full_llm_training"));
    assert!(text.contains("requires_cuda_hardware"));
    assert!(text.contains("not_compiled"));

    let mut policy_out: *mut i8 = std::ptr::null_mut();
    let mut policy_len: usize = 0;
    let rc =
        unsafe { enkai_tensor::enkai_mixed_precision_policy(&mut policy_out, &mut policy_len) };
    assert_eq!(rc, 0);
    let policy = unsafe {
        let slice = std::slice::from_raw_parts(policy_out as *const u8, policy_len);
        std::str::from_utf8(slice).unwrap().to_string()
    };
    unsafe { enkai_tensor::enkai_free(policy_out as *mut u8, policy_len) };
    assert!(policy.contains("enkai_amp_v1"));
    assert!(policy.contains("fp16"));
    assert!(policy.contains("bf16"));
    assert!(policy.contains("E_AMP_NONFINITE_GRADIENT"));
}

#[test]
fn listed_cuda_kernel_parity_gaps_are_closed_in_source_and_manifest() {
    let source = include_str!("../cuda/enkai_kernels.cu");
    let manifest_source = include_str!("../src/cuda_kernels.rs");
    let required_symbols = [
        // fp16/bf16 CUDA variants for the bounded mixed-precision frontier.
        "enkai_cuda_vec_add_f16",
        "enkai_cuda_vec_mul_f16",
        "enkai_cuda_vec_scale_f16",
        "enkai_cuda_vec_add_bf16",
        "enkai_cuda_vec_mul_bf16",
        "enkai_cuda_vec_scale_bf16",
        "enkai_cuda_bias_gelu_f16",
        "enkai_cuda_bias_gelu_bf16",
        "enkai_cuda_matmul_bias_f16",
        "enkai_cuda_matmul_bias_bf16",
        // Fused loss forward/backward.
        "enkai_cuda_cross_entropy_forward_f32",
        "enkai_cuda_cross_entropy_backward_f32",
        // Embedding gather/scatter backward.
        "enkai_cuda_embedding_forward_f32",
        "enkai_cuda_embedding_backward_f32",
        "atomicAdd",
        // Attention backward.
        "enkai_cuda_causal_attention_backward_f32",
        "enkai_causal_attention_backward_q_f32_kernel",
        "enkai_causal_attention_backward_k_f32_kernel",
        "enkai_causal_attention_backward_value_f32_kernel",
        // Gradient clipping.
        "enkai_cuda_clip_grad_norm_f32",
        "enkai_clip_grad_norm_f32_kernel",
    ];
    for symbol in required_symbols {
        assert!(
            source.contains(symbol) || manifest_source.contains(symbol),
            "missing CUDA parity symbol: {symbol}"
        );
    }
    assert!(manifest_source.contains("source_complete_hardware_gated"));
    assert!(!manifest_source.contains("missing_for_full_llm_training"));
}

#[test]
fn gpu_memory_planner_policy_and_deterministic_budget_errors_are_stable() {
    let mut policy_out: *mut i8 = std::ptr::null_mut();
    let mut policy_len: usize = 0;
    let rc =
        unsafe { enkai_tensor::enkai_gpu_memory_planner_policy(&mut policy_out, &mut policy_len) };
    assert_eq!(rc, 0);
    let policy = unsafe {
        let slice = std::slice::from_raw_parts(policy_out as *const u8, policy_len);
        std::str::from_utf8(slice).unwrap().to_string()
    };
    unsafe { enkai_tensor::enkai_free(policy_out as *mut u8, policy_len) };
    assert!(policy.contains("enkai_gpu_memory_planner_v1"));
    assert!(policy.contains("static_lifetime_pool"));
    assert!(policy.contains("E_GPU_PLAN_BUDGET_EXCEEDED"));

    let spec = std::ffi::CString::new(
        r#"{"dtype":"fp16","vocab_size":256,"hidden_size":64,"layers":2,"heads":4,"batch_size":2,"seq_len":32,"budget_bytes":100000000}"#,
    )
    .unwrap();
    let mut plan_out: *mut i8 = std::ptr::null_mut();
    let mut plan_len: usize = 0;
    let rc =
        unsafe { enkai_tensor::enkai_gpu_memory_plan(spec.as_ptr(), &mut plan_out, &mut plan_len) };
    assert_eq!(rc, 0);
    let plan = unsafe {
        let slice = std::slice::from_raw_parts(plan_out as *const u8, plan_len);
        std::str::from_utf8(slice).unwrap().to_string()
    };
    unsafe { enkai_tensor::enkai_free(plan_out as *mut u8, plan_len) };
    assert!(plan.contains("\"peak_estimated_bytes\""));
    assert!(plan.contains("\"pool_reuse_enabled\":true"));

    let too_small = std::ffi::CString::new(
        r#"{"dtype":"fp16","vocab_size":256,"hidden_size":64,"layers":2,"heads":4,"batch_size":2,"seq_len":32,"budget_bytes":1}"#,
    )
    .unwrap();
    let mut err_out: *mut i8 = std::ptr::null_mut();
    let mut err_len: usize = 0;
    let rc = unsafe {
        enkai_tensor::enkai_gpu_memory_plan(too_small.as_ptr(), &mut err_out, &mut err_len)
    };
    assert_eq!(rc, 1);
    let err = unsafe { std::ffi::CStr::from_ptr(enkai_tensor::enkai_tensor_last_error()) }
        .to_str()
        .unwrap()
        .to_string();
    assert!(err.contains("E_GPU_PLAN_BUDGET_EXCEEDED"));
    assert!(err_out.is_null());
    assert_eq!(err_len, 0);
}

#[test]
fn kv_cache_checkpoint_and_distributed_contracts_are_stable() {
    let mut out: *mut i8 = std::ptr::null_mut();
    let mut len: usize = 0;
    let rc = unsafe { enkai_tensor::enkai_kv_cache_attention_policy(&mut out, &mut len) };
    assert_eq!(rc, 0);
    let policy = unsafe {
        let slice = std::slice::from_raw_parts(out as *const u8, len);
        std::str::from_utf8(slice).unwrap().to_string()
    };
    unsafe { enkai_tensor::enkai_free(out as *mut u8, len) };
    assert!(policy.contains("enkai_kv_cache_attention_v1"));
    assert!(policy.contains("decode_with_kv_cache"));
    assert!(policy.contains("E_KV_BUDGET_EXCEEDED"));

    let kv_spec = std::ffi::CString::new(
        r#"{"dtype":"fp16","batch_size":2,"layers":4,"heads":8,"max_seq_len":128,"head_dim":32,"budget_bytes":100000000}"#,
    )
    .unwrap();
    let mut kv_out: *mut i8 = std::ptr::null_mut();
    let mut kv_len: usize = 0;
    let rc =
        unsafe { enkai_tensor::enkai_kv_cache_plan(kv_spec.as_ptr(), &mut kv_out, &mut kv_len) };
    assert_eq!(rc, 0);
    let kv_plan = unsafe {
        let slice = std::slice::from_raw_parts(kv_out as *const u8, kv_len);
        std::str::from_utf8(slice).unwrap().to_string()
    };
    unsafe { enkai_tensor::enkai_free(kv_out as *mut u8, kv_len) };
    assert!(kv_plan.contains("\"kv_cache_bytes\""));
    assert!(kv_plan.contains("enkai_cuda_kv_cache_decode_f32"));

    let mut ckpt_policy_out: *mut i8 = std::ptr::null_mut();
    let mut ckpt_policy_len: usize = 0;
    let rc = unsafe {
        enkai_tensor::enkai_large_checkpoint_format_policy(
            &mut ckpt_policy_out,
            &mut ckpt_policy_len,
        )
    };
    assert_eq!(rc, 0);
    let ckpt_policy = unsafe {
        let slice = std::slice::from_raw_parts(ckpt_policy_out as *const u8, ckpt_policy_len);
        std::str::from_utf8(slice).unwrap().to_string()
    };
    unsafe { enkai_tensor::enkai_free(ckpt_policy_out as *mut u8, ckpt_policy_len) };
    assert!(ckpt_policy.contains("enkai_large_checkpoint_v1"));
    assert!(ckpt_policy.contains("data_cursor.json"));

    let ckpt_spec = std::ffi::CString::new(
        r#"{"parameter_bytes":10485760,"optimizer_bytes":20971520,"shard_target_bytes":4194304,"world_size":4}"#,
    )
    .unwrap();
    let mut ckpt_out: *mut i8 = std::ptr::null_mut();
    let mut ckpt_len: usize = 0;
    let rc = unsafe {
        enkai_tensor::enkai_large_checkpoint_plan(ckpt_spec.as_ptr(), &mut ckpt_out, &mut ckpt_len)
    };
    assert_eq!(rc, 0);
    let ckpt_plan = unsafe {
        let slice = std::slice::from_raw_parts(ckpt_out as *const u8, ckpt_len);
        std::str::from_utf8(slice).unwrap().to_string()
    };
    unsafe { enkai_tensor::enkai_free(ckpt_out as *mut u8, ckpt_len) };
    assert!(ckpt_plan.contains("\"model_shard_count\""));
    assert!(ckpt_plan.contains("\"requires_resume_validation\":true"));

    let mut dist_policy_out: *mut i8 = std::ptr::null_mut();
    let mut dist_policy_len: usize = 0;
    let rc = unsafe {
        enkai_tensor::enkai_distributed_training_policy(&mut dist_policy_out, &mut dist_policy_len)
    };
    assert_eq!(rc, 0);
    let dist_policy = unsafe {
        let slice = std::slice::from_raw_parts(dist_policy_out as *const u8, dist_policy_len);
        std::str::from_utf8(slice).unwrap().to_string()
    };
    unsafe { enkai_tensor::enkai_free(dist_policy_out as *mut u8, dist_policy_len) };
    assert!(dist_policy.contains("enkai_distributed_training_v1"));
    assert!(dist_policy.contains("stale_gradient_payload"));

    let dist_spec = std::ffi::CString::new(
        r#"{"backend":"tcp","world_size":4,"gradient_bytes":16777216,"bucket_bytes":4194304}"#,
    )
    .unwrap();
    let mut dist_out: *mut i8 = std::ptr::null_mut();
    let mut dist_len: usize = 0;
    let rc = unsafe {
        enkai_tensor::enkai_distributed_training_plan(
            dist_spec.as_ptr(),
            &mut dist_out,
            &mut dist_len,
        )
    };
    assert_eq!(rc, 0);
    let dist_plan = unsafe {
        let slice = std::slice::from_raw_parts(dist_out as *const u8, dist_len);
        std::str::from_utf8(slice).unwrap().to_string()
    };
    unsafe { enkai_tensor::enkai_free(dist_out as *mut u8, dist_len) };
    assert!(dist_plan.contains("\"allreduce_bucket_count\":4"));
    assert!(dist_plan.contains("\"checkpoint_merge_replay_required\":true"));
}

#[test]
fn accelerator_backend_policy_locks_rocm_and_metal_interfaces() {
    let mut out: *mut i8 = std::ptr::null_mut();
    let mut len: usize = 0;
    let rc = unsafe { enkai_tensor::enkai_accelerator_backend_policy(&mut out, &mut len) };
    assert_eq!(rc, 0);
    let policy = unsafe {
        let slice = std::slice::from_raw_parts(out as *const u8, len);
        std::str::from_utf8(slice).unwrap().to_string()
    };
    unsafe { enkai_tensor::enkai_free(out as *mut u8, len) };
    assert!(policy.contains("enkai_accelerator_backends_v1"));
    assert!(policy.contains("rocm-kernels"));
    assert!(policy.contains("metal-kernels"));
    assert!(policy.contains("claim_without_green_verifier"));

    let spec = std::ffi::CString::new(r#"{"backend":"metal"}"#).unwrap();
    let mut plan_out: *mut i8 = std::ptr::null_mut();
    let mut plan_len: usize = 0;
    let rc = unsafe {
        enkai_tensor::enkai_accelerator_backend_plan(spec.as_ptr(), &mut plan_out, &mut plan_len)
    };
    assert_eq!(rc, 0);
    let plan = unsafe {
        let slice = std::slice::from_raw_parts(plan_out as *const u8, plan_len);
        std::str::from_utf8(slice).unwrap().to_string()
    };
    unsafe { enkai_tensor::enkai_free(plan_out as *mut u8, plan_len) };
    assert!(plan.contains("\"backend\":\"metal\""));
    assert!(plan.contains("\"required_feature\":\"metal-kernels\""));
    assert!(plan.contains("\"hardware_proof_required\":true"));
    assert!(plan.contains("\"production_supported\":false"));
}

#[test]
fn rocm_and_metal_have_real_source_kernel_manifests() {
    let mut rocm_out: *mut i8 = std::ptr::null_mut();
    let mut rocm_len: usize = 0;
    let rc = unsafe { enkai_tensor::enkai_rocm_kernel_manifest(&mut rocm_out, &mut rocm_len) };
    assert_eq!(rc, 0);
    let rocm = unsafe {
        let slice = std::slice::from_raw_parts(rocm_out as *const u8, rocm_len);
        std::str::from_utf8(slice).unwrap().to_string()
    };
    unsafe { enkai_tensor::enkai_free(rocm_out as *mut u8, rocm_len) };
    assert!(rocm.contains("enkai_rocm_first_party"));
    assert!(rocm.contains("enkai_tensor/rocm/enkai_kernels.hip.cpp"));
    assert!(rocm.contains("enkai_rocm_matmul_bias_f32"));
    assert!(rocm.contains("enkai_rocm_cross_entropy_backward_f32"));
    assert!(rocm.contains("enkai_rocm_embedding_backward_f32"));
    assert!(rocm.contains("enkai_rocm_clip_grad_norm_f32"));
    assert!(rocm.contains("requires_hipcc"));
    assert!(rocm.contains("claim_without_green_verifier"));

    let mut metal_out: *mut i8 = std::ptr::null_mut();
    let mut metal_len: usize = 0;
    let rc = unsafe { enkai_tensor::enkai_metal_kernel_manifest(&mut metal_out, &mut metal_len) };
    assert_eq!(rc, 0);
    let metal = unsafe {
        let slice = std::slice::from_raw_parts(metal_out as *const u8, metal_len);
        std::str::from_utf8(slice).unwrap().to_string()
    };
    unsafe { enkai_tensor::enkai_free(metal_out as *mut u8, metal_len) };
    assert!(metal.contains("enkai_metal_first_party"));
    assert!(metal.contains("enkai_tensor/metal/enkai_kernels.metal"));
    assert!(metal.contains("enkai_metal_matmul_bias_f32"));
    assert!(metal.contains("enkai_metal_cross_entropy_backward_f32"));
    assert!(metal.contains("enkai_metal_embedding_backward_f32"));
    assert!(metal.contains("enkai_metal_clip_grad_norm_f32"));
    assert!(metal.contains("requires_xcrun_metal"));
    assert!(metal.contains("claim_without_green_verifier"));

    let rocm_source = include_str!("../rocm/enkai_kernels.hip.cpp");
    assert!(rocm_source.contains("hipLaunchKernelGGL"));
    assert!(rocm_source.contains("atomicAdd"));
    assert!(rocm_source.contains("enkai_rocm_cross_entropy_forward_f32"));

    let metal_source = include_str!("../metal/enkai_kernels.metal");
    assert!(metal_source.contains("kernel void enkai_metal_matmul_bias_f32"));
    assert!(metal_source.contains("kernel void enkai_metal_cross_entropy_backward_f32"));
    assert!(metal_source.contains("atomic_fetch_add_explicit"));
}
