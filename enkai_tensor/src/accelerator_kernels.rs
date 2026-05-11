#![allow(dead_code)]

pub fn rocm_kernel_manifest() -> serde_json::Value {
    serde_json::json!({
        "schema_version": 1,
        "backend": "enkai_rocm_first_party",
        "build_status": if cfg!(feature = "rocm-kernels") { "compiled" } else { "not_compiled" },
        "source": "enkai_tensor/rocm/enkai_kernels.hip.cpp",
        "production_gate": {
            "requires_feature": "rocm-kernels",
            "requires_hipcc": true,
            "requires_rocm_hardware": true,
            "requires_pytorch_rocm_comparison": true,
            "claim_without_green_verifier": false
        },
        "kernels": [
            {"name": "vec_add", "symbol": "enkai_rocm_vec_add_f32", "dtype": "fp32", "class": "elementwise"},
            {"name": "vec_mul", "symbol": "enkai_rocm_vec_mul_f32", "dtype": "fp32", "class": "elementwise"},
            {"name": "vec_scale", "symbol": "enkai_rocm_vec_scale_f32", "dtype": "fp32", "class": "elementwise"},
            {"name": "bias_gelu", "symbol": "enkai_rocm_bias_gelu_f32", "dtype": "fp32", "class": "fusion"},
            {"name": "matmul_bias", "symbol": "enkai_rocm_matmul_bias_f32", "dtype": "fp32", "class": "fusion"},
            {"name": "softmax", "symbol": "enkai_rocm_softmax_f32", "dtype": "fp32", "class": "attention_core"},
            {"name": "cross_entropy_forward", "symbol": "enkai_rocm_cross_entropy_forward_f32", "dtype": "fp32", "class": "loss"},
            {"name": "cross_entropy_backward", "symbol": "enkai_rocm_cross_entropy_backward_f32", "dtype": "fp32", "class": "loss_backward"},
            {"name": "embedding_forward", "symbol": "enkai_rocm_embedding_forward_f32", "dtype": "fp32", "class": "embedding"},
            {"name": "embedding_backward", "symbol": "enkai_rocm_embedding_backward_f32", "dtype": "fp32", "class": "embedding_backward"},
            {"name": "adamw_update", "symbol": "enkai_rocm_adamw_update_f32", "dtype": "fp32", "class": "optimizer_fusion"},
            {"name": "clip_grad_norm", "symbol": "enkai_rocm_clip_grad_norm_f32", "dtype": "fp32", "class": "gradient_safety"}
        ],
        "bounded_llm_training_kernel_set": "source_complete_hardware_gated"
    })
}

pub fn metal_kernel_manifest() -> serde_json::Value {
    serde_json::json!({
        "schema_version": 1,
        "backend": "enkai_metal_first_party",
        "build_status": if cfg!(feature = "metal-kernels") { "compiled" } else { "not_compiled" },
        "source": "enkai_tensor/metal/enkai_kernels.metal",
        "production_gate": {
            "requires_feature": "metal-kernels",
            "requires_xcrun_metal": true,
            "requires_apple_gpu_hardware": true,
            "requires_pytorch_mps_comparison": true,
            "claim_without_green_verifier": false
        },
        "kernels": [
            {"name": "vec_add", "symbol": "enkai_metal_vec_add_f32", "dtype": "fp32", "class": "elementwise"},
            {"name": "vec_mul", "symbol": "enkai_metal_vec_mul_f32", "dtype": "fp32", "class": "elementwise"},
            {"name": "vec_scale", "symbol": "enkai_metal_vec_scale_f32", "dtype": "fp32", "class": "elementwise"},
            {"name": "bias_gelu", "symbol": "enkai_metal_bias_gelu_f32", "dtype": "fp32", "class": "fusion"},
            {"name": "matmul_bias", "symbol": "enkai_metal_matmul_bias_f32", "dtype": "fp32", "class": "fusion"},
            {"name": "softmax", "symbol": "enkai_metal_softmax_f32", "dtype": "fp32", "class": "attention_core"},
            {"name": "cross_entropy_forward", "symbol": "enkai_metal_cross_entropy_forward_f32", "dtype": "fp32", "class": "loss"},
            {"name": "cross_entropy_backward", "symbol": "enkai_metal_cross_entropy_backward_f32", "dtype": "fp32", "class": "loss_backward"},
            {"name": "embedding_forward", "symbol": "enkai_metal_embedding_forward_f32", "dtype": "fp32", "class": "embedding"},
            {"name": "embedding_backward", "symbol": "enkai_metal_embedding_backward_f32", "dtype": "fp32", "class": "embedding_backward"},
            {"name": "adamw_update", "symbol": "enkai_metal_adamw_update_f32", "dtype": "fp32", "class": "optimizer_fusion"},
            {"name": "clip_grad_norm", "symbol": "enkai_metal_clip_grad_norm_f32", "dtype": "fp32", "class": "gradient_safety"}
        ],
        "bounded_llm_training_kernel_set": "source_complete_hardware_gated"
    })
}

#[cfg(feature = "rocm-kernels")]
#[link(name = "amdhip64")]
extern "C" {
    fn hipGetDeviceCount(count: *mut std::ffi::c_int) -> std::ffi::c_int;
}

#[cfg(feature = "rocm-kernels")]
pub fn rocm_available() -> bool {
    let mut count = 0;
    unsafe { hipGetDeviceCount(&mut count) == 0 && count > 0 }
}

#[cfg(not(feature = "rocm-kernels"))]
pub fn rocm_available() -> bool {
    false
}

#[cfg(all(feature = "metal-kernels", target_os = "macos"))]
pub fn metal_available() -> bool {
    std::path::Path::new("/System/Library/Frameworks/Metal.framework").exists()
}

#[cfg(not(all(feature = "metal-kernels", target_os = "macos")))]
pub fn metal_available() -> bool {
    false
}
