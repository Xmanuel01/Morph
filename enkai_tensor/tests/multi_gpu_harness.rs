#![cfg(feature = "torch")]

#[test]
fn multi_gpu_harness_gate_smoke() {
    if std::env::var("ENKAI_RUN_MULTI_GPU_TESTS").ok().as_deref() != Some("1") {
        eprintln!("SKIPPED: ENKAI_RUN_MULTI_GPU_TESTS not set");
        return;
    }
    if std::env::var("ENKAI_ENABLE_DIST").ok().as_deref() != Some("1") {
        eprintln!("SKIPPED: ENKAI_ENABLE_DIST not set to 1");
        return;
    }
    if !tch::Cuda::is_available() {
        eprintln!("SKIPPED: CUDA not available");
        return;
    }
    if tch::Cuda::device_count() < 2 {
        eprintln!("SKIPPED: fewer than 2 CUDA devices");
        return;
    }

    #[cfg(not(feature = "dist"))]
    {
        let rc = enkai_tensor::enkai_dist_config(2, 0, -1, 42);
        assert_ne!(rc, 0, "dist feature must be enabled for multi-rank smoke");
        return;
    }

    #[cfg(feature = "dist")]
    {
        let rc = enkai_tensor::enkai_dist_config(2, 0, -1, 42);
        assert_eq!(rc, 0, "dist init must succeed for rank0 smoke");
        let _ = enkai_tensor::enkai_dist_shutdown();
    }
}
