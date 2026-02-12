#![cfg(feature = "torch")]

#[test]
fn multi_gpu_harness_scaffolded() {
    if std::env::var("ENKAI_RUN_MULTI_GPU_TESTS").ok().as_deref() != Some("1") {
        eprintln!("skipped: ENKAI_RUN_MULTI_GPU_TESTS not set");
        return;
    }
    if !tch::Cuda::is_available() {
        eprintln!("skipped: CUDA not available");
        return;
    }
    if tch::Cuda::device_count() < 2 {
        eprintln!("skipped: fewer than 2 CUDA devices");
        return;
    }
    eprintln!(
        "multi-gpu harness is scaffolded but intentionally disabled until single-GPU gate is green"
    );
    return;
}
