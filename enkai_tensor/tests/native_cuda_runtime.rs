#[cfg(feature = "cuda-kernels")]
use enkai_tensor::native_runtime::CpuBackend;
use enkai_tensor::native_runtime::{CudaNativeBackend, ExecutionBackend, NativeTensor};

#[cfg(feature = "cuda-kernels")]
fn assert_close(left: &[f32], right: &[f32], tolerance: f32) {
    assert_eq!(left.len(), right.len());
    for (idx, (a, b)) in left.iter().zip(right.iter()).enumerate() {
        assert!((a - b).abs() <= tolerance, "mismatch at {idx}: {a} vs {b}");
    }
}

#[cfg(not(feature = "cuda-kernels"))]
#[test]
fn native_cuda_backend_is_explicitly_feature_gated() {
    assert!(!CudaNativeBackend::available());
    let mut backend = CudaNativeBackend::new();
    let a = NativeTensor::from_vec(&[2], vec![1.0, 2.0]).unwrap();
    let b = NativeTensor::from_vec(&[2], vec![3.0, 4.0]).unwrap();
    let err = backend.add(&a, &b).unwrap_err();
    assert!(err.contains("E_CUDA_UNAVAILABLE"));
}

#[cfg(feature = "cuda-kernels")]
#[test]
fn native_cuda_backend_matches_cpu_for_core_training_ops() {
    if !CudaNativeBackend::available() {
        println!(
            "ENKAI_NATIVE_CUDA_EVIDENCE={}",
            serde_json::json!({"skipped": true, "reason": "no CUDA device"})
        );
        return;
    }

    let mut cpu = CpuBackend::with_parallelism(1);
    let mut cuda = CudaNativeBackend::new();
    let a = NativeTensor::from_vec(
        &[32, 32],
        (0..1024).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect(),
    )
    .unwrap();
    let b = NativeTensor::from_vec(
        &[32, 32],
        (0..1024).map(|i| ((i % 19) as f32 - 9.0) * 0.01).collect(),
    )
    .unwrap();
    let v0 = NativeTensor::from_vec(&[1024], a.data.clone()).unwrap();
    let v1 = NativeTensor::from_vec(&[1024], b.data.clone()).unwrap();

    let started = std::time::Instant::now();
    let cpu_add = cpu.add(&v0, &v1).unwrap();
    let cpu_matmul = cpu.matmul(&a, &b).unwrap();
    let cpu_softmax = cpu.softmax(&a).unwrap();
    let cpu_ce = cpu
        .cross_entropy(&a, &(0..32).map(|i| i % 32).collect::<Vec<_>>())
        .unwrap();
    let cpu_ms = started.elapsed().as_secs_f64() * 1000.0;

    let started = std::time::Instant::now();
    let cuda_add = cuda.add(&v0, &v1).unwrap();
    let cuda_matmul = cuda.matmul(&a, &b).unwrap();
    let cuda_softmax = cuda.softmax(&a).unwrap();
    let cuda_ce = cuda
        .cross_entropy(&a, &(0..32).map(|i| i % 32).collect::<Vec<_>>())
        .unwrap();
    let cuda_ms = started.elapsed().as_secs_f64() * 1000.0;

    assert_close(&cpu_add.data, &cuda_add.data, 1e-6);
    assert_close(&cpu_matmul.data, &cuda_matmul.data, 2e-5);
    assert_close(&cpu_softmax.data, &cuda_softmax.data, 2e-6);
    assert!((cpu_ce - cuda_ce).abs() < 1e-5, "{cpu_ce} vs {cuda_ce}");

    println!(
        "ENKAI_NATIVE_CUDA_EVIDENCE={}",
        serde_json::json!({
            "skipped": false,
            "backend": cuda.name(),
            "ops": ["vec_add", "matmul", "matmul_cublas", "softmax", "cross_entropy"],
            "cpu_ms": cpu_ms,
            "cuda_ms": cuda_ms,
            "cuda_memory": {
                "peak_bytes": cuda.memory_stats().peak_bytes,
                "allocated_bytes": cuda.memory_stats().allocated_bytes,
                "freed_bytes": cuda.memory_stats().freed_bytes
            },
            "pytorch_core_execution_dependency": false
        })
    );
}

#[cfg(feature = "cuda-kernels")]
#[test]
fn native_cuda_cublas_large_matmul_gate() {
    if !CudaNativeBackend::available() {
        return;
    }
    let n = 512usize;
    let iters = 5usize;
    let a = NativeTensor::from_vec(
        &[n, n],
        (0..n * n).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect(),
    )
    .unwrap();
    let b = NativeTensor::from_vec(
        &[n, n],
        (0..n * n).map(|i| ((i % 19) as f32 - 9.0) * 0.01).collect(),
    )
    .unwrap();
    let mut cuda = CudaNativeBackend::new();
    let warmup = cuda.matmul(&a, &b).unwrap();
    cuda.release(warmup);
    let started = std::time::Instant::now();
    let mut checksum = 0.0f32;
    for _ in 0..iters {
        let out = cuda.matmul(&a, &b).unwrap();
        checksum += out.data.iter().copied().sum::<f32>();
        cuda.release(out);
    }
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    println!(
        "ENKAI_NATIVE_CUDA_LARGE_MATMUL={}",
        serde_json::json!({
            "backend": cuda.name(),
            "kernel": "enkai_cuda_matmul_bias_cublas_f32",
            "shape": [n, n],
            "iterations": iters,
            "elapsed_ms": elapsed_ms,
            "checksum": checksum,
            "pytorch_core_execution_dependency": false
        })
    );
}
