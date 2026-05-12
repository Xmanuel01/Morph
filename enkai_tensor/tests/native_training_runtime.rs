use enkai_tensor::native_runtime::{
    benchmark_native_runtime, fused_add_relu, train_mlp_adamw, train_mlp_sgd, AdamWConfig,
    CpuBackend, CudaBackendHook, ExecutionBackend, GraphOp, GraphValue, NativeTensor, TensorGraph,
};

#[test]
fn graph_ir_executes_native_cpu_ops() {
    let a = NativeTensor::from_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = NativeTensor::from_vec(&[2, 2], vec![2.0, 0.0, 1.0, 2.0]).unwrap();
    let mut graph = TensorGraph::new();
    let a_id = graph.push(GraphOp::Input(a));
    let b_id = graph.push(GraphOp::Input(b));
    let mm = graph.push(GraphOp::Matmul(a_id, b_id));
    let relu = graph.push(GraphOp::Relu(mm));
    let sum = graph.push(GraphOp::Sum(relu));
    let mut backend = CpuBackend::new();
    let values = graph.execute(&mut backend).unwrap();
    match &values[sum] {
        GraphValue::Scalar(v) => assert!((*v - 26.0).abs() < 1e-6),
        _ => panic!("expected scalar"),
    }
    assert!(backend.memory_stats().allocated_bytes > 0);
}

#[test]
fn fusion_matches_unfused_add_relu() {
    let a = NativeTensor::from_vec(&[4], vec![-2.0, -0.5, 1.0, 2.0]).unwrap();
    let b = NativeTensor::from_vec(&[4], vec![1.0, 1.0, -2.0, 3.0]).unwrap();
    let mut backend = CpuBackend::new();
    let add = backend.add(&a, &b).unwrap();
    let unfused = backend.relu(&add).unwrap();
    let fused = fused_add_relu(&mut CpuBackend::new(), &a, &b).unwrap();
    assert_eq!(unfused.data, fused.data);
}

#[test]
fn mlp_sgd_training_reduces_loss_without_torch() {
    let report = train_mlp_sgd(300, 0.4).unwrap();
    assert!(report.loss_final < report.loss_initial, "{report:?}");
    assert!(report.peak_memory_bytes > 0);
}

#[test]
fn adamw_training_reduces_loss_more_than_sgd_on_bounded_mlp() {
    let sgd = train_mlp_sgd(200, 0.4).unwrap();
    let adamw = train_mlp_adamw(
        200,
        AdamWConfig {
            lr: 0.05,
            ..AdamWConfig::default()
        },
    )
    .unwrap();
    assert!(adamw.loss_final < adamw.loss_initial, "{adamw:?}");
    assert!(
        adamw.loss_final < sgd.loss_final,
        "adamw={adamw:?} sgd={sgd:?}"
    );
    assert!(adamw.reuse_count > 0);
}

#[test]
fn parallel_cpu_matmul_matches_single_threaded_backend() {
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
    let mut serial = CpuBackend::with_parallelism(1);
    let mut parallel = CpuBackend::with_parallelism(4);
    let serial_out = serial.matmul(&a, &b).unwrap();
    let parallel_out = parallel.matmul(&a, &b).unwrap();
    assert_eq!(serial_out.shape, parallel_out.shape);
    for (left, right) in serial_out.data.iter().zip(parallel_out.data.iter()) {
        assert!((left - right).abs() < 1e-6);
    }
}

#[test]
fn benchmark_emits_evidence_and_cuda_hook_is_not_pytorch() {
    let evidence = benchmark_native_runtime(2).unwrap();
    assert_eq!(evidence["runtime"], "enkai_native_training_runtime_cpu");
    assert_eq!(
        evidence["claims"]["pytorch_core_execution_dependency"],
        false
    );
    assert!(
        evidence["training"]["loss_final"].as_f64().unwrap()
            < evidence["training"]["loss_initial"].as_f64().unwrap()
    );
    assert!(
        evidence["memory"]["unfused"]["reuse_count"]
            .as_u64()
            .unwrap()
            > 0
    );
    for key in [
        "vector_add",
        "matmul",
        "mlp_forward",
        "mlp_training_step",
        "mlp_training_adamw",
        "softmax_cross_entropy",
        "memory_stress",
    ] {
        assert!(
            evidence["benchmarks"].get(key).is_some(),
            "missing benchmark evidence for {key}"
        );
    }
    assert!(
        evidence["benchmarks"]["memory_stress"]["memory"]["reuse_count"]
            .as_u64()
            .unwrap()
            > 0
    );
    let hook = CudaBackendHook::default();
    assert!(!hook.available);
    assert!(hook
        .dispatch_error()
        .contains("no PyTorch execution dependency"));
    println!("ENKAI_NATIVE_RUNTIME_EVIDENCE={}", evidence);
}
