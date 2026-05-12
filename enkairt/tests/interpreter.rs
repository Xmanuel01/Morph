use enkaic::compiler::compile_module;
use enkaic::parser::parse_module;
use enkairt::error::RuntimeError;
use enkairt::object::Obj;
use enkairt::{Value, VM};
use std::sync::{Mutex, OnceLock};

fn interpreter_test_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn run_value(source: &str) -> Value {
    run_value_with_accel(source, false)
}

fn run_value_with_accel(source: &str, accel_enabled: bool) -> Value {
    let _guard = interpreter_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let module = parse_module(source).expect("parse");
    let program = compile_module(&module).expect("compile");
    let mut vm = VM::new(false, false, false, false);
    vm.set_sim_accel_enabled(accel_enabled);
    vm.run(&program).expect("run")
}

fn run_result(source: &str) -> Result<Value, RuntimeError> {
    run_result_with_accel(source, false)
}

fn run_result_with_accel(source: &str, accel_enabled: bool) -> Result<Value, RuntimeError> {
    let _guard = interpreter_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let module = parse_module(source).expect("parse");
    let program = compile_module(&module).expect("compile");
    let mut vm = VM::new(false, false, false, false);
    vm.set_sim_accel_enabled(accel_enabled);
    vm.run(&program)
}

fn next_seed(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed
}

fn rand_index(seed: &mut u64, upper: i64) -> i64 {
    (next_seed(seed) % (upper as u64)) as i64
}

fn rand_float(seed: &mut u64) -> f64 {
    let raw = (next_seed(seed) % 2001) as i64 - 1000;
    raw as f64 / 100.0
}

#[test]
fn let_and_arithmetic() {
    let result = run_value("let x := 2 + 3\nx");
    assert_eq!(result, Value::Int(5));
}

#[test]
fn assignment_updates() {
    let result = run_value("mut x := 1\nx := x + 4\nx");
    assert_eq!(result, Value::Int(5));
}

#[test]
fn function_call_works() {
    let result = run_value("fn add(a: Int, b: Int) -> Int ::\n    return a + b\n::\nadd(2, 3)");
    assert_eq!(result, Value::Int(5));
}

#[test]
fn return_default_null() {
    let result = run_value("fn noop() -> Int ::\n    // no return\n::\nnoop()");
    assert_eq!(result, Value::Null);
}

#[test]
fn if_else_branches() {
    let result = run_value(
        "let x := 1\nmut y := 0\nif x == 1 ::\n    y := 10\n::\nelse ::\n    y := 5\n::\ny",
    );
    assert_eq!(result, Value::Int(10));
}

#[test]
fn while_loop_counts() {
    let result = run_value("mut x := 0\nwhile x < 3 ::\n    x := x + 1\n::\nx");
    assert_eq!(result, Value::Int(3));
}

#[test]
fn globals_work() {
    let result = run_value("let g := 2\nfn add1(a: Int) -> Int ::\n    return a + g\n::\nadd1(3)");
    assert_eq!(result, Value::Int(5));
}

#[test]
fn logic_short_circuit_and() {
    let result = run_value(
        "mut x := 0\nfn set() -> Bool ::\n    x := 1\n    return true\n::\nfalse and set()\nx",
    );
    assert_eq!(result, Value::Int(0));
}

#[test]
fn logic_short_circuit_or() {
    let result = run_value(
        "mut x := 0\nfn set() -> Bool ::\n    x := 1\n    return true\n::\ntrue or set()\nx",
    );
    assert_eq!(result, Value::Int(0));
}

#[test]
fn list_literal_and_index() {
    let result = run_value("let xs := [1, 2, 3]\nxs[1]");
    assert_eq!(result, Value::Int(2));
}

#[test]
fn list_index_assignment() {
    let result = run_value("let xs := [1, 2]\nxs[0] := 5\nxs[0]");
    assert_eq!(result, Value::Int(5));
}

#[test]
fn array_runtime_introspection_reports_element_type() {
    let result = run_value(
        "import std::array\n\
         let names := [\"Nairobi\", \"Mombasa\"]\n\
         if array.is_homogeneous(names) ::\n\
             return array.element_type(names)\n\
         ::\n\
         return \"bad\"\n",
    );
    match result {
        Value::Obj(obj) => match obj.as_obj() {
            Obj::String(value) => assert_eq!(value, "String"),
            other => panic!("expected String object, got {other:?}"),
        },
        other => panic!("expected String value, got {other:?}"),
    }
}

#[test]
fn dense_vector_runtime_ops_work() {
    let result = run_value(
        "import std::vector\n\
         let a := vector.from_array([1.0, 2.0, 3.0])\n\
         let b := vector.scale(a, 2.0)\n\
         vector.dot(a, b) + vector.get(b, 1) + vector.len(a)\n",
    );
    assert_eq!(result, Value::Float(35.0));
}

#[test]
fn tensor_runtime_shape_and_matmul_work() {
    let result = run_value(
        "import std::tensor\n\
         let a := tensor.from_array([1.0, 2.0, 3.0, 4.0], [2, 2])\n\
         let b := tensor.from_array([5.0, 6.0, 7.0, 8.0], [2, 2])\n\
         let c := tensor.matmul(a, b)\n\
         let s := tensor.shape(c)\n\
         tensor.get_flat(c, 0) + tensor.get_flat(c, 3) + s[0] + s[1]\n",
    );
    assert_eq!(result, Value::Float(73.0));
}

#[test]
fn tensor_runtime_rejects_shape_mismatch() {
    let err = run_result(
        "import std::tensor\n\
         let a := tensor.from_array([1.0, 2.0], [2])\n\
         let b := tensor.from_array([1.0, 2.0, 3.0], [3])\n\
         tensor.add(a, b)\n",
    )
    .expect_err("shape mismatch should fail");
    assert!(
        err.message.contains("broadcasting shape mismatch"),
        "{}",
        err.message
    );
}

#[test]
fn tensor_runtime_broader_ops_are_deterministic() {
    let result = run_value(
        "import std::tensor\n\
         let a := tensor.from_array([1.0, 2.0, 3.0, 4.0], [2, 2])\n\
         let transposed := tensor.transpose(a, 0, 1)\n\
         let row_sum := tensor.sum(a, 1, false)\n\
         let row_mean := tensor.mean(a, 1, false)\n\
         let sliced := tensor.slice(a, 0, 1, 2, 1)\n\
         let joined := tensor.concat([sliced, sliced], 0)\n\
         tensor.get_flat(transposed, 1) + tensor.get_flat(row_sum, 1) + tensor.get_flat(row_mean, 0) + tensor.get_flat(joined, 3)\n",
    );
    assert_eq!(result, Value::Float(15.5));
}

#[test]
fn tensor_runtime_activation_ops_work() {
    let result = run_value(
        "import std::tensor\n\
         let a := tensor.from_array([-1.0, 0.0, 1.0, 2.0], [2, 2])\n\
         let r := tensor.relu(a)\n\
         let s := tensor.sigmoid(a)\n\
         let p := tensor.softmax(a, 1)\n\
         tensor.get_flat(r, 0) + tensor.get_flat(r, 3) + tensor.get_flat(s, 1) + tensor.get_flat(p, 0) + tensor.get_flat(p, 1)\n",
    );
    match result {
        Value::Float(value) => assert!((value - 3.5).abs() < 1e-9, "{value}"),
        other => panic!("expected Float value, got {other:?}"),
    }
}

#[test]
fn tensor_runtime_rejects_invalid_dimensions() {
    let err = run_result(
        "import std::tensor\n\
         let a := tensor.from_array([1.0, 2.0, 3.0, 4.0], [2, 2])\n\
         tensor.transpose(a, 0, 3)\n",
    )
    .expect_err("invalid dimension should fail");
    assert!(
        err.message.contains("out of bounds for rank 2"),
        "{}",
        err.message
    );
}

#[test]
fn tensor_runtime_broadcasting_and_scalar_style_ops_work() {
    let result = run_value(
        "import std::tensor\n\
         let a := tensor.from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])\n\
         let b := tensor.from_array([10.0, 20.0, 30.0], [1, 3])\n\
         let added := tensor.add(a, b)\n\
         let scaled := tensor.scale(added, 0.5)\n\
         let broadcasted := tensor.broadcast_to(tensor.from_array([1.0, 2.0, 3.0], [1, 3]), [2, 3])\n\
         tensor.get_flat(scaled, 0) + tensor.get_flat(scaled, 5) + tensor.get_flat(broadcasted, 3)\n",
    );
    assert_eq!(result, Value::Float(24.5));
}

#[test]
fn tensor_runtime_ai_core_fallbacks_work() {
    let result = run_value(
        "import std::tensor\n\
         let x := tensor.from_array([1.0, 2.0, 3.0, 4.0], [2, 2])\n\
         let w := tensor.from_array([1.0, 0.0, 0.0, 1.0], [2, 2])\n\
         let b := tensor.from_array([0.5, -0.5], [2])\n\
         let projected := tensor.linear(x, w, b)\n\
         let normalized := tensor.layernorm(projected, tensor.from_array([1.0, 1.0], [2]), tensor.from_array([0.0, 0.0], [2]), 0.00001)\n\
         let embedded := tensor.embedding(tensor.from_array([1.0, 2.0, 3.0, 4.0], [2, 2]), tensor.from_array([1.0, 0.0], [2]))\n\
         let loss := tensor.cross_entropy(tensor.from_array([3.0, 1.0, 1.0, 3.0], [2, 2]), tensor.from_array([0.0, 1.0], [2]))\n\
         tensor.get_flat(projected, 0) + tensor.get_flat(normalized, 0) + tensor.get_flat(embedded, 0) + tensor.get_flat(loss, 0)\n",
    );
    match result {
        Value::Float(value) => assert!((value - 4.626_928).abs() < 1e-5, "{value}"),
        other => panic!("expected Float value, got {other:?}"),
    }
}

#[test]
fn tensor_runtime_rejects_broadcast_shape_mismatch() {
    let err = run_result(
        "import std::tensor\n\
         let a := tensor.from_array([1.0, 2.0], [2])\n\
         let b := tensor.from_array([1.0, 2.0, 3.0], [3])\n\
         tensor.add(a, b)\n",
    )
    .expect_err("broadcast mismatch should fail");
    assert!(
        err.message.contains("broadcasting shape mismatch"),
        "{}",
        err.message
    );
}

#[test]
fn tensor_runtime_indexing_selection_and_mask_ops_work() {
    let result = run_value(
        "import std::tensor\n\
         let x := tensor.from_array([1.0, 4.0, 3.0, 2.0], [2, 2])\n\
         let mask := tensor.from_array([1.0, 0.0], [1, 2])\n\
         let filled := tensor.masked_fill(x, mask, -1.0)\n\
         let chosen := tensor.where(mask, x, filled)\n\
         let clipped := tensor.clip(chosen, 0.0, 3.0)\n\
         let sorted := tensor.sort(clipped, 1, true)\n\
         let top := tensor.topk(clipped, 1, 1)\n\
         let arg := tensor.argmax(clipped, 1, false)\n\
         let gathered := tensor.gather(clipped, 1, tensor.from_array([0.0, 1.0, 1.0, 0.0], [2, 2]))\n\
         let scattered := tensor.scatter(clipped, 1, tensor.from_array([1.0, 0.0, 1.0, 0.0], [2, 2]), tensor.from_array([8.0, 9.0, 7.0, 6.0], [2, 2]))\n\
         tensor.get_flat(sorted, 0) + tensor.get_flat(top, 1) + tensor.get_flat(arg, 0) + tensor.get_flat(gathered, 2) + tensor.get_flat(scattered, 0)\n",
    );
    assert_eq!(result, Value::Float(18.0));
}

#[test]
fn tensor_runtime_conv_pool_batchnorm_einsum_attention_work() {
    let result = run_value(
        "import std::tensor\n\
         let image := tensor.from_array([1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2])\n\
         let kernel := tensor.from_array([1.0], [1, 1, 1, 1])\n\
         let conv := tensor.conv2d(image, kernel, tensor.from_array([0.5], [1]), 1, 0)\n\
         let maxp := tensor.max_pool2d(image, 2, 1)\n\
         let avgp := tensor.avg_pool2d(image, 2, 1)\n\
         let mat := tensor.einsum(\"ij,jk->ik\", tensor.from_array([1.0, 2.0, 3.0, 4.0], [2, 2]), tensor.from_array([1.0, 0.0, 0.0, 1.0], [2, 2]))\n\
         let bn := tensor.batchnorm1d(tensor.from_array([1.0, 2.0, 3.0, 4.0], [2, 2]), tensor.from_array([1.0, 1.0], [2]), tensor.from_array([0.0, 0.0], [2]), 0.00001, true)\n\
         let attn := tensor.attention(tensor.from_array([1.0, 0.0, 0.0, 1.0], [1, 2, 2]), tensor.from_array([1.0, 0.0, 0.0, 1.0], [1, 2, 2]), tensor.from_array([1.0, 2.0, 3.0, 4.0], [1, 2, 2]))\n\
         tensor.get_flat(conv, 0) + tensor.get_flat(maxp, 0) + tensor.get_flat(avgp, 0) + tensor.get_flat(mat, 3) + tensor.get_flat(bn, 0) + tensor.get_flat(attn, 0)\n",
    );
    match result {
        Value::Float(value) => assert!((value - 12.660_482).abs() < 1e-5, "{value}"),
        other => panic!("expected Float value, got {other:?}"),
    }
}

#[test]
fn tensor_runtime_autodiff_accumulates_gradients() {
    let result = run_value(
        "import std::tensor\n\
         let x := tensor.requires_grad(tensor.from_array([2.0, 3.0], [2]))\n\
         let y := tensor.mul(x, x)\n\
         let loss := tensor.sum(y, 0, false)\n\
         tensor.backward(loss)\n\
         let gx := tensor.grad(x)\n\
         let expected := tensor.from_array([4.0, 6.0], [2])\n\
         if tensor.grad_check(gx, expected, 0.00001) ::\n\
             return tensor.get_flat(gx, 0) + tensor.get_flat(gx, 1)\n\
         ::\n\
         return -1.0\n",
    );
    assert_eq!(result, Value::Float(10.0));
}

#[test]
fn tensor_runtime_autodiff_matmul_backward_works() {
    let result = run_value(
        "import std::tensor\n\
         let x := tensor.requires_grad(tensor.from_array([1.0, 2.0], [1, 2]))\n\
         let w := tensor.requires_grad(tensor.from_array([3.0, 4.0], [2, 1]))\n\
         let y := tensor.matmul(x, w)\n\
         tensor.backward(y)\n\
         let gx := tensor.grad(x)\n\
         let gw := tensor.grad(w)\n\
         tensor.get_flat(gx, 0) + tensor.get_flat(gx, 1) + tensor.get_flat(gw, 0) + tensor.get_flat(gw, 1)\n",
    );
    assert_eq!(result, Value::Float(10.0));
}

#[test]
fn tensor_runtime_autodiff_model_ops_backward_work() {
    let result = run_value(
        "import std::tensor\n\
         let x := tensor.requires_grad(tensor.from_array([1.0, 2.0], [1, 2]))\n\
         let w := tensor.requires_grad(tensor.from_array([3.0, 4.0, 5.0, 6.0], [2, 2]))\n\
         let b := tensor.requires_grad(tensor.from_array([0.5, -0.5], [2]))\n\
         let y := tensor.linear(x, w, b)\n\
         let loss := tensor.sum(y, 1, false)\n\
         tensor.backward(loss)\n\
         let gx := tensor.grad(x)\n\
         let gw := tensor.grad(w)\n\
         let gb := tensor.grad(b)\n\
         tensor.get_flat(gx, 0) + tensor.get_flat(gx, 1) + tensor.get_flat(gw, 0) + tensor.get_flat(gw, 1) + tensor.get_flat(gw, 2) + tensor.get_flat(gw, 3) + tensor.get_flat(gb, 0) + tensor.get_flat(gb, 1)\n",
    );
    assert_eq!(result, Value::Float(26.0));
}

#[test]
fn tensor_runtime_autodiff_cross_entropy_embedding_and_layernorm_work() {
    let ce = run_value(
        "import std::tensor\n\
         let logits := tensor.requires_grad(tensor.from_array([0.0, 0.0, 0.0, 0.0], [2, 2]))\n\
         let loss := tensor.cross_entropy(logits, tensor.from_array([0.0, 1.0], [2]))\n\
         tensor.backward(loss)\n\
         let grad := tensor.grad(logits)\n\
         tensor.get_flat(grad, 0) + tensor.get_flat(grad, 1) * 10.0 + tensor.get_flat(grad, 2) * 100.0 + tensor.get_flat(grad, 3) * 1000.0\n",
    );
    assert_eq!(ce, Value::Float(-222.75));

    let embedding = run_value(
        "import std::tensor\n\
         let weights := tensor.requires_grad(tensor.from_array([1.0, 2.0, 3.0, 4.0], [2, 2]))\n\
         let out := tensor.embedding(weights, tensor.from_array([1.0, 0.0, 1.0], [3]))\n\
         let loss := tensor.sum(out, 0, false)\n\
         let loss2 := tensor.sum(loss, 0, false)\n\
         tensor.backward(loss2)\n\
         let grad := tensor.grad(weights)\n\
         tensor.get_flat(grad, 0) + tensor.get_flat(grad, 1) + tensor.get_flat(grad, 2) + tensor.get_flat(grad, 3)\n",
    );
    assert_eq!(embedding, Value::Float(6.0));

    let layernorm = run_value(
        "import std::tensor\n\
         let x := tensor.requires_grad(tensor.from_array([1.0, 3.0], [1, 2]))\n\
         let w := tensor.requires_grad(tensor.from_array([1.0, 1.0], [2]))\n\
         let b := tensor.requires_grad(tensor.from_array([0.0, 0.0], [2]))\n\
         let y := tensor.layernorm(x, w, b, 0.000001)\n\
         let loss := tensor.sum(y, 1, false)\n\
         tensor.backward(loss)\n\
         let gw := tensor.grad(w)\n\
         let gb := tensor.grad(b)\n\
         tensor.get_flat(gw, 0) + tensor.get_flat(gw, 1) + tensor.get_flat(gb, 0) + tensor.get_flat(gb, 1)\n",
    );
    match layernorm {
        Value::Float(value) => assert!((value - 2.0).abs() < 1e-6, "{value}"),
        other => panic!("expected Float value, got {other:?}"),
    }
}

#[test]
fn tensor_runtime_autodiff_conv_pool_attention_and_optimizer_work() {
    let conv = run_value(
        "import std::tensor\n\
         let image := tensor.requires_grad(tensor.from_array([1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2]))\n\
         let kernel := tensor.requires_grad(tensor.from_array([2.0], [1, 1, 1, 1]))\n\
         let bias := tensor.requires_grad(tensor.from_array([0.5], [1]))\n\
         let out := tensor.conv2d(image, kernel, bias, 1, 0)\n\
         let loss := tensor.sum(tensor.sum(tensor.sum(out, 3, false), 2, false), 1, false)\n\
         tensor.backward(loss)\n\
         let gi := tensor.grad(image)\n\
         let gw := tensor.grad(kernel)\n\
         let gb := tensor.grad(bias)\n\
         tensor.get_flat(gi, 0) + tensor.get_flat(gi, 1) + tensor.get_flat(gi, 2) + tensor.get_flat(gi, 3) + tensor.get_flat(gw, 0) + tensor.get_flat(gb, 0)\n",
    );
    assert_eq!(conv, Value::Float(22.0));

    let pool = run_value(
        "import std::tensor\n\
         let image := tensor.requires_grad(tensor.from_array([1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2]))\n\
         let avg := tensor.avg_pool2d(image, 2, 1)\n\
         tensor.backward(tensor.sum(tensor.sum(tensor.sum(avg, 3, false), 2, false), 1, false))\n\
         let ga := tensor.grad(image)\n\
         tensor.zero_grad(image)\n\
         let maxp := tensor.max_pool2d(image, 2, 1)\n\
         tensor.backward(tensor.sum(tensor.sum(tensor.sum(maxp, 3, false), 2, false), 1, false))\n\
         let gm := tensor.grad(image)\n\
         tensor.get_flat(ga, 0) + tensor.get_flat(ga, 1) + tensor.get_flat(ga, 2) + tensor.get_flat(ga, 3) + tensor.get_flat(gm, 3)\n",
    );
    assert_eq!(pool, Value::Float(2.0));

    let attention = run_value(
        "import std::tensor\n\
         let q := tensor.requires_grad(tensor.from_array([1.0, 0.0, 0.0, 1.0], [1, 2, 2]))\n\
         let k := tensor.requires_grad(tensor.from_array([1.0, 0.0, 0.0, 1.0], [1, 2, 2]))\n\
         let v := tensor.requires_grad(tensor.from_array([1.0, 2.0, 3.0, 4.0], [1, 2, 2]))\n\
         let out := tensor.attention(q, k, v)\n\
         let loss := tensor.sum(tensor.sum(out, 2, false), 1, false)\n\
         tensor.backward(loss)\n\
         let gv := tensor.grad(v)\n\
         tensor.get_flat(gv, 0) + tensor.get_flat(gv, 1) + tensor.get_flat(gv, 2) + tensor.get_flat(gv, 3)\n",
    );
    match attention {
        Value::Float(value) => assert!((value - 4.0).abs() < 1e-6, "{value}"),
        other => panic!("expected Float value, got {other:?}"),
    }

    let opt = run_value(
        "import std::tensor\n\
         let p := tensor.from_array([1.0, 2.0], [2])\n\
         let g := tensor.from_array([0.1, 0.2], [2])\n\
         tensor.sgd_step(p, g, 0.5, 0.0)\n\
         tensor.get_flat(p, 0) + tensor.get_flat(p, 1)\n",
    );
    match opt {
        Value::Float(value) => assert!((value - 2.85).abs() < 1e-9, "{value}"),
        other => panic!("expected Float value, got {other:?}"),
    }
}

#[test]
fn tensor_runtime_memory_accounting_and_limits_work() {
    let result = run_value(
        "import std::tensor\n\
         tensor.memory_clear_limit()\n\
         tensor.memory_reset_peak()\n\
         let before := tensor.memory_current()\n\
         let t := tensor.from_array([1.0, 2.0, 3.0, 4.0], [2, 2])\n\
         let after := tensor.memory_current()\n\
         let peak := tensor.memory_peak()\n\
         if peak < after ::\n\
             return -1\n\
         ::\n\
         after - before\n",
    );
    assert_eq!(result, Value::Int(32));

    let err = run_result(
        "import std::tensor\n\
         let base := tensor.memory_current()\n\
         tensor.memory_set_limit(base + 8)\n\
         tensor.from_array([1.0, 2.0], [2])\n",
    )
    .expect_err("memory limit should reject allocation");
    assert!(
        err.message
            .contains("tensor.from_array would exceed tensor memory limit"),
        "{}",
        err.message
    );
    let _ = run_value("import std::tensor\ntensor.memory_clear_limit()\n");
}

#[test]
fn tensor_runtime_multi_optimizer_and_grad_clipping_work() {
    let result = run_value(
        "import std::tensor\n\
         let p1 := tensor.requires_grad(tensor.from_array([1.0, 2.0], [2]))\n\
         let p2 := tensor.requires_grad(tensor.from_array([3.0], [1]))\n\
         let l1 := tensor.sum(tensor.mul(p1, p1), 0, false)\n\
         let l2 := tensor.sum(tensor.mul(p2, p2), 0, false)\n\
         let loss := tensor.add(l1, l2)\n\
         tensor.backward(loss)\n\
         let before_norm := tensor.clip_grad_norm([p1, p2], 1.0, 0.000000001)\n\
         let g1 := tensor.grad(p1)\n\
         let g2 := tensor.grad(p2)\n\
         mut state := tensor.adamw_state()\n\
         state := tensor.adamw_step_multi([p1, p2], [g1, g2], state, 0.01, 0.9, 0.999, 0.00000001, 0.0)\n\
         tensor.zero_grad_multi([p1, p2])\n\
         let zg1 := tensor.grad(p1)\n\
         let zg2 := tensor.grad(p2)\n\
         before_norm + tensor.get_flat(zg1, 0) + tensor.get_flat(zg1, 1) + tensor.get_flat(zg2, 0)\n",
    );
    match result {
        Value::Float(value) => assert!((value - 7.483_314_773_547883).abs() < 1e-6, "{value}"),
        other => panic!("expected Float value, got {other:?}"),
    }
}

#[test]
fn tensor_runtime_multi_optimizer_errors_are_deterministic() {
    let err = run_result(
        "import std::tensor\n\
         let p := tensor.from_array([1.0], [1])\n\
         let g := tensor.from_array([0.1], [1])\n\
         tensor.sgd_step_multi([p], [g, g], 0.1, 0.0)\n",
    )
    .expect_err("mismatched optimizer inputs should fail");
    assert!(
        err.message.contains("matching params/grads lengths"),
        "{}",
        err.message
    );
}

#[test]
fn tensor_runtime_autodiff_detach_and_no_grad_are_deterministic() {
    let detached = run_result(
        "import std::tensor\n\
         let x := tensor.requires_grad(tensor.from_array([2.0], [1]))\n\
         let y := tensor.detach(x)\n\
         let loss := tensor.sum(tensor.mul(y, y), 0, false)\n\
         tensor.backward(loss)\n",
    )
    .expect_err("detached graph should not backpropagate");
    assert!(
        detached.message.contains("tensor without gradient graph"),
        "{}",
        detached.message
    );

    let no_grad = run_result(
        "import std::tensor\n\
         let _prev := tensor.no_grad(true)\n\
         let x := tensor.requires_grad(tensor.from_array([2.0], [1]))\n\
         let y := tensor.mul(x, x)\n\
         let _restore := tensor.no_grad(false)\n\
         tensor.backward(y)\n",
    )
    .expect_err("no_grad graph should not backpropagate");
    assert!(
        no_grad.message.contains("tensor without gradient graph"),
        "{}",
        no_grad.message
    );
}

#[test]
fn record_field_assignment() {
    let result = run_value("let r := json.parse(\"{\\\"a\\\":0}\")\nr.a := 7\nr.a");
    assert_eq!(result, Value::Int(7));
}

#[test]
fn try_unwrap_value() {
    let result = run_value("let x := 5\nx?");
    assert_eq!(result, Value::Int(5));
}

#[test]
fn try_unwrap_none_errors() {
    let result = run_result("let x := none\nx?");
    assert!(result.is_err());
}

#[test]
fn sparse_matrix_nonzero_and_matvec_work() {
    let result = run_value(
        "import std::sparse\n\
         let m := sparse.matrix()\n\
         sparse.set(m, 1, 0, 3.0)\n\
         sparse.set(m, 0, 2, 4.0)\n\
         sparse.set(m, 1, 0, 0.0)\n\
         let items := sparse.nonzero(m)\n\
         let out := sparse.matvec(m, [2, 5, 7])\n\
         items[0].row * 100 + items[0].col * 10 + out[0]\n",
    );
    assert_eq!(result, Value::Float(48.0));
}

#[test]
fn sparse_vector_dot_and_nnz_work() {
    let result = run_value(
        "import std::sparse\n\
         let v := sparse.vector()\n\
         sparse.set_vector(v, 0, 1.5)\n\
         sparse.set_vector(v, 3, 2.0)\n\
         sparse.dot(v, [4.0, 1.0, 9.0, 5.0]) + sparse.nnz(v)\n",
    );
    assert_eq!(result, Value::Float(18.0));
}

#[test]
fn sparse_access_and_dense_length_mismatch_are_stable() {
    let result = run_value(
        "import std::sparse\n\
         let m := sparse.matrix()\n\
         sparse.set(m, 0, 0, 2.0)\n\
         sparse.set(m, 0, 3, 5.0)\n\
         sparse.set(m, 2, 1, 7.0)\n\
         sparse.set(m, 0, 3, 0.0)\n\
         let v := sparse.vector()\n\
         sparse.set_vector(v, 0, 1.0)\n\
         sparse.set_vector(v, 4, 3.0)\n\
         let miss_m := sparse.get(m, 1, 1)\n\
         let miss_v := sparse.get_vector(v, 3)\n\
         let mat := sparse.matvec(m, [2.0, 4.0])\n\
         let dot := sparse.dot(v, [5.0, 0.0, 0.0])\n\
         let items := sparse.nonzero(m)\n\
         mut out := 0\n\
         if miss_m == none and miss_v == none ::\n\
             out := sparse.nnz(m) * 100000 + sparse.nnz(v) * 10000 + items[0].row * 1000 + items[0].col * 100 + mat[0] * 10 + dot\n\
         ::\n\
         out\n",
    );
    assert_eq!(result, Value::Float(220045.0));
}

#[test]
fn sparse_invalid_indices_fail_consistently() {
    for accel_enabled in [true, false] {
        let matrix_err = run_result_with_accel(
            "import std::sparse\n\
             let m := sparse.matrix()\n\
             sparse.get(m, -1, 0)\n",
            accel_enabled,
        )
        .expect_err("negative sparse matrix index should fail");
        assert_eq!(matrix_err.message, "sparse.get expects row >= 0");

        let vector_err = run_result_with_accel(
            "import std::sparse\n\
             let v := sparse.vector()\n\
             sparse.set_vector(v, -1, 1.0)\n",
            accel_enabled,
        )
        .expect_err("negative sparse vector index should fail");
        assert_eq!(vector_err.message, "sparse.set_vector expects index >= 0");
    }
}

#[test]
fn sparse_and_pool_guardrails_reject_unbounded_runtime_shapes() {
    let row_err = run_result(
        "import std::sparse\n\
         let m := sparse.matrix()\n\
         sparse.set(m, 1000001, 0, 1.0)\n",
    )
    .expect_err("sparse matrix row above default guardrail should fail");
    assert_eq!(
        row_err.message,
        "sparse.set row exceeds ENKAI_SPARSE_MAX_INDEX"
    );

    let vector_err = run_result(
        "import std::sparse\n\
         let v := sparse.vector()\n\
         sparse.set_vector(v, 1000001, 1.0)\n",
    )
    .expect_err("sparse vector index above default guardrail should fail");
    assert_eq!(
        vector_err.message,
        "sparse.set_vector index exceeds ENKAI_SPARSE_MAX_INDEX"
    );

    let pool_err = run_result(
        "import std::pool\n\
         pool.make(1000001)\n",
    )
    .expect_err("pool capacity above default guardrail should fail");
    assert_eq!(
        pool_err.message,
        "pool.make exceeds ENKAI_POOL_MAX_CAPACITY"
    );

    let dense_err = run_result(
        "import std::sparse\n\
         let v := sparse.vector()\n\
         sparse.set_vector(v, 0, 1.0)\n\
         sparse.dot(v, [1e309])\n",
    )
    .expect_err("non-finite dense values should fail before sparse compute");
    assert_eq!(dense_err.message, "Dense values must be finite");
}

#[test]
fn event_queue_orders_by_time_then_insertion() {
    let result = run_value(
        "import std::event\n\
         let q := event.make()\n\
         event.push(q, 1.0, 10)\n\
         event.push(q, 1.0, 20)\n\
         event.push(q, 0.5, 5)\n\
         let a := event.pop(q)?\n\
         let b := event.pop(q)?\n\
         let c := event.pop(q)?\n\
         a.event * 10000 + b.event * 100 + c.event\n",
    );
    assert_eq!(result, Value::Int(51020));
}

#[test]
fn event_queue_peek_len_and_empty_are_stable() {
    let result = run_value(
        "import std::event\n\
         let q := event.make()\n\
         let empty_before := event.is_empty(q)\n\
         event.push(q, 2.0, 20)\n\
         event.push(q, 1.0, 10)\n\
         let peeked := event.peek(q)?\n\
         let len_before := event.len(q)\n\
         let popped := event.pop(q)?\n\
         let len_after := event.len(q)\n\
         let empty_after := event.is_empty(q)\n\
         mut out := 0\n\
         if empty_before and not empty_after ::\n\
             out := peeked.event * 10000 + popped.event * 100 + len_before * 10 + len_after\n\
         ::\n\
         out\n",
    );
    assert_eq!(result, Value::Int(101021));
}

#[test]
fn pool_fixed_and_growable_behaviors_work() {
    let fixed = run_value(
        "import std::pool\n\
         let p := pool.make(1)\n\
         let first := pool.release(p, 7)\n\
         let second := pool.release(p, 9)\n\
         let got := pool.acquire(p)?\n\
         let stats := pool.stats(p)\n\
         mut out := 0\n\
         if first and not second ::\n\
             out := got * 100 + stats.dropped_on_full\n\
         ::\n\
         out\n",
    );
    assert_eq!(fixed, Value::Int(701));

    let growable = run_value(
        "import std::pool\n\
         let p := pool.make_growable(1)\n\
         pool.release(p, 1)\n\
         pool.release(p, 2)\n\
         let stats := pool.stats(p)\n\
         stats.capacity * 10 + stats.available\n",
    );
    assert_eq!(growable, Value::Int(22));
}

#[test]
fn pool_reset_preserves_counters_and_clears_available_items() {
    let result = run_value(
        "import std::pool\n\
         let p := pool.make(2)\n\
         pool.release(p, 1)\n\
         pool.release(p, 2)\n\
         let before := pool.stats(p)\n\
         pool.reset(p)\n\
         let after := pool.stats(p)\n\
         before.available * 100000 + before.high_watermark * 10000 + before.releases * 1000 + after.available * 100 + after.high_watermark * 10 + after.releases\n",
    );
    assert_eq!(result, Value::Int(222022));
}

#[test]
fn sim_scheduler_snapshot_restore_and_replay_work() {
    let result = run_value(
        "import std::sim\n\
         let w := sim.make_seeded(8, 42)\n\
         sim.schedule(w, 2.0, 20)\n\
         sim.schedule(w, 1.0, 10)\n\
         let first := sim.step(w)?\n\
         let snap := sim.snapshot(w)\n\
         let restored := sim.restore(snap)\n\
         let second := sim.step(restored)?\n\
         let log := sim.log(restored)\n\
         let replayed := sim.replay(log, 8, sim.seed(restored))\n\
         let replay_log := sim.log(replayed)\n\
         first.event * 100000 + second.event * 1000 + replay_log[1].event * 10 + sim.pending(replayed)\n",
    );
    assert_eq!(result, Value::Int(1020200));
}

#[test]
fn sim_entities_round_trip_through_world_state() {
    let result = run_value(
        "import std::sim\n\
         let w := sim.make(4)\n\
         sim.entity_set(w, 7, 99)\n\
         let before := sim.entity_get(w, 7)?\n\
         let removed := sim.entity_remove(w, 7)\n\
         let after := sim.entity_get(w, 7)\n\
         mut out := 0\n\
         if removed and after == none ::\n\
             out := before * 100 + sim.pending(w)\n\
         ::\n\
         out\n",
    );
    assert_eq!(result, Value::Int(9900));
}

#[test]
fn sim_run_reports_stable_starvation_error_code() {
    let result = run_result(
        "import std::sim\n\
         let w := sim.make(4)\n\
         sim.schedule(w, 1.0, 1)\n\
         sim.schedule(w, 2.0, 2)\n\
         sim.run(w, 1)\n",
    );
    let err = result.expect_err("run should fail when work remains after max_steps");
    assert_eq!(err.code(), Some("E_SIM_STARVATION"));
}

#[test]
fn sim_restore_rejects_corrupted_snapshots() {
    let result = run_result(
        "import std::sim\n\
         import std::json\n\
         sim.restore(json.parse(\"{\\\"seed\\\":1,\\\"now\\\":0.0,\\\"max_events\\\":2,\\\"next_seq\\\":0,\\\"queue\\\":[{\\\"time\\\":-1.0,\\\"seq\\\":0,\\\"event\\\":1}],\\\"log\\\":[],\\\"entities\\\":[]}\"))\n",
    );
    let err = result.expect_err("restore should fail");
    assert_eq!(err.code(), Some("E_SIM_CORRUPTED_REPLAY"));
}

#[test]
fn sim_coroutines_yield_and_join_through_stdlib() {
    let result = run_value(
        "import std::sim\n\
         fn worker(coro: SimCoroutine, state: Int) -> Int ::\n\
             let world := sim.world(coro)\n\
             mut step := 0\n\
             while step < state ::\n\
                 sim.schedule(world, sim.time(world) + 1.0, [state, step])\n\
                 sim.emit(coro, [state, step])\n\
                 step := step + 1\n\
             ::\n\
             return sim.pending(world)\n\
         ::\n\
         let world := sim.make_seeded(16, 9)\n\
         let coro := sim.coroutine_with(world, worker, 3)\n\
         let a := sim.next(coro)?\n\
         let b := sim.next(coro)?\n\
         let c := sim.next(coro)?\n\
         let done_before := sim.done(coro)\n\
         let joined := sim.join(coro)?\n\
         let done_after := sim.done(coro)\n\
         mut out := a[0] * 100000 + b[1] * 10000 + c[1] * 1000 + joined * 10\n\
         if done_before ::\n\
             out := out + 1\n\
         ::\n\
         if done_after ::\n\
             out := out + 1\n\
         ::\n\
         out\n",
    );
    assert_eq!(result, Value::Int(312031));
}

#[test]
fn sim_next_surfaces_coroutine_errors() {
    let result = run_result(
        "import std::sim\n\
         fn worker(coro: SimCoroutine) -> Int ::\n\
             let world := sim.world(coro)\n\
             sim.schedule(world, 1.0, 7)\n\
             return none?\n\
         ::\n\
         let world := sim.make(8)\n\
         let coro := sim.coroutine(world, worker)\n\
         sim.next(coro)\n",
    );
    assert!(result.is_err());
}

#[test]
fn sim_coroutine_native_and_vm_paths_match() {
    let source = "import std::sim\n\
         fn worker(coro: SimCoroutine, state: Int) -> Int ::\n\
             let world := sim.world(coro)\n\
             mut step := 0\n\
             while step < state ::\n\
                 sim.schedule(world, sim.time(world) + 1.0 + step, [state, step])\n\
                 sim.emit(coro, [state, step])\n\
                 step := step + 1\n\
             ::\n\
             return sim.pending(world)\n\
         ::\n\
         let world := sim.make_seeded(16, 9)\n\
         let coro := sim.coroutine_with(world, worker, 3)\n\
         let a := sim.next(coro)?\n\
         let b := sim.next(coro)?\n\
         let c := sim.next(coro)?\n\
         let joined := sim.join(coro)?\n\
         let first := sim.step(world)?\n\
         let second := sim.step(world)?\n\
         let third := sim.step(world)?\n\
         let replayed := sim.replay(sim.log(world), 16, sim.seed(world))\n\
         joined * 100000 + a[0] * 10000 + b[1] * 1000 + c[1] * 100 + first.event[1] * 10 + sim.snapshot(replayed).seed\n";
    let native = run_value_with_accel(source, cfg!(not(windows)));
    let fallback = run_value_with_accel(source, false);
    assert_eq!(native, fallback);
}

#[test]
fn spatial_queries_and_rng_streams_are_deterministic() {
    let spatial = run_value(
        "import std::spatial\n\
         let idx := spatial.make()\n\
         spatial.upsert(idx, 1, 0.0, 0.0)\n\
         spatial.upsert(idx, 2, 0.5, 0.0)\n\
         spatial.upsert(idx, 3, 4.0, 0.0)\n\
         let near := spatial.radius(idx, 0.0, 0.0, 1.0)\n\
         let nearest := spatial.nearest(idx, 0.2, 0.0)?\n\
         spatial.occupancy(idx, -1.0, -1.0, 1.0, 1.0) * 1000 + near[0] * 100 + near[1] * 10 + nearest\n",
    );
    assert_eq!(spatial, Value::Int(2121));

    let rng = run_value(
        "import std::agent\n\
         import std::sim\n\
         import std::spatial\n\
         let world := sim.make_seeded(8, 11)\n\
         let idx := spatial.make()\n\
         let env := agent.make(world, idx)\n\
         agent.register(env, 7, 1, 2, 0.0, 0.0)\n\
         let a := agent.stream(env, 7, \"sensor\")\n\
         let b := agent.stream(env, 7, \"sensor\")\n\
         let x := agent.next_int(a, 100)\n\
         let y := agent.next_int(b, 100)\n\
         if x == y ::\n\
             return 1\n\
         ::\n\
         return 0\n",
    );
    assert_eq!(rng, Value::Int(1));
}

#[test]
fn spatial_rtree_frontier_is_deterministic_and_bounded() {
    let mut source = String::from("import std::spatial\nlet idx := spatial.make()\n");
    for id in 0..64 {
        let x = (id % 8) as f64;
        let y = (id / 8) as f64;
        source.push_str(&format!("spatial.upsert(idx, {id}, {x}, {y})\n"));
    }
    source.push_str(
        "spatial.upsert(idx, 63, 0.25, 0.25)\n\
         let near := spatial.radius(idx, 0.0, 0.0, 1.5)\n\
         let nearest := spatial.nearest(idx, 0.2, 0.2)?\n\
         let occ := spatial.occupancy(idx, 0.0, 0.0, 2.0, 2.0)\n\
         let removed := spatial.remove(idx, 0)\n\
         let nearest_after := spatial.nearest(idx, 0.0, 0.0)?\n\
         mut out := near[0] * 1000000 + near[1] * 10000 + nearest * 100 + occ\n\
         if removed ::\n\
             out := out * 100 + nearest_after\n\
         ::\n\
         out\n",
    );
    let result = run_value(&source);
    assert_eq!(result, Value::Int(63631063));

    let invalid = run_result(
        "import std::spatial\n\
         let idx := spatial.make()\n\
         spatial.upsert(idx, 1, 1e309, 0.0)\n",
    )
    .expect_err("non-finite spatial coordinates should fail");
    assert_eq!(invalid.message, "spatial.upsert expects finite coordinates");

    let bounds = run_result(
        "import std::spatial\n\
         let idx := spatial.make()\n\
         spatial.occupancy(idx, 1.0, 0.0, 0.0, 1.0)\n",
    )
    .expect_err("invalid occupancy rectangle should fail");
    assert_eq!(
        bounds.message,
        "spatial.occupancy expects min bounds <= max bounds"
    );
}

#[test]
fn snn_runtime_and_agent_environment_kernel_work() {
    let snn = run_value(
        "import std::snn\n\
         import std::sparse\n\
         let net := snn.make(3)\n\
         snn.set_threshold(net, 0, 0.4)\n\
         snn.set_threshold(net, 1, 0.4)\n\
         snn.set_threshold(net, 2, 0.5)\n\
         snn.connect(net, 0, 1, 0.5)\n\
         snn.connect(net, 1, 2, 0.75)\n\
         let a := snn.step(net, [1.0, 0.0, 0.0])\n\
         let b := snn.step(net, [0.0, 0.0, 0.0])\n\
         let c := snn.step(net, [0.0, 0.0, 0.0])\n\
         sparse.nnz(snn.synapses(net)) * 1000 + a[0] * 100 + b[0] * 10 + c[0]\n",
    );
    assert_eq!(snn, Value::Int(2012));

    let agent_kernel = run_value(
        "import std::json\n\
         import std::agent\n\
         import std::sim\n\
         import std::spatial\n\
         let world := sim.make_seeded(16, 3)\n\
         let idx := spatial.make()\n\
         let env := agent.make(world, idx)\n\
         agent.register(env, 1, json.parse(\"{\\\"role\\\":\\\"worker\\\"}\"), json.parse(\"{}\"), 0.0, 0.0)\n\
         agent.register(env, 2, json.parse(\"{\\\"role\\\":\\\"scout\\\"}\"), json.parse(\"{}\"), 0.5, 0.0)\n\
         agent.reward_add(env, 1, 1.25)\n\
         let reward := agent.reward_take(env, 1)\n\
         agent.sense_push(env, 1, \"food\")\n\
         let sense := agent.sense_take(env, 1)?\n\
         agent.action_push(env, 1, \"move\")\n\
         let action := agent.action_take(env, 1)?\n\
         agent.set_position(env, 1, 0.25, 0.0)\n\
         let nearest := spatial.nearest(idx, 0.2, 0.0)?\n\
         let neighbors := agent.neighbors(env, 1, 1.0)\n\
         mut out := 0\n\
         if reward > 1.0 and sense == \"food\" and action == \"move\" and nearest == 1 ::\n\
             out := neighbors[0] * 10 + nearest\n\
         ::\n\
         out\n",
    );
    assert_eq!(agent_kernel, Value::Int(21));
}

#[test]
fn snn_batched_kernel_matches_scalar_frontier_and_rejects_invalid_policy() {
    let result = run_value(
        "import std::snn\n\
         let net := snn.make(3)\n\
         snn.set_threshold(net, 0, 0.4)\n\
         snn.set_threshold(net, 1, 0.4)\n\
         snn.set_threshold(net, 2, 0.5)\n\
         snn.connect(net, 0, 1, 0.5)\n\
         snn.connect(net, 1, 2, 0.75)\n\
         let batch := snn.step_batch(net, [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])\n\
         let potentials := snn.potentials(net)\n\
         batch[0][0] * 100000 + batch[1][0] * 1000 + batch[2][0] * 10 + potentials[0]\n",
    );
    assert_eq!(result, Value::Float(1020.0));

    let decay = run_result(
        "import std::snn\n\
         let net := snn.make(2)\n\
         snn.set_decay(net, 1.5)\n",
    )
    .expect_err("invalid SNN decay should fail");
    assert_eq!(
        decay.message,
        "snn.set_decay expects finite value in [0, 1]"
    );

    let weight = run_result(
        "import std::snn\n\
         let net := snn.make(2)\n\
         snn.connect(net, 0, 1, 1e309)\n",
    )
    .expect_err("invalid SNN weight should fail");
    assert_eq!(weight.message, "snn.connect expects finite weight");
}

#[test]
fn snn_batch_native_and_vm_paths_match_for_seeded_scenarios() {
    let source = "import std::snn\n\
         let net := snn.make(4)\n\
         snn.set_threshold(net, 0, 0.4)\n\
         snn.set_threshold(net, 1, 0.5)\n\
         snn.set_threshold(net, 2, 0.6)\n\
         snn.set_threshold(net, 3, 0.7)\n\
         snn.connect(net, 0, 1, 0.25)\n\
         snn.connect(net, 1, 2, 0.50)\n\
         snn.connect(net, 2, 3, 0.75)\n\
         let batch := snn.step_batch(net, [[1.0, 0.0, 0.0, 0.0], [0.0, 0.3, 0.0, 0.0], [0.0, 0.0, 0.3, 0.0], [0.0, 0.0, 0.0, 0.3]])\n\
         let potentials := snn.potentials(net)\n\
         batch[0][0] * 100000 + batch[1][0] * 10000 + batch[2][0] * 1000 + batch[3][0] * 100 + potentials[3] * 10\n";
    let native = run_value_with_accel(source, cfg!(not(windows)));
    let fallback = run_value_with_accel(source, false);
    assert_eq!(native, fallback);
}

#[test]
fn sparse_native_and_vm_paths_match_for_seeded_scenarios() {
    for case in 0..8_u64 {
        let mut seed = 100 + case;
        let mut source = String::from(
            "import std::sparse\nlet v := sparse.vector()\nlet m := sparse.matrix()\n",
        );
        for row in 0..4 {
            let seed_value = (row as f64) + 1.0;
            source.push_str(&format!("sparse.set(m, {row}, {row}, {seed_value})\n"));
        }
        for _ in 0..12 {
            let index = rand_index(&mut seed, 8);
            let value = rand_float(&mut seed);
            source.push_str(&format!("sparse.set_vector(v, {index}, {value})\n"));
        }
        for _ in 0..16 {
            let row = rand_index(&mut seed, 4);
            let col = rand_index(&mut seed, 8);
            let value = rand_float(&mut seed);
            source.push_str(&format!("sparse.set(m, {row}, {col}, {value})\n"));
        }
        source.push_str("let dense := [");
        for idx in 0..8 {
            if idx > 0 {
                source.push_str(", ");
            }
            source.push_str(&format!("{}", rand_float(&mut seed)));
        }
        source.push_str("]\n");
        source.push_str(
            "let rows := sparse.matvec(m, dense)\n\
             mut acc := sparse.dot(v, dense)\n\
             let head := sparse.nonzero(m)\n\
             acc := acc + sparse.nnz(v) * 0.01 + sparse.nnz(m) * 0.001\n\
             let v0 := sparse.get_vector(v, 0)\n\
             if v0 != none ::\n\
                 acc := acc + v0? * 0.0001\n\
             ::\n\
             let m00 := sparse.get(m, 0, 0)\n\
             if m00 != none ::\n\
                 acc := acc + m00? * 0.00001\n\
             ::\n\
             acc := acc + head[0].row * 0.000001 + head[0].col * 0.0000001 + head[0].value * 0.00000001\n\
             acc := acc + rows[0] * 0.1 + rows[1] * 0.01 + rows[2] * 0.001 + rows[3] * 0.0001\n\
             acc\n",
        );
        let native = run_value_with_accel(&source, cfg!(not(windows)));
        let fallback = run_value_with_accel(&source, false);
        assert_eq!(native, fallback, "seeded sparse case {}", case);
    }
}

#[test]
fn event_queue_native_and_vm_paths_match_for_seeded_scenarios() {
    for case in 0..8_u64 {
        let mut seed = 200 + case;
        let mut source = String::from("import std::event\nlet q := event.make()\n");
        for _ in 0..12 {
            let bucket = rand_index(&mut seed, 4) as f64;
            let time = bucket / 10.0;
            let event = rand_index(&mut seed, 50);
            source.push_str(&format!("event.push(q, {time}, {event})\n"));
        }
        source.push_str(
            "mut acc := 0\n\
             while not event.is_empty(q) ::\n\
                 let item := event.pop(q)?\n\
                 acc := acc * 131 + item.event * 7 + item.time * 100\n\
             ::\n\
             acc\n",
        );
        let native = run_value_with_accel(&source, cfg!(not(windows)));
        let fallback = run_value_with_accel(&source, false);
        assert_eq!(native, fallback, "seeded event case {}", case);
    }
}

#[test]
fn pool_native_and_vm_paths_match_for_seeded_scenarios() {
    for case in 0..8_u64 {
        let mut seed = 300 + case;
        let mut source =
            String::from("import std::pool\nlet p := pool.make_growable(2)\nmut acc := 0\n");
        for _ in 0..32 {
            if (next_seed(&mut seed) & 1) == 0 {
                let value = rand_index(&mut seed, 100);
                source.push_str(&format!(
                    "if pool.release(p, {value}) ::\n    acc := acc + {value}\n::\n"
                ));
            } else {
                source.push_str(
                    "let item := pool.acquire(p)\n\
                     if item != none ::\n\
                         acc := acc * 17 + item?\n\
                     ::\n",
                );
            }
        }
        source.push_str(
            "let stats := pool.stats(p)\n\
             acc + stats.available * 100000 + stats.capacity * 1000 + stats.high_watermark * 10 + stats.dropped_on_full\n",
        );
        let native = run_value_with_accel(&source, cfg!(not(windows)));
        let fallback = run_value_with_accel(&source, false);
        assert_eq!(native, fallback, "seeded pool case {}", case);
    }
}
