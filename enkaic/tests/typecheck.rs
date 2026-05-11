use enkaic::parser::parse_module;
use enkaic::TypeChecker;

fn type_ok(src: &str) -> bool {
    let m = parse_module(src).expect("parse");
    let mut tc = TypeChecker::new();
    tc.check_module(&m).is_ok()
}

fn type_err(src: &str) -> String {
    let m = parse_module(src).expect("parse");
    let mut tc = TypeChecker::new();
    tc.check_module(&m).unwrap_err().message
}

fn compile_err(src: &str) -> String {
    let m = parse_module(src).expect("parse");
    enkaic::compiler::compile_module(&m).unwrap_err().message
}

#[test]
fn arithmetic_types_ok() {
    assert!(type_ok("let x := 1 + 2\n"));
}

#[test]
fn logic_types_ok() {
    assert!(type_ok("let x := true and false\n"));
}

#[test]
fn optional_allows_none() {
    assert!(type_ok("let x: String? := none\n"));
}

#[test]
fn reassignment_requires_mut_binding() {
    let msg = type_err("let count := 6\ncount := 7\n");
    assert!(msg.contains("cannot assign to immutable variable `count`"));
    assert!(msg.contains("declare it with `mut`"));
}

#[test]
fn mut_binding_allows_reassignment() {
    assert!(type_ok("mut count := 6\ncount := count + 1\n"));
    assert!(type_ok("let mut score := 0.0\nscore := score + 1.0\n"));
}

#[test]
fn const_binding_accepts_compile_time_expression() {
    assert!(type_ok(
        "const BASE := 40\n\
         const ANSWER: Int := BASE + 2\n\
         let value := ANSWER\n"
    ));
    assert!(type_ok("const FLAGS: Array[Bool] := [true, false]\n"));
}

#[test]
fn const_binding_rejects_runtime_expression() {
    let msg = type_err(
        "fn answer() -> Int ::\n    return 42\n::fn\n\
         const VALUE := answer()\n",
    );
    assert!(msg.contains("const binding `VALUE` requires a compile-time constant expression"));
}

#[test]
fn const_binding_cannot_depend_on_let_binding() {
    let msg = type_err("let base := 40\nconst VALUE := base + 2\n");
    assert!(msg.contains("const binding `VALUE` requires a compile-time constant expression"));
}

#[test]
fn const_binding_is_immutable() {
    let msg = type_err("const LIMIT := 6\nLIMIT := 7\n");
    assert!(msg.contains("cannot assign to immutable variable `LIMIT`"));
}

#[test]
fn compiler_rejects_immutable_reassignment_without_typecheck() {
    let msg = compile_err("let count := 6\ncount := 7\n");
    assert!(msg.contains("cannot assign to immutable variable `count`"));
}

#[test]
fn compiler_rejects_runtime_const_initializer_without_typecheck() {
    let msg = compile_err("fn answer() -> Int ::\n    return 42\n::fn\nconst VALUE := answer()\n");
    assert!(msg.contains("const binding `VALUE` requires a compile-time constant expression"));
}

#[test]
fn json_namespace_requires_std_json_import() {
    let msg = type_err("let value := json.parse(\"{}\")\n");
    assert!(msg.contains("ImportError: `json.parse` requires `import std::json`"));
}

#[test]
fn std_json_import_enables_json_namespace() {
    assert!(type_ok(
        "import std::json\n\
         let value := json.parse(\"{}\")\n\
         let text := json.enkai(value)\n"
    ));
}

#[test]
fn policy_static_check_allows_io_write() {
    assert!(type_ok(
        "import std::io\n\
         policy default ::\n    allow io.write\n::policy\n\
         let _ := io.stdout_write_text(\"hello\")\n"
    ));
}

#[test]
fn policy_static_check_rejects_missing_allow() {
    let msg = type_err(
        "import std::io\n\
         policy default ::\n    allow fs.read\n::policy\n\
         let _ := io.stdout_write_text(\"hello\")\n",
    );
    assert!(msg.contains("PolicyError: io.write is not allowed by policy `default`"));
}

#[test]
fn policy_static_check_rejects_explicit_deny() {
    let msg = type_err(
        "import std::io\n\
         policy default ::\n    allow io\n    deny io.write\n::policy\n\
         let _ := io.stdout_write_text(\"hello\")\n",
    );
    assert!(msg.contains("PolicyError: io.write is denied by policy `default`"));
}

#[test]
fn policy_static_check_rejects_missing_default_policy() {
    let msg = type_err("import std::io\nlet _ := io.stdout_write_text(\"hello\")\n");
    assert!(msg.contains("requires `policy default`"));
}

#[test]
fn policy_static_check_validates_filtered_paths() {
    assert!(type_ok(
        "import std::io\n\
         policy default ::\n    allow fs.read path_prefix=\"/safe/\"\n::policy\n\
         let data := io.read_text(\"/safe/input.txt\")\n"
    ));
    let msg = type_err(
        "import std::io\n\
         policy default ::\n    allow fs.read path_prefix=\"/safe/\"\n::policy\n\
         let data := io.read_text(\"/tmp/input.txt\")\n",
    );
    assert!(msg.contains("PolicyError: fs.read is not allowed by policy `default`"));
}

#[test]
fn policy_static_check_rejects_unknown_filter() {
    let msg = type_err("policy default ::\n    allow fs.read owner=\"me\"\n::policy\n");
    assert!(msg.contains("unsupported policy filter `owner`"));
}

#[test]
fn function_params_are_immutable_by_default() {
    let msg =
        type_err("fn bump(count: Int) -> Int ::\n    count := count + 1\n    return count\n::\n");
    assert!(msg.contains("cannot assign to immutable variable `count`"));
}

#[test]
fn mutable_function_params_allow_reassignment() {
    assert!(type_ok(
        "fn bump(mut count: Int) -> Int ::\n    count := count + 1\n    return count\n::\n"
    ));
}

#[test]
fn function_call_type_error() {
    let msg = type_err("fn add(a: Int, b: Int) -> Int ::\n    return a + b\n::\nadd(1, true)\n");
    assert!(msg.contains("Argument type mismatch"));
}

#[test]
fn return_type_error() {
    let msg = type_err("fn f() -> Int ::\n    return true\n::\n");
    assert!(msg.contains("Return type mismatch"));
}

#[test]
fn native_import_rejects_invalid_type() {
    let msg = type_err("native::import \"libdemo\" ::\n    fn bad(a: List) -> Int\n::\n");
    assert!(msg.contains("Invalid FFI parameter type"));
}

#[test]
fn native_import_rejects_optional_scalar_param() {
    let msg = type_err("native::import \"libdemo\" ::\n    fn bad(a: Int?) -> Int\n::\n");
    assert!(msg.contains("Invalid FFI parameter type"));
}

#[test]
fn native_import_rejects_optional_scalar_return() {
    let msg = type_err("native::import \"libdemo\" ::\n    fn bad() -> Bool?\n::\n");
    assert!(msg.contains("Invalid FFI return type"));
}

#[test]
fn native_import_accepts_handle_types() {
    assert!(type_ok(
        "native::import \"libdemo\" ::\n    fn make(seed: Int) -> Handle\n    fn read(handle: Handle) -> Int\n    fn maybe(flag: Bool) -> Handle?\n::\n"
    ));
}

#[test]
fn simulation_stdlib_modules_typecheck() {
    assert!(type_ok(
        "import std::sparse\n\
         import std::event\n\
         import std::pool\n\
         let m: SparseMatrix := sparse.matrix()\n\
         let v: SparseVector := sparse.vector()\n\
         let q: EventQueue := event.make()\n\
         let p: Pool := pool.make(4)\n\
         sparse.set(m, 0, 1, 2.0)\n\
         sparse.set_vector(v, 1, 3.0)\n\
         event.push(q, 1.0, 7)\n\
         let hit := sparse.get(m, 0, 1)\n\
         let item := event.pop(q)\n\
         let maybe := pool.acquire(p)\n\
         let ok := pool.release(p, 4)\n"
    ));
}

#[test]
fn simulation_world_stdlib_typechecks() {
    assert!(type_ok(
        "import std::sim\n\
         import std::json\n\
         let world: SimWorld := sim.make_seeded(16, 7)\n\
         sim.schedule(world, 1.0, json.parse(\"{\\\"kind\\\":\\\"tick\\\"}\"))\n\
         let next := sim.step(world)\n\
         let snap := sim.snapshot(world)\n\
         let restored := sim.restore(snap)\n\
         let replayed := sim.replay(sim.log(restored), 16, sim.seed(restored))\n\
         sim.entity_set(replayed, 1, json.parse(\"{\\\"state\\\":2}\"))\n\
         let entity := sim.entity_get(replayed, 1)\n\
         let ids := sim.entity_ids(replayed)\n"
    ));
}

#[test]
fn simulation_coroutines_typecheck() {
    assert!(type_ok(
        "import std::sim\n\
         fn worker(coro: SimCoroutine, state: Any) -> Int ::\n\
             let world: SimWorld := sim.world(coro)\n\
             let cfg := sim.state(coro)\n\
             sim.emit(coro, state)\n\
             let done := sim.done(coro)\n\
             return sim.pending(world)\n\
         ::\n\
         let world: SimWorld := sim.make_seeded(16, 7)\n\
         let coro: SimCoroutine := sim.coroutine_with(world, worker, 3)\n\
         let next := sim.next(coro)\n\
         let joined := sim.join(coro)\n"
    ));
}

#[test]
fn spatial_snn_and_agent_modules_typecheck() {
    assert!(type_ok(
        "import std::json\n\
         import std::agent\n\
         import std::sim\n\
         import std::snn\n\
         import std::spatial\n\
         let world: SimWorld := sim.make_seeded(16, 9)\n\
         let idx: SpatialIndex := spatial.make()\n\
         let env: AgentEnv := agent.make(world, idx)\n\
         agent.register(env, 1, json.parse(\"{}\"), json.parse(\"{}\"), 0.0, 0.0)\n\
         let stream: RngStream := agent.stream(env, 1, \"sense\")\n\
         let n: SnnNetwork := snn.make(4)\n\
         snn.connect(n, 0, 1, 0.5)\n\
         let spikes := snn.step(n, [1.0, 0.0, 0.0, 0.0])\n\
         let nearest := spatial.nearest(idx, 0.0, 0.0)\n\
         let state := agent.state(env, 1)\n\
         let reward: Float := agent.reward_take(env, 1)\n\
         let value: Int := agent.next_int(stream, 8)\n"
    ));
}

#[test]
fn unknown_callee_call_is_permitted() {
    assert!(type_ok(
        "type Boxed ::\n    value: Int\n::\n\
         impl Boxed ::\n    fn add(self: Boxed, x: Int) -> Int ::\n        return self.value + x\n    ::\n::\n\
         fn main() -> Int ::\n    let b := Boxed(3)\n    let out := b.add(2)\n    return 0\n::\n\
         main()\n"
    ));
}

#[test]
fn record_constructor_and_field_access_typecheck() {
    assert!(type_ok(
        "type Pair ::\n    left: Int\n    right: Int\n::\n\
         fn main() -> Int ::\n    let pair := Pair(4, 5)\n    return pair.left + pair.right\n::\n\
         main()\n"
    ));
}

#[test]
fn unknown_record_field_is_rejected() {
    let msg = type_err(
        "type Pair ::\n    left: Int\n    right: Int\n::\n\
         fn main() -> Int ::\n    let pair := Pair(4, 5)\n    return pair.missing\n::\n\
         main()\n",
    );
    assert!(msg.contains("Unknown field Pair.missing"));
}

#[test]
fn record_field_mutation_typechecks() {
    assert!(type_ok(
        "type Pair ::\n    left: Int\n    right: Int\n::\n\
         fn main() -> Int ::\n    let pair := Pair(4, 5)\n    pair.left := pair.left + 3\n    return pair.left + pair.right\n::\n\
         main()\n"
    ));
}

#[test]
fn record_field_mutation_rejects_type_mismatch() {
    let msg = type_err(
        "type Pair ::\n    left: Int\n    right: Int\n::\n\
         fn main() -> Int ::\n    let pair := Pair(4, 5)\n    pair.left := true\n    return pair.left\n::\n\
         main()\n",
    );
    assert!(msg.contains("Type mismatch: variable pair is Int, assigned Bool"));
}

#[test]
fn list_index_typechecks() {
    assert!(type_ok(
        "fn main() -> Int ::\n    let xs := [4, 5, 6]\n    return xs[1]\n::\nmain()\n"
    ));
}

#[test]
fn list_index_rejects_non_int_index() {
    let msg =
        type_err("fn main() -> Int ::\n    let xs := [4, 5, 6]\n    return xs[true]\n::\nmain()\n");
    assert!(msg.contains("Index expects Int"));
}

#[test]
fn arrays_infer_homogeneous_element_types() {
    assert!(type_ok(
        "let names: Array[String] := [\"Nairobi\", \"Mombasa\"]\n\
         let scores: Array[Float] := [0.7, 0.8, 1]\n\
         let first: String := names[0]\n\
         let score: Float := scores[2]\n"
    ));
}

#[test]
fn empty_array_requires_explicit_type() {
    let msg = type_err("let empty := []\n");
    assert!(msg.contains("cannot infer type of empty array"));
    assert!(type_ok("let empty: Array[String] := []\n"));
}

#[test]
fn mixed_array_binding_requires_explicit_dynamic_type() {
    let msg = type_err("let values := [1, \"two\"]\n");
    assert!(msg.contains("mixed-type arrays require an explicit dynamic type"));
    assert!(type_ok("let values: Array[Any] := [1, \"two\"]\n"));
}

#[test]
fn vector_sparse_vector_and_tensor_annotations_typecheck() {
    assert!(type_ok(
        "import std::sparse\n\
         let scores: Vector[Float] := [0.75, 0.68, 0.62]\n\
         let weights: SparseVector[Float] := sparse.vector()\n\
         let matrix: Tensor[Float, 2] := 0\n"
    ));
}

#[test]
fn array_vector_and_tensor_stdlib_exports_typecheck() {
    assert!(type_ok(
        "import std::array\n\
         import std::vector\n\
         import std::tensor\n\
         let kind: String := array.element_type([1, 2])\n\
         let homogeneous: Bool := array.is_homogeneous([1.0, 2.0])\n\
         let v: Vector[Float] := vector.from_array([1.0, 2.0])\n\
         let dot: Float := vector.dot(v, v)\n\
         let t: Tensor := tensor.from_array([1.0, 2.0, 3.0, 4.0], [2, 2])\n\
         let shape: Array[Int] := tensor.shape(t)\n\
         let first: Float := tensor.get_flat(t, 0)\n\
         let transposed: Tensor := tensor.transpose(t, 0, 1)\n\
         let sliced: Tensor := tensor.slice(t, 0, 0, 1, 1)\n\
         let joined: Tensor := tensor.concat([t, t], 0)\n\
         let reduced: Tensor := tensor.sum(t, 1, false)\n\
         let averaged: Tensor := tensor.mean(t, 1, true)\n\
         let probs: Tensor := tensor.softmax(t, 1)\n\
         let activated: Tensor := tensor.gelu(t)\n\
         let shifted: Tensor := tensor.sub(t, t)\n\
         let divided: Tensor := tensor.div(t, t)\n\
         let scaled: Tensor := tensor.scale(t, 0.5)\n\
         let broadcasted: Tensor := tensor.broadcast_to(tensor.from_array([1.0, 2.0], [1, 2]), [2, 2])\n\
         let logged: Tensor := tensor.log(t)\n\
         let normalized: Tensor := tensor.layernorm(t, tensor.from_array([1.0, 1.0], [2]), tensor.from_array([0.0, 0.0], [2]), 0.00001)\n\
         let projected: Tensor := tensor.linear(t, t, tensor.from_array([0.0, 0.0], [2]))\n\
         let loss: Tensor := tensor.cross_entropy(t, tensor.from_array([0.0, 1.0], [2]))\n\
         let chosen: Tensor := tensor.where(t, t, t)\n\
         let clipped: Tensor := tensor.clip(t, 0.0, 1.0)\n\
         let arg: Tensor := tensor.argmax(t, 1, false)\n\
         let sorted: Tensor := tensor.sort(t, 1, true)\n\
         let top: Tensor := tensor.topk(t, 1, 1)\n\
         let gathered: Tensor := tensor.gather(t, 1, tensor.from_array([0.0, 1.0, 1.0, 0.0], [2, 2]))\n\
         let scattered: Tensor := tensor.scatter(t, 1, tensor.from_array([0.0, 1.0, 1.0, 0.0], [2, 2]), t)\n\
         let masked: Tensor := tensor.masked_fill(t, t, -1.0)\n\
         let eins: Tensor := tensor.einsum(\"ij,jk->ik\", t, t)\n\
         let image: Tensor := tensor.from_array([1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2])\n\
         let kernel: Tensor := tensor.from_array([1.0], [1, 1, 1, 1])\n\
         let conv: Tensor := tensor.conv2d(image, kernel, tensor.from_array([0.0], [1]), 1, 0)\n\
         let pool: Tensor := tensor.max_pool2d(image, 2, 1)\n\
         let bn: Tensor := tensor.batchnorm1d(t, tensor.from_array([1.0, 1.0], [2]), tensor.from_array([0.0, 0.0], [2]), 0.00001, true)\n\
         let attn: Tensor := tensor.attention(tensor.from_array([1.0, 0.0, 0.0, 1.0], [1, 2, 2]), tensor.from_array([1.0, 0.0, 0.0, 1.0], [1, 2, 2]), tensor.from_array([1.0, 2.0, 3.0, 4.0], [1, 2, 2]))\n\
         let tracked: Tensor := tensor.requires_grad(t)\n\
         let detached: Tensor := tensor.detach(tracked)\n\
         let grad: Tensor := tensor.grad(tracked)\n\
         let previous: Bool := tensor.no_grad(true)\n\
         let grad_ok: Bool := tensor.grad_check(grad, grad, 0.00001)\n"
    ));
}

#[test]
fn tensor_handle_compatibility_applies_to_return_types() {
    assert!(type_ok(
        "fn device_handle() -> Device ::\n\
             return 0\n\
         ::\n\
         fn tensor_handle() -> Tensor ::\n\
             return 0\n\
         ::\n"
    ));
}

#[test]
fn coroutine_args_accepts_heterogeneous_dynamic_list() {
    assert!(type_ok(
        "import std::sim\n\
         import std::sparse\n\
         fn worker(coro: SimCoroutine, start: Int, weights: SparseMatrix) -> Int ::\n\
             return start\n\
         ::\n\
         fn main() -> Int ::\n\
             let world := sim.make_seeded(16, 7)\n\
             let weights := sparse.matrix()\n\
             let coro := sim.coroutine_args(world, worker, [3, weights])\n\
             sim.join(coro)?\n\
             return 0\n\
         ::\n\
         main()\n"
    ));
}

#[test]
fn capturing_lambda_typechecks() {
    assert!(type_ok(
        "fn main() -> Int ::\n    let base := 3\n    let f := (x: Int) -> Int => x + base\n    return f(5)\n::\nmain()\n"
    ));
}
