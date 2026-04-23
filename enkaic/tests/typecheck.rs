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
         import json\n\
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
        "import json\n\
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
