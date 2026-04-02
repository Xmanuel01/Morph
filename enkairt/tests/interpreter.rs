use enkaic::compiler::compile_module;
use enkaic::parser::parse_module;
use enkairt::error::RuntimeError;
use enkairt::{Value, VM};

fn run_value(source: &str) -> Value {
    let module = parse_module(source).expect("parse");
    let program = compile_module(&module).expect("compile");
    let mut vm = VM::new(false, false, false, false);
    vm.run(&program).expect("run")
}

fn run_result(source: &str) -> Result<Value, RuntimeError> {
    let module = parse_module(source).expect("parse");
    let program = compile_module(&module).expect("compile");
    let mut vm = VM::new(false, false, false, false);
    vm.run(&program)
}

#[test]
fn let_and_arithmetic() {
    let result = run_value("let x := 2 + 3\nx");
    assert_eq!(result, Value::Int(5));
}

#[test]
fn assignment_updates() {
    let result = run_value("let x := 1\nx := x + 4\nx");
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
        "let x := 1\nlet y := 0\nif x == 1 ::\n    y := 10\n::\nelse ::\n    y := 5\n::\ny",
    );
    assert_eq!(result, Value::Int(10));
}

#[test]
fn while_loop_counts() {
    let result = run_value("let x := 0\nwhile x < 3 ::\n    x := x + 1\n::\nx");
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
        "let x := 0\nfn set() -> Bool ::\n    x := 1\n    return true\n::\nfalse and set()\nx",
    );
    assert_eq!(result, Value::Int(0));
}

#[test]
fn logic_short_circuit_or() {
    let result = run_value(
        "let x := 0\nfn set() -> Bool ::\n    x := 1\n    return true\n::\ntrue or set()\nx",
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
fn pool_fixed_and_growable_behaviors_work() {
    let fixed = run_value(
        "import std::pool\n\
         let p := pool.make(1)\n\
         let first := pool.release(p, 7)\n\
         let second := pool.release(p, 9)\n\
         let got := pool.acquire(p)?\n\
         let stats := pool.stats(p)\n\
         let out := 0\n\
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
         let out := 0\n\
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
         import json\n\
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
             let step := 0\n\
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
         let out := a[0] * 100000 + b[1] * 10000 + c[1] * 1000 + joined * 10\n\
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
        "import json\n\
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
         let out := 0\n\
         if reward > 1.0 and sense == \"food\" and action == \"move\" and nearest == 1 ::\n\
             out := neighbors[0] * 10 + nearest\n\
         ::\n\
         out\n",
    );
    assert_eq!(agent_kernel, Value::Int(21));
}
