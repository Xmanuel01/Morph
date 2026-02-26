use enkaic::compiler::compile_module;
use enkaic::parser::parse_module;
use enkairt::{Value, VM};

fn run_value(source: &str) -> Value {
    let module = parse_module(source).expect("parse");
    let program = compile_module(&module).expect("compile");
    let mut vm = VM::new(false, false, false, false);
    vm.run(&program).expect("run")
}

#[test]
fn spawn_join_returns_value() {
    let value = run_value(
        "fn work() -> Int ::\n    return 2 + 3\n::\nlet h := task.spawn(work)\ntask.join(h)\n",
    );
    assert_eq!(value, Value::Int(5));
}

#[test]
fn sleep_yields_and_tasks_complete() {
    let value = run_value(
        "fn slow() -> Int ::\n    task.sleep(5)\n    return 1\n::\nfn fast() -> Int ::\n    return 2\n::\nlet h1 := task.spawn(slow)\nlet h2 := task.spawn(fast)\nlet a := task.join(h2)\nlet b := task.join(h1)\na + b\n",
    );
    assert_eq!(value, Value::Int(3));
}

#[test]
fn unary_spawn_and_await() {
    let value = run_value(
        "fn work() -> Int ::\n    return 7\n::\nlet h := spawn work\nawait h\n",
    );
    assert_eq!(value, Value::Int(7));
}
