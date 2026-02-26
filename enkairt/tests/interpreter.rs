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
