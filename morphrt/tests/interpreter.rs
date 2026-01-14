use morphc::parser::parse_module;
use morphrt::{Interpreter, Value};

fn eval(source: &str) -> Interpreter {
    let module = parse_module(source).expect("parse");
    let mut interpreter = Interpreter::new();
    interpreter.eval_module(&module).expect("run");
    interpreter
}

#[test]
fn runs_control_flow_and_loops() {
    let source = "\
let x := 0
while x < 3 ::
    x := x + 1
::
let total := 0
for item in [1, 2, 3] ::
    total := total + item
::
";
    let interpreter = eval(source);
    assert_eq!(interpreter.get_global("x"), Some(Value::Int(3)));
    assert_eq!(interpreter.get_global("total"), Some(Value::Int(6)));
}

#[test]
fn runs_functions_and_calls() {
    let source = "\
fn add(a: Int, b: Int) -> Int ::
    return a + b
::
let result := add(2, 3)
";
    let interpreter = eval(source);
    assert_eq!(interpreter.get_global("result"), Some(Value::Int(5)));
}

#[test]
fn runs_match_expression() {
    let source = "\
let x := 2
let y := match x ::
    1 => 10
    2 => 20
    _ => 0
::
";
    let interpreter = eval(source);
    assert_eq!(interpreter.get_global("y"), Some(Value::Int(20)));
}

#[test]
fn runs_try_catch() {
    let source = "\
let value := 0
let result := 0
try ::
    result := 1 / value
::
catch e ::
    result := 5
::
";
    let interpreter = eval(source);
    assert_eq!(interpreter.get_global("result"), Some(Value::Int(5)));
}

#[test]
fn runs_nested_blocks() {
    let source = "\
let x := 0
if true ::
    while x < 2 ::
        x := x + 1
    ::
::
";
    let interpreter = eval(source);
    assert_eq!(interpreter.get_global("x"), Some(Value::Int(2)));
}

#[test]
fn runs_match_statement_block_arms() {
    let source = "\
let result := 0
match 2 ::
    1 => ::
        result := 10
    ::
    2 => ::
        let inner := 5
        result := inner + 15
    ::
    _ => ::
        result := 99
    ::
::
";
    let interpreter = eval(source);
    assert_eq!(interpreter.get_global("result"), Some(Value::Int(20)));
}

#[test]
fn runs_match_inside_if() {
    let source = "\
let result := 0
if true ::
    let value := match 3 ::
        1 => 10
        3 => 30
        _ => 0
    ::
    result := value
::
";
    let interpreter = eval(source);
    assert_eq!(interpreter.get_global("result"), Some(Value::Int(30)));
}

#[test]
fn runs_if_inside_match_block_arm() {
    let source = "\
let result := 0
match 1 ::
    1 => ::
        if true ::
            result := 42
        ::
    ::
    _ => ::
        result := 0
    ::
::
";
    let interpreter = eval(source);
    assert_eq!(interpreter.get_global("result"), Some(Value::Int(42)));
}

#[test]
fn runs_string_functions() {
    let source = "\
let a := std.string.len(\"hello\")
let b := std.string.contains(\"hello\", \"ell\")
let c := std.string.slice(\"hello\", 1, 4)
";
    let interpreter = eval(source);
    assert_eq!(interpreter.get_global("a"), Some(Value::Int(5)));
    assert_eq!(interpreter.get_global("b"), Some(Value::Bool(true)));
    assert_eq!(
        interpreter.get_global("c"),
        Some(Value::String("ell".to_string()))
    );
}
