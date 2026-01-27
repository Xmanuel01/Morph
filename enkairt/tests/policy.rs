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
fn repeated_string_allocations_do_not_crash() {
    let mut last = Value::Null;
    for _ in 0..1000 {
        last = run_value("let s := \"hello\"\ns");
    }
    if let Value::Obj(obj) = last {
        assert!(obj.strong_count() >= 1);
    }
}
