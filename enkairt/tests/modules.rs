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
fn strings_are_refcounted() {
    let val = run_value("let s := \"hi\"\ns");
    if let Value::Obj(obj) = val {
        let weak = obj.downgrade();
        assert!(weak.upgrade().is_some());
        drop(obj);
        assert!(weak.upgrade().is_none());
    } else {
        panic!("expected string obj");
    }
}

#[test]
fn objref_clone_drops() {
    let val = run_value("let s := \"hi\"\ns");
    if let Value::Obj(obj) = val {
        let cloned = obj.clone();
        assert!(cloned.strong_count() >= 2);
        drop(cloned);
        assert_eq!(obj.strong_count(), 1);
    } else {
        panic!("expected string obj");
    }
}
