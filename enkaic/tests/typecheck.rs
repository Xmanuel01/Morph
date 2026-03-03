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
fn unknown_callee_call_is_permitted() {
    assert!(type_ok(
        "type Boxed ::\n    value: Int\n::\n\
         impl Boxed ::\n    fn add(self: Boxed, x: Int) -> Int ::\n        return self.value + x\n    ::\n::\n\
         fn main() -> Int ::\n    let b := Boxed(3)\n    let out := b.add(2)\n    return 0\n::\n\
         main()\n"
    ));
}
