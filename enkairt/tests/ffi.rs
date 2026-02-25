use enkaic::compiler::compile_module;
use enkaic::parser::parse_module;
use enkairt::error::RuntimeError;
use enkairt::object::Obj;
use enkairt::{Value, VM};

fn run(source: &str) -> Result<Value, RuntimeError> {
    let module = parse_module(source).expect("parse");
    let program = compile_module(&module).expect("compile");
    let mut vm = VM::new(false, false, false, false);
    vm.run(&program)
}

fn expect_string(value: Value) -> String {
    match value {
        Value::Obj(obj) => match obj.as_obj() {
            Obj::String(s) => s.clone(),
            _ => panic!("expected string"),
        },
        _ => panic!("expected string"),
    }
}

#[test]
fn ffi_add_i64() {
    let value = run(
        "native::import \"enkai_native\" ::\n    fn add_i64(a: Int, b: Int) -> Int\n::\nadd_i64(2, 3)\n",
    )
    .expect("run");
    assert_eq!(value, Value::Int(5));
}

#[test]
fn ffi_string_roundtrip() {
    let value = run(
        "native::import \"enkai_native\" ::\n    fn echo_string(data: String) -> String\n::\necho_string(\"hi\")\n",
    )
    .expect("run");
    assert_eq!(expect_string(value), "hi");
}

#[test]
fn ffi_buffer_len_roundtrip() {
    let value = run(
        "native::import \"enkai_native\" ::\n    fn buffer_from_string(data: String) -> Buffer\n    fn buffer_len(data: Buffer) -> Int\n::\nlet buf := buffer_from_string(\"hello\")\nbuffer_len(buf)\n",
    )
    .expect("run");
    assert_eq!(value, Value::Int(5));
}

#[test]
fn ffi_hash_sha256_len() {
    let value = run(
        "native::import \"enkai_native\" ::\n    fn buffer_from_string(data: String) -> Buffer\n    fn hash_sha256(data: Buffer) -> Buffer\n    fn buffer_len(data: Buffer) -> Int\n::\nlet buf := buffer_from_string(\"abc\")\nlet digest := hash_sha256(buf)\nbuffer_len(digest)\n",
    )
    .expect("run");
    assert_eq!(value, Value::Int(32));
}

#[test]
fn ffi_missing_symbol_error() {
    let err = run(
        "native::import \"enkai_native\" ::\n    fn missing_symbol() -> Int\n::\nmissing_symbol()\n",
    )
    .expect_err("expected error");
    assert!(err.message.contains("Failed to resolve symbol"));
}

#[test]
fn ffi_missing_library_error() {
    let err = run(
        "native::import \"definitely_missing\" ::\n    fn add_i64(a: Int, b: Int) -> Int\n::\nadd_i64(1, 2)\n",
    )
    .expect_err("expected error");
    assert!(err.message.contains("Failed to load library"));
}

#[test]
fn ffi_optional_scalar_signature_rejected_at_compile() {
    let module =
        parse_module("native::import \"enkai_native\" ::\n    fn bad(a: Int?) -> Int\n::\n0\n")
            .expect("parse");
    let err = compile_module(&module).expect_err("expected compile failure");
    assert!(err.message.contains("Unsupported FFI parameter type"));
}
