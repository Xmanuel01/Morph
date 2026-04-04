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
fn ffi_handle_roundtrip_and_drop() {
    {
        let value = run(
            "native::import \"enkai_native\" ::\n    fn handle_new(value: Int) -> Handle\n    fn handle_read(handle: Handle) -> Int\n::\nlet handle := handle_new(41)\nhandle_read(handle)\n",
        )
        .expect("run");
        assert_eq!(value, Value::Int(41));
    }

    let live = run(
        "native::import \"enkai_native\" ::\n    fn handle_live_count() -> Int\n::\nhandle_live_count()\n",
    )
    .expect("run");
    assert_eq!(live, Value::Int(0));
}

#[test]
fn ffi_optional_handle_roundtrip() {
    let value = run(
        "native::import \"enkai_native\" ::\n    fn handle_maybe_new(flag: Bool, value: Int) -> Handle?\n::\nhandle_maybe_new(false, 9)\n",
    )
    .expect("run");
    assert_eq!(value, Value::Null);
}

#[test]
fn ffi_missing_symbol_error() {
    let err = run(
        "native::import \"enkai_native\" ::\n    fn missing_symbol() -> Int\n::\nmissing_symbol()\n",
    )
    .expect_err("expected error");
    assert_eq!(err.code(), Some("E_FFI_SYMBOL_MISSING"));
    assert!(err.message.contains("Failed to resolve symbol"));
}

#[test]
fn ffi_missing_library_error() {
    let err = run(
        "native::import \"definitely_missing\" ::\n    fn add_i64(a: Int, b: Int) -> Int\n::\nadd_i64(1, 2)\n",
    )
    .expect_err("expected error");
    assert_eq!(err.code(), Some("E_FFI_LIBRARY_LOAD"));
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

#[test]
fn ffi_wrong_handle_kind_is_rejected_and_counted() {
    let value = run(
        "native::import \"enkai_native\" ::\n    fn handle_reset_stale_count() -> Void\n    fn handle_stale_count() -> Int\n    fn handle_new(value: Int) -> Handle\n    fn sim_sparse_vector_set(handle: Handle, index: Int, value: Float) -> Bool\n::\nhandle_reset_stale_count()\nlet handle := handle_new(9)\nlet ok := sim_sparse_vector_set(handle, 0, 1.0)\nlet out := 0\nif ok ::\n    out := -100\n::\nout := out + handle_stale_count()\nout\n",
    )
    .expect("run");
    assert_eq!(value, Value::Int(1));

    let live = run(
        "native::import \"enkai_native\" ::\n    fn handle_live_count() -> Int\n::\nhandle_live_count()\n",
    )
    .expect("run");
    assert_eq!(live, Value::Int(0));
}

#[test]
fn ffi_fault_injection_errors_are_stable() {
    let null_err = run(
        "native::import \"enkai_native\" ::\n    fn fault_string_null() -> String\n::\nfault_string_null()\n",
    )
    .expect_err("expected null pointer error");
    assert_eq!(null_err.code(), Some("E_FFI_RETURN_NULL"));

    let oversized_err = run(
        "native::import \"enkai_native\" ::\n    fn fault_buffer_oversized() -> Buffer\n::\nfault_buffer_oversized()\n",
    )
    .expect_err("expected oversized buffer error");
    assert_eq!(oversized_err.code(), Some("E_FFI_RETURN_OVERSIZED"));

    let utf8_err = run(
        "native::import \"enkai_native\" ::\n    fn fault_string_invalid_utf8() -> String\n::\nfault_string_invalid_utf8()\n",
    )
    .expect_err("expected utf8 error");
    assert_eq!(utf8_err.code(), Some("E_FFI_UTF8"));
}
