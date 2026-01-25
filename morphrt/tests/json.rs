use morphc::compiler::compile_module;
use morphc::parser::parse_module;
use morphrt::VM;

fn run_value(source: &str) -> Result<morphrt::Value, morphrt::error::RuntimeError> {
    let module = parse_module(source).expect("parse");
    let program = compile_module(&module).expect("compile");
    let mut vm = VM::new(false, false, false, false);
    vm.run(&program)
}

#[test]
fn json_parse_stringify_roundtrip() {
    let source = "let v := json.parse(\"{\\\"a\\\":1,\\\"b\\\":[true,null]}\")\n\
                  let out := json.stringify(v)\n\
                  return out\n";
    let value = run_value(source).expect("run");
    let out = match value {
        morphrt::Value::Obj(obj) => match obj.as_obj() {
            morphrt::object::Obj::String(s) => s.clone(),
            _ => panic!("expected string"),
        },
        _ => panic!("expected string"),
    };
    let parsed: serde_json::Value = serde_json::from_str(&out).expect("json");
    let expected: serde_json::Value =
        serde_json::from_str("{\"a\":1,\"b\":[true,null]}").expect("expected");
    assert_eq!(parsed, expected);
}

#[test]
fn json_parse_invalid_errors() {
    let source = "let v := json.parse(\"{not_json}\")\n\
                  return v\n";
    let result = run_value(source);
    assert!(result.is_err());
}
