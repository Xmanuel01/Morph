use enkaic::compiler::compile_module;
use enkaic::parser::parse_module;
use enkairt::error::RuntimeError;
use enkairt::object::Obj;
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

fn assert_string(value: Value, expected: &str) {
    match value {
        Value::Obj(obj) => match obj.as_obj() {
            Obj::String(s) => assert_eq!(s, expected),
            _ => panic!("expected string"),
        },
        _ => panic!("expected string"),
    }
}

#[test]
fn type_impl_method_dispatch() {
    let value = run_value(
        "type Point ::\n    x: Int\n    y: Int\n::\n\
         impl Point ::\n    fn sum() -> Int ::\n        return self.x + self.y\n    ::\n::\n\
         let p := Point(2, 3)\n\
         p.sum()\n",
    );
    assert_eq!(value, Value::Int(5));
}

#[test]
fn tool_stub_errors() {
    let result = run_result("tool tools.echo(a: Int) -> Int\ntools.echo(1)\n");
    assert!(result.is_err());
}

#[test]
fn prompt_decl_creates_record() {
    let value = run_value(
        "prompt Greeting ::\n\
         input ::\n\
             name: String\n\
         ::\n\
         template ::\n\
             \"Hello\"\n\
         ::\n\
         ::\n\
         Greeting.template\n",
    );
    assert_string(value, "Hello");
}

#[test]
fn model_decl_creates_record() {
    let value = run_value("model MyModel := 2 + 3\nMyModel.value\n");
    assert_eq!(value, Value::Int(5));
}

#[test]
fn agent_decl_creates_record() {
    let value = run_value(
        "agent Bot ::\n\
         policy default\n\
         memory store(\"disk\")\n\
         fn ping() -> Int ::\n\
             return 1\n\
         ::\n\
         ::\n\
         Bot.policy_name\n",
    );
    assert_string(value, "default");
}

#[test]
fn agent_memory_and_method_access() {
    let value = run_value(
        "agent Bot ::\n\
         memory store(\"disk\")\n\
         fn ping() -> Int ::\n\
             return 1\n\
         ::\n\
         ::\n\
         Bot.store.path\n",
    );
    assert_string(value, "disk");
}
