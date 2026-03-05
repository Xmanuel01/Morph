use enkaic::compiler::compile_module;
use enkaic::parser::parse_module;
use enkairt::error::RuntimeError;
use enkairt::object::Obj;
use enkairt::{Value, VM};
use std::sync::{Mutex, OnceLock};

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

fn tool_test_guard() -> std::sync::MutexGuard<'static, ()> {
    static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    GUARD
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|err| err.into_inner())
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
fn tool_decl_invokes_host_runner() {
    let _guard = tool_test_guard();
    let key = "ENKAI_TOOL_TOOL_TOOLS_ECHO";
    #[cfg(windows)]
    let command = r#"["powershell","-NoProfile","-Command","$null=[Console]::In.ReadToEnd(); Write-Output '{\"ok\":true}'"]"#;
    #[cfg(not(windows))]
    let command = r#"["sh","-c","cat >/dev/null; printf '{\"ok\":true}'"]"#;
    std::env::set_var(key, command);
    let value = run_value(
        "policy default ::\n    allow tool\n::\n\
         tool tools.echo(a: Int) -> Any\n\
         let out := tools.echo(1)\n\
         out.ok\n",
    );
    std::env::remove_var(key);
    assert_eq!(value, Value::Bool(true));
}

#[test]
fn tool_decl_rejects_legacy_split_runner_without_opt_in() {
    let _guard = tool_test_guard();
    std::env::remove_var("ENKAI_TOOL_ALLOW_LEGACY_SPLIT");
    #[cfg(windows)]
    std::env::set_var(
        "ENKAI_TOOL_RUNNER",
        r#"powershell -NoProfile -Command "Write-Output '{}'"#,
    );
    #[cfg(not(windows))]
    std::env::set_var("ENKAI_TOOL_RUNNER", "sh -c 'printf {}'");
    let result = run_result(
        "policy default ::\n    allow tool\n::\n\
         tool tools.echo(a: Int) -> Any\n\
         tools.echo(1)\n",
    );
    std::env::remove_var("ENKAI_TOOL_RUNNER");
    assert!(result.is_err());
    let message = result.err().unwrap().to_string();
    assert!(message.contains("[E_TOOL_CONFIG]"));
    assert!(message.contains("Tool command must be a JSON array"));
}

#[test]
fn tool_decl_spawn_failure_has_stable_error_code() {
    let _guard = tool_test_guard();
    std::env::remove_var("ENKAI_TOOL_ALLOW_LEGACY_SPLIT");
    std::env::set_var("ENKAI_TOOL_RUNNER", r#"["__enkai_missing_tool_runner__"]"#);
    let result = run_result(
        "policy default ::\n    allow tool\n::\n\
         tool tools.echo(a: Int) -> Any\n\
         tools.echo(1)\n",
    );
    std::env::remove_var("ENKAI_TOOL_RUNNER");
    assert!(result.is_err());
    let message = result.err().unwrap().to_string();
    assert!(message.contains("[E_TOOL_SPAWN]"));
}

#[test]
fn tool_decl_timeout_has_stable_error_code() {
    let _guard = tool_test_guard();
    let old_runner = std::env::var("ENKAI_TOOL_RUNNER").ok();
    let old_timeout = std::env::var("ENKAI_TOOL_TIMEOUT_MS").ok();
    #[cfg(windows)]
    std::env::set_var(
        "ENKAI_TOOL_RUNNER",
        r#"["powershell","-NoProfile","-Command","$null=[Console]::In.ReadToEnd(); Start-Sleep -Milliseconds 200; Write-Output '{\"ok\":true}'"]"#,
    );
    #[cfg(not(windows))]
    std::env::set_var(
        "ENKAI_TOOL_RUNNER",
        r#"["sh","-c","cat >/dev/null; sleep 1; printf '{\"ok\":true}'"]"#,
    );
    std::env::set_var("ENKAI_TOOL_TIMEOUT_MS", "10");
    let result = run_result(
        "policy default ::\n    allow tool\n::\n\
         tool tools.echo(a: Int) -> Any\n\
         tools.echo(1)\n",
    );
    if let Some(value) = old_runner {
        std::env::set_var("ENKAI_TOOL_RUNNER", value);
    } else {
        std::env::remove_var("ENKAI_TOOL_RUNNER");
    }
    if let Some(value) = old_timeout {
        std::env::set_var("ENKAI_TOOL_TIMEOUT_MS", value);
    } else {
        std::env::remove_var("ENKAI_TOOL_TIMEOUT_MS");
    }
    assert!(result.is_err());
    let message = result.err().unwrap().to_string();
    assert!(message.contains("[E_TOOL_TIMEOUT]"));
}

#[test]
fn tool_decl_policy_denial_has_stable_error_code() {
    let _guard = tool_test_guard();
    let key = "ENKAI_TOOL_TOOL_TOOLS_ECHO";
    #[cfg(windows)]
    let command = r#"["powershell","-NoProfile","-Command","$null=[Console]::In.ReadToEnd(); Write-Output '{\"ok\":true}'"]"#;
    #[cfg(not(windows))]
    let command = r#"["sh","-c","cat >/dev/null; printf '{\"ok\":true}'"]"#;
    std::env::set_var(key, command);
    let result = run_result("tool tools.echo(a: Int) -> Any\ntools.echo(1)\n");
    std::env::remove_var(key);
    assert!(result.is_err());
    let message = result.err().unwrap().to_string();
    assert!(message.contains("[E_POLICY_DENIED]"));
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
