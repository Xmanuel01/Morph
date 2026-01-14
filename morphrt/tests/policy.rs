use morphc::parser::parse_module;
use morphrt::{Interpreter, Value};

fn eval_main(source: &str) -> Result<Option<Value>, String> {
    let module = parse_module(source).map_err(|err| err.to_string())?;
    let mut interpreter = Interpreter::new();
    interpreter
        .eval_module(&module)
        .map_err(|err| err.to_string())?;
    interpreter.call_main().map_err(|err| err.to_string())
}

fn eval_err(source: &str) -> String {
    eval_main(source).expect_err("expected error")
}

#[test]
fn denies_io_without_policy() {
    let source = "fn main() -> Int ::\n    print(\"hi\")\n    return 0\n::\n";
    let err = eval_err(source);
    assert!(err.contains("Policy denied"));
}

#[test]
fn allows_io_with_default_policy() {
    let source = "\
policy default ::
    allow io.print
::
fn main() -> Int ::
    print(\"hi\")
    return 0
::
";
    assert!(eval_main(source).is_ok());
}

#[test]
fn denies_tool_without_policy() {
    let source = "\
tool web.search(query: String) -> String
fn main() -> Int ::
    web.search(\"hi\")
    return 0
::
";
    let err = eval_err(source);
    assert!(err.contains("Policy denied"));
}

#[test]
fn allows_tool_with_policy_but_stub_fails() {
    let source = "\
policy default ::
    allow tool.web.search
::
tool web.search(query: String) -> String
fn main() -> Int ::
    web.search(\"hi\")
    return 0
::
";
    let err = eval_err(source);
    assert!(err.contains("Tool not implemented"));
}

#[test]
fn includes_stack_trace_on_error() {
    let source = "\
fn inner() -> Int ::
    return 1 / 0
::
fn outer() -> Int ::
    return inner()
::
fn main() -> Int ::
    return outer()
::
";
    let message = eval_err(source);
    assert!(message.contains("Stack trace"));
    assert!(message.contains("inner"));
    assert!(message.contains("outer"));
    assert!(message.contains("main"));
}

#[test]
fn default_deny_blocks_restricted_operations() {
    let cases = [
        ("fs.read", "fn main() -> Int ::\n    std.fs.read(\"a\")\n    return 0\n::\n"),
        ("net.connect", "fn main() -> Int ::\n    std.net.connect(\"a\")\n    return 0\n::\n"),
        (
            "process.exec",
            "fn main() -> Int ::\n    std.process.exec(\"a\")\n    return 0\n::\n",
        ),
        (
            "model.invoke",
            "fn main() -> Int ::\n    std.ai.model_invoke(\"a\")\n    return 0\n::\n",
        ),
        (
            "memory.read",
            "fn main() -> Int ::\n    std.ai.memory_read(\"a\")\n    return 0\n::\n",
        ),
        (
            "tool.web.search",
            "tool web.search(query: String) -> String\nfn main() -> Int ::\n    web.search(\"a\")\n    return 0\n::\n",
        ),
    ];
    for (capability, source) in cases {
        let err = eval_err(source);
        assert!(err.contains("Policy denied"), "{}", capability);
    }
}

#[test]
fn allowlist_allows_fs_read_but_stub_fails() {
    let source = "\
policy default ::
    allow fs.read
::
fn main() -> Int ::
    std.fs.read(\"a\")
    return 0
::
";
    let err = eval_err(source);
    assert!(err.contains("not implemented"));
    assert!(!err.contains("Policy denied"));
}

#[test]
fn deny_overrides_allow() {
    let source = "\
policy default ::
    allow fs
    deny fs.read
::
fn main() -> Int ::
    std.fs.read(\"a\")
    return 0
::
";
    let err = eval_err(source);
    assert!(err.contains("Policy denied"));
}
