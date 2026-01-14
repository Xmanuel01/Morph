use std::fs;

use morphc::parser::parse_module;
use morphrt::{Interpreter, Value};
use tempfile::tempdir;

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

fn escape_path(path: &std::path::Path) -> String {
    path.to_string_lossy().replace('\\', "\\\\")
}

fn escape_backslash_path(raw: &str) -> String {
    raw.replace('\\', "\\\\")
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
fn allowlist_allows_fs_read() {
    let dir = tempdir().expect("tempdir");
    let file = dir.path().join("data.txt");
    fs::write(&file, "hello").expect("write");
    let path = escape_path(&file);
    let source = format!(
        "policy default ::\n    allow fs.read\n::\nfn main() -> Int ::\n    let data := std.fs.read(\"{}\")\n    return std.collections.len(data)\n::\n",
        path
    );
    let result = eval_main(&source).expect("run").expect("value");
    assert_eq!(result, Value::Int(5));
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

#[test]
fn allows_path_prefix_filter() {
    let dir = tempdir().expect("tempdir");
    let file = dir.path().join("data.txt");
    fs::write(&file, "ok").expect("write");
    let prefix = escape_path(dir.path());
    let path = escape_path(&file);
    let source = format!(
        "policy default ::\n    allow fs.read path_prefix=\"{}\"\n::\nfn main() -> Int ::\n    let data := std.fs.read(\"{}\")\n    return std.collections.len(data)\n::\n",
        prefix, path
    );
    let result = eval_main(&source).expect("run").expect("value");
    assert_eq!(result, Value::Int(2));
}

#[test]
fn denies_when_path_prefix_mismatches() {
    let dir = tempdir().expect("tempdir");
    let file = dir.path().join("data.txt");
    fs::write(&file, "no").expect("write");
    let path = escape_path(&file);
    let source = format!(
        "policy default ::\n    allow fs.read path_prefix=\"/tmp/other\"\n::\nfn main() -> Int ::\n    std.fs.read(\"{}\")\n    return 0\n::\n",
        path
    );
    let err = eval_err(&source);
    assert!(err.contains("Policy denied"));
}

#[test]
fn allows_domain_filter() {
    let source = "\
policy default ::
    allow net.connect domain=\".example.com\"
::
fn main() -> Int ::
    std.net.connect(\"api.example.com\")
    return 0
::
";
    let err = eval_err(source);
    assert!(err.contains("not implemented"));
    assert!(!err.contains("Policy denied"));
}

#[test]
fn allows_domain_suffix_without_dot() {
    let source = "\
policy default ::
    allow net.connect domain=\"example.com\"
::
fn main() -> Int ::
    std.net.connect(\"api.example.com\")
    return 0
::
";
    let err = eval_err(source);
    assert!(err.contains("not implemented"));
    assert!(!err.contains("Policy denied"));
}

#[test]
fn allows_path_prefix_with_normalization() {
    let dir = tempdir().expect("tempdir");
    let file = dir.path().join("data.txt");
    fs::write(&file, "ok").expect("write");
    let prefix_raw = dir.path().join("sub").join("..");
    let mut prefix = prefix_raw.to_string_lossy().replace('/', "\\");
    if cfg!(windows) {
        prefix = prefix.to_ascii_uppercase();
    }
    let prefix = escape_backslash_path(&prefix);
    let path = escape_path(&file);
    let source = format!(
        "policy default ::\n    allow fs.read path_prefix=\"{}\"\n::\nfn main() -> Int ::\n    let data := std.fs.read(\"{}\")\n    return std.collections.len(data)\n::\n",
        prefix, path
    );
    let result = eval_main(&source).expect("run").expect("value");
    assert_eq!(result, Value::Int(2));
}
