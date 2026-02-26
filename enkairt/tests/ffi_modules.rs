use std::fs;
use std::path::{Path, PathBuf};

use enkaic::compiler::compile_package;
use enkaic::modules::load_package;
use enkairt::error::RuntimeError;
use enkairt::object::Obj;
use enkairt::{Value, VM};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("repo root")
        .to_path_buf()
}

fn copy_std_modules(dest: &Path) {
    let std_src = repo_root().join("std");
    let std_dst = dest.join("std");
    fs::create_dir_all(&std_dst).expect("std dst");
    for entry in fs::read_dir(&std_src).expect("std entries") {
        let entry = entry.expect("std entry");
        let path = entry.path();
        if matches!(
            path.extension().and_then(|ext| ext.to_str()),
            Some("enk") | Some("en") | Some("enkai")
        ) {
            let name = path.file_name().expect("name");
            fs::copy(&path, std_dst.join(name)).expect("copy std module");
        }
    }
}

const POLICY_ALLOW_ALL: &str = "policy default ::\n    allow io\n    allow fs\n    allow env\n    allow process\n    allow net\n::\n\n";

fn inject_policy(source: &str) -> String {
    if source.contains("policy ") {
        return source.to_string();
    }
    let mut insert_at = 0usize;
    let mut in_native = false;
    for (idx, line) in source.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }
        if in_native {
            if trimmed == "::" {
                in_native = false;
                insert_at = idx + 1;
            }
            continue;
        }
        if trimmed.starts_with("import ") {
            insert_at = idx + 1;
            continue;
        }
        if trimmed.starts_with("native::import ") {
            in_native = true;
            insert_at = idx + 1;
            continue;
        }
        break;
    }
    let mut out = String::new();
    let mut inserted = false;
    for (idx, line) in source.lines().enumerate() {
        if !inserted && idx == insert_at {
            out.push_str(POLICY_ALLOW_ALL);
            inserted = true;
        }
        out.push_str(line);
        out.push('\n');
    }
    if !inserted {
        out.push_str(POLICY_ALLOW_ALL);
    }
    out
}

fn run_package(root: &Path, entry_name: &str, source: &str) -> Result<Value, RuntimeError> {
    let entry = root.join(entry_name);
    let source = inject_policy(source);
    fs::write(&entry, source).expect("write entry");
    let package = load_package(&entry).expect("load package");
    let program = compile_package(&package).expect("compile");
    let mut vm = VM::new(false, false, false, false);
    vm.run(&program)
}

fn run_package_raw(root: &Path, entry_name: &str, source: &str) -> Result<Value, RuntimeError> {
    let entry = root.join(entry_name);
    fs::write(&entry, source).expect("write entry");
    let package = load_package(&entry).expect("load package");
    let program = compile_package(&package).expect("compile");
    let mut vm = VM::new(false, false, false, false);
    vm.run(&program)
}

#[test]
fn std_fsx_roundtrip() {
    let temp = tempfile::tempdir().expect("tempdir");
    copy_std_modules(temp.path());
    let data_path = temp.path().join("data.bin");
    let data_path = data_path.to_string_lossy().replace('\\', "/");
    let source = format!(
        "import std::fsx\n\n\
        native::import \"enkai_native\" ::\n    fn buffer_from_string(data: String) -> Buffer\n    fn buffer_eq(a: Buffer, b: Buffer) -> Bool\n::\n\
        let path := \"{}\"\n\
        let buf := buffer_from_string(\"hello\")\n\
        fsx.write_bytes(path, buf)\n\
        let out := fsx.read_bytes(path)\n\
        buffer_eq(buf, out)\n",
        data_path
    );
    let value = run_package(temp.path(), "main.enk", &source).expect("run");
    assert_eq!(value, Value::Bool(true));
}

#[test]
fn std_zstd_roundtrip() {
    let temp = tempfile::tempdir().expect("tempdir");
    copy_std_modules(temp.path());
    let source = "import std::zstd\n\
        native::import \"enkai_native\" ::\n    fn buffer_from_string(data: String) -> Buffer\n    fn buffer_eq(a: Buffer, b: Buffer) -> Bool\n::\n\
        let input := buffer_from_string(\"hello\")\n\
        let compressed := zstd.compress(input, 1)\n\
        let output := zstd.decompress(compressed)\n\
        buffer_eq(input, output)\n";
    let value = run_package(temp.path(), "main.enk", source).expect("run");
    assert_eq!(value, Value::Bool(true));
}

#[test]
fn std_hash_sha256_len() {
    let temp = tempfile::tempdir().expect("tempdir");
    copy_std_modules(temp.path());
    let source = "import std::hash\n\
        native::import \"enkai_native\" ::\n    fn buffer_len(data: Buffer) -> Int\n::\n\
        let digest := hash.sha256_from_string(\"abc\")\n\
        buffer_len(digest)\n";
    let value = run_package(temp.path(), "main.enk", source).expect("run");
    assert_eq!(value, Value::Int(32));
}

#[test]
fn std_env_get_set_roundtrip() {
    let temp = tempfile::tempdir().expect("tempdir");
    copy_std_modules(temp.path());
    let source = "import std::env\n\
        native::import \"enkai_native\" ::\n    fn buffer_from_string(data: String) -> Buffer\n    fn buffer_eq(a: Buffer, b: Buffer) -> Bool\n::\n\
        let ok := env.set(\"ENKAI_TEST_KEY\", \"hello\")\n\
        let val := env.get(\"ENKAI_TEST_KEY\")?\n\
        env.remove(\"ENKAI_TEST_KEY\")\n\
        let expected := buffer_from_string(\"hello\")\n\
        let actual := buffer_from_string(val)\n\
        ok and buffer_eq(expected, actual)\n";
    let value = run_package(temp.path(), "main.enk", source).expect("run");
    assert_eq!(value, Value::Bool(true));
}

#[test]
fn std_path_basename() {
    let temp = tempfile::tempdir().expect("tempdir");
    copy_std_modules(temp.path());
    let source = "import std::path\n\
        native::import \"enkai_native\" ::\n    fn buffer_from_string(data: String) -> Buffer\n    fn buffer_eq(a: Buffer, b: Buffer) -> Bool\n::\n\
        let joined := path.join(\"foo\", \"bar\")\n\
        let base := path.basename(joined)?\n\
        let expected := buffer_from_string(\"bar\")\n\
        let actual := buffer_from_string(base)\n\
        buffer_eq(expected, actual)\n";
    let value = run_package(temp.path(), "main.enk", source).expect("run");
    assert_eq!(value, Value::Bool(true));
}

#[test]
fn std_time_now_ms() {
    let temp = tempfile::tempdir().expect("tempdir");
    copy_std_modules(temp.path());
    let source = "import std::time\n\
        let now := time.now_ms()\n\
        now > 0\n";
    let value = run_package(temp.path(), "main.enk", source).expect("run");
    assert_eq!(value, Value::Bool(true));
}

#[test]
fn std_log_emit() {
    let temp = tempfile::tempdir().expect("tempdir");
    copy_std_modules(temp.path());
    let source = "import std::log\n\
        log.info(\"hello\")\n\
        log.warn(\"warn\")\n\
        log.error(\"err\")\n\
        true\n";
    let value = run_package(temp.path(), "main.enk", source).expect("run");
    assert_eq!(value, Value::Bool(true));
}

#[test]
fn std_io_read_write_text() {
    let temp = tempfile::tempdir().expect("tempdir");
    copy_std_modules(temp.path());
    let data_path = temp.path().join("data.txt");
    let data_path = data_path.to_string_lossy().replace('\\', "/");
    let source = format!(
        "import std::io\n\
        native::import \"enkai_native\" ::\n    fn buffer_from_string(data: String) -> Buffer\n    fn buffer_eq(a: Buffer, b: Buffer) -> Bool\n::\n\
        io.write_text(\"{}\", \"hello\")\n\
        let text := io.read_text(\"{}\")?\n\
        let expected := buffer_from_string(\"hello\")\n\
        let actual := buffer_from_string(text)\n\
        buffer_eq(expected, actual)\n",
        data_path, data_path
    );
    let value = run_package(temp.path(), "main.enk", &source).expect("run");
    assert_eq!(value, Value::Bool(true));
}

#[test]
fn std_process_run_echo() {
    let temp = tempfile::tempdir().expect("tempdir");
    copy_std_modules(temp.path());
    let (cmd, args) = if cfg!(windows) {
        ("cmd", "[\"/C\",\"echo\",\"hello\"]")
    } else {
        ("sh", "[\"-c\",\"echo hello\"]")
    };
    let source = format!(
        "import std::process\n\
        native::import \"enkai_native\" ::\n    fn buffer_from_string(data: String) -> Buffer\n    fn buffer_eq(a: Buffer, b: Buffer) -> Bool\n::\n\
        let out := process.run(\"{}\", {}, none)\n\
        out.stdout\n",
        cmd, args
    );
    let value = run_package(temp.path(), "main.enk", &source).expect("run");
    let stdout = match value {
        Value::Obj(obj) => match obj.as_obj() {
            Obj::String(text) => text.clone(),
            _ => panic!("expected stdout string"),
        },
        _ => panic!("expected stdout string"),
    };
    let normalized = stdout.replace("\r\n", "\n");
    assert_eq!(normalized, "hello\n");
}

#[test]
fn policy_blocks_fs_without_allow() {
    let temp = tempfile::tempdir().expect("tempdir");
    copy_std_modules(temp.path());
    let data_path = temp.path().join("data.txt");
    fs::write(&data_path, "hello").expect("write data");
    let data_path = data_path.to_string_lossy().replace('\\', "/");
    let source = format!(
        "import std::io\n\
        io.read_text(\"{}\")\n",
        data_path
    );
    let result = run_package_raw(temp.path(), "main.enk", &source);
    assert!(result.is_err());
    let message = result.err().unwrap().to_string();
    assert!(message.contains("Policy denied"));
}
