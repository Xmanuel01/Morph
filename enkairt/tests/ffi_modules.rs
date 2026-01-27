use std::fs;
use std::path::{Path, PathBuf};

use enkaic::compiler::compile_package;
use enkaic::modules::load_package;
use enkairt::error::RuntimeError;
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

fn run_package(root: &Path, entry_name: &str, source: &str) -> Result<Value, RuntimeError> {
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
