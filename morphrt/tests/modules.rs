use std::fs;

use morphc::loader::load_package;
use morphrt::{Interpreter, Value};
use tempfile::tempdir;

#[test]
fn runs_with_module_use_resolution() {
    let dir = tempdir().expect("tempdir");
    fs::write(
        dir.path().join("morph.toml"),
        "[package]\nname = \"demo\"\n",
    )
    .unwrap();
    let src = dir.path().join("src");
    fs::create_dir_all(&src).unwrap();
    fs::write(
        src.join("main.morph"),
        "use util\nfn main() -> Int ::\n    return util.add(2, 3)\n::\n",
    )
    .unwrap();
    fs::write(
        src.join("util.morph"),
        "pub fn add(a: Int, b: Int) -> Int ::\n    return a + b\n::\n",
    )
    .unwrap();

    let package = load_package(&src.join("main.morph"), dir.path()).expect("package");
    let mut interpreter = Interpreter::new();
    interpreter.eval_package(&package).expect("eval");
    let result = interpreter.call_main().expect("call").expect("value");
    assert_eq!(result, Value::Int(5));
}

#[test]
fn runs_with_symbol_import() {
    let dir = tempdir().expect("tempdir");
    fs::write(
        dir.path().join("morph.toml"),
        "[package]\nname = \"demo\"\n",
    )
    .unwrap();
    let src = dir.path().join("src");
    fs::create_dir_all(&src).unwrap();
    fs::write(
        src.join("main.morph"),
        "use math.add\nfn main() -> Int ::\n    return add(4, 1)\n::\n",
    )
    .unwrap();
    fs::write(
        src.join("math.morph"),
        "pub fn add(a: Int, b: Int) -> Int ::\n    return a + b\n::\n",
    )
    .unwrap();

    let package = load_package(&src.join("main.morph"), dir.path()).expect("package");
    let mut interpreter = Interpreter::new();
    interpreter.eval_package(&package).expect("eval");
    let result = interpreter.call_main().expect("call").expect("value");
    assert_eq!(result, Value::Int(5));
}

#[test]
fn runs_with_reexported_symbol() {
    let dir = tempdir().expect("tempdir");
    fs::write(
        dir.path().join("morph.toml"),
        "[package]\nname = \"demo\"\n",
    )
    .unwrap();
    let src = dir.path().join("src");
    fs::create_dir_all(&src).unwrap();
    fs::write(
        src.join("main.morph"),
        "use reexp.add\nfn main() -> Int ::\n    return add(7, 3)\n::\n",
    )
    .unwrap();
    fs::write(
        src.join("util.morph"),
        "pub fn add(a: Int, b: Int) -> Int ::\n    return a + b\n::\n",
    )
    .unwrap();
    fs::write(src.join("reexp.morph"), "pub use util::{add}\n").unwrap();

    let package = load_package(&src.join("main.morph"), dir.path()).expect("package");
    let mut interpreter = Interpreter::new();
    interpreter.eval_package(&package).expect("eval");
    let result = interpreter.call_main().expect("call").expect("value");
    assert_eq!(result, Value::Int(10));
}

#[test]
fn runs_with_use_list_symbol_import() {
    let dir = tempdir().expect("tempdir");
    fs::write(
        dir.path().join("morph.toml"),
        "[package]\nname = \"demo\"\n",
    )
    .unwrap();
    let src = dir.path().join("src");
    fs::create_dir_all(&src).unwrap();
    fs::write(
        src.join("main.morph"),
        "use math::{add}\nfn main() -> Int ::\n    return add(9, 1)\n::\n",
    )
    .unwrap();
    fs::write(
        src.join("math.morph"),
        "pub fn add(a: Int, b: Int) -> Int ::\n    return a + b\n::\n",
    )
    .unwrap();

    let package = load_package(&src.join("main.morph"), dir.path()).expect("package");
    let mut interpreter = Interpreter::new();
    interpreter.eval_package(&package).expect("eval");
    let result = interpreter.call_main().expect("call").expect("value");
    assert_eq!(result, Value::Int(10));
}
