use std::fs;

use morphc::diagnostic::Span;
use morphc::modules::load_package;
use morphc::{compiler::compile_package, TypeChecker};
use tempfile::tempdir;

#[test]
fn resolves_import_path() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    fs::write(
        root.join("main.morph"),
        "import app::utils\nfn main() -> Int ::\n    return utils.add(1, 2)\n::\n",
    )
    .unwrap();
    fs::create_dir_all(root.join("app")).unwrap();
    fs::write(
        root.join("app").join("utils.morph"),
        "pub fn add(a: Int, b: Int) -> Int ::\n    return a + b\n::\n",
    )
    .unwrap();

    let package = load_package(&root.join("main.morph")).expect("package");
    assert!(package.modules.len() >= 2);
}

#[test]
fn detects_circular_import() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    fs::write(root.join("main.morph"), "import app::utils\n").unwrap();
    fs::create_dir_all(root.join("app")).unwrap();
    fs::write(root.join("app").join("utils.morph"), "import main\n").unwrap();
    let err = load_package(&root.join("main.morph")).unwrap_err();
    assert!(err.to_string().contains("Circular import detected"));
}

#[test]
fn enforces_public_visibility() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    fs::write(
        root.join("main.morph"),
        "import app::utils as utils\nfn main() -> Int ::\n    return utils.secret(1)\n::\n",
    )
    .unwrap();
    fs::create_dir_all(root.join("app")).unwrap();
    fs::write(
        root.join("app").join("utils.morph"),
        "fn secret(x: Int) -> Int ::\n    return x\n::\n",
    )
    .unwrap();
    let package = load_package(&root.join("main.morph")).expect("package");
    let err = TypeChecker::check_package(&package).unwrap_err();
    assert!(err.message.contains("private to module"));
}

#[test]
fn compiler_rejects_private_symbol_access() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    fs::write(
        root.join("main.morph"),
        "import app::utils as utils\nfn main() -> Int ::\n    return utils.secret(1)\n::\n",
    )
    .unwrap();
    fs::create_dir_all(root.join("app")).unwrap();
    fs::write(
        root.join("app").join("utils.morph"),
        "fn secret(x: Int) -> Int ::\n    return x\n::\n",
    )
    .unwrap();
    let package = load_package(&root.join("main.morph")).expect("package");
    let err = compile_package(&package).unwrap_err();
    assert!(err.message.contains("private to module"));
}

#[test]
fn compiler_private_access_includes_span() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    fs::write(
        root.join("main.morph"),
        "import app::utils as utils\nfn main() -> Int ::\n    return utils.secret(1)\n::\n",
    )
    .unwrap();
    fs::create_dir_all(root.join("app")).unwrap();
    fs::write(
        root.join("app").join("utils.morph"),
        "fn secret(x: Int) -> Int ::\n    return x\n::\n",
    )
    .unwrap();
    let package = load_package(&root.join("main.morph")).expect("package");
    let err = compile_package(&package).unwrap_err();
    assert!(err.message.contains("private to module"));
    assert_eq!(
        err.span,
        Some(Span {
            line: 3,
            col: 17,
            end_col: 17
        })
    );
}

#[test]
fn compiler_private_access_span_shifted() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    fs::write(
        root.join("main.morph"),
        "import app::utils as utils\n\nfn main() -> Int ::\n    let x := 1\n    return  utils.secret(x)\n::\n",
    )
    .unwrap();
    fs::create_dir_all(root.join("app")).unwrap();
    fs::write(
        root.join("app").join("utils.morph"),
        "fn secret(x: Int) -> Int ::\n    return x\n::\n",
    )
    .unwrap();
    let package = load_package(&root.join("main.morph")).expect("package");
    let err = compile_package(&package).unwrap_err();
    assert!(err.message.contains("private to module"));
    assert_eq!(
        err.span,
        Some(Span {
            line: 5,
            col: 18,
            end_col: 18
        })
    );
}
