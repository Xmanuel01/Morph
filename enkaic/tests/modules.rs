use std::fs;

use enkaic::loader::load_package;
use tempfile::tempdir;

#[test]
fn loads_modules_from_src() {
    let dir = tempdir().expect("tempdir");
    fs::write(
        dir.path().join("enkai.toml"),
        "[package]\nname = \"demo\"\n",
    )
    .unwrap();
    let src = dir.path().join("src");
    fs::create_dir_all(src.join("foo")).unwrap();
    fs::write(
        src.join("main.enk"),
        "use foo.bar\nfn main() -> Int ::\n    return 0\n::\n",
    )
    .unwrap();
    fs::write(
        src.join("foo").join("bar.enk"),
        "pub fn ping() -> Int ::\n    return 1\n::\n",
    )
    .unwrap();

    let package = load_package(&src.join("main.enk"), dir.path()).expect("package");
    assert_eq!(package.entry, vec!["main".to_string()]);
    assert!(package.modules.contains_key(&vec!["main".to_string()]));
    assert!(package
        .modules
        .contains_key(&vec!["foo".to_string(), "bar".to_string()]));
}

#[test]
fn reports_missing_module() {
    let dir = tempdir().expect("tempdir");
    fs::write(
        dir.path().join("enkai.toml"),
        "[package]\nname = \"demo\"\n",
    )
    .unwrap();
    let src = dir.path().join("src");
    fs::create_dir_all(&src).unwrap();
    fs::write(
        src.join("main.enk"),
        "use missing.module\nfn main() -> Int ::\n    return 0\n::\n",
    )
    .unwrap();

    let err = load_package(&src.join("main.enk"), dir.path()).unwrap_err();
    assert!(err.to_string().contains("Module not found"));
}

#[test]
fn rejects_private_symbol_import_with_labels() {
    let dir = tempdir().expect("tempdir");
    fs::write(
        dir.path().join("enkai.toml"),
        "[package]\nname = \"demo\"\n",
    )
    .unwrap();
    let src = dir.path().join("src");
    fs::create_dir_all(&src).unwrap();
    fs::write(
        src.join("main.enk"),
        "use foo.secret\nfn main() -> Int ::\n    return 0\n::\n",
    )
    .unwrap();
    fs::write(
        src.join("foo.enk"),
        "fn secret() -> Int ::\n    return 1\n::\n",
    )
    .unwrap();

    let err = load_package(&src.join("main.enk"), dir.path()).unwrap_err();
    let message = err.to_string();
    assert!(message.contains("Symbol is private"));
    assert!(message.contains("^ module"));
    assert!(message.contains("^ symbol"));
}

#[test]
fn resolves_path_dependency() {
    let dir = tempdir().expect("tempdir");
    let dep_dir = dir.path().join("dep");
    fs::create_dir_all(dep_dir.join("src")).unwrap();
    fs::write(
        dep_dir.join("src").join("lib.enk"),
        "pub fn add(a: Int, b: Int) -> Int ::\n    return a + b\n::\n",
    )
    .unwrap();

    fs::write(
        dir.path().join("enkai.toml"),
        "[package]\nname = \"demo\"\n\n[dependencies]\nutil = { path = \"dep\" }\n",
    )
    .unwrap();
    let src = dir.path().join("src");
    fs::create_dir_all(&src).unwrap();
    fs::write(
        src.join("main.enk"),
        "use util\nfn main() -> Int ::\n    return util.add(2, 3)\n::\n",
    )
    .unwrap();

    let package = load_package(&src.join("main.enk"), dir.path()).expect("package");
    assert!(package.modules.contains_key(&vec!["util".to_string()]));
}

#[test]
fn resolves_bare_string_path_dependency() {
    let dir = tempdir().expect("tempdir");
    let dep_dir = dir.path().join("dep");
    fs::create_dir_all(dep_dir.join("src")).unwrap();
    fs::write(
        dep_dir.join("src").join("lib.enk"),
        "pub fn add(a: Int, b: Int) -> Int ::\n    return a + b\n::\n",
    )
    .unwrap();

    fs::write(
        dir.path().join("enkai.toml"),
        "[package]\nname = \"demo\"\n\n[dependencies]\nutil = \"dep\"\n",
    )
    .unwrap();

    let src = dir.path().join("src");
    fs::create_dir_all(&src).unwrap();
    fs::write(
        src.join("main.enk"),
        "use util\nfn main() -> Int ::\n    return util.add(2, 3)\n::\n",
    )
    .unwrap();

    let package = load_package(&src.join("main.enk"), dir.path()).expect("package");
    assert!(package.modules.contains_key(&vec!["util".to_string()]));
}

#[test]
fn errors_on_missing_path_dependency() {
    let dir = tempdir().expect("tempdir");
    fs::write(
        dir.path().join("enkai.toml"),
        "[package]\nname = \"demo\"\n\n[dependencies]\nutil = { path = \"missing\" }\n",
    )
    .unwrap();

    let src = dir.path().join("src");
    fs::create_dir_all(&src).unwrap();
    fs::write(
        src.join("main.enk"),
        "use util\nfn main() -> Int ::\n    return util.add(2, 3)\n::\n",
    )
    .unwrap();

    let err = load_package(&src.join("main.enk"), dir.path()).unwrap_err();
    assert!(err.to_string().contains("Dependency path not found"));
}
