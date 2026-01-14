use std::fs;

use morphc::loader::load_package;
use tempfile::tempdir;

#[test]
fn loads_modules_from_src() {
    let dir = tempdir().expect("tempdir");
    fs::write(
        dir.path().join("morph.toml"),
        "[package]\nname = \"demo\"\n",
    )
    .unwrap();
    let src = dir.path().join("src");
    fs::create_dir_all(src.join("foo")).unwrap();
    fs::write(
        src.join("main.morph"),
        "use foo.bar\nfn main() -> Int ::\n    return 0\n::\n",
    )
    .unwrap();
    fs::write(
        src.join("foo").join("bar.morph"),
        "pub fn ping() -> Int ::\n    return 1\n::\n",
    )
    .unwrap();

    let package = load_package(&src.join("main.morph"), dir.path()).expect("package");
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
        dir.path().join("morph.toml"),
        "[package]\nname = \"demo\"\n",
    )
    .unwrap();
    let src = dir.path().join("src");
    fs::create_dir_all(&src).unwrap();
    fs::write(
        src.join("main.morph"),
        "use missing.module\nfn main() -> Int ::\n    return 0\n::\n",
    )
    .unwrap();

    let err = load_package(&src.join("main.morph"), dir.path()).unwrap_err();
    assert!(err.to_string().contains("Module not found"));
}
