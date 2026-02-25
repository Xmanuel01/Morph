use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

use enkai_compiler::compiler::{compile_package, CompileError};
use enkai_compiler::formatter::{check_format, format_source};
use enkai_compiler::modules::load_package;
use enkai_compiler::{TypeChecker, TypeError};
use enkai_runtime::{Value, VM};

mod train;

const LANG_VERSION: &str = "0.9.3";

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() == 2 && (args[1] == "--version" || args[1] == "-V") {
        print_version();
        process::exit(0);
    }
    if args.len() < 2 {
        print_usage();
        process::exit(1);
    }

    let exit_code = match args[1].as_str() {
        "run" => run_command(&args[2..]),
        "check" => check_command(&args[2..]),
        "fmt" => fmt_command(&args[2..]),
        "test" => test_command(&args[2..]),
        "train" => train_command(&args[2..]),
        "eval" => eval_command(&args[2..]),
        _ => {
            print_usage();
            1
        }
    };
    process::exit(exit_code);
}

fn run_command(args: &[String]) -> i32 {
    if args.is_empty() {
        eprintln!("enkai run requires a file or directory");
        return 1;
    }
    let mut trace_vm = false;
    let mut disasm = false;
    let mut trace_task = false;
    let mut trace_net = false;
    let mut file_arg: Option<String> = None;
    for arg in args {
        match arg.as_str() {
            "--trace-vm" => trace_vm = true,
            "--disasm" => disasm = true,
            "--trace-task" => trace_task = true,
            "--trace-net" => trace_net = true,
            _ => file_arg = Some(arg.clone()),
        }
    }
    let target = match file_arg {
        Some(t) => PathBuf::from(t),
        None => {
            eprintln!("enkai run requires a file or directory");
            return 1;
        }
    };
    let (_root, entry) = match resolve_entry(&target) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{}", err);
            return 1;
        }
    };
    let package = match load_package(&entry) {
        Ok(p) => p,
        Err(err) => {
            eprintln!("{}", err);
            return 1;
        }
    };
    if let Err(err) = TypeChecker::check_package(&package) {
        print_type_error(&err);
        return 1;
    }
    let program = match compile_package(&package) {
        Ok(p) => p,
        Err(err) => {
            print_compile_error(&err);
            return 1;
        }
    };
    let mut vm = VM::new(trace_vm, disasm, trace_task, trace_net);
    match vm.run(&program) {
        Ok(Value::Int(code)) => code as i32,
        Ok(_) => 0,
        Err(err) => {
            eprintln!("Runtime error: {}", err);
            1
        }
    }
}

fn check_command(args: &[String]) -> i32 {
    if args.is_empty() {
        eprintln!("enkai check requires a file or directory");
        return 1;
    }
    let target = PathBuf::from(&args[0]);
    let (_root, entry) = match resolve_entry(&target) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{}", err);
            return 1;
        }
    };
    let package = match load_package(&entry) {
        Ok(p) => p,
        Err(err) => {
            eprintln!("{}", err);
            return 1;
        }
    };
    match TypeChecker::check_package(&package) {
        Ok(_) => 0,
        Err(err) => {
            print_type_error(&err);
            1
        }
    }
}

fn test_command(args: &[String]) -> i32 {
    let target = if args.is_empty() {
        PathBuf::from(".")
    } else {
        PathBuf::from(&args[0])
    };
    let root = if target.is_dir() {
        find_project_root(&target).unwrap_or_else(|| target.clone())
    } else {
        let parent = target
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));
        find_project_root(&parent).unwrap_or(parent)
    };
    let tests_dir = root.join("tests");
    if !tests_dir.is_dir() {
        eprintln!("tests directory not found: {}", tests_dir.display());
        return 1;
    }
    let mut files = Vec::new();
    if let Err(err) = collect_source_files_in_dir(&tests_dir, &mut files) {
        eprintln!("{}", err);
        return 1;
    }
    if files.is_empty() {
        eprintln!("No .enk/.en tests found in {}", tests_dir.display());
        return 1;
    }
    let mut passed = 0usize;
    let mut failed = 0usize;
    for file in files {
        let package = match load_package(&file) {
            Ok(p) => p,
            Err(err) => {
                eprintln!("[fail] {}: {}", file.display(), err);
                failed += 1;
                continue;
            }
        };
        if let Err(err) = TypeChecker::check_package(&package) {
            eprintln!("[fail] {}:", file.display());
            print_type_error(&err);
            failed += 1;
            continue;
        }
        let program = match compile_package(&package) {
            Ok(p) => p,
            Err(err) => {
                eprintln!("[fail] {}:", file.display());
                print_compile_error(&err);
                failed += 1;
                continue;
            }
        };
        let mut vm = VM::new(false, false, false, false);
        match vm.run(&program) {
            Ok(_) => {
                println!("[pass] {}", file.display());
                passed += 1;
            }
            Err(err) => {
                eprintln!("[fail] {}: Runtime error: {}", file.display(), err);
                failed += 1;
            }
        }
    }
    println!("Tests: {} passed, {} failed", passed, failed);
    if failed == 0 {
        0
    } else {
        1
    }
}

fn print_type_error(err: &TypeError) {
    if let Some(diagnostic) = err.diagnostic() {
        eprintln!("{}", diagnostic);
    } else {
        eprintln!(
            "Type error: {} at {}:{}",
            err.message, err.span.line, err.span.col
        );
    }
}

fn print_compile_error(err: &CompileError) {
    if let Some(diagnostic) = err.diagnostic() {
        eprintln!("{}", diagnostic);
    } else if let Some(span) = &err.span {
        eprintln!(
            "Compile error: {} at {}:{}",
            err.message, span.line, span.col
        );
    } else {
        eprintln!("Compile error: {}", err.message);
    }
}

fn fmt_command(args: &[String]) -> i32 {
    if args.is_empty() {
        eprintln!("enkai fmt requires a file or directory");
        return 1;
    }
    let mut check = false;
    let mut target: Option<PathBuf> = None;
    for arg in args {
        if arg == "--check" {
            check = true;
        } else if target.is_none() {
            target = Some(PathBuf::from(arg));
        } else {
            eprintln!("Unexpected argument: {}", arg);
            return 1;
        }
    }
    let target = match target {
        Some(path) => path,
        None => {
            eprintln!("enkai fmt requires a file or directory");
            return 1;
        }
    };
    let files = match collect_source_files(&target) {
        Ok(files) => files,
        Err(err) => {
            eprintln!("{}", err);
            return 1;
        }
    };
    let mut failed = false;
    for file in files {
        let source = match fs::read_to_string(&file) {
            Ok(source) => source,
            Err(err) => {
                eprintln!("Failed to read {}: {}", file.display(), err);
                failed = true;
                continue;
            }
        };
        if check {
            if let Err(err) = check_format(&source) {
                eprintln!("{}: {}", file.display(), err);
                failed = true;
            }
            continue;
        }
        match format_source(&source) {
            Ok(formatted) => {
                let normalized = normalize_line_endings(&source);
                if formatted != normalized {
                    if let Err(err) = fs::write(&file, formatted) {
                        eprintln!("Failed to write {}: {}", file.display(), err);
                        failed = true;
                    }
                }
            }
            Err(err) => {
                eprintln!("{}: {}", file.display(), err);
                failed = true;
            }
        }
    }
    if failed {
        1
    } else {
        0
    }
}

fn train_command(args: &[String]) -> i32 {
    if args.is_empty() {
        eprintln!("enkai train requires a config file");
        return 1;
    }
    let path = PathBuf::from(&args[0]);
    match train::train(&path) {
        Ok(_) => 0,
        Err(err) => {
            eprintln!("Train error: {}", err);
            1
        }
    }
}

fn eval_command(args: &[String]) -> i32 {
    if args.is_empty() {
        eprintln!("enkai eval requires a config file");
        return 1;
    }
    let path = PathBuf::from(&args[0]);
    match train::eval(&path) {
        Ok(_) => 0,
        Err(err) => {
            eprintln!("Eval error: {}", err);
            1
        }
    }
}

fn resolve_entry(target: &Path) -> Result<(PathBuf, PathBuf), String> {
    let metadata = fs::metadata(target)
        .map_err(|err| format!("Failed to read {}: {}", target.display(), err))?;
    if metadata.is_dir() {
        let root = target.to_path_buf();
        let _manifest = find_manifest(&root)
            .ok_or_else(|| format!("enkai.toml not found in {}", root.display()))?;
        let entry = find_entry_file(&root)
            .ok_or_else(|| "Entry file not found (main.enk/main.en/main.enkai)".to_string())?;
        Ok((root, entry))
    } else {
        let file = target.to_path_buf();
        let parent = file
            .parent()
            .ok_or_else(|| "Invalid file path".to_string())?;
        let root = find_project_root(parent).unwrap_or_else(|| parent.to_path_buf());
        Ok((root, file))
    }
}

fn find_project_root(start: &Path) -> Option<PathBuf> {
    let mut current = Some(start);
    while let Some(dir) = current {
        if find_manifest(dir).is_some() {
            return Some(dir.to_path_buf());
        }
        current = dir.parent();
    }
    None
}

fn collect_source_files(path: &Path) -> Result<Vec<PathBuf>, String> {
    if path.is_file() {
        if is_source_extension(path) {
            return Ok(vec![path.to_path_buf()]);
        }
        return Err(format!("Not an .enk/.en file: {}", path.display()));
    }
    if !path.is_dir() {
        return Err(format!("Path not found: {}", path.display()));
    }
    let scan_root = if find_manifest(path).is_some() {
        let src = path.join("src");
        if src.is_dir() {
            src
        } else {
            path.to_path_buf()
        }
    } else {
        path.to_path_buf()
    };
    let mut files = Vec::new();
    collect_source_files_in_dir(&scan_root, &mut files)?;
    if files.is_empty() {
        return Err(format!(
            "No .enk/.en files found in {}",
            scan_root.display()
        ));
    }
    Ok(files)
}

fn collect_source_files_in_dir(dir: &Path, files: &mut Vec<PathBuf>) -> Result<(), String> {
    for entry in
        fs::read_dir(dir).map_err(|err| format!("Failed to read {}: {}", dir.display(), err))?
    {
        let entry = entry.map_err(|err| err.to_string())?;
        let path = entry.path();
        if path.is_dir() {
            collect_source_files_in_dir(&path, files)?;
        } else if is_source_extension(&path) {
            files.push(path);
        }
    }
    Ok(())
}

fn normalize_line_endings(input: &str) -> String {
    input.replace("\r\n", "\n")
}

fn print_usage() {
    eprintln!("Enkai CLI");
    eprintln!("Usage:");
    eprintln!("  enkai --version");
    eprintln!("  enkai run [--trace-vm] [--disasm] [--trace-task] [--trace-net] <file|dir>");
    eprintln!("  enkai check <file|dir>");
    eprintln!("  enkai fmt [--check] <file|dir>");
    eprintln!("  enkai test [dir]");
    eprintln!("  enkai train <config.enk>");
    eprintln!("  enkai eval <config.enk>");
}

fn print_version() {
    println!(
        "Enkai v{} (cli {})",
        LANG_VERSION,
        env!("CARGO_PKG_VERSION")
    );
}

fn is_source_extension(path: &Path) -> bool {
    let ext = path.extension().and_then(|ext| ext.to_str());
    matches!(ext, Some("enk") | Some("en") | Some("enkai"))
}

fn find_manifest(root: &Path) -> Option<PathBuf> {
    let enkai = root.join("enkai.toml");
    if enkai.is_file() {
        return Some(enkai);
    }
    let legacy = root.join("enkai.toml");
    if legacy.is_file() {
        return Some(legacy);
    }
    None
}

fn find_entry_file(root: &Path) -> Option<PathBuf> {
    let candidates = if root.join("src").is_dir() {
        root.join("src")
    } else {
        root.to_path_buf()
    };
    let enk = candidates.join("main.enk");
    if enk.is_file() {
        return Some(enk);
    }
    let en = candidates.join("main.en");
    if en.is_file() {
        return Some(en);
    }
    let legacy = candidates.join("main.enkai");
    if legacy.is_file() {
        return Some(legacy);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    use tempfile::tempdir;

    #[test]
    fn resolves_project_entry_from_directory() {
        let dir = tempdir().expect("tempdir");
        fs::write(
            dir.path().join("enkai.toml"),
            "[package]\nname = \"demo\"\n",
        )
        .expect("manifest");
        let src = dir.path().join("src");
        fs::create_dir_all(&src).expect("src");
        fs::write(
            src.join("main.enk"),
            "fn main() -> Int ::\n    return 0\n::\n",
        )
        .expect("main");

        let (root, entry) = resolve_entry(dir.path()).expect("resolve");
        assert_eq!(root, dir.path());
        assert_eq!(entry, src.join("main.enk"));
    }

    #[test]
    fn run_rejects_type_error() {
        let dir = tempdir().expect("tempdir");
        let file = dir.path().join("bad.enk");
        fs::write(&file, "fn f() -> Int ::\n    return true\n::\n").unwrap();
        let code = run_command(&vec![file.to_string_lossy().to_string()]);
        assert_ne!(code, 0);
    }

    #[test]
    fn test_command_passes() {
        let dir = tempdir().expect("tempdir");
        let tests_dir = dir.path().join("tests");
        fs::create_dir_all(&tests_dir).expect("tests dir");
        fs::write(
            tests_dir.join("ok.enk"),
            "fn main() -> Int ::\n    return 0\n::\n",
        )
        .unwrap();
        let code = test_command(&vec![dir.path().to_string_lossy().to_string()]);
        assert_eq!(code, 0);
    }

    #[test]
    fn test_command_fails_on_type_error() {
        let dir = tempdir().expect("tempdir");
        let tests_dir = dir.path().join("tests");
        fs::create_dir_all(&tests_dir).expect("tests dir");
        fs::write(
            tests_dir.join("bad.enk"),
            "fn main() -> Int ::\n    return true\n::\n",
        )
        .unwrap();
        let code = test_command(&vec![dir.path().to_string_lossy().to_string()]);
        assert_ne!(code, 0);
    }

    #[test]
    fn fmt_check_fails_on_unformatted_source() {
        let dir = tempdir().expect("tempdir");
        let file = dir.path().join("bad.enk");
        fs::write(&file, "if true ::\nprint(\"hi\")\n::\n").unwrap();
        let code = fmt_command(&vec![
            "--check".to_string(),
            file.to_string_lossy().to_string(),
        ]);
        assert_ne!(code, 0);
    }

    #[test]
    fn fmt_formats_file() {
        let dir = tempdir().expect("tempdir");
        let file = dir.path().join("fix.enk");
        fs::write(&file, "if true ::\nprint(\"hi\")\n::\n").unwrap();
        let code = fmt_command(&vec![file.to_string_lossy().to_string()]);
        assert_eq!(code, 0);
        let updated = fs::read_to_string(&file).expect("read");
        assert!(updated.contains("\n    print(\"hi\")\n"));
    }
}
