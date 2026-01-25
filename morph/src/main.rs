use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

use morphc::compiler::{compile_package, CompileError};
use morphc::formatter::{check_format, format_source};
use morphc::modules::load_package;
use morphc::{TypeChecker, TypeError};
use morphrt::{Value, VM};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        process::exit(1);
    }

    let exit_code = match args[1].as_str() {
        "run" => run_command(&args[2..]),
        "check" => check_command(&args[2..]),
        "fmt" => fmt_command(&args[2..]),
        "test" => test_command(&args[2..]),
        _ => {
            print_usage();
            1
        }
    };
    process::exit(exit_code);
}

fn run_command(args: &[String]) -> i32 {
    if args.is_empty() {
        eprintln!("morph run requires a file or directory");
        return 1;
    }
    let mut trace_vm = false;
    let mut disasm = false;
    let mut file_arg: Option<String> = None;
    for arg in args {
        match arg.as_str() {
            "--trace-vm" => trace_vm = true,
            "--disasm" => disasm = true,
            _ => file_arg = Some(arg.clone()),
        }
    }
    let target = match file_arg {
        Some(t) => PathBuf::from(t),
        None => {
            eprintln!("morph run requires a file or directory");
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
    let mut vm = VM::new(trace_vm, disasm);
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
        eprintln!("morph check requires a file or directory");
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
    if let Err(err) = collect_morph_files_in_dir(&tests_dir, &mut files) {
        eprintln!("{}", err);
        return 1;
    }
    if files.is_empty() {
        eprintln!("No .morph tests found in {}", tests_dir.display());
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
        let mut vm = VM::new(false, false);
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
        eprintln!("morph fmt requires a file or directory");
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
            eprintln!("morph fmt requires a file or directory");
            return 1;
        }
    };
    let files = match collect_morph_files(&target) {
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

fn resolve_entry(target: &Path) -> Result<(PathBuf, PathBuf), String> {
    let metadata = fs::metadata(target)
        .map_err(|err| format!("Failed to read {}: {}", target.display(), err))?;
    if metadata.is_dir() {
        let root = target.to_path_buf();
        let manifest = root.join("morph.toml");
        if !manifest.is_file() {
            return Err(format!("morph.toml not found in {}", root.display()));
        }
        let entry = root.join("src").join("main.morph");
        if !entry.is_file() {
            return Err(format!("Entry file not found: {}", entry.display()));
        }
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
        if dir.join("morph.toml").is_file() {
            return Some(dir.to_path_buf());
        }
        current = dir.parent();
    }
    None
}

fn collect_morph_files(path: &Path) -> Result<Vec<PathBuf>, String> {
    if path.is_file() {
        if path.extension().and_then(|ext| ext.to_str()) == Some("morph") {
            return Ok(vec![path.to_path_buf()]);
        }
        return Err(format!("Not a .morph file: {}", path.display()));
    }
    if !path.is_dir() {
        return Err(format!("Path not found: {}", path.display()));
    }
    let scan_root = if path.join("morph.toml").is_file() {
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
    collect_morph_files_in_dir(&scan_root, &mut files)?;
    if files.is_empty() {
        return Err(format!("No .morph files found in {}", scan_root.display()));
    }
    Ok(files)
}

fn collect_morph_files_in_dir(dir: &Path, files: &mut Vec<PathBuf>) -> Result<(), String> {
    for entry in
        fs::read_dir(dir).map_err(|err| format!("Failed to read {}: {}", dir.display(), err))?
    {
        let entry = entry.map_err(|err| err.to_string())?;
        let path = entry.path();
        if path.is_dir() {
            collect_morph_files_in_dir(&path, files)?;
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("morph") {
            files.push(path);
        }
    }
    Ok(())
}

fn normalize_line_endings(input: &str) -> String {
    input.replace("\r\n", "\n")
}

fn print_usage() {
    eprintln!("Morph CLI");
    eprintln!("Usage:");
    eprintln!("  morph run [--trace-vm] [--disasm] <file|dir>");
    eprintln!("  morph check <file|dir>");
    eprintln!("  morph fmt [--check] <file|dir>");
    eprintln!("  morph test [dir]");
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
            dir.path().join("morph.toml"),
            "[package]\nname = \"demo\"\n",
        )
        .expect("manifest");
        let src = dir.path().join("src");
        fs::create_dir_all(&src).expect("src");
        fs::write(
            src.join("main.morph"),
            "fn main() -> Int ::\n    return 0\n::\n",
        )
        .expect("main");

        let (root, entry) = resolve_entry(dir.path()).expect("resolve");
        assert_eq!(root, dir.path());
        assert_eq!(entry, src.join("main.morph"));
    }

    #[test]
    fn run_rejects_type_error() {
        let dir = tempdir().expect("tempdir");
        let file = dir.path().join("bad.morph");
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
            tests_dir.join("ok.morph"),
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
            tests_dir.join("bad.morph"),
            "fn main() -> Int ::\n    return true\n::\n",
        )
        .unwrap();
        let code = test_command(&vec![dir.path().to_string_lossy().to_string()]);
        assert_ne!(code, 0);
    }

    #[test]
    fn fmt_check_fails_on_unformatted_source() {
        let dir = tempdir().expect("tempdir");
        let file = dir.path().join("bad.morph");
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
        let file = dir.path().join("fix.morph");
        fs::write(&file, "if true ::\nprint(\"hi\")\n::\n").unwrap();
        let code = fmt_command(&vec![file.to_string_lossy().to_string()]);
        assert_eq!(code, 0);
        let updated = fs::read_to_string(&file).expect("read");
        assert!(updated.contains("\n    print(\"hi\")\n"));
    }
}
