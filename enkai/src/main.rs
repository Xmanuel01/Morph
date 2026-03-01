use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

use semver::Version;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use toml::Value as TomlValue;

use enkai_compiler::bytecode::Program;
use enkai_compiler::compiler::{compile_package, CompileError};
use enkai_compiler::formatter::{check_format, format_source};
use enkai_compiler::modules::load_package;
use enkai_compiler::{TypeChecker, TypeError};
use enkai_runtime::{Value, VM};

mod train;

const LANG_VERSION: &str = "1.3.0";

#[derive(Debug, Clone)]
struct DependencySpec {
    name: String,
    path: PathBuf,
    version_req: Option<String>,
}

#[derive(Debug, Clone)]
struct ResolvedDependency {
    name: String,
    path: PathBuf,
    version: Option<String>,
    package: Option<String>,
}

#[derive(Debug, Clone)]
struct ManifestInfo {
    dependencies: Vec<DependencySpec>,
}

#[derive(Debug, Clone, Default)]
struct PackageInfo {
    name: Option<String>,
    version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BuildCacheMeta {
    cache_version: u32,
    lang_version: String,
    entry: String,
    hash: String,
    program: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ServeModelSelection {
    checkpoint_path: PathBuf,
    model_name: Option<String>,
    model_version: Option<String>,
    registry: Option<PathBuf>,
}

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
        "serve" => serve_command(&args[2..]),
        "check" => check_command(&args[2..]),
        "fmt" => fmt_command(&args[2..]),
        "build" => build_command(&args[2..]),
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
    let (root, entry) = match resolve_entry(&target) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{}", err);
            return 1;
        }
    };
    let program = match load_cached_program(&root, &entry) {
        Ok(Some(program)) => program,
        Ok(None) => {
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
            match compile_package(&package) {
                Ok(p) => p,
                Err(err) => {
                    print_compile_error(&err);
                    return 1;
                }
            }
        }
        Err(err) => {
            eprintln!("cache disabled: {}", err);
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
            match compile_package(&package) {
                Ok(p) => p,
                Err(err) => {
                    print_compile_error(&err);
                    return 1;
                }
            }
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

fn serve_command(args: &[String]) -> i32 {
    let mut host: Option<String> = None;
    let mut port: Option<String> = None;
    let mut registry: Option<String> = None;
    let mut model: Option<String> = None;
    let mut model_version: Option<String> = None;
    let mut latest = false;
    let mut checkpoint: Option<String> = None;
    let mut run_args = Vec::new();
    let mut target: Option<String> = None;
    let mut idx = 0usize;
    while idx < args.len() {
        let arg = &args[idx];
        match arg.as_str() {
            "--host" => {
                idx += 1;
                if idx >= args.len() {
                    eprintln!("enkai serve --host requires a value");
                    return 1;
                }
                host = Some(args[idx].clone());
            }
            "--port" => {
                idx += 1;
                if idx >= args.len() {
                    eprintln!("enkai serve --port requires a value");
                    return 1;
                }
                port = Some(args[idx].clone());
            }
            "--registry" => {
                idx += 1;
                if idx >= args.len() {
                    eprintln!("enkai serve --registry requires a value");
                    return 1;
                }
                registry = Some(args[idx].clone());
            }
            "--model" => {
                idx += 1;
                if idx >= args.len() {
                    eprintln!("enkai serve --model requires a value");
                    return 1;
                }
                model = Some(args[idx].clone());
            }
            "--model-version" => {
                idx += 1;
                if idx >= args.len() {
                    eprintln!("enkai serve --model-version requires a value");
                    return 1;
                }
                model_version = Some(args[idx].clone());
            }
            "--latest" => {
                latest = true;
            }
            "--checkpoint" => {
                idx += 1;
                if idx >= args.len() {
                    eprintln!("enkai serve --checkpoint requires a value");
                    return 1;
                }
                checkpoint = Some(args[idx].clone());
            }
            "--trace-vm" | "--disasm" | "--trace-task" | "--trace-net" => {
                run_args.push(arg.clone());
            }
            _ => {
                if arg.starts_with("--") {
                    eprintln!("Unknown serve option: {}", arg);
                    return 1;
                }
                if target.is_some() {
                    eprintln!("enkai serve accepts only one file or directory target");
                    return 1;
                }
                target = Some(arg.clone());
            }
        }
        idx += 1;
    }
    let model_selection = match resolve_serve_model_selection(
        checkpoint.as_deref(),
        registry.as_deref(),
        model.as_deref(),
        model_version.as_deref(),
        latest,
    ) {
        Ok(selection) => selection,
        Err(err) => {
            eprintln!("enkai serve: {}", err);
            return 1;
        }
    };
    if let Some(host) = host {
        env::set_var("ENKAI_SERVE_HOST", host);
    }
    if let Some(port) = port {
        env::set_var("ENKAI_SERVE_PORT", port);
    }
    if let Some(selection) = model_selection {
        env::set_var(
            "ENKAI_SERVE_MODEL_PATH",
            selection.checkpoint_path.to_string_lossy().to_string(),
        );
        if let Some(name) = selection.model_name {
            env::set_var("ENKAI_SERVE_MODEL_NAME", name);
        }
        if let Some(version) = selection.model_version {
            env::set_var("ENKAI_SERVE_MODEL_VERSION", version);
        }
        if let Some(registry_path) = selection.registry {
            env::set_var(
                "ENKAI_SERVE_MODEL_REGISTRY",
                registry_path.to_string_lossy().to_string(),
            );
        }
    }
    run_args.push(target.unwrap_or_else(|| ".".to_string()));
    run_command(&run_args)
}

fn resolve_serve_model_selection(
    checkpoint: Option<&str>,
    registry: Option<&str>,
    model: Option<&str>,
    model_version: Option<&str>,
    latest: bool,
) -> Result<Option<ServeModelSelection>, String> {
    if checkpoint.is_some()
        && (registry.is_some() || model.is_some() || model_version.is_some() || latest)
    {
        return Err(
            "use either --checkpoint or --registry/--model/--model-version/--latest".to_string(),
        );
    }

    if let Some(path) = checkpoint {
        let checkpoint_path = canonical_existing_path(Path::new(path), "--checkpoint")?;
        return Ok(Some(ServeModelSelection {
            checkpoint_path,
            model_name: None,
            model_version: None,
            registry: None,
        }));
    }

    if registry.is_none() && model.is_none() && model_version.is_none() && !latest {
        return Ok(None);
    }

    let registry =
        registry.ok_or_else(|| "missing --registry for model registry resolution".to_string())?;
    let model = model.ok_or_else(|| "missing --model for model registry resolution".to_string())?;
    if latest && model_version.is_some() {
        return Err("use --latest or --model-version, not both".to_string());
    }

    let registry_path = canonical_existing_path(Path::new(registry), "--registry")?;
    let model_root = registry_path.join(model);
    if !model_root.is_dir() {
        return Err(format!(
            "model '{}' not found under registry {}",
            model,
            registry_path.display()
        ));
    }

    let version = if let Some(version) = model_version {
        version.to_string()
    } else {
        select_latest_model_version(&model_root)?
    };
    let version_dir = model_root.join(&version);
    if !version_dir.exists() {
        return Err(format!(
            "version '{}' for model '{}' does not exist in {}",
            version,
            model,
            model_root.display()
        ));
    }
    let checkpoint_path = if version_dir.join("checkpoint").exists() {
        canonical_existing_path(&version_dir.join("checkpoint"), "checkpoint directory")?
    } else {
        canonical_existing_path(&version_dir, "model version directory")?
    };
    Ok(Some(ServeModelSelection {
        checkpoint_path,
        model_name: Some(model.to_string()),
        model_version: Some(version),
        registry: Some(registry_path),
    }))
}

fn canonical_existing_path(path: &Path, label: &str) -> Result<PathBuf, String> {
    if !path.exists() {
        return Err(format!("{} not found: {}", label, path.display()));
    }
    fs::canonicalize(path)
        .map_err(|err| format!("failed to resolve {} {}: {}", label, path.display(), err))
}

fn select_latest_model_version(model_root: &Path) -> Result<String, String> {
    let mut versions = Vec::new();
    let entries = fs::read_dir(model_root).map_err(|err| {
        format!(
            "failed to read model registry {}: {}",
            model_root.display(),
            err
        )
    })?;
    for entry in entries {
        let entry = entry.map_err(|err| {
            format!(
                "failed to read model registry entry in {}: {}",
                model_root.display(),
                err
            )
        })?;
        if !entry.path().is_dir() {
            continue;
        }
        let name = entry.file_name();
        let name = name.to_string_lossy().to_string();
        if !name.is_empty() {
            versions.push(name);
        }
    }
    if versions.is_empty() {
        return Err(format!(
            "no versions found for model directory {}",
            model_root.display()
        ));
    }
    versions.sort_by(compare_version_labels);
    versions
        .pop()
        .ok_or_else(|| "failed to select latest model version".to_string())
}

fn compare_version_labels(a: &String, b: &String) -> std::cmp::Ordering {
    let parsed_a = parse_semver_label(a);
    let parsed_b = parse_semver_label(b);
    match (parsed_a, parsed_b) {
        (Some(left), Some(right)) => left.cmp(&right),
        (Some(_), None) => std::cmp::Ordering::Greater,
        (None, Some(_)) => std::cmp::Ordering::Less,
        (None, None) => a.cmp(b),
    }
}

fn parse_semver_label(value: &str) -> Option<Version> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    let candidate = trimmed.strip_prefix('v').unwrap_or(trimmed);
    Version::parse(candidate).ok()
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

fn build_command(args: &[String]) -> i32 {
    let target = if args.is_empty() {
        PathBuf::from(".")
    } else {
        PathBuf::from(&args[0])
    };
    let (root, entry) = match resolve_entry(&target) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{}", err);
            return 1;
        }
    };
    if find_manifest(&root).is_none() {
        eprintln!("enkai.toml not found in {}", root.display());
        return 1;
    }
    let manifest = match read_manifest_info(&root) {
        Ok(info) => info,
        Err(err) => {
            eprintln!("{}", err);
            return 1;
        }
    };
    let deps = match resolve_dependencies(&root, &manifest) {
        Ok(list) => list,
        Err(err) => {
            eprintln!("{}", err);
            return 1;
        }
    };
    if let Err(err) = write_lockfile(&root, &deps) {
        eprintln!("{}", err);
        return 1;
    }
    let hash = match compute_build_hash(&root, &deps) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{}", err);
            return 1;
        }
    };
    let entry_rel = entry
        .strip_prefix(&root)
        .unwrap_or(&entry)
        .to_string_lossy()
        .to_string();
    if let Ok(Some(meta)) = load_build_cache(&root) {
        if is_cache_valid(&meta, &entry_rel, &hash) {
            println!("build up to date");
            return 0;
        }
    }
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
    if let Err(err) = write_build_cache(&root, &entry_rel, &hash, &program) {
        eprintln!("{}", err);
        return 1;
    }
    println!("build ok");
    0
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
    eprintln!(
        "  enkai serve [--host <host>] [--port <port>] [--registry <dir> --model <name> [--model-version <v>|--latest] | --checkpoint <path>] [--trace-vm] [--disasm] [--trace-task] [--trace-net] [file|dir]"
    );
    eprintln!("  enkai check <file|dir>");
    eprintln!("  enkai fmt [--check] <file|dir>");
    eprintln!("  enkai build [dir]");
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

const BUILD_CACHE_VERSION: u32 = 1;
const LOCK_VERSION: i64 = 1;

fn cache_dir(root: &Path) -> PathBuf {
    root.join("target").join("enkai")
}

fn cache_meta_path(root: &Path) -> PathBuf {
    cache_dir(root).join("build.json")
}

fn cache_program_path(root: &Path, meta: &BuildCacheMeta) -> PathBuf {
    cache_dir(root).join(&meta.program)
}

fn load_build_cache(root: &Path) -> Result<Option<BuildCacheMeta>, String> {
    let path = cache_meta_path(root);
    if !path.is_file() {
        return Ok(None);
    }
    let text = fs::read_to_string(&path)
        .map_err(|err| format!("Failed to read {}: {}", path.display(), err))?;
    let meta: BuildCacheMeta =
        serde_json::from_str(&text).map_err(|err| format!("Invalid cache: {}", err))?;
    Ok(Some(meta))
}

fn is_cache_valid(meta: &BuildCacheMeta, entry: &str, hash: &str) -> bool {
    meta.cache_version == BUILD_CACHE_VERSION
        && meta.lang_version == LANG_VERSION
        && meta.entry == entry
        && meta.hash == hash
}

fn write_build_cache(
    root: &Path,
    entry: &str,
    hash: &str,
    program: &Program,
) -> Result<(), String> {
    let dir = cache_dir(root);
    fs::create_dir_all(&dir)
        .map_err(|err| format!("Failed to create {}: {}", dir.display(), err))?;
    let program_path = dir.join("program.bin");
    let encoded =
        bincode::serialize(program).map_err(|err| format!("Cache serialize failed: {}", err))?;
    fs::write(&program_path, encoded)
        .map_err(|err| format!("Failed to write {}: {}", program_path.display(), err))?;
    let meta = BuildCacheMeta {
        cache_version: BUILD_CACHE_VERSION,
        lang_version: LANG_VERSION.to_string(),
        entry: entry.to_string(),
        hash: hash.to_string(),
        program: "program.bin".to_string(),
    };
    let meta_text =
        serde_json::to_string_pretty(&meta).map_err(|err| format!("Cache meta error: {}", err))?;
    let meta_path = cache_meta_path(root);
    fs::write(&meta_path, meta_text)
        .map_err(|err| format!("Failed to write {}: {}", meta_path.display(), err))?;
    Ok(())
}

fn load_cached_program(root: &Path, entry: &Path) -> Result<Option<Program>, String> {
    let Some(meta) = load_build_cache(root)? else {
        return Ok(None);
    };
    let lock_path = root.join("enkai.lock");
    if !lock_path.is_file() {
        return Ok(None);
    }
    let entry_rel = entry
        .strip_prefix(root)
        .unwrap_or(entry)
        .to_string_lossy()
        .to_string();
    if !is_cache_valid(&meta, &entry_rel, &meta.hash) {
        return Ok(None);
    }
    let deps = read_lock_dependencies(root)?;
    let hash = compute_build_hash(root, &deps)?;
    if meta.hash != hash {
        return Ok(None);
    }
    let program_path = cache_program_path(root, &meta);
    if !program_path.is_file() {
        return Ok(None);
    }
    let bytes = fs::read(&program_path)
        .map_err(|err| format!("Failed to read {}: {}", program_path.display(), err))?;
    let program: Program =
        bincode::deserialize(&bytes).map_err(|err| format!("Cache decode failed: {}", err))?;
    Ok(Some(program))
}

fn read_manifest_info(root: &Path) -> Result<ManifestInfo, String> {
    let path = root.join("enkai.toml");
    let source = fs::read_to_string(&path)
        .map_err(|err| format!("Failed to read {}: {}", path.display(), err))?;
    let value = source
        .parse::<TomlValue>()
        .map_err(|err| format!("Failed to parse {}: {}", path.display(), err))?;
    let mut dependencies = Vec::new();
    if let Some(table) = value.get("dependencies").and_then(|val| val.as_table()) {
        for (name, entry) in table {
            let (path, version_req) = if let Some(path) = entry.as_str() {
                (PathBuf::from(path), None)
            } else if let Some(dep_table) = entry.as_table() {
                let path = dep_table
                    .get("path")
                    .and_then(|val| val.as_str())
                    .ok_or_else(|| format!("Dependency {} missing path", name))?;
                let version_req = dep_table
                    .get("version")
                    .and_then(|val| val.as_str())
                    .map(|s| s.to_string());
                (PathBuf::from(path), version_req)
            } else {
                return Err(format!("Dependency {} must be path or table", name));
            };
            dependencies.push(DependencySpec {
                name: name.to_string(),
                path,
                version_req,
            });
        }
    }
    Ok(ManifestInfo { dependencies })
}

fn read_package_info(root: &Path) -> Result<PackageInfo, String> {
    let path = root.join("enkai.toml");
    if !path.is_file() {
        return Ok(PackageInfo::default());
    }
    let source = fs::read_to_string(&path)
        .map_err(|err| format!("Failed to read {}: {}", path.display(), err))?;
    let value = source
        .parse::<TomlValue>()
        .map_err(|err| format!("Failed to parse {}: {}", path.display(), err))?;
    let mut info = PackageInfo::default();
    if let Some(table) = value.get("package").and_then(|val| val.as_table()) {
        info.name = table
            .get("name")
            .and_then(|val| val.as_str())
            .map(|s| s.to_string());
        info.version = table
            .get("version")
            .and_then(|val| val.as_str())
            .map(|s| s.to_string());
    }
    Ok(info)
}

fn resolve_dependencies(
    root: &Path,
    manifest: &ManifestInfo,
) -> Result<Vec<ResolvedDependency>, String> {
    let mut resolved = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for dep in &manifest.dependencies {
        if !seen.insert(dep.name.clone()) {
            return Err(format!("Duplicate dependency name {}", dep.name));
        }
        let dep_root = root.join(&dep.path);
        if !dep_root.is_dir() {
            return Err(format!(
                "Dependency {} path not found: {}",
                dep.name,
                dep_root.display()
            ));
        }
        let info = read_package_info(&dep_root)?;
        if let Some(req) = &dep.version_req {
            match info.version.as_ref() {
                Some(version) if version == req => {}
                Some(version) => {
                    return Err(format!(
                        "Dependency {} version mismatch (expected {}, found {})",
                        dep.name, req, version
                    ))
                }
                None => {
                    return Err(format!(
                        "Dependency {} missing package.version (expected {})",
                        dep.name, req
                    ))
                }
            }
        }
        resolved.push(ResolvedDependency {
            name: dep.name.clone(),
            path: dep.path.clone(),
            version: info.version,
            package: info.name,
        });
    }
    resolved.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(resolved)
}

fn write_lockfile(root: &Path, deps: &[ResolvedDependency]) -> Result<(), String> {
    let mut out = String::new();
    out.push_str("lock_version = 1\n");
    for dep in deps {
        out.push_str("\n[[dependency]]\n");
        out.push_str(&format!("name = \"{}\"\n", toml_escape(&dep.name)));
        out.push_str(&format!(
            "path = \"{}\"\n",
            toml_escape(&dep.path.to_string_lossy())
        ));
        if let Some(version) = &dep.version {
            out.push_str(&format!("version = \"{}\"\n", toml_escape(version)));
        }
        if let Some(package) = &dep.package {
            out.push_str(&format!("package = \"{}\"\n", toml_escape(package)));
        }
    }
    let path = root.join("enkai.lock");
    fs::write(&path, out).map_err(|err| format!("Failed to write {}: {}", path.display(), err))?;
    Ok(())
}

fn read_lock_dependencies(root: &Path) -> Result<Vec<ResolvedDependency>, String> {
    let path = root.join("enkai.lock");
    let source = fs::read_to_string(&path)
        .map_err(|err| format!("Failed to read {}: {}", path.display(), err))?;
    let value = source
        .parse::<TomlValue>()
        .map_err(|err| format!("Failed to parse {}: {}", path.display(), err))?;
    if let Some(lock_version) = value.get("lock_version").and_then(|v| v.as_integer()) {
        if lock_version != LOCK_VERSION {
            return Err(format!("Unsupported lock_version {}", lock_version));
        }
    }
    let mut deps = Vec::new();
    if let Some(entries) = value.get("dependency").and_then(|v| v.as_array()) {
        for entry in entries {
            let name = entry
                .get("name")
                .and_then(|val| val.as_str())
                .ok_or_else(|| "Dependency entry missing name".to_string())?;
            let path_value = entry
                .get("path")
                .and_then(|val| val.as_str())
                .ok_or_else(|| format!("Dependency {} missing path", name))?;
            let version = entry
                .get("version")
                .and_then(|val| val.as_str())
                .map(|s| s.to_string());
            let package = entry
                .get("package")
                .and_then(|val| val.as_str())
                .map(|s| s.to_string());
            deps.push(ResolvedDependency {
                name: name.to_string(),
                path: PathBuf::from(path_value),
                version,
                package,
            });
        }
    }
    deps.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(deps)
}

fn compute_build_hash(root: &Path, deps: &[ResolvedDependency]) -> Result<String, String> {
    let mut files = collect_build_inputs(root, deps)?;
    files.sort_by(|a, b| a.to_string_lossy().cmp(&b.to_string_lossy()));
    files.dedup();
    let mut hasher = Sha256::new();
    for path in files {
        let name = path.to_string_lossy();
        hasher.update(name.as_bytes());
        hasher.update([0u8]);
        let data =
            fs::read(&path).map_err(|err| format!("Failed to read {}: {}", path.display(), err))?;
        hasher.update(&data);
    }
    let digest = hasher.finalize();
    Ok(format!("{:x}", digest))
}

fn collect_build_inputs(root: &Path, deps: &[ResolvedDependency]) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    let src_dir = if root.join("src").is_dir() {
        root.join("src")
    } else {
        root.to_path_buf()
    };
    collect_source_files_in_dir(&src_dir, &mut files)?;
    let manifest = root.join("enkai.toml");
    if manifest.is_file() {
        files.push(manifest);
    }
    let lock = root.join("enkai.lock");
    if lock.is_file() {
        files.push(lock);
    }
    for dep in deps {
        let dep_root = if dep.path.is_absolute() {
            dep.path.clone()
        } else {
            root.join(&dep.path)
        };
        let dep_src = if dep_root.join("src").is_dir() {
            dep_root.join("src")
        } else {
            dep_root.clone()
        };
        if dep_src.is_dir() {
            collect_source_files_in_dir(&dep_src, &mut files)?;
        }
        let dep_manifest = dep_root.join("enkai.toml");
        if dep_manifest.is_file() {
            files.push(dep_manifest);
        }
    }
    Ok(files)
}

fn toml_escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('\"', "\\\"")
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

    #[test]
    fn serve_requires_flag_values() {
        let code = serve_command(&["--host".to_string()]);
        assert_ne!(code, 0);
        let code = serve_command(&["--port".to_string()]);
        assert_ne!(code, 0);
        let code = serve_command(&["--registry".to_string()]);
        assert_ne!(code, 0);
        let code = serve_command(&["--model".to_string()]);
        assert_ne!(code, 0);
        let code = serve_command(&["--checkpoint".to_string()]);
        assert_ne!(code, 0);
    }

    #[test]
    fn serve_rejects_conflicting_model_flags() {
        let code = serve_command(&[
            "--checkpoint".to_string(),
            ".".to_string(),
            "--registry".to_string(),
            ".".to_string(),
            "--model".to_string(),
            "demo".to_string(),
        ]);
        assert_ne!(code, 0);
    }

    #[test]
    fn resolve_model_selection_checkpoint_path() {
        let dir = tempdir().expect("tempdir");
        let ckpt = dir.path().join("checkpoint");
        fs::create_dir_all(&ckpt).expect("checkpoint dir");
        let selected = resolve_serve_model_selection(
            Some(ckpt.to_string_lossy().as_ref()),
            None,
            None,
            None,
            false,
        )
        .expect("selection")
        .expect("some");
        assert!(selected.checkpoint_path.ends_with("checkpoint"));
        assert!(selected.model_name.is_none());
    }

    #[test]
    fn resolve_model_selection_latest_semver() {
        let dir = tempdir().expect("tempdir");
        let model_root = dir.path().join("registry").join("chat");
        fs::create_dir_all(model_root.join("v1.2.0")).expect("v1");
        fs::create_dir_all(model_root.join("v1.10.0")).expect("v2");
        fs::create_dir_all(model_root.join("dev")).expect("dev");
        let selected = resolve_serve_model_selection(
            None,
            Some(dir.path().join("registry").to_string_lossy().as_ref()),
            Some("chat"),
            None,
            true,
        )
        .expect("selection")
        .expect("some");
        assert_eq!(selected.model_version.as_deref(), Some("v1.10.0"));
        assert!(selected
            .checkpoint_path
            .ends_with(std::path::Path::new("chat").join("v1.10.0")));
    }
}
