use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::{Mutex, OnceLock};

use enkai_compiler::bytecode::Program;
use enkai_compiler::compiler::CompileError;
use enkai_compiler::formatter::{check_format, format_source};
use enkai_compiler::TypeError;

mod bench;
mod bootstrap;
mod cluster;
mod cluster_runtime;
mod cluster_sim_runtime;
mod deploy;
mod deploy_runtime;
mod frontend;
mod grpc;
mod grpc_runtime;
mod install_diag;
mod migrate;
mod model;
mod model_runtime;
mod program_runtime;
mod project_entrypoints;
mod queue_backend;
mod queue_runtime;
mod readiness;
mod runtime_exec;
mod service_runtime;
mod sim;
mod systems;
mod train;
mod train_runtime;
mod validate;
mod worker;
mod worker_handler_runtime;

pub(crate) fn env_guard() -> std::sync::MutexGuard<'static, ()> {
    static ENV_GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    ENV_GUARD
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|err| err.into_inner())
}

#[cfg(test)]
#[allow(dead_code)]
type ServeModelSelection = systems::ResolvedServeModelSelection;

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
        "cluster" => cluster::cluster_command(&args[2..]),
        "bench" => bench::bench_command(&args[2..]),
        "model" => model::model_command(&args[2..]),
        "new" => frontend::new_command(&args[2..]),
        "sdk" => frontend::sdk_command(&args[2..]),
        "readiness" => readiness::readiness_command(&args[2..]),
        "deploy" => deploy::deploy_command(&args[2..]),
        "grpc" => grpc::grpc_command(&args[2..]),
        "install-diagnostics" => install_diag::install_diagnostics_command(&args[2..]),
        "sim" => sim::sim_command(&args[2..]),
        "systems" => systems::systems_command(&args[2..]),
        "validate" => validate::validate_command(&args[2..]),
        "worker" => worker::worker_command(&args[2..]),
        "fmt-lite" => bootstrap::fmt_lite_command(&args[2..]),
        "lint-lite" => bootstrap::lint_lite_command(&args[2..]),
        "tokenizer-lite" => bootstrap::tokenizer_lite_command(&args[2..]),
        "dataset-lite" => bootstrap::dataset_lite_command(&args[2..]),
        "litec" => bootstrap::litec_command(&args[2..]),
        "check" => check_command(&args[2..]),
        "fmt" => fmt_command(&args[2..]),
        "build" => build_command(&args[2..]),
        "test" => test_command(&args[2..]),
        "train" => train_command(&args[2..]),
        "train-manifest" => train_manifest_command(&args[2..]),
        "train-exec" => train_exec_command(&args[2..]),
        "pretrain" => pretrain_command(&args[2..]),
        "eval" => eval_command(&args[2..]),
        "migrate" => migrate::migrate_command(&args[2..]),
        "doctor" => migrate::doctor_command(&args[2..]),
        _ => {
            print_usage();
            1
        }
    };
    process::exit(exit_code);
}

pub(crate) fn run_command(args: &[String]) -> i32 {
    if args.is_empty() {
        eprintln!("enkai run requires a file or directory");
        return 1;
    }
    let mut trace_vm = false;
    let mut disasm = false;
    let mut trace_task = false;
    let mut trace_net = false;
    let mut runtime_backend = runtime_exec::RuntimeBackendMode::Auto;
    let mut file_arg: Option<String> = None;
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "--trace-vm" => trace_vm = true,
            "--disasm" => disasm = true,
            "--trace-task" => trace_task = true,
            "--trace-net" => trace_net = true,
            "--runtime-backend" => {
                index += 1;
                let Some(value) = args.get(index) else {
                    eprintln!("enkai run --runtime-backend requires a value");
                    return 1;
                };
                match value.as_str() {
                    "auto" => runtime_backend = runtime_exec::RuntimeBackendMode::Auto,
                    "selfhost" => runtime_backend = runtime_exec::RuntimeBackendMode::Selfhost,
                    "rust" => runtime_backend = runtime_exec::RuntimeBackendMode::Rust,
                    other => {
                        eprintln!(
                            "enkai run --runtime-backend must be one of auto|selfhost|rust, got {}",
                            other
                        );
                        return 1;
                    }
                }
            }
            other => file_arg = Some(other.to_string()),
        }
        index += 1;
    }
    let target = match file_arg {
        Some(t) => PathBuf::from(t),
        None => {
            eprintln!("enkai run requires a file or directory");
            return 1;
        }
    };
    match execute_runtime_target(
        &target,
        runtime_exec::RuntimeExecOptions {
            trace_vm,
            disasm,
            trace_task,
            trace_net,
            runtime_backend,
            report_command: Some("run"),
        },
    ) {
        Ok(exit_code) => exit_code,
        Err(err) => {
            eprintln!("{}", err);
            1
        }
    }
}

pub(crate) fn emit_command_backend_report(
    command: &str,
    entry: &Path,
    root: &Path,
    backend: bootstrap::SelfhostRunBackend,
) {
    let env_key = match command {
        "run" => "ENKAI_RUN_BACKEND_REPORT",
        "check" => "ENKAI_CHECK_BACKEND_REPORT",
        "build" => "ENKAI_BUILD_BACKEND_REPORT",
        _ => return,
    };
    let report = serde_json::json!({
        "command": command,
        "entry": entry.to_string_lossy(),
        "root": root.to_string_lossy(),
        "backend": backend.as_str(),
    });
    if let Err(err) = write_json_report_to_env_path(env_key, &report) {
        eprintln!("[{}] failed to write backend report: {}", command, err);
    }
}

pub(crate) fn write_json_report_to_env_path(
    env_key: &str,
    payload: &serde_json::Value,
) -> Result<(), String> {
    let Some(path) = env::var_os(env_key) else {
        return Ok(());
    };
    let path_buf = PathBuf::from(path);
    if let Some(parent) = path_buf.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|err| {
                format!(
                    "failed to create report directory {}: {}",
                    parent.display(),
                    err
                )
            })?;
        }
    }
    let json = serde_json::to_string_pretty(payload)
        .map_err(|err| format!("failed to serialize backend report: {}", err))?;
    fs::write(&path_buf, json).map_err(|err| {
        format!(
            "failed to write backend report {}: {}",
            path_buf.display(),
            err
        )
    })
}

fn serve_command(args: &[String]) -> i32 {
    let manifest = match systems::build_serve_runtime_manifest(args) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{}", err);
            return 1;
        }
    };
    systems::execute_serve_runtime_manifest(&manifest)
}

#[cfg(test)]
#[allow(dead_code)]
fn resolve_serve_model_selection(
    checkpoint: Option<&str>,
    registry: Option<&str>,
    model: Option<&str>,
    model_version: Option<&str>,
    latest: bool,
    require_loaded: bool,
) -> Result<Option<ServeModelSelection>, String> {
    systems::resolve_serve_model_selection(
        checkpoint,
        registry,
        model,
        model_version,
        latest,
        require_loaded,
    )
}

pub(crate) fn execute_runtime_target(
    target: &Path,
    options: runtime_exec::RuntimeExecOptions,
) -> Result<i32, String> {
    runtime_exec::execute_runtime_target(target, options)
}

fn check_command(args: &[String]) -> i32 {
    project_entrypoints::check_command(args)
}

fn test_command(args: &[String]) -> i32 {
    project_entrypoints::test_command(args)
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
    project_entrypoints::build_command(args)
}

pub(crate) fn compile_program_prefer_selfhost(
    entry: &Path,
) -> Result<(Program, bootstrap::SelfhostRunBackend), String> {
    runtime_exec::compile_program_prefer_selfhost(entry)
}

pub(crate) fn type_error_message(err: TypeError) -> String {
    if let Some(diagnostic) = err.diagnostic() {
        diagnostic.to_string()
    } else {
        format!(
            "Type error: {} at {}:{}",
            err.message, err.span.line, err.span.col
        )
    }
}

pub(crate) fn compile_error_message(err: CompileError) -> String {
    if let Some(diagnostic) = err.diagnostic() {
        diagnostic.to_string()
    } else if let Some(span) = &err.span {
        format!(
            "Compile error: {} at {}:{}",
            err.message, span.line, span.col
        )
    } else {
        format!("Compile error: {}", err.message)
    }
}

fn train_command(args: &[String]) -> i32 {
    let (path, strict_contracts) = match parse_train_eval_args("train", args) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("{}", err);
            return 1;
        }
    };
    let manifest = match train::build_train_command_manifest(
        train::TrainManifestCommand::Train,
        &path,
        strict_contracts,
    ) {
        Ok(manifest) => manifest,
        Err(err) => {
            eprintln!("Train error: {}", err);
            return 1;
        }
    };
    let result = train_runtime::execute_train_command_manifest(&manifest);
    match result {
        Ok(_) => 0,
        Err(err) => {
            eprintln!("Train error: {}", err);
            1
        }
    }
}

fn pretrain_command(args: &[String]) -> i32 {
    let (path, strict_contracts) = match parse_train_eval_args("pretrain", args) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("{}", err);
            return 1;
        }
    };
    let manifest = match train::build_train_command_manifest(
        train::TrainManifestCommand::Pretrain,
        &path,
        strict_contracts,
    ) {
        Ok(manifest) => manifest,
        Err(err) => {
            eprintln!("Pretrain error: {}", err);
            return 1;
        }
    };
    let result = train_runtime::execute_train_command_manifest(&manifest);
    match result {
        Ok(_) => 0,
        Err(err) => {
            eprintln!("Pretrain error: {}", err);
            1
        }
    }
}

fn eval_command(args: &[String]) -> i32 {
    let (path, strict_contracts) = match parse_train_eval_args("eval", args) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("{}", err);
            return 1;
        }
    };
    let manifest = match train::build_train_command_manifest(
        train::TrainManifestCommand::Eval,
        &path,
        strict_contracts,
    ) {
        Ok(manifest) => manifest,
        Err(err) => {
            eprintln!("Eval error: {}", err);
            return 1;
        }
    };
    let result = train_runtime::execute_train_command_manifest(&manifest);
    match result {
        Ok(_) => 0,
        Err(err) => {
            eprintln!("Eval error: {}", err);
            1
        }
    }
}

fn train_manifest_command(args: &[String]) -> i32 {
    let Some(mode) = args.first().map(|value| value.as_str()) else {
        eprintln!("enkai train-manifest: missing mode (train|pretrain|eval)");
        return 1;
    };
    let command = match mode {
        "train" => train::TrainManifestCommand::Train,
        "pretrain" => train::TrainManifestCommand::Pretrain,
        "eval" => train::TrainManifestCommand::Eval,
        other => {
            eprintln!(
                "enkai train-manifest: unknown mode '{}'; expected train|pretrain|eval",
                other
            );
            return 1;
        }
    };
    let (sub_args, manifest_output) = match parse_train_manifest_args(&args[1..]) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("enkai train-manifest: {}", err);
            return 1;
        }
    };
    let (path, strict_contracts) = match parse_train_eval_args(mode, &sub_args) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("{}", err);
            return 1;
        }
    };
    let manifest = match train::build_train_command_manifest(command, &path, strict_contracts) {
        Ok(manifest) => manifest,
        Err(err) => {
            eprintln!("enkai train-manifest: {}", err);
            return 1;
        }
    };
    emit_train_manifest(&manifest, manifest_output.as_deref())
}

fn train_exec_command(args: &[String]) -> i32 {
    let manifest_path = match parse_train_exec_manifest_flag(args) {
        Ok(path) => path,
        Err(err) => {
            eprintln!("enkai train-exec: {}", err);
            return 1;
        }
    };
    let manifest = match train_runtime::load_train_command_manifest(&manifest_path) {
        Ok(manifest) => manifest,
        Err(err) => {
            eprintln!("enkai train-exec: {}", err);
            return 1;
        }
    };
    match train_runtime::execute_train_command_manifest(&manifest) {
        Ok(()) => 0,
        Err(err) => {
            eprintln!("Train exec error: {}", err);
            1
        }
    }
}

fn parse_train_manifest_args(args: &[String]) -> Result<(Vec<String>, Option<PathBuf>), String> {
    let mut sub_args = Vec::new();
    let mut manifest_output = None;
    let mut idx = 0usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--manifest-output" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--manifest-output requires a value".to_string());
                }
                manifest_output = Some(PathBuf::from(&args[idx]));
            }
            other => sub_args.push(other.to_string()),
        }
        idx += 1;
    }
    Ok((sub_args, manifest_output))
}

fn emit_train_manifest(manifest: &train::TrainCommandManifest, output: Option<&Path>) -> i32 {
    let text = match serde_json::to_string_pretty(manifest) {
        Ok(text) => text,
        Err(err) => {
            eprintln!(
                "enkai train-manifest: failed to serialize manifest: {}",
                err
            );
            return 1;
        }
    };
    if let Some(path) = output {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                if let Err(err) = fs::create_dir_all(parent) {
                    eprintln!(
                        "enkai train-manifest: failed to create output directory {}: {}",
                        parent.display(),
                        err
                    );
                    return 1;
                }
            }
        }
        if let Err(err) = fs::write(path, text.as_bytes()) {
            eprintln!(
                "enkai train-manifest: failed to write manifest {}: {}",
                path.display(),
                err
            );
            return 1;
        }
    } else {
        println!("{}", text);
    }
    0
}

fn parse_train_exec_manifest_flag(args: &[String]) -> Result<PathBuf, String> {
    let mut manifest_path = None;
    let mut idx = 0usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--manifest" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--manifest requires a value".to_string());
                }
                manifest_path = Some(PathBuf::from(&args[idx]));
            }
            other => return Err(format!("unknown option '{}'", other)),
        }
        idx += 1;
    }
    manifest_path.ok_or_else(|| "--manifest is required".to_string())
}

fn parse_train_eval_args(command: &str, args: &[String]) -> Result<(PathBuf, bool), String> {
    let mut path: Option<PathBuf> = None;
    let mut strict_contracts = true;
    let allow_legacy = allow_legacy_contracts_from_env();
    for arg in args {
        match arg.as_str() {
            "--strict-contracts" => strict_contracts = true,
            "--lenient-contracts" => {
                if !allow_legacy {
                    return Err(format!(
                        "{}: --lenient-contracts is disabled in v{}. Migrate with `enkai migrate config-v1` / `enkai migrate checkpoint-meta-v1`, or set ENKAI_ALLOW_LEGACY_CONTRACTS=1 for temporary recovery.",
                        command,
                        language_version()
                    ));
                }
                strict_contracts = false;
            }
            _ if arg.starts_with('-') => {
                return Err(format!(
                    "Unknown {} option: {} (supported: --strict-contracts, --lenient-contracts)",
                    command, arg
                ))
            }
            _ => {
                if path.is_some() {
                    return Err(format!(
                        "enkai {} accepts exactly one config path (plus optional flags)",
                        command
                    ));
                }
                path = Some(PathBuf::from(arg));
            }
        }
    }
    let Some(path) = path else {
        return Err(format!(
            "enkai {} requires a config file (usage: enkai {} <config> [--strict-contracts|--lenient-contracts])",
            command, command
        ));
    };
    Ok((path, strict_contracts))
}

fn allow_legacy_contracts_from_env() -> bool {
    match env::var("ENKAI_ALLOW_LEGACY_CONTRACTS") {
        Ok(value) => {
            let normalized = value.trim().to_ascii_lowercase();
            matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
        }
        Err(_) => false,
    }
}

pub(crate) fn resolve_entry(target: &Path) -> Result<(PathBuf, PathBuf), String> {
    project_entrypoints::resolve_entry(target)
}

pub(crate) fn find_project_root(start: &Path) -> Option<PathBuf> {
    project_entrypoints::find_project_root(start)
}

pub(crate) fn collect_source_files(path: &Path) -> Result<Vec<PathBuf>, String> {
    project_entrypoints::collect_source_files(path)
}

fn normalize_line_endings(input: &str) -> String {
    input.replace("\r\n", "\n")
}

fn print_usage() {
    eprintln!("Enkai CLI");
    eprintln!("Usage:");
    eprintln!("  enkai --version");
    eprintln!(
        "  enkai run [--runtime-backend <auto|selfhost|rust>] [--trace-vm] [--disasm] [--trace-task] [--trace-net] <file|dir>"
    );
    eprintln!(
        "  enkai serve [--host <host>] [--port <port>] [--grpc-host <host>] [--grpc-port <port>] [--registry <dir> --model <name> [--model-version <v>|--latest] [--require-loaded] | --multi-model --registry <dir> | --checkpoint <path>] [--trace-vm] [--disasm] [--trace-task] [--trace-net] [file|dir]"
    );
    bench::print_bench_usage();
    model::print_model_usage();
    frontend::print_new_usage();
    frontend::print_sdk_usage();
    readiness::print_readiness_usage();
    eprintln!("  enkai cluster <validate|plan|run> <config.enk> [--json] [--dry-run]");
    deploy::print_deploy_usage();
    grpc::print_grpc_usage();
    install_diag::print_install_diagnostics_usage();
    sim::print_sim_usage();
    systems::print_systems_usage();
    validate::print_validate_usage();
    worker::print_worker_usage();
    bootstrap::print_usage();
    eprintln!("  enkai check <file|dir>");
    eprintln!("  enkai fmt [--check] <file|dir>");
    eprintln!("  enkai build [dir]");
    eprintln!("  enkai test [dir]");
    eprintln!("  enkai train <config.enk> [--strict-contracts|--lenient-contracts]");
    eprintln!("  enkai train-manifest <train|pretrain|eval> <config.enk> [--strict-contracts|--lenient-contracts] [--manifest-output <file>]");
    eprintln!("  enkai train-exec --manifest <file>");
    eprintln!("  enkai pretrain <config.enk> [--strict-contracts|--lenient-contracts]");
    eprintln!("  enkai eval <config.enk> [--strict-contracts|--lenient-contracts]");
    migrate::print_usage();
}

fn print_version() {
    println!("{}", format_version_string());
}

fn language_version() -> &'static str {
    env!("ENKAI_LANG_VERSION")
}

fn cli_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

fn format_version_string() -> String {
    format!("Enkai v{} (cli {})", language_version(), cli_version())
}

pub(crate) fn load_cached_program(root: &Path, entry: &Path) -> Result<Option<Program>, String> {
    project_entrypoints::load_cached_program(root, entry)
}

#[cfg(all(test, not(windows)))]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;

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
        let code = run_command(&[file.to_string_lossy().to_string()]);
        assert_ne!(code, 0);
    }

    #[test]
    fn run_prefers_selfhost_backend_for_supported_programs() {
        let _guard = env_guard();
        let dir = tempdir().expect("tempdir");
        let file = dir.path().join("ok.enk");
        let report = dir.path().join("run_backend.json");
        fs::write(&file, "fn main() -> Int ::\n    return 0\n::\n").expect("program");
        env::set_var("ENKAI_RUN_BACKEND_REPORT", &report);
        let code = run_command(&[file.to_string_lossy().to_string()]);
        env::remove_var("ENKAI_RUN_BACKEND_REPORT");
        assert_eq!(code, 0);
        let payload: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&report).expect("report")).expect("json");
        assert_eq!(payload.get("backend"), Some(&serde_json::json!("selfhost")));
    }

    #[test]
    fn run_rust_backend_flag_skips_selfhost_path() {
        let _guard = env_guard();
        let dir = tempdir().expect("tempdir");
        let file = dir.path().join("ok.enk");
        let report = dir.path().join("run_backend_rust.json");
        fs::write(&file, "fn main() -> Int ::\n    return 0\n::\n").expect("program");
        env::set_var("ENKAI_RUN_BACKEND_REPORT", &report);
        let code = run_command(&[
            "--runtime-backend".to_string(),
            "rust".to_string(),
            file.to_string_lossy().to_string(),
        ]);
        env::remove_var("ENKAI_RUN_BACKEND_REPORT");
        assert_eq!(code, 0);
        let payload: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&report).expect("report")).expect("json");
        assert_eq!(payload.get("backend"), Some(&serde_json::json!("rust")));
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
        let code = test_command(&[dir.path().to_string_lossy().to_string()]);
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
        let code = test_command(&[dir.path().to_string_lossy().to_string()]);
        assert_ne!(code, 0);
    }

    #[test]
    fn fmt_check_fails_on_unformatted_source() {
        let dir = tempdir().expect("tempdir");
        let file = dir.path().join("bad.enk");
        fs::write(&file, "if true ::\nprint(\"hi\")\n::\n").unwrap();
        let code = fmt_command(&["--check".to_string(), file.to_string_lossy().to_string()]);
        assert_ne!(code, 0);
    }

    #[test]
    fn fmt_formats_file() {
        let dir = tempdir().expect("tempdir");
        let file = dir.path().join("fix.enk");
        fs::write(&file, "if true ::\nprint(\"hi\")\n::\n").unwrap();
        let code = fmt_command(&[file.to_string_lossy().to_string()]);
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
    fn serve_rejects_multi_model_without_registry() {
        let code = serve_command(&["--multi-model".to_string()]);
        assert_ne!(code, 0);
    }

    #[test]
    fn resolve_model_selection_requires_loaded_when_requested() {
        let dir = tempdir().expect("tempdir");
        let registry = dir.path().join("registry");
        let checkpoint = dir.path().join("checkpoint");
        fs::create_dir_all(&checkpoint).expect("checkpoint");
        let register = super::model::model_command(&[
            "register".to_string(),
            registry.to_string_lossy().to_string(),
            "chat".to_string(),
            "v1.0.0".to_string(),
            checkpoint.to_string_lossy().to_string(),
            "--activate".to_string(),
        ]);
        assert_eq!(register, 0);

        let err = resolve_serve_model_selection(
            None,
            Some(registry.to_string_lossy().as_ref()),
            Some("chat"),
            Some("v1.0.0"),
            false,
            true,
        )
        .expect_err("must reject unloaded model");
        assert!(err.contains("not loaded for serving"));

        let load = super::model::model_command(&[
            "load".to_string(),
            registry.to_string_lossy().to_string(),
            "chat".to_string(),
            "v1.0.0".to_string(),
        ]);
        assert_eq!(load, 0);

        let selected = resolve_serve_model_selection(
            None,
            Some(registry.to_string_lossy().as_ref()),
            Some("chat"),
            Some("v1.0.0"),
            false,
            true,
        )
        .expect("selection")
        .expect("some");
        assert_eq!(selected.model_version.as_deref(), Some("v1.0.0"));
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
            false,
        )
        .expect("selection")
        .expect("some");
        assert_eq!(selected.model_version.as_deref(), Some("v1.10.0"));
        assert!(selected
            .checkpoint_path
            .ends_with(std::path::Path::new("chat").join("v1.10.0")));
    }

    #[test]
    fn resolve_model_selection_prefers_active_pointer_without_latest_flag() {
        let dir = tempdir().expect("tempdir");
        let model_root = dir.path().join("registry").join("chat");
        fs::create_dir_all(model_root.join("v1.0.0")).expect("v1");
        fs::create_dir_all(model_root.join("v1.1.0")).expect("v2");
        fs::write(model_root.join(".active_version"), "v1.0.0\n").expect("active pointer");
        let selected = resolve_serve_model_selection(
            None,
            Some(dir.path().join("registry").to_string_lossy().as_ref()),
            Some("chat"),
            None,
            false,
            false,
        )
        .expect("selection")
        .expect("some");
        assert_eq!(selected.model_version.as_deref(), Some("v1.0.0"));
    }

    #[test]
    fn resolve_model_selection_honors_checkpoint_pointer() {
        let dir = tempdir().expect("tempdir");
        let registry = dir.path().join("registry");
        let model_root = registry.join("chat");
        let version_tag = format!("v{}", env!("CARGO_PKG_VERSION"));
        let version_dir = model_root.join(&version_tag);
        fs::create_dir_all(&version_dir).expect("version dir");
        fs::write(model_root.join(".active_version"), &version_tag).expect("active");
        let checkpoint = dir.path().join("external_ckpt");
        fs::create_dir_all(&checkpoint).expect("checkpoint dir");
        fs::write(
            version_dir.join("checkpoint_path.txt"),
            checkpoint.to_string_lossy().to_string(),
        )
        .expect("pointer");
        let selected = resolve_serve_model_selection(
            None,
            Some(registry.to_string_lossy().as_ref()),
            Some("chat"),
            None,
            false,
            false,
        )
        .expect("selection")
        .expect("some");
        assert_eq!(
            selected.checkpoint_path,
            fs::canonicalize(checkpoint).expect("canon")
        );
    }

    #[test]
    fn version_output_matches_language_and_cli_versions() {
        let expected = format!(
            "Enkai v{} (cli {})",
            language_version(),
            env!("CARGO_PKG_VERSION")
        );
        assert_eq!(format_version_string(), expected);
    }

    #[test]
    fn parse_train_eval_args_accepts_strict_flag() {
        let (path, strict) = parse_train_eval_args(
            "train",
            &["cfg.enk".to_string(), "--strict-contracts".to_string()],
        )
        .expect("parse");
        assert_eq!(path, PathBuf::from("cfg.enk"));
        assert!(strict);
    }

    #[test]
    fn parse_train_eval_args_defaults_to_strict() {
        let parsed = parse_train_eval_args("eval", &["cfg.enk".to_string()]).expect("parse");
        assert!(parsed.1);
    }

    #[test]
    fn parse_train_eval_args_supports_pretrain_command_name() {
        let parsed = parse_train_eval_args("pretrain", &["cfg.enk".to_string()]).expect("parse");
        assert_eq!(parsed.0, PathBuf::from("cfg.enk"));
        assert!(parsed.1);
    }

    #[test]
    fn parse_train_eval_args_rejects_lenient_without_gate() {
        let _guard = env_guard();
        let prev = env::var("ENKAI_ALLOW_LEGACY_CONTRACTS").ok();
        env::remove_var("ENKAI_ALLOW_LEGACY_CONTRACTS");
        let err = parse_train_eval_args(
            "train",
            &["cfg.enk".to_string(), "--lenient-contracts".to_string()],
        )
        .expect_err("must reject");
        assert!(err.contains("ENKAI_ALLOW_LEGACY_CONTRACTS=1"));
        if let Some(value) = prev {
            env::set_var("ENKAI_ALLOW_LEGACY_CONTRACTS", value);
        } else {
            env::remove_var("ENKAI_ALLOW_LEGACY_CONTRACTS");
        }
    }

    #[test]
    fn parse_train_eval_args_allows_lenient_with_gate() {
        let _guard = env_guard();
        let prev = env::var("ENKAI_ALLOW_LEGACY_CONTRACTS").ok();
        env::set_var("ENKAI_ALLOW_LEGACY_CONTRACTS", "1");
        let parsed = parse_train_eval_args(
            "eval",
            &["cfg.enk".to_string(), "--lenient-contracts".to_string()],
        )
        .expect("parse");
        if let Some(value) = prev {
            env::set_var("ENKAI_ALLOW_LEGACY_CONTRACTS", value);
        } else {
            env::remove_var("ENKAI_ALLOW_LEGACY_CONTRACTS");
        }
        assert!(!parsed.1);
    }

    #[test]
    fn master_pipeline_cpu_smoke() {
        let dir = tempdir().expect("tempdir");
        let data = dir.path().join("data.txt");
        fs::write(&data, "alpha beta\ngamma delta\n").expect("dataset");
        let ckpt = dir.path().join("ckpt");
        let config_path = dir.path().join("config.enk");
        let json = serde_json::json!({
            "config_version": 1,
            "backend": "cpu",
            "vocab_size": 8,
            "hidden_size": 4,
            "seq_len": 4,
            "batch_size": 2,
            "lr": 0.1,
            "dataset_path": data.to_string_lossy(),
            "checkpoint_dir": ckpt.to_string_lossy(),
            "max_steps": 1,
            "save_every": 1,
            "log_every": 1,
            "eval_steps": 1,
            "drop_remainder": false,
            "tokenizer_train": { "path": data.to_string_lossy(), "vocab_size": 8 }
        });
        let escaped = json.to_string().replace('\\', "\\\\").replace('\"', "\\\"");
        let source = format!("fn main() ::\n    return json.parse(\"{}\")\n::\n", escaped);
        fs::write(&config_path, source).expect("config");
        super::train::train(&config_path).expect("train");
        super::train::eval(&config_path).expect("eval");

        let frontend_out = dir.path().join("frontend");
        let frontend_code = super::frontend::new_command(&[
            "frontend-chat".to_string(),
            frontend_out.to_string_lossy().to_string(),
        ]);
        assert_eq!(frontend_code, 0);
        assert!(frontend_out.join("src").join("App.tsx").is_file());

        let corpus = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tools")
            .join("bootstrap")
            .join("selfhost_corpus");
        let selfhost_code = super::bootstrap::litec_command(&[
            "selfhost-ci".to_string(),
            corpus.to_string_lossy().to_string(),
            "--no-compare-stage0".to_string(),
        ]);
        assert_eq!(selfhost_code, 0);
    }
}
