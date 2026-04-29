use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use semver::Version;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::{cluster, deploy, model, worker};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct ResolvedServeModelSelection {
    pub(crate) checkpoint_path: PathBuf,
    pub(crate) model_name: Option<String>,
    pub(crate) model_version: Option<String>,
    pub(crate) registry: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct ServiceBindingManifest {
    pub(crate) host: Option<String>,
    pub(crate) port: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub(crate) enum ServeModelManifest {
    None,
    Single {
        checkpoint_path: String,
        model_name: Option<String>,
        model_version: Option<String>,
        registry: Option<String>,
        require_loaded: bool,
    },
    Multi {
        registry: String,
        require_model_version_header: bool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct ServeRuntimeManifest {
    pub(crate) schema_version: u32,
    pub(crate) profile: String,
    pub(crate) target: String,
    pub(crate) runtime_flags: Vec<String>,
    pub(crate) http: ServiceBindingManifest,
    pub(crate) grpc: ServiceBindingManifest,
    pub(crate) model: ServeModelManifest,
    pub(crate) http_runtime: HttpRuntimeManifest,
    pub(crate) grpc_runtime: Option<GrpcRuntimeManifest>,
    pub(crate) env_projection: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct HttpRuntimeManifest {
    pub(crate) api_version: String,
    pub(crate) conversation_dir: String,
    pub(crate) log_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct GrpcRuntimeManifest {
    pub(crate) api_version: String,
    pub(crate) conversation_dir: String,
    pub(crate) log_path: Option<String>,
}

#[derive(Debug, Clone)]
struct RuntimeDefaults {
    api_version: String,
    conversation_dir: String,
    log_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct WorkerRetryPolicyManifest {
    pub(crate) max_attempts: u32,
    pub(crate) delay_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct WorkerRunPolicyManifest {
    pub(crate) drain_mode: String,
    pub(crate) max_messages: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub(crate) enum WorkerQueueManifest {
    Enqueue {
        dir: String,
        queue: String,
        backend_kind: String,
        queue_root: String,
        pending_path: String,
        inflight_path: String,
        schedule_path: String,
        dead_letter_path: String,
        state_path: String,
        payload: JsonValue,
        tenant: Option<String>,
        id: Option<String>,
        max_attempts: u32,
        retry_policy: WorkerRetryPolicyManifest,
        json: bool,
        output: Option<String>,
    },
    Run {
        dir: String,
        queue: String,
        backend_kind: String,
        queue_root: String,
        pending_path: String,
        inflight_path: String,
        schedule_path: String,
        dead_letter_path: String,
        state_path: String,
        handler: String,
        once: bool,
        run_policy: WorkerRunPolicyManifest,
        json: bool,
        output: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct DeployValidateManifest {
    pub(crate) schema_version: u32,
    pub(crate) profile: String,
    pub(crate) project_dir: String,
    pub(crate) target_profile: String,
    pub(crate) strict: bool,
    pub(crate) json: bool,
    pub(crate) output: Option<String>,
    pub(crate) evaluated_project: JsonValue,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ClusterCommandManifest {
    pub(crate) schema_version: u32,
    pub(crate) profile: String,
    pub(crate) subcommand: String,
    pub(crate) config_path: String,
    pub(crate) json: bool,
    pub(crate) dry_run: bool,
    pub(crate) output: Option<String>,
    pub(crate) evaluated_plan: JsonValue,
}

#[derive(Debug, Clone, Default)]
struct ServeParseState {
    host: Option<String>,
    port: Option<String>,
    grpc_host: Option<String>,
    grpc_port: Option<String>,
    registry: Option<String>,
    model: Option<String>,
    model_version: Option<String>,
    latest: bool,
    checkpoint: Option<String>,
    multi_model: bool,
    require_loaded: bool,
    runtime_flags: Vec<String>,
    target: Option<String>,
}

#[derive(Debug, Clone)]
struct ServeManifestCommandArgs {
    manifest: ServeRuntimeManifest,
    json: bool,
    output: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct WorkerManifestCommandArgs {
    manifest: WorkerQueueManifest,
    json: bool,
    output: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct DeployManifestCommandArgs {
    manifest: DeployValidateManifest,
    json: bool,
    output: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct ClusterManifestCommandArgs {
    manifest: ClusterCommandManifest,
    json: bool,
    output: Option<PathBuf>,
}

pub(crate) fn print_systems_usage() {
    eprintln!(
        "  enkai systems serve-manifest [--host <host>] [--port <port>] [--grpc-host <host>] [--grpc-port <port>] [--registry <dir> --model <name> [--model-version <v>|--latest] [--require-loaded] | --multi-model --registry <dir> | --checkpoint <path>] [--trace-vm] [--disasm] [--trace-task] [--trace-net] [file|dir] [--json] [--output <file>]"
    );
    eprintln!("  enkai systems serve-exec --manifest <file>");
    eprintln!("  enkai systems worker-manifest <enqueue|run> ... [--json] [--output <file>]");
    eprintln!("  enkai systems worker-exec --manifest <file>");
    eprintln!(
        "  enkai systems deploy-manifest validate <project_dir> --profile <backend|fullstack|mobile> --strict [--json] [--output <file>]"
    );
    eprintln!("  enkai systems deploy-exec --manifest <file>");
    eprintln!(
        "  enkai systems cluster-manifest <validate|plan|run> <config.enk> [--json] [--dry-run] [--output <file>]"
    );
    eprintln!("  enkai systems cluster-exec --manifest <file>");
}

pub(crate) fn systems_command(args: &[String]) -> i32 {
    let Some(subcommand) = args.first().map(|value| value.as_str()) else {
        print_systems_usage();
        return 1;
    };
    match subcommand {
        "serve-manifest" => serve_manifest_command(&args[1..]),
        "serve-exec" => serve_exec_command(&args[1..]),
        "worker-manifest" => worker_manifest_command(&args[1..]),
        "worker-exec" => worker_exec_command(&args[1..]),
        "deploy-manifest" => deploy_manifest_command(&args[1..]),
        "deploy-exec" => deploy_exec_command(&args[1..]),
        "cluster-manifest" => cluster_manifest_command(&args[1..]),
        "cluster-exec" => cluster_exec_command(&args[1..]),
        other => {
            eprintln!("enkai systems: unknown subcommand '{}'", other);
            print_systems_usage();
            1
        }
    }
}

fn serve_exec_command(args: &[String]) -> i32 {
    let manifest_path = match parse_manifest_flag("serve-exec", args) {
        Ok(path) => path,
        Err(err) => {
            eprintln!("{}", err);
            print_systems_usage();
            return 1;
        }
    };
    let manifest = match load_json_manifest::<ServeRuntimeManifest>(&manifest_path) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("enkai systems serve-exec: {}", err);
            return 1;
        }
    };
    execute_serve_runtime_manifest(&manifest)
}

fn worker_exec_command(args: &[String]) -> i32 {
    let manifest_path = match parse_manifest_flag("worker-exec", args) {
        Ok(path) => path,
        Err(err) => {
            eprintln!("{}", err);
            print_systems_usage();
            return 1;
        }
    };
    let manifest = match load_json_manifest::<WorkerQueueManifest>(&manifest_path) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("enkai systems worker-exec: {}", err);
            return 1;
        }
    };
    worker::execute_worker_manifest(&manifest)
}

fn deploy_exec_command(args: &[String]) -> i32 {
    let manifest_path = match parse_manifest_flag("deploy-exec", args) {
        Ok(path) => path,
        Err(err) => {
            eprintln!("{}", err);
            print_systems_usage();
            return 1;
        }
    };
    let manifest = match load_json_manifest::<DeployValidateManifest>(&manifest_path) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("enkai systems deploy-exec: {}", err);
            return 1;
        }
    };
    deploy::execute_deploy_manifest(&manifest)
}

fn cluster_exec_command(args: &[String]) -> i32 {
    let manifest_path = match parse_manifest_flag("cluster-exec", args) {
        Ok(path) => path,
        Err(err) => {
            eprintln!("{}", err);
            print_systems_usage();
            return 1;
        }
    };
    let manifest = match load_json_manifest::<ClusterCommandManifest>(&manifest_path) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("enkai systems cluster-exec: {}", err);
            return 1;
        }
    };
    cluster::execute_cluster_manifest(&manifest)
}

pub(crate) fn build_serve_runtime_manifest(
    args: &[String],
) -> Result<ServeRuntimeManifest, String> {
    let state = parse_serve_state(args)?;
    serve_manifest_from_state(state)
}

pub(crate) fn apply_serve_runtime_manifest_env(manifest: &ServeRuntimeManifest) {
    for key in [
        "ENKAI_SERVE_HOST",
        "ENKAI_SERVE_PORT",
        "ENKAI_GRPC_HOST",
        "ENKAI_GRPC_PORT",
        "ENKAI_API_VERSION",
        "ENKAI_CONVERSATION_DIR",
        "ENKAI_LOG_PATH",
        "ENKAI_SERVE_MULTI_MODEL",
        "ENKAI_REQUIRE_MODEL_VERSION_HEADER",
        "ENKAI_SERVE_MODEL_REGISTRY",
        "ENKAI_SERVE_MODEL_PATH",
        "ENKAI_SERVE_MODEL_NAME",
        "ENKAI_SERVE_MODEL_VERSION",
    ] {
        env::remove_var(key);
    }
    for (key, value) in &manifest.env_projection {
        env::set_var(key, value);
    }
}

pub(crate) fn build_worker_queue_manifest(args: &[String]) -> Result<WorkerQueueManifest, String> {
    let Some(subcommand) = args.first().map(|value| value.as_str()) else {
        return Err("missing subcommand (enqueue|run)".to_string());
    };
    match subcommand {
        "enqueue" => build_worker_enqueue_manifest(&args[1..]),
        "run" => build_worker_run_manifest(&args[1..]),
        other => Err(format!(
            "unknown worker subcommand '{}'; expected enqueue|run",
            other
        )),
    }
}

pub(crate) fn build_deploy_validate_manifest(
    args: &[String],
) -> Result<DeployValidateManifest, String> {
    let Some(subcommand) = args.first().map(|value| value.as_str()) else {
        return Err("missing subcommand".to_string());
    };
    if subcommand != "validate" {
        return Err(format!(
            "unknown deploy subcommand '{}'; expected validate",
            subcommand
        ));
    }
    parse_deploy_validate_manifest(&args[1..])
}

pub(crate) fn build_cluster_command_manifest(
    args: &[String],
) -> Result<ClusterCommandManifest, String> {
    let Some(subcommand) = args.first().map(|value| value.as_str()) else {
        return Err("missing subcommand".to_string());
    };
    if subcommand != "validate" && subcommand != "plan" && subcommand != "run" {
        return Err(format!(
            "unknown cluster subcommand '{}'; expected validate|plan|run",
            subcommand
        ));
    }
    parse_cluster_manifest(subcommand, &args[1..])
}

pub(crate) fn execute_serve_runtime_manifest(manifest: &ServeRuntimeManifest) -> i32 {
    let env_guard = match serve_env_lock().lock() {
        Ok(value) => value,
        Err(_) => {
            eprintln!("enkai serve: serve env lock poisoned");
            return 1;
        }
    };
    let previous_env = capture_manifest_env(&manifest.env_projection);
    apply_serve_runtime_manifest_env(manifest);
    let grpc_handle = match crate::grpc_runtime::start_from_serve_manifest(manifest) {
        Ok(handle) => handle,
        Err(err) => {
            restore_manifest_env(previous_env);
            drop(env_guard);
            eprintln!("enkai serve: {}", err);
            return 1;
        }
    };
    let runtime_options = runtime_exec_options_from_manifest(manifest);
    let exit_code = match crate::service_runtime::maybe_execute_backend_contract_service(manifest) {
        Ok(Some(code)) => code,
        Ok(None) => {
            match crate::runtime_exec::execute_runtime_target(
                Path::new(&manifest.target),
                runtime_options,
            ) {
                Ok(code) => code,
                Err(err) => {
                    eprintln!("{}", err);
                    1
                }
            }
        }
        Err(err) => {
            eprintln!("enkai serve: {}", err);
            1
        }
    };
    if let Some(handle) = grpc_handle {
        handle.shutdown();
    }
    restore_manifest_env(previous_env);
    drop(env_guard);
    exit_code
}

fn runtime_exec_options_from_manifest(
    manifest: &ServeRuntimeManifest,
) -> crate::runtime_exec::RuntimeExecOptions {
    let mut options = crate::runtime_exec::RuntimeExecOptions {
        trace_vm: false,
        disasm: false,
        trace_task: false,
        trace_net: false,
        runtime_backend: crate::runtime_exec::RuntimeBackendMode::Auto,
        report_command: Some("run"),
    };
    for flag in &manifest.runtime_flags {
        match flag.as_str() {
            "--trace-vm" => options.trace_vm = true,
            "--disasm" => options.disasm = true,
            "--trace-task" => options.trace_task = true,
            "--trace-net" => options.trace_net = true,
            _ => {}
        }
    }
    options
}

fn default_runtime_defaults(target: &str) -> RuntimeDefaults {
    RuntimeDefaults {
        api_version: env::var("ENKAI_API_VERSION")
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "v1".to_string()),
        conversation_dir: default_conversation_dir_for_target(target)
            .display()
            .to_string(),
        log_path: env::var("ENKAI_LOG_PATH")
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty()),
    }
}

fn default_http_runtime_manifest(target: &str) -> HttpRuntimeManifest {
    let defaults = default_runtime_defaults(target);
    HttpRuntimeManifest {
        api_version: defaults.api_version,
        conversation_dir: defaults.conversation_dir,
        log_path: defaults.log_path,
    }
}

pub(crate) fn default_grpc_runtime_manifest(target: &str) -> GrpcRuntimeManifest {
    let defaults = default_runtime_defaults(target);
    GrpcRuntimeManifest {
        api_version: defaults.api_version,
        conversation_dir: defaults.conversation_dir,
        log_path: defaults.log_path,
    }
}

fn default_conversation_dir_for_target(target: &str) -> PathBuf {
    if let Some(path) = env::var_os("ENKAI_CONVERSATION_DIR") {
        return PathBuf::from(path);
    }
    let target_path = PathBuf::from(target);
    if target_path.is_dir() {
        return target_path;
    }
    target_path
        .parent()
        .map(Path::to_path_buf)
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| PathBuf::from("."))
}

fn serve_env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn capture_manifest_env(
    env_projection: &BTreeMap<String, String>,
) -> Vec<(String, Option<std::ffi::OsString>)> {
    env_projection
        .keys()
        .map(|key| (key.clone(), env::var_os(key)))
        .collect()
}

fn restore_manifest_env(previous_env: Vec<(String, Option<std::ffi::OsString>)>) {
    for (key, value) in previous_env {
        unsafe {
            if let Some(value) = value {
                env::set_var(key, value);
            } else {
                env::remove_var(key);
            }
        }
    }
}

pub(crate) fn resolve_serve_model_selection(
    checkpoint: Option<&str>,
    registry: Option<&str>,
    model_name: Option<&str>,
    model_version: Option<&str>,
    latest: bool,
    require_loaded: bool,
) -> Result<Option<ResolvedServeModelSelection>, String> {
    if checkpoint.is_some()
        && (registry.is_some() || model_name.is_some() || model_version.is_some() || latest)
    {
        return Err(
            "use either --checkpoint or --registry/--model/--model-version/--latest".to_string(),
        );
    }

    if let Some(path) = checkpoint {
        let checkpoint_path = canonical_existing_path(Path::new(path), "--checkpoint")?;
        return Ok(Some(ResolvedServeModelSelection {
            checkpoint_path,
            model_name: None,
            model_version: None,
            registry: None,
        }));
    }

    if registry.is_none() && model_name.is_none() && model_version.is_none() && !latest {
        return Ok(None);
    }

    let registry =
        registry.ok_or_else(|| "missing --registry for model registry resolution".to_string())?;
    let model_name =
        model_name.ok_or_else(|| "missing --model for model registry resolution".to_string())?;
    if latest && model_version.is_some() {
        return Err("use --latest or --model-version, not both".to_string());
    }

    let registry_path = canonical_existing_path(Path::new(registry), "--registry")?;
    let model_root = registry_path.join(model_name);
    if !model_root.is_dir() {
        return Err(format!(
            "model '{}' not found under registry {}",
            model_name,
            registry_path.display()
        ));
    }

    let version = if let Some(version) = model_version {
        version.to_string()
    } else if !latest {
        select_active_or_latest_model_version(&model_root)?
    } else {
        select_latest_model_version(&model_root)?
    };
    let version_dir = model_root.join(&version);
    if !version_dir.exists() {
        return Err(format!(
            "version '{}' for model '{}' does not exist in {}",
            version,
            model_name,
            model_root.display()
        ));
    }
    let checkpoint_path = if version_dir.join("checkpoint").exists() {
        canonical_existing_path(&version_dir.join("checkpoint"), "checkpoint directory")?
    } else if let Some(pointer) = model::resolve_checkpoint_pointer(&version_dir) {
        canonical_existing_path(&pointer, "checkpoint pointer")?
    } else {
        canonical_existing_path(&version_dir, "model version directory")?
    };
    if require_loaded {
        let loaded = model::is_model_loaded(&registry_path, model_name, &version)
            .map_err(|err| format!("failed to read serve load state: {}", err))?;
        if !loaded {
            return Err(format!(
                "model '{}' version '{}' is not loaded for serving (run: enkai model load {} {} {})",
                model_name,
                version,
                registry_path.display(),
                model_name,
                version
            ));
        }
    }
    Ok(Some(ResolvedServeModelSelection {
        checkpoint_path,
        model_name: Some(model_name.to_string()),
        model_version: Some(version),
        registry: Some(registry_path),
    }))
}

fn serve_manifest_command(args: &[String]) -> i32 {
    let parsed = match parse_serve_manifest_command_args(args) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("enkai systems serve-manifest: {}", err);
            print_systems_usage();
            return 1;
        }
    };

    let payload = match serde_json::to_string_pretty(&parsed.manifest) {
        Ok(value) => value,
        Err(err) => {
            eprintln!(
                "enkai systems serve-manifest: failed to serialize manifest: {}",
                err
            );
            return 1;
        }
    };
    if let Some(output) = parsed.output.as_ref() {
        if let Some(parent) = output.parent() {
            if !parent.as_os_str().is_empty() {
                if let Err(err) = fs::create_dir_all(parent) {
                    eprintln!(
                        "enkai systems serve-manifest: failed to create output directory {}: {}",
                        parent.display(),
                        err
                    );
                    return 1;
                }
            }
        }
        if let Err(err) = fs::write(output, &payload) {
            eprintln!(
                "enkai systems serve-manifest: failed to write {}: {}",
                output.display(),
                err
            );
            return 1;
        }
    }
    if parsed.json || parsed.output.is_none() {
        println!("{}", payload);
    }
    0
}

fn worker_manifest_command(args: &[String]) -> i32 {
    let parsed = match parse_worker_manifest_command_args(args) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("enkai systems worker-manifest: {}", err);
            print_systems_usage();
            return 1;
        }
    };
    emit_manifest_payload(
        &parsed.manifest,
        parsed.json,
        parsed.output.as_deref(),
        "worker-manifest",
    )
}

fn deploy_manifest_command(args: &[String]) -> i32 {
    let parsed = match parse_deploy_manifest_command_args(args) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("enkai systems deploy-manifest: {}", err);
            print_systems_usage();
            return 1;
        }
    };
    emit_manifest_payload(
        &parsed.manifest,
        parsed.json,
        parsed.output.as_deref(),
        "deploy-manifest",
    )
}

fn cluster_manifest_command(args: &[String]) -> i32 {
    let parsed = match parse_cluster_manifest_command_args(args) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("enkai systems cluster-manifest: {}", err);
            print_systems_usage();
            return 1;
        }
    };
    emit_manifest_payload(
        &parsed.manifest,
        parsed.json,
        parsed.output.as_deref(),
        "cluster-manifest",
    )
}

fn parse_serve_manifest_command_args(args: &[String]) -> Result<ServeManifestCommandArgs, String> {
    let mut output: Option<PathBuf> = None;
    let mut json = false;
    let mut serve_args = Vec::new();
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "--json" => json = true,
            "--output" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| "--output requires a value".to_string())?;
                output = Some(PathBuf::from(value));
            }
            other => serve_args.push(other.to_string()),
        }
        index += 1;
    }
    let manifest = build_serve_runtime_manifest(&serve_args)?;
    Ok(ServeManifestCommandArgs {
        manifest,
        json,
        output,
    })
}

fn parse_worker_manifest_command_args(
    args: &[String],
) -> Result<WorkerManifestCommandArgs, String> {
    let mut output: Option<PathBuf> = None;
    let mut json = false;
    let mut worker_args = Vec::new();
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "--json" => json = true,
            "--output" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| "--output requires a value".to_string())?;
                output = Some(PathBuf::from(value));
            }
            other => worker_args.push(other.to_string()),
        }
        index += 1;
    }
    let manifest = build_worker_queue_manifest(&worker_args)?;
    Ok(WorkerManifestCommandArgs {
        manifest,
        json,
        output,
    })
}

fn parse_deploy_manifest_command_args(
    args: &[String],
) -> Result<DeployManifestCommandArgs, String> {
    let mut output: Option<PathBuf> = None;
    let mut json = false;
    let mut deploy_args = Vec::new();
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "--json" => json = true,
            "--output" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| "--output requires a value".to_string())?;
                output = Some(PathBuf::from(value));
            }
            other => deploy_args.push(other.to_string()),
        }
        index += 1;
    }
    let manifest = build_deploy_validate_manifest(&deploy_args)?;
    Ok(DeployManifestCommandArgs {
        manifest,
        json,
        output,
    })
}

fn parse_cluster_manifest_command_args(
    args: &[String],
) -> Result<ClusterManifestCommandArgs, String> {
    let mut output: Option<PathBuf> = None;
    let mut json = false;
    let mut cluster_args = Vec::new();
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "--json" => json = true,
            "--output" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| "--output requires a value".to_string())?;
                output = Some(PathBuf::from(value));
            }
            other => cluster_args.push(other.to_string()),
        }
        index += 1;
    }
    let manifest = build_cluster_command_manifest(&cluster_args)?;
    Ok(ClusterManifestCommandArgs {
        manifest,
        json,
        output,
    })
}

fn emit_manifest_payload<T: Serialize>(
    manifest: &T,
    json: bool,
    output: Option<&Path>,
    label: &str,
) -> i32 {
    let payload = match serde_json::to_string_pretty(manifest) {
        Ok(value) => value,
        Err(err) => {
            eprintln!(
                "enkai systems {}: failed to serialize manifest: {}",
                label, err
            );
            return 1;
        }
    };
    if let Some(output_path) = output {
        if let Some(parent) = output_path.parent() {
            if !parent.as_os_str().is_empty() {
                if let Err(err) = fs::create_dir_all(parent) {
                    eprintln!(
                        "enkai systems {}: failed to create output directory {}: {}",
                        label,
                        parent.display(),
                        err
                    );
                    return 1;
                }
            }
        }
        if let Err(err) = fs::write(output_path, &payload) {
            eprintln!(
                "enkai systems {}: failed to write {}: {}",
                label,
                output_path.display(),
                err
            );
            return 1;
        }
    }
    if json || output.is_none() {
        println!("{}", payload);
    }
    0
}

fn parse_serve_state(args: &[String]) -> Result<ServeParseState, String> {
    let mut state = ServeParseState::default();
    let mut idx = 0usize;
    while idx < args.len() {
        let arg = &args[idx];
        match arg.as_str() {
            "--host" => {
                idx += 1;
                state.host = Some(
                    args.get(idx)
                        .ok_or_else(|| "enkai serve --host requires a value".to_string())?
                        .clone(),
                );
            }
            "--port" => {
                idx += 1;
                state.port = Some(
                    args.get(idx)
                        .ok_or_else(|| "enkai serve --port requires a value".to_string())?
                        .clone(),
                );
            }
            "--grpc-host" => {
                idx += 1;
                state.grpc_host = Some(
                    args.get(idx)
                        .ok_or_else(|| "enkai serve --grpc-host requires a value".to_string())?
                        .clone(),
                );
            }
            "--grpc-port" => {
                idx += 1;
                state.grpc_port = Some(
                    args.get(idx)
                        .ok_or_else(|| "enkai serve --grpc-port requires a value".to_string())?
                        .clone(),
                );
            }
            "--registry" => {
                idx += 1;
                state.registry = Some(
                    args.get(idx)
                        .ok_or_else(|| "enkai serve --registry requires a value".to_string())?
                        .clone(),
                );
            }
            "--model" => {
                idx += 1;
                state.model = Some(
                    args.get(idx)
                        .ok_or_else(|| "enkai serve --model requires a value".to_string())?
                        .clone(),
                );
            }
            "--model-version" => {
                idx += 1;
                state.model_version = Some(
                    args.get(idx)
                        .ok_or_else(|| "enkai serve --model-version requires a value".to_string())?
                        .clone(),
                );
            }
            "--latest" => state.latest = true,
            "--multi-model" => state.multi_model = true,
            "--require-loaded" => state.require_loaded = true,
            "--checkpoint" => {
                idx += 1;
                state.checkpoint = Some(
                    args.get(idx)
                        .ok_or_else(|| "enkai serve --checkpoint requires a value".to_string())?
                        .clone(),
                );
            }
            "--trace-vm" | "--disasm" | "--trace-task" | "--trace-net" => {
                state.runtime_flags.push(arg.clone());
            }
            other => {
                if other.starts_with("--") {
                    return Err(format!("Unknown serve option: {}", other));
                }
                if state.target.is_some() {
                    return Err("enkai serve accepts only one file or directory target".to_string());
                }
                state.target = Some(other.to_string());
            }
        }
        idx += 1;
    }
    Ok(state)
}

fn build_worker_enqueue_manifest(args: &[String]) -> Result<WorkerQueueManifest, String> {
    let mut dir: Option<PathBuf> = None;
    let mut queue: Option<String> = None;
    let mut payload: Option<JsonValue> = None;
    let mut tenant: Option<String> = None;
    let mut id: Option<String> = None;
    let mut max_attempts = 3u32;
    let mut json = false;
    let mut output: Option<PathBuf> = None;
    let mut idx = 0usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--dir" => {
                idx += 1;
                dir = Some(PathBuf::from(
                    args.get(idx)
                        .ok_or_else(|| "--dir requires a value".to_string())?,
                ));
            }
            "--queue" => {
                idx += 1;
                queue = Some(
                    args.get(idx)
                        .ok_or_else(|| "--queue requires a value".to_string())?
                        .trim()
                        .to_string(),
                );
            }
            "--payload" => {
                idx += 1;
                let raw = args
                    .get(idx)
                    .ok_or_else(|| "--payload requires a value".to_string())?;
                payload = Some(
                    serde_json::from_str(raw)
                        .map_err(|err| format!("--payload must be valid JSON: {}", err))?,
                );
            }
            "--tenant" => {
                idx += 1;
                tenant = Some(
                    args.get(idx)
                        .ok_or_else(|| "--tenant requires a value".to_string())?
                        .to_string(),
                );
            }
            "--id" => {
                idx += 1;
                id = Some(
                    args.get(idx)
                        .ok_or_else(|| "--id requires a value".to_string())?
                        .to_string(),
                );
            }
            "--max-attempts" => {
                idx += 1;
                max_attempts = args
                    .get(idx)
                    .ok_or_else(|| "--max-attempts requires a value".to_string())?
                    .parse::<u32>()
                    .map_err(|_| "--max-attempts must be an integer".to_string())?;
                if max_attempts == 0 {
                    return Err("--max-attempts must be >= 1".to_string());
                }
            }
            "--json" => json = true,
            "--output" => {
                idx += 1;
                output = Some(PathBuf::from(
                    args.get(idx)
                        .ok_or_else(|| "--output requires a value".to_string())?,
                ));
            }
            other => return Err(format!("unknown option '{}'", other)),
        }
        idx += 1;
    }
    let dir = dir.ok_or_else(|| "--dir is required".to_string())?;
    let queue = queue.ok_or_else(|| "--queue is required".to_string())?;
    let payload = payload.ok_or_else(|| "--payload is required".to_string())?;
    let queue_root = worker_queue_root(&dir, &queue);
    let pending_path = queue_pending_path(&dir, &queue);
    let inflight_path = worker_inflight_path(&dir, &queue);
    let schedule_path = worker_schedule_path(&dir, &queue);
    let dead_letter_path = worker_dead_letter_path(&dir, &queue);
    let state_path = worker_state_path(&dir, &queue);
    Ok(WorkerQueueManifest::Enqueue {
        dir: dir.display().to_string(),
        queue,
        backend_kind: "selfhost_jsonl_queue_v2".to_string(),
        queue_root: queue_root.display().to_string(),
        pending_path: pending_path.display().to_string(),
        inflight_path: inflight_path.display().to_string(),
        schedule_path: schedule_path.display().to_string(),
        dead_letter_path: dead_letter_path.display().to_string(),
        state_path: state_path.display().to_string(),
        payload,
        tenant,
        id,
        max_attempts,
        retry_policy: WorkerRetryPolicyManifest {
            max_attempts,
            delay_ms: 0,
        },
        json,
        output: output.map(|path| path.display().to_string()),
    })
}

fn build_worker_run_manifest(args: &[String]) -> Result<WorkerQueueManifest, String> {
    let mut dir: Option<PathBuf> = None;
    let mut queue: Option<String> = None;
    let mut handler: Option<PathBuf> = None;
    let mut once = false;
    let mut json = false;
    let mut output: Option<PathBuf> = None;
    let mut idx = 0usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--dir" => {
                idx += 1;
                dir = Some(PathBuf::from(
                    args.get(idx)
                        .ok_or_else(|| "--dir requires a value".to_string())?,
                ));
            }
            "--queue" => {
                idx += 1;
                queue = Some(
                    args.get(idx)
                        .ok_or_else(|| "--queue requires a value".to_string())?
                        .trim()
                        .to_string(),
                );
            }
            "--handler" => {
                idx += 1;
                handler = Some(PathBuf::from(
                    args.get(idx)
                        .ok_or_else(|| "--handler requires a value".to_string())?,
                ));
            }
            "--once" => once = true,
            "--json" => json = true,
            "--output" => {
                idx += 1;
                output = Some(PathBuf::from(
                    args.get(idx)
                        .ok_or_else(|| "--output requires a value".to_string())?,
                ));
            }
            other => return Err(format!("unknown option '{}'", other)),
        }
        idx += 1;
    }
    let dir = dir.ok_or_else(|| "--dir is required".to_string())?;
    let queue = queue.ok_or_else(|| "--queue is required".to_string())?;
    let handler = handler.ok_or_else(|| "--handler is required".to_string())?;
    let queue_root = worker_queue_root(&dir, &queue);
    let pending_path = queue_pending_path(&dir, &queue);
    let inflight_path = worker_inflight_path(&dir, &queue);
    let schedule_path = worker_schedule_path(&dir, &queue);
    let dead_letter_path = worker_dead_letter_path(&dir, &queue);
    let state_path = worker_state_path(&dir, &queue);
    Ok(WorkerQueueManifest::Run {
        dir: dir.display().to_string(),
        queue,
        backend_kind: "selfhost_jsonl_queue_v2".to_string(),
        queue_root: queue_root.display().to_string(),
        pending_path: pending_path.display().to_string(),
        inflight_path: inflight_path.display().to_string(),
        schedule_path: schedule_path.display().to_string(),
        dead_letter_path: dead_letter_path.display().to_string(),
        state_path: state_path.display().to_string(),
        handler: handler.display().to_string(),
        once,
        run_policy: WorkerRunPolicyManifest {
            drain_mode: if once {
                "once".to_string()
            } else {
                "until_idle".to_string()
            },
            max_messages: if once { Some(1) } else { None },
        },
        json,
        output: output.map(|path| path.display().to_string()),
    })
}

fn parse_deploy_validate_manifest(args: &[String]) -> Result<DeployValidateManifest, String> {
    if args.is_empty() {
        return Err("missing project directory".to_string());
    }
    let mut project_dir: Option<PathBuf> = None;
    let mut profile: Option<String> = None;
    let mut strict = false;
    let mut json = false;
    let mut output: Option<PathBuf> = None;
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "--profile" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| "--profile requires a value".to_string())?;
                match value.trim() {
                    "backend" | "fullstack" | "mobile" => profile = Some(value.trim().to_string()),
                    other => {
                        return Err(format!(
                            "invalid --profile '{}'; expected backend|fullstack|mobile",
                            other
                        ))
                    }
                }
            }
            "--output" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| "--output requires a value".to_string())?;
                output = Some(PathBuf::from(value));
            }
            "--strict" => strict = true,
            "--json" => json = true,
            other if other.starts_with("--") => {
                return Err(format!("unknown option '{}'", other));
            }
            path => {
                if project_dir.is_some() {
                    return Err(format!("unexpected argument '{}'", path));
                }
                project_dir = Some(PathBuf::from(path));
            }
        }
        index += 1;
    }
    let project_dir = project_dir.ok_or_else(|| "missing project directory".to_string())?;
    if !project_dir.is_dir() {
        return Err(format!(
            "project directory not found: {}",
            project_dir.display()
        ));
    }
    let profile = profile.ok_or_else(|| "--profile is required".to_string())?;
    let evaluated_project =
        crate::deploy_runtime::evaluate_project_layout_json(&project_dir, &profile)?;
    Ok(DeployValidateManifest {
        schema_version: 1,
        profile: "deploy_validate_manifest".to_string(),
        project_dir: project_dir.display().to_string(),
        target_profile: profile,
        strict,
        json,
        output: output.map(|path| path.display().to_string()),
        evaluated_project,
    })
}

fn parse_cluster_manifest(
    subcommand: &str,
    args: &[String],
) -> Result<ClusterCommandManifest, String> {
    let allow_dry_run = subcommand == "run";
    let mut json = false;
    let mut dry_run = false;
    let mut output: Option<PathBuf> = None;
    let mut config_path: Option<PathBuf> = None;
    let mut idx = 0usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--json" => json = true,
            "--dry-run" if allow_dry_run => dry_run = true,
            "--output" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--output requires a value".to_string());
                }
                output = Some(PathBuf::from(&args[idx]));
            }
            flag if flag.starts_with("--") => {
                return Err(format!(
                    "unknown option {} (supported: --json, --output{} )",
                    flag,
                    if allow_dry_run { ", --dry-run" } else { "" }
                ));
            }
            value => {
                if config_path.is_some() {
                    return Err("expected exactly one config file path".to_string());
                }
                config_path = Some(PathBuf::from(value));
            }
        }
        idx += 1;
    }
    let config_path = config_path.ok_or_else(|| "missing config file path".to_string())?;
    let evaluated_plan = crate::cluster_runtime::build_cluster_plan_json(&config_path)?;
    Ok(ClusterCommandManifest {
        schema_version: 1,
        profile: "cluster_command_manifest".to_string(),
        subcommand: subcommand.to_string(),
        config_path: config_path.display().to_string(),
        json,
        dry_run,
        output: output.map(|path| path.display().to_string()),
        evaluated_plan,
    })
}

fn serve_manifest_from_state(state: ServeParseState) -> Result<ServeRuntimeManifest, String> {
    let target = state.target.unwrap_or_else(|| ".".to_string());
    let http_runtime = default_http_runtime_manifest(&target);
    let http = ServiceBindingManifest {
        host: state.host.clone(),
        port: state.port.clone(),
    };
    let grpc = ServiceBindingManifest {
        host: state.grpc_host.clone(),
        port: state.grpc_port.clone(),
    };
    if let (Some(http_port), Some(grpc_port)) = (http.port.as_deref(), grpc.port.as_deref()) {
        if http_port.trim() == grpc_port.trim() && !http_port.trim().is_empty() {
            return Err("enkai serve: --port and --grpc-port must be different".to_string());
        }
    }

    let mut env_projection = BTreeMap::new();
    if let Some(host) = &http.host {
        env_projection.insert("ENKAI_SERVE_HOST".to_string(), host.clone());
    }
    if let Some(port) = &http.port {
        env_projection.insert("ENKAI_SERVE_PORT".to_string(), port.clone());
    }
    if let Some(host) = &grpc.host {
        env_projection.insert("ENKAI_GRPC_HOST".to_string(), host.clone());
    }
    if let Some(port) = &grpc.port {
        env_projection.insert("ENKAI_GRPC_PORT".to_string(), port.clone());
    }
    env_projection.insert(
        "ENKAI_API_VERSION".to_string(),
        http_runtime.api_version.clone(),
    );
    env_projection.insert(
        "ENKAI_CONVERSATION_DIR".to_string(),
        http_runtime.conversation_dir.clone(),
    );
    if let Some(log_path) = &http_runtime.log_path {
        env_projection.insert("ENKAI_LOG_PATH".to_string(), log_path.clone());
    }

    let model = if state.multi_model {
        if state.checkpoint.is_some()
            || state.model.is_some()
            || state.model_version.is_some()
            || state.latest
        {
            return Err(
                "enkai serve: --multi-model cannot be combined with --checkpoint/--model/--model-version/--latest"
                    .to_string(),
            );
        }
        let Some(registry_raw) = state.registry.as_deref() else {
            return Err("enkai serve: --multi-model requires --registry <dir>".to_string());
        };
        let registry_path = canonical_existing_path(Path::new(registry_raw), "--registry")?;
        env_projection.insert("ENKAI_SERVE_MULTI_MODEL".to_string(), "1".to_string());
        env_projection.insert(
            "ENKAI_REQUIRE_MODEL_VERSION_HEADER".to_string(),
            "1".to_string(),
        );
        env_projection.insert(
            "ENKAI_SERVE_MODEL_REGISTRY".to_string(),
            registry_path.to_string_lossy().to_string(),
        );
        ServeModelManifest::Multi {
            registry: registry_path.to_string_lossy().to_string(),
            require_model_version_header: true,
        }
    } else if let Some(selection) = resolve_serve_model_selection(
        state.checkpoint.as_deref(),
        state.registry.as_deref(),
        state.model.as_deref(),
        state.model_version.as_deref(),
        state.latest,
        state.require_loaded,
    )? {
        env_projection.insert(
            "ENKAI_SERVE_MODEL_PATH".to_string(),
            selection.checkpoint_path.to_string_lossy().to_string(),
        );
        if let Some(name) = &selection.model_name {
            env_projection.insert("ENKAI_SERVE_MODEL_NAME".to_string(), name.clone());
        }
        if let Some(version) = &selection.model_version {
            env_projection.insert("ENKAI_SERVE_MODEL_VERSION".to_string(), version.clone());
        }
        if let Some(registry) = &selection.registry {
            env_projection.insert(
                "ENKAI_SERVE_MODEL_REGISTRY".to_string(),
                registry.to_string_lossy().to_string(),
            );
        }
        ServeModelManifest::Single {
            checkpoint_path: selection.checkpoint_path.to_string_lossy().to_string(),
            model_name: selection.model_name,
            model_version: selection.model_version,
            registry: selection
                .registry
                .map(|value| value.to_string_lossy().to_string()),
            require_loaded: state.require_loaded,
        }
    } else {
        ServeModelManifest::None
    };

    Ok(ServeRuntimeManifest {
        schema_version: 1,
        profile: "serve_runtime_manifest".to_string(),
        target: target.clone(),
        runtime_flags: state.runtime_flags,
        http,
        grpc,
        model,
        http_runtime,
        grpc_runtime: Some(default_grpc_runtime_manifest(&target)),
        env_projection,
    })
}

fn worker_queue_root(dir: &Path, queue: &str) -> PathBuf {
    dir.join("queues").join(queue)
}

fn queue_pending_path(dir: &Path, queue: &str) -> PathBuf {
    worker_queue_root(dir, queue).join("pending.jsonl")
}

fn worker_inflight_path(dir: &Path, queue: &str) -> PathBuf {
    worker_queue_root(dir, queue).join("inflight.jsonl")
}

fn worker_schedule_path(dir: &Path, queue: &str) -> PathBuf {
    worker_queue_root(dir, queue).join("scheduled.jsonl")
}

fn worker_dead_letter_path(dir: &Path, queue: &str) -> PathBuf {
    worker_queue_root(dir, queue).join("dead_letter.jsonl")
}

fn worker_state_path(dir: &Path, queue: &str) -> PathBuf {
    worker_queue_root(dir, queue).join("queue_state.json")
}

fn canonical_existing_path(path: &Path, label: &str) -> Result<PathBuf, String> {
    if !path.exists() {
        return Err(format!("{} not found: {}", label, path.display()));
    }
    fs::canonicalize(path)
        .map_err(|err| format!("failed to resolve {} {}: {}", label, path.display(), err))
}

fn load_json_manifest<T>(path: &Path) -> Result<T, String>
where
    T: for<'de> Deserialize<'de>,
{
    let text = fs::read_to_string(path)
        .map_err(|err| format!("failed to read manifest {}: {}", path.display(), err))?;
    serde_json::from_str(&text)
        .map_err(|err| format!("failed to parse manifest {}: {}", path.display(), err))
}

fn parse_manifest_flag(command_name: &str, args: &[String]) -> Result<PathBuf, String> {
    let mut manifest_path: Option<PathBuf> = None;
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "--manifest" => {
                index += 1;
                let Some(value) = args.get(index) else {
                    return Err(format!(
                        "enkai systems {}: --manifest requires a value",
                        command_name
                    ));
                };
                manifest_path = Some(PathBuf::from(value));
            }
            other => {
                return Err(format!(
                    "enkai systems {}: unknown option '{}'",
                    command_name, other
                ));
            }
        }
        index += 1;
    }
    manifest_path.ok_or_else(|| format!("enkai systems {}: --manifest is required", command_name))
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

fn select_active_or_latest_model_version(model_root: &Path) -> Result<String, String> {
    if let Some(active) = model::read_active_model_version(model_root) {
        if model_root.join(&active).is_dir() {
            return Ok(active);
        }
    }
    select_latest_model_version(model_root)
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

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use tempfile::{tempdir, tempdir_in};

    use super::{
        build_cluster_command_manifest, build_deploy_validate_manifest,
        build_serve_runtime_manifest, build_worker_queue_manifest, resolve_serve_model_selection,
        ClusterCommandManifest, ServeModelManifest, WorkerQueueManifest,
    };

    fn write_cluster_config(dir: &Path) -> std::path::PathBuf {
        let config = dir.join("cluster_config.enk");
        fs::write(
            &config,
            r#"import json
fn main() ::
    return json.parse("{\"config_version\":1,\"backend\":\"cpu\",\"vocab_size\":8,\"hidden_size\":4,\"seq_len\":4,\"batch_size\":2,\"lr\":0.1,\"dataset_path\":\"data.txt\",\"checkpoint_dir\":\"ckpt\",\"max_steps\":2,\"save_every\":1,\"log_every\":1,\"tokenizer_train\":{\"path\":\"data.txt\",\"vocab_size\":8},\"world_size\":2,\"rank\":0,\"dist\":{\"topology\":\"multi-node\",\"rendezvous\":\"tcp://127.0.0.1:29500\",\"retry_budget\":2,\"device_map\":[0,1],\"hosts\":[\"node-a\",\"node-b\"],\"host_map\":[0,1]}}")
::
main()
"#,
        )
        .expect("write cluster config");
        config
    }

    fn create_backend_project(root: &Path) {
        let files = [
            root.join("enkai.toml"),
            root.join("src").join("main.enk"),
            root.join("contracts").join("backend_api.snapshot.json"),
            root.join("contracts")
                .join("conversation_state.schema.json"),
            root.join("contracts").join("grpc_api.snapshot.json"),
            root.join("contracts").join("worker_queue.snapshot.json"),
            root.join("contracts").join("db_engines.snapshot.json"),
            root.join("contracts").join("enkai_chat.proto"),
            root.join("contracts").join("deploy_env.snapshot.json"),
            root.join("scripts").join("validate_env_contract.py"),
            root.join(".env.example"),
            root.join("migrations").join("001_conversation_state.sql"),
            root.join("migrations")
                .join("002_conversation_state_index.sql"),
            root.join("worker").join("handler.enk"),
            root.join("deploy").join("docker").join("Dockerfile"),
            root.join("deploy").join("docker-compose.yml"),
            root.join("deploy")
                .join("systemd")
                .join("enkai-worker.service"),
            root.join("deploy")
                .join("systemd")
                .join("enkai-backend.service"),
        ];
        for path in files {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).expect("mkdir");
            }
            fs::write(path, "{}\n").expect("write");
        }
    }

    #[test]
    fn build_serve_manifest_defaults_target_and_flags() {
        let manifest = build_serve_runtime_manifest(&[
            "--host".to_string(),
            "127.0.0.1".to_string(),
            "--port".to_string(),
            "8080".to_string(),
            "--grpc-port".to_string(),
            "9090".to_string(),
            "--trace-vm".to_string(),
        ])
        .expect("manifest");
        assert_eq!(manifest.target, ".");
        assert_eq!(manifest.runtime_flags, vec!["--trace-vm".to_string()]);
        assert_eq!(manifest.http.host.as_deref(), Some("127.0.0.1"));
        assert_eq!(manifest.http.port.as_deref(), Some("8080"));
        assert_eq!(manifest.grpc.port.as_deref(), Some("9090"));
        assert!(matches!(manifest.model, ServeModelManifest::None));
        assert_eq!(manifest.http_runtime.api_version, "v1");
        assert_eq!(
            manifest
                .env_projection
                .get("ENKAI_SERVE_PORT")
                .map(String::as_str),
            Some("8080")
        );
        assert_eq!(
            manifest
                .env_projection
                .get("ENKAI_API_VERSION")
                .map(String::as_str),
            Some("v1")
        );
    }

    #[test]
    fn build_serve_manifest_resolves_single_model_selection() {
        let dir = tempdir().expect("tempdir");
        let registry = dir.path().join("registry");
        let version_dir = registry.join("chat").join("v1.0.0");
        fs::create_dir_all(version_dir.join("checkpoint")).expect("checkpoint dir");
        fs::create_dir_all(registry.join("chat")).expect("chat dir");
        fs::write(registry.join("chat").join(".active_version"), "v1.0.0\n").expect("active");
        let manifest = build_serve_runtime_manifest(&[
            "--registry".to_string(),
            registry.to_string_lossy().to_string(),
            "--model".to_string(),
            "chat".to_string(),
            "--latest".to_string(),
            "examples/hello/main.enk".to_string(),
        ])
        .expect("manifest");
        match manifest.model {
            ServeModelManifest::Single {
                model_name,
                model_version,
                registry,
                ..
            } => {
                assert_eq!(model_name.as_deref(), Some("chat"));
                assert_eq!(model_version.as_deref(), Some("v1.0.0"));
                assert!(registry.expect("registry").contains("registry"));
            }
            other => panic!("unexpected model manifest: {:?}", other),
        }
    }

    #[test]
    fn resolve_serve_model_selection_prefers_latest_semver() {
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
    }

    #[test]
    fn build_serve_manifest_rejects_conflicting_ports() {
        let err = build_serve_runtime_manifest(&[
            "--port".to_string(),
            "8080".to_string(),
            "--grpc-port".to_string(),
            "8080".to_string(),
        ])
        .expect_err("must reject identical ports");
        assert!(err.contains("--port and --grpc-port must be different"));
    }

    #[test]
    fn build_serve_manifest_rejects_multi_model_without_registry() {
        let err = build_serve_runtime_manifest(&["--multi-model".to_string()])
            .expect_err("must require registry");
        assert!(err.contains("--multi-model requires --registry"));
    }

    #[test]
    fn build_worker_manifest_captures_retry_and_handler_contracts() {
        let manifest = build_worker_queue_manifest(&[
            "enqueue".to_string(),
            "--queue".to_string(),
            "jobs".to_string(),
            "--dir".to_string(),
            "state".to_string(),
            "--payload".to_string(),
            "{\"job\":1}".to_string(),
            "--max-attempts".to_string(),
            "5".to_string(),
        ])
        .expect("worker manifest");
        match manifest {
            WorkerQueueManifest::Enqueue {
                queue,
                backend_kind,
                max_attempts,
                retry_policy,
                inflight_path,
                schedule_path,
                state_path,
                payload,
                ..
            } => {
                assert_eq!(queue, "jobs");
                assert_eq!(backend_kind, "selfhost_jsonl_queue_v2");
                assert_eq!(max_attempts, 5);
                assert_eq!(retry_policy.max_attempts, 5);
                assert_eq!(retry_policy.delay_ms, 0);
                assert!(inflight_path.ends_with("inflight.jsonl"));
                assert!(schedule_path.ends_with("scheduled.jsonl"));
                assert!(state_path.ends_with("queue_state.json"));
                assert_eq!(payload["job"], 1);
            }
            other => panic!("unexpected worker manifest: {:?}", other),
        }
    }

    #[test]
    fn build_worker_run_manifest_captures_once_policy() {
        let manifest = build_worker_queue_manifest(&[
            "run".to_string(),
            "--queue".to_string(),
            "jobs".to_string(),
            "--dir".to_string(),
            "state".to_string(),
            "--handler".to_string(),
            "handler.enk".to_string(),
            "--once".to_string(),
        ])
        .expect("worker manifest");
        match manifest {
            WorkerQueueManifest::Run {
                queue,
                backend_kind,
                once,
                run_policy,
                inflight_path,
                schedule_path,
                state_path,
                ..
            } => {
                assert_eq!(queue, "jobs");
                assert_eq!(backend_kind, "selfhost_jsonl_queue_v2");
                assert!(once);
                assert_eq!(run_policy.drain_mode, "once");
                assert_eq!(run_policy.max_messages, Some(1));
                assert!(inflight_path.ends_with("inflight.jsonl"));
                assert!(schedule_path.ends_with("scheduled.jsonl"));
                assert!(state_path.ends_with("queue_state.json"));
            }
            other => panic!("unexpected worker manifest: {:?}", other),
        }
    }

    #[test]
    fn build_worker_run_manifest_defaults_to_until_idle_policy() {
        let manifest = build_worker_queue_manifest(&[
            "run".to_string(),
            "--queue".to_string(),
            "jobs".to_string(),
            "--dir".to_string(),
            "state".to_string(),
            "--handler".to_string(),
            "handler.enk".to_string(),
        ])
        .expect("worker manifest");
        match manifest {
            WorkerQueueManifest::Run {
                once, run_policy, ..
            } => {
                assert!(!once);
                assert_eq!(run_policy.drain_mode, "until_idle");
                assert_eq!(run_policy.max_messages, None);
            }
            other => panic!("unexpected worker manifest: {:?}", other),
        }
    }

    #[test]
    fn build_deploy_manifest_evaluates_project_layout() {
        let dir = tempdir().expect("tempdir");
        create_backend_project(dir.path());
        let manifest = build_deploy_validate_manifest(&[
            "validate".to_string(),
            dir.path().to_string_lossy().to_string(),
            "--profile".to_string(),
            "backend".to_string(),
            "--strict".to_string(),
        ])
        .expect("deploy manifest");
        assert_eq!(manifest.target_profile, "backend");
        assert_eq!(manifest.evaluated_project["profile"], "backend");
        assert_eq!(manifest.evaluated_project["missing_required_paths"], 0);
    }

    #[test]
    fn build_cluster_manifest_embeds_evaluated_plan() {
        let dir = tempdir_in(".").expect("tempdir");
        let config = write_cluster_config(dir.path());
        let manifest = build_cluster_command_manifest(&[
            "plan".to_string(),
            "--json".to_string(),
            config.to_string_lossy().to_string(),
        ])
        .expect("cluster manifest");
        let ClusterCommandManifest { evaluated_plan, .. } = manifest;
        assert_eq!(evaluated_plan["topology"], "multi-node");
        assert_eq!(evaluated_plan["world_size"], 2);
        assert_eq!(
            evaluated_plan["rank_plans"].as_array().map(Vec::len),
            Some(2)
        );
    }
}
