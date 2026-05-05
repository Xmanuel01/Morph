use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use enkai_runtime::checkpoint::{
    latest_checkpoint, load_checkpoint, save_checkpoint, CheckpointMeta, CheckpointState,
};
use enkai_runtime::dataset::{resolve_dataset_paths, Batch, DatasetConfig, DatasetStream};
use enkai_runtime::engine::{
    checkpoint_save as rt_checkpoint_save, eval_step as rt_eval_step, init,
    train_step as rt_train_step,
};
use enkai_runtime::tokenizer::{Tokenizer, TrainConfig as TokenizerTrainConfig};

use crate::runtime_exec::{with_scoped_env, ScopedEnv};
use crate::train::{
    apply_checkpoint_retention, build_tokenizer as build_train_tokenizer, data_seed,
    dataset_fingerprint, load_config_value, native_checkpoint_meta, parse_train_config_with_mode,
    register_checkpoint_lifecycle, rt_config_from, shard_paths, try_load_native_checkpoint,
    validate_checkpoint_integrity, validate_native_saved_checkpoint, write_error_checkpoint,
    DivergenceGuard, LogEvent, RunContext, TrainCommandManifest, TrainConfig, TrainManifestCommand,
    TrainMode,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TrainExecutionBackend {
    Cpu,
    EnkaiAccel,
    Native,
}

#[derive(Debug, Clone)]
struct CpuRuntimeConfig {
    requested_backend: String,
    executed_backend: String,
    fallback_reason: Option<String>,
    command_name: String,
    suite_id: Option<String>,
    vocab_size: usize,
    hidden_size: usize,
    seq_len: usize,
    batch_size: usize,
    lr: f32,
    dataset_path: String,
    eval_dataset_path: Option<String>,
    checkpoint_dir: String,
    world_size: usize,
    rank: usize,
    max_steps: usize,
    save_every: usize,
    log_every: usize,
    eval_steps: usize,
    drop_remainder: bool,
    add_eos: bool,
    pad_id: u32,
    seed: u64,
    shuffle: bool,
    prefetch_batches: usize,
    oom_budget_bytes: Option<u64>,
    grad_accum_steps: usize,
    grad_clip_norm: Option<f32>,
    dist_topology: String,
    dist_rendezvous: String,
    dist_retry_budget: usize,
    dist_preview_mode: String,
    shape: AccelModelShape,
    tokenizer: CpuTokenizerConfig,
    config_hash: String,
}

#[derive(Debug, Clone)]
enum CpuTokenizerConfig {
    Load(PathBuf),
    Train {
        path: PathBuf,
        vocab_size: usize,
        save_path: Option<PathBuf>,
    },
}

#[derive(Debug, Clone)]
struct AccelModelShape {
    preset: String,
    n_layers: usize,
    n_heads: usize,
    ff_mult: f32,
    activation: String,
    norm: String,
    tie_embeddings: bool,
}

#[derive(Debug, Clone, Copy)]
struct CpuParamGroup {
    start: usize,
    len: usize,
}

#[derive(Debug)]
struct CpuAdamW {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    m: Vec<f32>,
    v: Vec<f32>,
    beta1_pow: f32,
    beta2_pow: f32,
}

#[derive(Debug)]
struct CpuAdamWMulti {
    groups: Vec<CpuAdamW>,
}

#[derive(Debug)]
struct CpuTinyModel {
    vocab_size: usize,
    hidden_size: usize,
    params: Vec<f32>,
    embed_offset: usize,
    out_offset: usize,
    bias_offset: usize,
}

#[derive(Debug)]
struct EnkaiAccelTinyModel {
    vocab_size: usize,
    hidden_size: usize,
    n_layers: usize,
    n_heads: usize,
    ff_mult: f32,
    activation: String,
    norm: String,
    tie_embeddings: bool,
    params: Vec<f32>,
    embed_offset: usize,
    out_offset: usize,
    bias_offset: usize,
    gate_in_offset: usize,
    gate_out_offset: usize,
    out_transposed: Vec<f32>,
}

#[derive(Debug, Clone)]
struct EnkaiAccelForwardCache {
    hidden: Vec<f32>,
    residuals: Vec<Vec<f32>>,
    normalized: Vec<Vec<f32>>,
    scales: Vec<f32>,
}

#[derive(Debug, Clone, Serialize)]
struct CpuLogEvent {
    step: usize,
    loss: f32,
    tokens: u64,
    lr: f32,
    elapsed_ms: u128,
    event: String,
    tokens_per_sec: Option<f32>,
    step_time_ms: Option<f32>,
    grad_norm: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
struct TrainRuntimeReport {
    schema_version: u32,
    suite_id: Option<String>,
    command: String,
    requested_backend: String,
    executed_backend: String,
    fallback_reason: Option<String>,
    distributed_preview_mode: String,
    kernel: String,
    worker_count: usize,
    success: bool,
    step: usize,
    tokens: u64,
    eval_batches: usize,
    loss: Option<f32>,
    ppl: Option<f32>,
    elapsed_ms: u128,
    tokens_per_sec: Option<f32>,
    peak_memory_bytes_est: u64,
    checkpoint_dir: String,
    world_size: usize,
    rank: usize,
    networked_rendezvous: bool,
    networked_gradient_exchange: bool,
    networked_gradient_bytes: u64,
    dist_retry_count: usize,
    fault_injection_observed: bool,
    latest_checkpoint_path: Option<String>,
    checkpoint_bytes: Option<u64>,
    dataset_path: String,
    eval_dataset_path: Option<String>,
    config_hash: String,
    error_code: Option<String>,
    error_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GradSyncPayload {
    schema_version: u32,
    step: usize,
    rank: usize,
    world_size: usize,
    config_hash: String,
    grads: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MergedGradPayload {
    schema_version: u32,
    step: usize,
    world_size: usize,
    config_hash: String,
    grads: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RendezvousBarrierPayload {
    schema_version: u32,
    step: usize,
    phase: String,
    rank: usize,
    world_size: usize,
    group_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RendezvousBarrierAck {
    schema_version: u32,
    step: usize,
    phase: String,
    world_size: usize,
    group_id: String,
    ok: bool,
}

static NETWORKED_RENDEZVOUS_FAULT_USED: AtomicBool = AtomicBool::new(false);

pub(crate) fn execution_backend(manifest: &TrainCommandManifest) -> TrainExecutionBackend {
    match manifest
        .evaluated_config
        .get("backend")
        .and_then(|value| value.as_str())
        .unwrap_or("cpu")
    {
        "enkai_accel" => TrainExecutionBackend::EnkaiAccel,
        "native" => TrainExecutionBackend::Native,
        _ => TrainExecutionBackend::Cpu,
    }
}

pub(crate) fn execute_train_command_manifest(
    manifest: &TrainCommandManifest,
) -> Result<(), String> {
    match execution_backend(manifest) {
        TrainExecutionBackend::Cpu => execute_cpu_manifest(manifest),
        TrainExecutionBackend::EnkaiAccel => execute_enkai_accel_manifest(manifest),
        TrainExecutionBackend::Native => execute_native_manifest(manifest),
    }
}

pub(crate) fn load_train_command_manifest(path: &Path) -> Result<TrainCommandManifest, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|err| format!("failed to read manifest {}: {}", path.display(), err))?;
    serde_json::from_str(&text)
        .map_err(|err| format!("failed to parse manifest {}: {}", path.display(), err))
}

fn execute_native_manifest(manifest: &TrainCommandManifest) -> Result<(), String> {
    let config_value = load_config_value(Path::new(&manifest.config_path))?;
    let config = parse_train_config_with_mode(&config_value, manifest.strict_contracts)?;
    let env_scope = native_backend_env_scope();
    let config_path = Path::new(&manifest.config_path);
    let run_result = with_scoped_env(&env_scope, || match manifest.command {
        TrainManifestCommand::Train => {
            execute_native_train_or_pretrain(&config, config_path, TrainMode::Train)
        }
        TrainManifestCommand::Pretrain => {
            execute_native_train_or_pretrain(&config, config_path, TrainMode::Pretrain)
        }
        TrainManifestCommand::Eval => execute_native_eval_manifest(&config),
    });
    match run_result {
        Ok(()) => Ok(()),
        Err(err) if is_tensor_backend_unavailable_error(&err) => {
            let enriched = enrich_native_backend_error(err, &env_scope);
            eprintln!(
                "[train-runtime] native backend unavailable, falling back to enkai_accel execution: {}",
                enriched
            );
            execute_enkai_accel_manifest_with_context(
                manifest,
                "native".to_string(),
                Some(enriched),
            )
        }
        Err(err) => Err(enrich_native_backend_error(err, &env_scope)),
    }
}

fn execute_enkai_accel_manifest(manifest: &TrainCommandManifest) -> Result<(), String> {
    execute_enkai_accel_manifest_with_context(manifest, "enkai_accel".to_string(), None)
}

fn execute_enkai_accel_manifest_with_context(
    manifest: &TrainCommandManifest,
    requested_backend: String,
    fallback_reason: Option<String>,
) -> Result<(), String> {
    let config = parse_cpu_runtime_config(
        manifest,
        requested_backend,
        "enkai_accel".to_string(),
        fallback_reason,
    )?;
    match manifest.command {
        TrainManifestCommand::Train | TrainManifestCommand::Pretrain => {
            execute_enkai_accel_train(&config)
        }
        TrainManifestCommand::Eval => execute_enkai_accel_eval(&config),
    }
}

fn execute_native_train_or_pretrain(
    config: &TrainConfig,
    config_path: &Path,
    mode: TrainMode,
) -> Result<(), String> {
    let tokenizer = build_train_tokenizer(config)?;
    let dataset_paths_full = resolve_dataset_paths(&config.dataset_path)?;
    let dataset_hash = dataset_fingerprint(&dataset_paths_full)?;
    let mut run_context = RunContext::open(config, config_path, mode, dataset_hash)?;
    let dataset_paths = shard_paths(dataset_paths_full, config.world_size, config.rank);
    let mut data_cfg = DatasetConfig::new(config.seq_len, config.batch_size);
    data_cfg.add_eos = config.add_eos;
    data_cfg.drop_remainder = config.drop_remainder;
    data_cfg.pad_id = config.pad_id;
    data_cfg.seed = data_seed(config);
    data_cfg.shuffle = config.shuffle;
    data_cfg.prefetch_batches = config.prefetch_batches;
    let mut stream = DatasetStream::new(dataset_paths, tokenizer, data_cfg)?;
    let mut step = 0usize;
    let mut tokens = 0u64;
    fs::create_dir_all(&config.checkpoint_dir)
        .map_err(|err| format!("Failed to create checkpoint dir: {}", err))?;
    let mut log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(Path::new(&config.checkpoint_dir).join("train_log.jsonl"))
        .map_err(|err| format!("Failed to open log file: {}", err))?;
    let start = Instant::now();
    let run_result = (|| -> Result<(), String> {
        let mut engine = init(rt_config_from(config)).map_err(|e| e.to_string())?;
        if let Some(meta) = try_load_native_checkpoint(&mut engine, config, &config.checkpoint_dir)?
        {
            if config.validate_checkpoint_on_resume {
                validate_checkpoint_integrity(
                    config,
                    meta.step,
                    &Path::new(&config.checkpoint_dir).join("latest"),
                )?;
            }
            step = meta.step as usize;
            tokens = meta.tokens;
            run_context.update_progress(step, tokens)?;
        }
        let mut last_saved_step = step;
        let mut packing_sum = 0.0f32;
        let mut packing_count = 0usize;
        let mut loss_guard = DivergenceGuard::new(config);
        while step < config.max_steps {
            let batch = match stream.next_batch()? {
                Some(batch) => batch,
                None => break,
            };
            packing_sum += batch.packing_efficiency;
            packing_count += 1;
            let metrics = match rt_train_step(&mut engine, &batch) {
                Ok(m) => m,
                Err(err) => {
                    let _ = write_error_checkpoint(&engine, config, step, tokens, &err.to_string());
                    return Err(format!("Train error: {}", err));
                }
            };
            if metrics.found_inf {
                packing_sum = 0.0;
                packing_count = 0;
                continue;
            }
            if !metrics.loss.is_finite() {
                let _ = write_error_checkpoint(
                    &engine,
                    config,
                    step,
                    tokens,
                    &format!("non-finite loss at step {}", step),
                );
                return Err("non-finite loss detected".to_string());
            }
            tokens += batch.token_count as u64;
            if !metrics.did_update {
                continue;
            }
            step = metrics.step as usize;
            if let Err(err) = loss_guard.observe(step, metrics.loss) {
                let _ = write_error_checkpoint(&engine, config, step, tokens, &err);
                return Err(err);
            }
            let avg_packing = if packing_count == 0 {
                0.0
            } else {
                packing_sum / packing_count as f32
            };
            if step == 1 || step.is_multiple_of(config.log_every.max(1)) {
                let line = serde_json::to_string(&LogEvent {
                    step,
                    loss: metrics.loss,
                    tokens,
                    lr: config.optim.lr as f32,
                    elapsed_ms: start.elapsed().as_millis(),
                    event: "step".to_string(),
                    tokens_per_sec: Some(metrics.tokens_per_sec),
                    step_time_ms: Some(metrics.step_time_ms),
                    packing_efficiency: Some(avg_packing),
                    gpu_util: metrics.gpu_util,
                    grad_norm: metrics.grad_norm,
                    forward_time_ms: Some(metrics.forward_time_ms),
                    backward_time_ms: Some(metrics.backward_time_ms),
                    optim_time_ms: Some(metrics.optim_time_ms),
                    found_inf: Some(false),
                    ema_loss: loss_guard.ema_loss,
                    divergence_streak: Some(loss_guard.streak),
                })
                .map_err(|err| err.to_string())?;
                writeln!(log_file, "{}", line).map_err(|err| err.to_string())?;
                println!("step {} loss {:.4} tokens {}", step, metrics.loss, tokens);
            }
            if step == config.max_steps || step.is_multiple_of(config.save_every.max(1)) {
                let meta = native_checkpoint_meta(config, step as u64, tokens, metrics.loss)?;
                rt_checkpoint_save(&engine, &config.checkpoint_dir, &meta)
                    .map_err(|err| err.to_string())?;
                let latest_dir = Path::new(&config.checkpoint_dir).join("latest");
                if config.validate_checkpoint_on_save {
                    validate_native_saved_checkpoint(
                        config,
                        &mut engine,
                        step as u64,
                        &latest_dir,
                    )?;
                }
                register_checkpoint_lifecycle(
                    config,
                    step as u64,
                    tokens,
                    metrics.loss as f64,
                    &latest_dir,
                )?;
                run_context.record_checkpoint(step, tokens, &latest_dir, metrics.loss as f64)?;
                if step < last_saved_step {
                    return Err("checkpoint step went backwards".to_string());
                }
                last_saved_step = step;
                let line = serde_json::to_string(&LogEvent {
                    step,
                    loss: metrics.loss,
                    tokens,
                    lr: config.optim.lr as f32,
                    elapsed_ms: start.elapsed().as_millis(),
                    event: "checkpoint".to_string(),
                    tokens_per_sec: None,
                    step_time_ms: None,
                    packing_efficiency: Some(avg_packing),
                    gpu_util: metrics.gpu_util,
                    grad_norm: metrics.grad_norm,
                    forward_time_ms: None,
                    backward_time_ms: None,
                    optim_time_ms: None,
                    found_inf: Some(false),
                    ema_loss: loss_guard.ema_loss,
                    divergence_streak: Some(loss_guard.streak),
                })
                .map_err(|err| err.to_string())?;
                writeln!(log_file, "{}", line).map_err(|err| err.to_string())?;
                apply_native_checkpoint_retention(config, Path::new(&config.checkpoint_dir))?;
                println!("saved checkpoint latest");
            }
            packing_sum = 0.0;
            packing_count = 0;
        }
        Ok(())
    })();

    match run_result {
        Ok(()) => {
            run_context.mark_completed(step, tokens)?;
            Ok(())
        }
        Err(err) => {
            let _ = run_context.mark_failed(step, tokens, &err);
            Err(err)
        }
    }
}

fn execute_native_eval_manifest(config: &TrainConfig) -> Result<(), String> {
    let tokenizer = build_train_tokenizer(config)?;
    let dataset_path = config
        .eval_dataset_path
        .as_ref()
        .unwrap_or(&config.dataset_path);
    let dataset_paths = resolve_dataset_paths(dataset_path)?;
    let mut data_cfg = DatasetConfig::new(config.seq_len, config.batch_size);
    data_cfg.add_eos = config.add_eos;
    data_cfg.drop_remainder = config.drop_remainder;
    data_cfg.pad_id = config.pad_id;
    data_cfg.seed = data_seed(config);
    data_cfg.prefetch_batches = config.prefetch_batches;
    let mut stream = DatasetStream::new(dataset_paths, tokenizer, data_cfg)?;
    let mut engine = init(rt_config_from(config)).map_err(|e| e.to_string())?;
    let _meta = try_load_native_checkpoint(&mut engine, config, &config.checkpoint_dir)?
        .ok_or_else(|| "No checkpoint found".to_string())?;
    let mut total_loss = 0.0f32;
    let mut batches = 0usize;
    while batches < config.eval_steps {
        let batch = match stream.next_batch()? {
            Some(batch) => batch,
            None => break,
        };
        let metrics = rt_eval_step(&mut engine, &batch).map_err(|e| e.to_string())?;
        if !metrics.loss.is_finite() {
            return Err("non-finite loss detected".to_string());
        }
        total_loss += metrics.loss;
        batches += 1;
    }
    if batches == 0 {
        return Err("No eval batches produced".to_string());
    }
    let avg_loss = total_loss / batches as f32;
    let ppl = avg_loss.exp();
    println!("eval loss {:.4} ppl {:.4}", avg_loss, ppl);
    Ok(())
}

fn apply_native_checkpoint_retention(
    config: &TrainConfig,
    checkpoint_root: &Path,
) -> Result<(), String> {
    apply_checkpoint_retention(config, checkpoint_root)
}

fn native_backend_env_scope() -> ScopedEnv {
    let mut vars = BTreeMap::new();
    if env::var_os("ENKAI_TENSOR_PATH").is_none() {
        if let Some(path) = resolve_tensor_library_path() {
            vars.insert(
                "ENKAI_TENSOR_PATH".to_string(),
                path.to_string_lossy().into_owned(),
            );
        }
    }
    ScopedEnv {
        vars,
        std_override: None,
    }
}

fn resolve_tensor_library_path() -> Option<PathBuf> {
    candidate_tensor_library_paths()
        .into_iter()
        .find(|path| path.is_file())
}

fn candidate_tensor_library_paths() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    let current_exe = env::current_exe().ok();
    if let Some(exe) = current_exe.as_ref() {
        if let Some(exe_dir) = exe.parent() {
            roots.push(exe_dir.to_path_buf());
            roots.push(exe_dir.join("deps"));
        }
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if let Some(repo_root) = manifest_dir.parent() {
        roots.push(repo_root.join("target").join("debug"));
        roots.push(repo_root.join("target").join("release"));
        roots.push(repo_root.join("target_codex4").join("debug"));
        roots.push(repo_root.join("target_codex3").join("debug"));
        roots.push(repo_root.join("install"));
    }

    candidate_tensor_library_paths_from_roots(&roots)
}

fn candidate_tensor_library_paths_from_roots(roots: &[PathBuf]) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    for root in roots {
        push_tensor_candidates_from_root(root, &mut candidates);
    }
    dedupe_paths(candidates)
}

fn push_tensor_candidates_from_root(root: &Path, candidates: &mut Vec<PathBuf>) {
    let file_name = tensor_library_file_name();
    candidates.push(root.join(file_name));
    candidates.push(root.join("deps").join(file_name));
}

fn tensor_library_file_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "enkai_tensor.dll"
    } else if cfg!(target_os = "macos") {
        "libenkai_tensor.dylib"
    } else {
        "libenkai_tensor.so"
    }
}

fn dedupe_paths(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut seen = std::collections::BTreeSet::new();
    let mut deduped = Vec::new();
    for path in paths {
        if seen.insert(path.clone()) {
            deduped.push(path);
        }
    }
    deduped
}

fn enrich_native_backend_error(err: String, env_scope: &ScopedEnv) -> String {
    let mut detail = err;
    if detail.contains("Failed to load enkai_tensor (libtorch backend)") {
        detail.push_str(". train_runtime searched for enkai_tensor via ENKAI_TENSOR_PATH and common target/install bundle locations");
        if let Some(path) = env_scope.vars.get("ENKAI_TENSOR_PATH") {
            detail.push_str(&format!("; resolved candidate: {}", path));
        }
    } else if detail.contains("torch backend not enabled") {
        if let Some(path) = env_scope.vars.get("ENKAI_TENSOR_PATH") {
            detail.push_str(&format!(
                ". train_runtime resolved ENKAI_TENSOR_PATH to {} but the loaded backend was built without the torch feature",
                path
            ));
        } else {
            detail.push_str(
                ". train_runtime did not find a tensor library candidate to bind automatically",
            );
        }
    }
    detail
}

fn is_tensor_backend_unavailable_error(err: &str) -> bool {
    err.contains("Failed to load enkai_tensor (libtorch backend)")
        || err.contains("torch backend not enabled")
}

fn runtime_report_path(config: &CpuRuntimeConfig) -> PathBuf {
    Path::new(&config.checkpoint_dir).join("ai_runtime_report.json")
}

fn write_runtime_report(
    config: &CpuRuntimeConfig,
    report: &TrainRuntimeReport,
) -> Result<(), String> {
    let path = runtime_report_path(config);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            format!(
                "failed to create runtime report directory {}: {}",
                parent.display(),
                err
            )
        })?;
    }
    let text = serde_json::to_string_pretty(report)
        .map_err(|err| format!("failed to serialize runtime report: {}", err))?;
    fs::write(&path, text)
        .map_err(|err| format!("failed to write runtime report {}: {}", path.display(), err))
}

fn directory_size_bytes(root: &Path) -> Result<u64, String> {
    if !root.exists() {
        return Ok(0);
    }
    let mut total = 0u64;
    for entry in fs::read_dir(root)
        .map_err(|err| format!("failed to read directory {}: {}", root.display(), err))?
    {
        let entry = entry.map_err(|err| format!("failed to read directory entry: {}", err))?;
        let path = entry.path();
        let meta = entry
            .metadata()
            .map_err(|err| format!("failed to stat {}: {}", path.display(), err))?;
        if meta.is_dir() {
            total += directory_size_bytes(&path)?;
        } else {
            total += meta.len();
        }
    }
    Ok(total)
}

fn estimate_peak_memory_bytes(config: &CpuRuntimeConfig) -> u64 {
    let accel_extra = (config.shape.n_layers * config.hidden_size * 2) as u64;
    let param_count =
        (config.vocab_size * config.hidden_size * 2 + config.vocab_size) as u64 + accel_extra;
    let param_bytes = param_count * 4;
    let optimizer_bytes = param_count * 8;
    let batch_bytes = (config.batch_size * config.seq_len * 2) as u64 * 4;
    let working_bytes = (config.hidden_size * config.batch_size * config.seq_len) as u64 * 8;
    param_bytes + optimizer_bytes + batch_bytes + working_bytes
}

fn enforce_oom_budget(config: &CpuRuntimeConfig) -> Result<u64, String> {
    let estimate = estimate_peak_memory_bytes(config);
    if let Some(limit) = config.oom_budget_bytes {
        if estimate > limit {
            return Err(format!(
                "E_TRAIN_OOM_BUDGET: estimated peak memory {} bytes exceeds configured budget {} bytes",
                estimate, limit
            ));
        }
    }
    Ok(estimate)
}

fn ensure_runtime_state(config: &CpuRuntimeConfig) -> Result<(), String> {
    if config.vocab_size == 0
        || config.hidden_size == 0
        || config.seq_len == 0
        || config.batch_size == 0
    {
        return Err(
            "E_TRAIN_INVALID_STATE: vocab_size, hidden_size, seq_len, and batch_size must all be > 0"
                .to_string(),
        );
    }
    if config.world_size == 0 || config.rank >= config.world_size {
        return Err("E_TRAIN_INVALID_STATE: invalid distributed rank state".to_string());
    }
    if config.world_size > 1
        && config.executed_backend == "enkai_accel"
        && !matches!(
            config.dist_preview_mode.as_str(),
            "rank-sharded-preview" | "synchronized-grad-preview" | "networked-sync-preview"
        )
    {
        return Err(
            "E_TRAIN_INVALID_STATE: enkai_accel distributed execution requires rank-sharded-preview, synchronized-grad-preview, or networked-sync-preview mode"
                .to_string(),
        );
    }
    if config.dist_preview_mode == "networked-sync-preview" {
        if config.dist_topology != "multi-node" {
            return Err(
                "E_TRAIN_INVALID_STATE: networked-sync-preview requires dist.topology = \"multi-node\""
                    .to_string(),
            );
        }
        if !config
            .dist_rendezvous
            .to_ascii_lowercase()
            .starts_with("tcp://")
        {
            return Err(
                "E_TRAIN_INVALID_STATE: networked-sync-preview requires dist.rendezvous = tcp://<host>:<port>"
                    .to_string(),
            );
        }
    }
    Ok(())
}

fn execute_cpu_manifest(manifest: &TrainCommandManifest) -> Result<(), String> {
    let requested = manifest
        .evaluated_config
        .get("backend")
        .and_then(|value| value.as_str())
        .unwrap_or("cpu")
        .to_string();
    execute_cpu_manifest_with_context(manifest, requested, "cpu".to_string(), None)
}

fn execute_cpu_manifest_with_context(
    manifest: &TrainCommandManifest,
    requested_backend: String,
    executed_backend: String,
    fallback_reason: Option<String>,
) -> Result<(), String> {
    let config = parse_cpu_runtime_config(
        manifest,
        requested_backend,
        executed_backend,
        fallback_reason,
    )?;
    match manifest.command {
        TrainManifestCommand::Train | TrainManifestCommand::Pretrain => execute_cpu_train(&config),
        TrainManifestCommand::Eval => execute_cpu_eval(&config),
    }
}

fn execute_cpu_train(config: &CpuRuntimeConfig) -> Result<(), String> {
    let peak_memory_bytes_est = enforce_oom_budget(config)?;
    ensure_runtime_state(config)?;
    let tokenizer = build_tokenizer(&config.tokenizer)?;
    let dataset_paths = shard_paths(
        resolve_dataset_paths(&config.dataset_path)?,
        config.world_size,
        config.rank,
    );
    let mut data_cfg = DatasetConfig::new(config.seq_len, config.batch_size);
    data_cfg.add_eos = config.add_eos;
    data_cfg.drop_remainder = config.drop_remainder;
    data_cfg.pad_id = config.pad_id;
    data_cfg.seed = Some(config.seed);
    data_cfg.shuffle = config.shuffle;
    data_cfg.prefetch_batches = config.prefetch_batches;
    let mut stream = DatasetStream::new(dataset_paths, tokenizer, data_cfg)?;

    let checkpoint_root = Path::new(&config.checkpoint_dir);
    fs::create_dir_all(checkpoint_root).map_err(|err| {
        format!(
            "Failed to create checkpoint dir {}: {}",
            checkpoint_root.display(),
            err
        )
    })?;
    let mut log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(checkpoint_root.join("train_log.jsonl"))
        .map_err(|err| format!("Failed to open log file: {}", err))?;

    let mut model = CpuTinyModel::new(config.vocab_size, config.hidden_size, config.seed);
    let groups = model.param_groups();
    let group_sizes: Vec<usize> = groups.iter().map(|group| group.len).collect();
    let mut opt = CpuAdamWMulti::new(&group_sizes, config.lr);

    let mut step = 0usize;
    let mut tokens = 0u64;
    let mut latest_checkpoint_path = None;
    if let Some(latest) =
        latest_checkpoint(checkpoint_root).map_err(|err| format!("E_CHECKPOINT_IO: {}", err))?
    {
        let state =
            load_checkpoint(&latest).map_err(|err| format!("E_CHECKPOINT_CORRUPT: {}", err))?;
        model
            .set_params(&state.weights)
            .map_err(|err| format!("E_CHECKPOINT_CORRUPT: {}", err))?;
        opt.load_state(&state.optimizer)
            .map_err(|err| format!("E_CHECKPOINT_CORRUPT: {}", err))?;
        step = state.meta.step as usize;
        tokens = state.meta.tokens;
        latest_checkpoint_path = Some(latest.display().to_string());
    }

    let accum_steps = config.grad_accum_steps.max(1);
    let mut grad_accum = vec![0.0f32; model.params().len()];
    let start = Instant::now();
    let mut final_loss = None;
    while step < config.max_steps {
        let step_start = Instant::now();
        let mut step_loss_sum = 0.0f32;
        let mut step_tokens = 0u64;
        let mut micro_batches = 0usize;
        while micro_batches < accum_steps {
            let batch = match stream.next_batch()? {
                Some(batch) => batch,
                None => break,
            };
            let (loss, grads) = model.compute_grads(&batch);
            if !loss.is_finite() {
                return Err("E_TRAIN_INVALID_STATE: non-finite loss detected".to_string());
            }
            for (acc, grad) in grad_accum.iter_mut().zip(grads.iter()) {
                *acc += *grad;
            }
            step_loss_sum += loss;
            step_tokens += batch.token_count as u64;
            tokens += batch.token_count as u64;
            micro_batches += 1;
        }
        if micro_batches == 0 {
            break;
        }
        let scale = 1.0 / micro_batches as f32;
        for grad in &mut grad_accum {
            *grad *= scale;
        }
        let grad_norm = config
            .grad_clip_norm
            .and_then(|max_norm| clip_grad_norm(&mut grad_accum, max_norm));
        if config.world_size > 1 && config.dist_preview_mode == "synchronized-grad-preview" {
            grad_accum = synchronize_preview_gradients(config, step + 1, &grad_accum)?;
        }
        model.apply_grads(&grad_accum, &mut opt, &groups);
        grad_accum.fill(0.0);
        step += 1;
        let avg_loss = step_loss_sum / micro_batches as f32;
        let elapsed = step_start.elapsed();
        let tokens_per_sec = (step_tokens as f32) / elapsed.as_secs_f32().max(1e-6);
        let step_time_ms = elapsed.as_secs_f64() as f32 * 1000.0;

        if step == 1 || step.is_multiple_of(config.log_every.max(1)) {
            let line = serde_json::to_string(&CpuLogEvent {
                step,
                loss: avg_loss,
                tokens,
                lr: config.lr,
                elapsed_ms: start.elapsed().as_millis(),
                event: "step".to_string(),
                tokens_per_sec: Some(tokens_per_sec),
                step_time_ms: Some(step_time_ms),
                grad_norm,
            })
            .map_err(|err| err.to_string())?;
            writeln!(log_file, "{}", line).map_err(|err| err.to_string())?;
            println!("step {} loss {:.4} tokens {}", step, avg_loss, tokens);
        }

        if step == config.max_steps || step.is_multiple_of(config.save_every.max(1)) {
            let state = CheckpointState {
                weights: model.params().to_vec(),
                optimizer: opt.state_vec(),
                meta: CheckpointMeta {
                    format_version: 1,
                    step: step as u64,
                    tokens,
                    loss: avg_loss as f64,
                    config_hash: config.config_hash.clone(),
                    model_sig: model_signature(config),
                    dtype: "f32".to_string(),
                    device: "cpu".to_string(),
                    world_size: config.world_size,
                    rank: config.rank,
                    grad_accum_steps: config.grad_accum_steps,
                    grad_clip_norm: config.grad_clip_norm.map(|value| value as f64),
                    amp: None,
                },
            };
            let path = save_checkpoint(checkpoint_root, &state).map_err(|err| err.to_string())?;
            latest_checkpoint_path = Some(path.display().to_string());
            let line = serde_json::to_string(&CpuLogEvent {
                step,
                loss: avg_loss,
                tokens,
                lr: config.lr,
                elapsed_ms: start.elapsed().as_millis(),
                event: "checkpoint".to_string(),
                tokens_per_sec: None,
                step_time_ms: None,
                grad_norm,
            })
            .map_err(|err| err.to_string())?;
            writeln!(log_file, "{}", line).map_err(|err| err.to_string())?;
            println!("saved checkpoint {}", path.display());
        }
        final_loss = Some(avg_loss);
    }
    let elapsed_ms = start.elapsed().as_millis();
    let tokens_per_sec = if elapsed_ms > 0 {
        Some(tokens as f32 / (elapsed_ms as f32 / 1000.0))
    } else {
        None
    };
    write_runtime_report(
        config,
        &TrainRuntimeReport {
            schema_version: 1,
            suite_id: config.suite_id.clone(),
            command: config.command_name.clone(),
            requested_backend: config.requested_backend.clone(),
            executed_backend: config.executed_backend.clone(),
            fallback_reason: config.fallback_reason.clone(),
            distributed_preview_mode: config.dist_preview_mode.clone(),
            kernel: "cpu_scalar_v1".to_string(),
            worker_count: 1,
            success: true,
            step,
            tokens,
            eval_batches: 0,
            loss: final_loss,
            ppl: final_loss.map(|value| value.exp()),
            elapsed_ms,
            tokens_per_sec,
            peak_memory_bytes_est,
            checkpoint_dir: config.checkpoint_dir.clone(),
            world_size: config.world_size,
            rank: config.rank,
            networked_rendezvous: false,
            networked_gradient_exchange: false,
            networked_gradient_bytes: 0,
            dist_retry_count: 0,
            fault_injection_observed: false,
            latest_checkpoint_path,
            checkpoint_bytes: Some(directory_size_bytes(checkpoint_root)?),
            dataset_path: config.dataset_path.clone(),
            eval_dataset_path: config.eval_dataset_path.clone(),
            config_hash: config.config_hash.clone(),
            error_code: None,
            error_message: None,
        },
    )?;
    Ok(())
}

fn execute_cpu_eval(config: &CpuRuntimeConfig) -> Result<(), String> {
    let peak_memory_bytes_est = enforce_oom_budget(config)?;
    ensure_runtime_state(config)?;
    let tokenizer = build_tokenizer(&config.tokenizer)?;
    let dataset_path = config
        .eval_dataset_path
        .as_ref()
        .unwrap_or(&config.dataset_path);
    let dataset_paths = shard_paths(
        resolve_dataset_paths(dataset_path)?,
        config.world_size,
        config.rank,
    );
    let mut data_cfg = DatasetConfig::new(config.seq_len, config.batch_size);
    data_cfg.add_eos = config.add_eos;
    data_cfg.drop_remainder = config.drop_remainder;
    data_cfg.pad_id = config.pad_id;
    data_cfg.seed = Some(config.seed);
    data_cfg.prefetch_batches = config.prefetch_batches;
    let mut stream = DatasetStream::new(dataset_paths, tokenizer, data_cfg)?;

    let checkpoint_root = Path::new(&config.checkpoint_dir);
    let latest = latest_checkpoint(checkpoint_root)
        .map_err(|err| format!("E_CHECKPOINT_IO: {}", err))?
        .ok_or_else(|| "E_CHECKPOINT_MISSING: No checkpoint found".to_string())?;
    let state = load_checkpoint(&latest).map_err(|err| format!("E_CHECKPOINT_CORRUPT: {}", err))?;
    let mut model = CpuTinyModel::new(config.vocab_size, config.hidden_size, config.seed);
    model
        .set_params(&state.weights)
        .map_err(|err| format!("E_CHECKPOINT_CORRUPT: {}", err))?;

    let start = Instant::now();
    let mut total_loss = 0.0f32;
    let mut batches = 0usize;
    while batches < config.eval_steps.max(1) {
        let batch = match stream.next_batch()? {
            Some(batch) => batch,
            None => break,
        };
        total_loss += model.eval_loss(&batch);
        batches += 1;
    }
    if batches == 0 {
        return Err("E_EVAL_EMPTY: No eval batches produced".to_string());
    }
    let avg_loss = total_loss / batches as f32;
    let ppl = avg_loss.exp();
    println!("eval loss {:.4} ppl {:.4}", avg_loss, ppl);
    write_runtime_report(
        config,
        &TrainRuntimeReport {
            schema_version: 1,
            suite_id: config.suite_id.clone(),
            command: config.command_name.clone(),
            requested_backend: config.requested_backend.clone(),
            executed_backend: config.executed_backend.clone(),
            fallback_reason: config.fallback_reason.clone(),
            distributed_preview_mode: config.dist_preview_mode.clone(),
            kernel: "cpu_scalar_v1".to_string(),
            worker_count: 1,
            success: true,
            step: state.meta.step as usize,
            tokens: state.meta.tokens,
            eval_batches: batches,
            loss: Some(avg_loss),
            ppl: Some(ppl),
            elapsed_ms: start.elapsed().as_millis(),
            tokens_per_sec: None,
            peak_memory_bytes_est,
            checkpoint_dir: config.checkpoint_dir.clone(),
            world_size: config.world_size,
            rank: config.rank,
            networked_rendezvous: false,
            networked_gradient_exchange: false,
            networked_gradient_bytes: 0,
            dist_retry_count: 0,
            fault_injection_observed: false,
            latest_checkpoint_path: Some(latest.display().to_string()),
            checkpoint_bytes: Some(directory_size_bytes(checkpoint_root)?),
            dataset_path: config.dataset_path.clone(),
            eval_dataset_path: config.eval_dataset_path.clone(),
            config_hash: config.config_hash.clone(),
            error_code: None,
            error_message: None,
        },
    )?;
    Ok(())
}

fn execute_enkai_accel_train(config: &CpuRuntimeConfig) -> Result<(), String> {
    let peak_memory_bytes_est = enforce_oom_budget(config)?;
    ensure_runtime_state(config)?;
    let tokenizer = build_tokenizer(&config.tokenizer)?;
    let dataset_paths = shard_paths(
        resolve_dataset_paths(&config.dataset_path)?,
        config.world_size,
        config.rank,
    );
    let mut data_cfg = DatasetConfig::new(config.seq_len, config.batch_size);
    data_cfg.add_eos = config.add_eos;
    data_cfg.drop_remainder = config.drop_remainder;
    data_cfg.pad_id = config.pad_id;
    data_cfg.seed = Some(config.seed);
    data_cfg.shuffle = config.shuffle;
    data_cfg.prefetch_batches = config.prefetch_batches;
    let mut stream = DatasetStream::new(dataset_paths, tokenizer, data_cfg)?;

    let checkpoint_root = Path::new(&config.checkpoint_dir);
    fs::create_dir_all(checkpoint_root).map_err(|err| {
        format!(
            "Failed to create checkpoint dir {}: {}",
            checkpoint_root.display(),
            err
        )
    })?;
    let mut log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(checkpoint_root.join("train_log.jsonl"))
        .map_err(|err| format!("Failed to open log file: {}", err))?;

    let mut model = EnkaiAccelTinyModel::new(
        config.vocab_size,
        config.hidden_size,
        config.seed,
        &config.shape,
    );
    let groups = model.param_groups();
    let group_sizes: Vec<usize> = groups.iter().map(|group| group.len).collect();
    let mut opt = CpuAdamWMulti::new(&group_sizes, config.lr);

    let mut step = 0usize;
    let mut tokens = 0u64;
    let mut latest_checkpoint_path = None;
    if let Some(latest) =
        latest_checkpoint(checkpoint_root).map_err(|err| format!("E_CHECKPOINT_IO: {}", err))?
    {
        let state =
            load_checkpoint(&latest).map_err(|err| format!("E_CHECKPOINT_CORRUPT: {}", err))?;
        model
            .set_params(&state.weights)
            .map_err(|err| format!("E_CHECKPOINT_CORRUPT: {}", err))?;
        opt.load_state(&state.optimizer)
            .map_err(|err| format!("E_CHECKPOINT_CORRUPT: {}", err))?;
        step = state.meta.step as usize;
        tokens = state.meta.tokens;
        latest_checkpoint_path = Some(latest.display().to_string());
    }

    let accum_steps = config.grad_accum_steps.max(1);
    let mut grad_accum = vec![0.0f32; model.params().len()];
    let start = Instant::now();
    let mut final_loss = None;
    let mut dist_retry_count = 0usize;
    let mut fault_injection_observed = false;
    let mut networked_gradient_bytes = 0u64;
    let networked_rendezvous = config.dist_preview_mode == "networked-sync-preview";
    while step < config.max_steps {
        let step_start = Instant::now();
        let mut step_loss_sum = 0.0f32;
        let mut step_tokens = 0u64;
        let mut micro_batches = 0usize;
        while micro_batches < accum_steps {
            let batch = match stream.next_batch()? {
                Some(batch) => batch,
                None => break,
            };
            let (loss, grads) = model.compute_grads(&batch);
            if !loss.is_finite() {
                return Err("E_TRAIN_INVALID_STATE: non-finite loss detected".to_string());
            }
            for (acc, grad) in grad_accum.iter_mut().zip(grads.iter()) {
                *acc += *grad;
            }
            step_loss_sum += loss;
            step_tokens += batch.token_count as u64;
            tokens += batch.token_count as u64;
            micro_batches += 1;
        }
        if micro_batches == 0 {
            break;
        }
        let scale = 1.0 / micro_batches as f32;
        for grad in &mut grad_accum {
            *grad *= scale;
        }
        let grad_norm = config
            .grad_clip_norm
            .and_then(|max_norm| clip_grad_norm(&mut grad_accum, max_norm));
        if config.world_size > 1 {
            match config.dist_preview_mode.as_str() {
                "synchronized-grad-preview" => {
                    grad_accum = synchronize_preview_gradients(config, step + 1, &grad_accum)?;
                }
                "networked-sync-preview" => {
                    networked_barrier(
                        config,
                        step + 1,
                        "pre-sync",
                        &mut dist_retry_count,
                        &mut fault_injection_observed,
                    )?;
                    let (merged, exchanged_bytes) =
                        synchronize_networked_gradients(config, step + 1, &grad_accum)?;
                    grad_accum = merged;
                    networked_gradient_bytes =
                        networked_gradient_bytes.saturating_add(exchanged_bytes);
                    networked_barrier(
                        config,
                        step + 1,
                        "post-sync",
                        &mut dist_retry_count,
                        &mut fault_injection_observed,
                    )?;
                }
                _ => {}
            }
        }
        model.apply_grads(&grad_accum, &mut opt, &groups);
        grad_accum.fill(0.0);
        step += 1;
        let avg_loss = step_loss_sum / micro_batches as f32;
        let elapsed = step_start.elapsed();
        let tokens_per_sec = (step_tokens as f32) / elapsed.as_secs_f32().max(1e-6);
        let step_time_ms = elapsed.as_secs_f64() as f32 * 1000.0;

        if step == 1 || step.is_multiple_of(config.log_every.max(1)) {
            let line = serde_json::to_string(&CpuLogEvent {
                step,
                loss: avg_loss,
                tokens,
                lr: config.lr,
                elapsed_ms: start.elapsed().as_millis(),
                event: "step".to_string(),
                tokens_per_sec: Some(tokens_per_sec),
                step_time_ms: Some(step_time_ms),
                grad_norm,
            })
            .map_err(|err| err.to_string())?;
            writeln!(log_file, "{}", line).map_err(|err| err.to_string())?;
            println!("step {} loss {:.4} tokens {}", step, avg_loss, tokens);
        }

        if step == config.max_steps || step.is_multiple_of(config.save_every.max(1)) {
            let state = CheckpointState {
                weights: model.params().to_vec(),
                optimizer: opt.state_vec(),
                meta: CheckpointMeta {
                    format_version: 1,
                    step: step as u64,
                    tokens,
                    loss: avg_loss as f64,
                    config_hash: config.config_hash.clone(),
                    model_sig: model_signature(config),
                    dtype: "f32".to_string(),
                    device: "cpu".to_string(),
                    world_size: config.world_size,
                    rank: config.rank,
                    grad_accum_steps: config.grad_accum_steps,
                    grad_clip_norm: config.grad_clip_norm.map(|value| value as f64),
                    amp: None,
                },
            };
            let path = save_checkpoint(checkpoint_root, &state).map_err(|err| err.to_string())?;
            latest_checkpoint_path = Some(path.display().to_string());
            let line = serde_json::to_string(&CpuLogEvent {
                step,
                loss: avg_loss,
                tokens,
                lr: config.lr,
                elapsed_ms: start.elapsed().as_millis(),
                event: "checkpoint".to_string(),
                tokens_per_sec: None,
                step_time_ms: None,
                grad_norm,
            })
            .map_err(|err| err.to_string())?;
            writeln!(log_file, "{}", line).map_err(|err| err.to_string())?;
            println!("saved checkpoint {}", path.display());
        }
        final_loss = Some(avg_loss);
    }
    let elapsed_ms = start.elapsed().as_millis();
    let tokens_per_sec = if elapsed_ms > 0 {
        Some(tokens as f32 / (elapsed_ms as f32 / 1000.0))
    } else {
        None
    };
    write_runtime_report(
        config,
        &TrainRuntimeReport {
            schema_version: 1,
            suite_id: config.suite_id.clone(),
            command: config.command_name.clone(),
            requested_backend: config.requested_backend.clone(),
            executed_backend: config.executed_backend.clone(),
            fallback_reason: config.fallback_reason.clone(),
            distributed_preview_mode: config.dist_preview_mode.clone(),
            kernel: "enkai_accel_v1".to_string(),
            worker_count: EnkaiAccelTinyModel::worker_count(config.batch_size * config.seq_len),
            success: true,
            step,
            tokens,
            eval_batches: 0,
            loss: final_loss,
            ppl: final_loss.map(|value| value.exp()),
            elapsed_ms,
            tokens_per_sec,
            peak_memory_bytes_est,
            checkpoint_dir: config.checkpoint_dir.clone(),
            world_size: config.world_size,
            rank: config.rank,
            networked_rendezvous,
            networked_gradient_exchange: networked_rendezvous,
            networked_gradient_bytes,
            dist_retry_count,
            fault_injection_observed,
            latest_checkpoint_path,
            checkpoint_bytes: Some(directory_size_bytes(checkpoint_root)?),
            dataset_path: config.dataset_path.clone(),
            eval_dataset_path: config.eval_dataset_path.clone(),
            config_hash: config.config_hash.clone(),
            error_code: None,
            error_message: None,
        },
    )?;
    Ok(())
}

fn execute_enkai_accel_eval(config: &CpuRuntimeConfig) -> Result<(), String> {
    let peak_memory_bytes_est = enforce_oom_budget(config)?;
    ensure_runtime_state(config)?;
    let tokenizer = build_tokenizer(&config.tokenizer)?;
    let dataset_path = config
        .eval_dataset_path
        .as_ref()
        .unwrap_or(&config.dataset_path);
    let dataset_paths = shard_paths(
        resolve_dataset_paths(dataset_path)?,
        config.world_size,
        config.rank,
    );
    let mut data_cfg = DatasetConfig::new(config.seq_len, config.batch_size);
    data_cfg.add_eos = config.add_eos;
    data_cfg.drop_remainder = config.drop_remainder;
    data_cfg.pad_id = config.pad_id;
    data_cfg.seed = Some(config.seed);
    data_cfg.prefetch_batches = config.prefetch_batches;
    let mut stream = DatasetStream::new(dataset_paths, tokenizer, data_cfg)?;

    let checkpoint_root = Path::new(&config.checkpoint_dir);
    let latest = latest_checkpoint(checkpoint_root)
        .map_err(|err| format!("E_CHECKPOINT_IO: {}", err))?
        .ok_or_else(|| "E_CHECKPOINT_MISSING: No checkpoint found".to_string())?;
    let state = load_checkpoint(&latest).map_err(|err| format!("E_CHECKPOINT_CORRUPT: {}", err))?;
    let mut model = EnkaiAccelTinyModel::new(
        config.vocab_size,
        config.hidden_size,
        config.seed,
        &config.shape,
    );
    model
        .set_params(&state.weights)
        .map_err(|err| format!("E_CHECKPOINT_CORRUPT: {}", err))?;

    let start = Instant::now();
    let mut total_loss = 0.0f32;
    let mut batches = 0usize;
    while batches < config.eval_steps.max(1) {
        let batch = match stream.next_batch()? {
            Some(batch) => batch,
            None => break,
        };
        total_loss += model.eval_loss(&batch);
        batches += 1;
    }
    if batches == 0 {
        return Err("E_EVAL_EMPTY: No eval batches produced".to_string());
    }
    let avg_loss = total_loss / batches as f32;
    let ppl = avg_loss.exp();
    println!("eval loss {:.4} ppl {:.4}", avg_loss, ppl);
    write_runtime_report(
        config,
        &TrainRuntimeReport {
            schema_version: 1,
            suite_id: config.suite_id.clone(),
            command: config.command_name.clone(),
            requested_backend: config.requested_backend.clone(),
            executed_backend: config.executed_backend.clone(),
            fallback_reason: config.fallback_reason.clone(),
            distributed_preview_mode: config.dist_preview_mode.clone(),
            kernel: "enkai_accel_v1".to_string(),
            worker_count: EnkaiAccelTinyModel::worker_count(config.batch_size * config.seq_len),
            success: true,
            step: state.meta.step as usize,
            tokens: state.meta.tokens,
            eval_batches: batches,
            loss: Some(avg_loss),
            ppl: Some(ppl),
            elapsed_ms: start.elapsed().as_millis(),
            tokens_per_sec: None,
            peak_memory_bytes_est,
            checkpoint_dir: config.checkpoint_dir.clone(),
            world_size: config.world_size,
            rank: config.rank,
            networked_rendezvous: config.dist_preview_mode == "networked-sync-preview",
            networked_gradient_exchange: false,
            networked_gradient_bytes: 0,
            dist_retry_count: 0,
            fault_injection_observed: false,
            latest_checkpoint_path: Some(latest.display().to_string()),
            checkpoint_bytes: Some(directory_size_bytes(checkpoint_root)?),
            dataset_path: config.dataset_path.clone(),
            eval_dataset_path: config.eval_dataset_path.clone(),
            config_hash: config.config_hash.clone(),
            error_code: None,
            error_message: None,
        },
    )?;
    Ok(())
}

fn parse_cpu_runtime_config(
    manifest: &TrainCommandManifest,
    requested_backend: String,
    executed_backend: String,
    fallback_reason: Option<String>,
) -> Result<CpuRuntimeConfig, String> {
    let root = manifest
        .evaluated_config
        .as_object()
        .ok_or_else(|| "evaluated_config must be an object".to_string())?;
    let tokenizer = if let Some(path) = root.get("tokenizer_path").and_then(|value| value.as_str())
    {
        CpuTokenizerConfig::Load(PathBuf::from(path))
    } else if let Some(train) = root
        .get("tokenizer_train")
        .and_then(|value| value.as_object())
    {
        let path = train
            .get("path")
            .and_then(|value| value.as_str())
            .ok_or_else(|| "tokenizer_train.path must be a string".to_string())?;
        let vocab_size = as_usize(train.get("vocab_size"), "tokenizer_train.vocab_size")?;
        let save_path = train
            .get("save_path")
            .and_then(|value| value.as_str())
            .map(PathBuf::from);
        CpuTokenizerConfig::Train {
            path: PathBuf::from(path),
            vocab_size,
            save_path,
        }
    } else {
        return Err("Config missing tokenizer_path or tokenizer_train".to_string());
    };

    let model = root.get("model").and_then(|value| value.as_object());
    let dist = root.get("dist").and_then(|value| value.as_object());
    let shape = AccelModelShape {
        preset: model
            .and_then(|value| value.get("preset"))
            .and_then(|value| value.as_str())
            .unwrap_or("enkai_accel_v1")
            .to_string(),
        n_layers: model
            .and_then(|value| value.get("n_layers"))
            .and_then(|value| value.as_u64())
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(1)
            .max(1),
        n_heads: model
            .and_then(|value| value.get("n_heads"))
            .and_then(|value| value.as_u64())
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(1)
            .max(1),
        ff_mult: model
            .and_then(|value| value.get("ff_mult"))
            .and_then(|value| value.as_f64())
            .map(|value| value as f32)
            .unwrap_or(1.0)
            .max(1.0),
        activation: model
            .and_then(|value| value.get("activation"))
            .and_then(|value| value.as_str())
            .unwrap_or("gelu")
            .to_ascii_lowercase(),
        norm: model
            .and_then(|value| value.get("norm"))
            .and_then(|value| value.as_str())
            .unwrap_or("rmsnorm")
            .to_ascii_lowercase(),
        tie_embeddings: model
            .and_then(|value| value.get("tie_embeddings"))
            .and_then(|value| value.as_bool())
            .unwrap_or(false),
    };

    Ok(CpuRuntimeConfig {
        requested_backend,
        executed_backend,
        fallback_reason,
        command_name: match manifest.command {
            TrainManifestCommand::Train => "train".to_string(),
            TrainManifestCommand::Pretrain => "pretrain".to_string(),
            TrainManifestCommand::Eval => "eval".to_string(),
        },
        suite_id: root
            .get("suite_id")
            .and_then(|value| value.as_str())
            .map(|value| value.to_string()),
        vocab_size: as_usize(root.get("vocab_size"), "vocab_size")?,
        hidden_size: as_usize(root.get("hidden_size"), "hidden_size")?,
        seq_len: as_usize(root.get("seq_len"), "seq_len")?,
        batch_size: as_usize(root.get("batch_size"), "batch_size")?,
        lr: as_f32(root.get("lr"), "lr")?,
        dataset_path: as_string(root.get("dataset_path"), "dataset_path")?,
        eval_dataset_path: root
            .get("eval_dataset_path")
            .and_then(|value| value.as_str())
            .map(|value| value.to_string()),
        checkpoint_dir: as_string(root.get("checkpoint_dir"), "checkpoint_dir")?,
        world_size: as_usize_default(root.get("world_size"), 1)?,
        rank: as_usize_default(root.get("rank"), 0)?,
        max_steps: as_usize(root.get("max_steps"), "max_steps")?,
        save_every: as_usize_default(root.get("save_every"), 100)?,
        log_every: as_usize_default(root.get("log_every"), 10)?,
        eval_steps: as_usize_default(root.get("eval_steps"), 10)?,
        drop_remainder: as_bool_default(root.get("drop_remainder"), true),
        add_eos: as_bool_default(root.get("add_eos"), true),
        pad_id: as_u32_default(root.get("pad_id"), 0)?,
        seed: as_u64_default(root.get("seed"), 1337)?,
        shuffle: as_bool_default(root.get("shuffle"), true),
        prefetch_batches: as_usize_default(root.get("prefetch_batches"), 1)?,
        oom_budget_bytes: root
            .get("oom_budget_bytes")
            .and_then(|value| value.as_u64()),
        grad_accum_steps: as_usize_default(root.get("grad_accum_steps"), 1)?,
        grad_clip_norm: root
            .get("grad_clip_norm")
            .map(|value| as_f32(Some(value), "grad_clip_norm"))
            .transpose()?,
        dist_topology: dist
            .and_then(|value| value.get("topology"))
            .and_then(|value| value.as_str())
            .unwrap_or(if as_usize_default(root.get("world_size"), 1)? > 1 {
                "single-node"
            } else {
                "standalone"
            })
            .to_ascii_lowercase(),
        dist_rendezvous: dist
            .and_then(|value| value.get("rendezvous"))
            .and_then(|value| value.as_str())
            .unwrap_or("env://")
            .to_string(),
        dist_retry_budget: dist
            .and_then(|value| value.get("retry_budget"))
            .and_then(|value| value.as_u64())
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(3),
        dist_preview_mode: dist
            .and_then(|value| value.get("preview_mode"))
            .and_then(|value| value.as_str())
            .unwrap_or("none")
            .to_ascii_lowercase(),
        shape,
        tokenizer,
        config_hash: hash_manifest_config(&manifest.evaluated_config),
    })
}

fn build_tokenizer(config: &CpuTokenizerConfig) -> Result<Tokenizer, String> {
    match config {
        CpuTokenizerConfig::Load(path) => Tokenizer::load(path),
        CpuTokenizerConfig::Train {
            path,
            vocab_size,
            save_path,
        } => {
            let cfg = TokenizerTrainConfig {
                vocab_size: *vocab_size,
                ..TokenizerTrainConfig::default()
            };
            let tok = Tokenizer::train_from_path(path, &cfg)?;
            if let Some(save_path) = save_path {
                tok.save(save_path)?;
            }
            Ok(tok)
        }
    }
}

fn model_signature(config: &CpuRuntimeConfig) -> String {
    format!(
        "{}:{}:{}:{}:{}:{}:{:.3}:{}:{}:{}:{}",
        config.shape.preset,
        config.vocab_size,
        config.hidden_size,
        config.seq_len,
        config.batch_size,
        config.shape.n_layers,
        config.shape.ff_mult,
        config.shape.n_heads,
        config.shape.activation,
        config.shape.norm,
        config.shape.tie_embeddings
    )
}

fn hash_manifest_config(value: &serde_json::Value) -> String {
    let encoded = serde_json::to_vec(value).unwrap_or_default();
    let mut hasher = Sha256::new();
    hasher.update(encoded);
    format!("{:x}", hasher.finalize())
}

fn distributed_group_id(config: &CpuRuntimeConfig) -> String {
    let mut hasher = Sha256::new();
    hasher.update(config.command_name.as_bytes());
    hasher.update(config.requested_backend.as_bytes());
    hasher.update(config.executed_backend.as_bytes());
    hasher.update(config.suite_id.clone().unwrap_or_default().as_bytes());
    hasher.update(config.dataset_path.as_bytes());
    hasher.update(config.seq_len.to_le_bytes());
    hasher.update(config.batch_size.to_le_bytes());
    hasher.update(config.hidden_size.to_le_bytes());
    hasher.update(config.vocab_size.to_le_bytes());
    hasher.update(config.world_size.to_le_bytes());
    hasher.update(config.shape.preset.as_bytes());
    hasher.update(config.shape.n_layers.to_le_bytes());
    hasher.update(config.shape.n_heads.to_le_bytes());
    hasher.update(config.shape.ff_mult.to_le_bytes());
    hasher.update(config.shape.activation.as_bytes());
    hasher.update(config.shape.norm.as_bytes());
    hasher.update([config.shape.tie_embeddings as u8]);
    hasher.update(config.dist_preview_mode.as_bytes());
    hasher.update(config.dist_topology.as_bytes());
    hasher.update(config.dist_rendezvous.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn distributed_sync_root(config: &CpuRuntimeConfig) -> PathBuf {
    Path::new(&config.checkpoint_dir)
        .parent()
        .unwrap_or_else(|| Path::new(&config.checkpoint_dir))
        .join(".dist_sync")
        .join(distributed_group_id(config))
}

fn distributed_sync_step_dir(config: &CpuRuntimeConfig, step: usize) -> PathBuf {
    distributed_sync_root(config).join(format!("step_{:08}", step))
}

fn parse_tcp_rendezvous_uri(uri: &str) -> Result<(String, u16), String> {
    let trimmed = uri.trim();
    let rest = trimmed
        .strip_prefix("tcp://")
        .ok_or_else(|| "E_DIST_RENDEZVOUS_URI: rendezvous must start with tcp://".to_string())?;
    let (host, port) = rest.rsplit_once(':').ok_or_else(|| {
        "E_DIST_RENDEZVOUS_URI: rendezvous must be tcp://<host>:<port>".to_string()
    })?;
    let port = port
        .parse::<u16>()
        .map_err(|_| "E_DIST_RENDEZVOUS_URI: invalid tcp rendezvous port".to_string())?;
    Ok((host.to_string(), port))
}

fn phase_port(base_port: u16, step: usize, phase: &str) -> u16 {
    let phase_index = match phase {
        "pre-sync" => 0u16,
        "post-sync" => 1u16,
        _ => 2u16,
    };
    base_port
        .saturating_add((step as u16).saturating_mul(4))
        .saturating_add(phase_index)
}

fn fault_listener_delay_ms(config: &CpuRuntimeConfig, step: usize, phase: &str) -> Option<u64> {
    let mode = env::var("ENKAI_DIST_FAULT_MODE").ok()?.to_ascii_lowercase();
    if mode != "listener-delay-once" {
        return None;
    }
    let target_rank = env::var("ENKAI_DIST_FAULT_RANK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let target_step = env::var("ENKAI_DIST_FAULT_STEP")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1);
    let target_phase =
        env::var("ENKAI_DIST_FAULT_PHASE").unwrap_or_else(|_| "pre-sync".to_string());
    let delay_ms = env::var("ENKAI_DIST_FAULT_DELAY_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(250);
    if config.rank == target_rank
        && step == target_step
        && phase == target_phase
        && !NETWORKED_RENDEZVOUS_FAULT_USED.swap(true, Ordering::SeqCst)
    {
        Some(delay_ms)
    } else {
        None
    }
}

fn should_force_rendezvous_retry(config: &CpuRuntimeConfig, step: usize, phase: &str) -> bool {
    let mode = match env::var("ENKAI_DIST_FAULT_MODE") {
        Ok(value) => value.to_ascii_lowercase(),
        Err(_) => return false,
    };
    if mode != "listener-delay-once" {
        return false;
    }
    let target_rank = env::var("ENKAI_DIST_FAULT_RANK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let target_step = env::var("ENKAI_DIST_FAULT_STEP")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1);
    let target_phase =
        env::var("ENKAI_DIST_FAULT_PHASE").unwrap_or_else(|_| "pre-sync".to_string());
    config.rank != target_rank && step == target_step && phase == target_phase
}

fn write_json_line<T: Serialize>(stream: &mut TcpStream, payload: &T) -> Result<(), String> {
    let text =
        serde_json::to_string(payload).map_err(|err| format!("E_DIST_RENDEZVOUS_JSON: {}", err))?;
    stream
        .write_all(text.as_bytes())
        .map_err(|err| format!("E_DIST_RENDEZVOUS_IO: {}", err))?;
    stream
        .write_all(b"\n")
        .map_err(|err| format!("E_DIST_RENDEZVOUS_IO: {}", err))?;
    stream
        .flush()
        .map_err(|err| format!("E_DIST_RENDEZVOUS_IO: {}", err))
}

fn read_json_line<T: for<'de> Deserialize<'de>>(stream: &TcpStream) -> Result<T, String> {
    let mut reader = BufReader::new(
        stream
            .try_clone()
            .map_err(|err| format!("E_DIST_RENDEZVOUS_IO: {}", err))?,
    );
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .map_err(|err| format!("E_DIST_RENDEZVOUS_IO: {}", err))?;
    serde_json::from_str(line.trim_end())
        .map_err(|err| format!("E_DIST_RENDEZVOUS_CORRUPT: {}", err))
}

fn networked_barrier(
    config: &CpuRuntimeConfig,
    step: usize,
    phase: &str,
    retry_count: &mut usize,
    fault_injection_observed: &mut bool,
) -> Result<(), String> {
    let (host, base_port) = parse_tcp_rendezvous_uri(&config.dist_rendezvous)?;
    let port = phase_port(base_port, step, phase);
    let group_id = distributed_group_id(config);
    let payload = RendezvousBarrierPayload {
        schema_version: 1,
        step,
        phase: phase.to_string(),
        rank: config.rank,
        world_size: config.world_size,
        group_id: group_id.clone(),
    };

    if config.rank == 0 {
        if let Some(delay_ms) = fault_listener_delay_ms(config, step, phase) {
            *fault_injection_observed = true;
            std::thread::sleep(std::time::Duration::from_millis(delay_ms));
        }
        let listener = TcpListener::bind((host.as_str(), port))
            .map_err(|err| format!("E_DIST_RENDEZVOUS_BIND: {}", err))?;
        listener
            .set_nonblocking(false)
            .map_err(|err| format!("E_DIST_RENDEZVOUS_IO: {}", err))?;
        let mut peers = Vec::with_capacity(config.world_size.saturating_sub(1));
        let mut peer_ranks = BTreeSet::new();
        let mut group_mismatch = false;
        for _ in 1..config.world_size {
            let (stream, _) = listener
                .accept()
                .map_err(|err| format!("E_DIST_RENDEZVOUS_ACCEPT: {}", err))?;
            stream
                .set_read_timeout(Some(std::time::Duration::from_secs(30)))
                .map_err(|err| format!("E_DIST_RENDEZVOUS_IO: {}", err))?;
            stream
                .set_write_timeout(Some(std::time::Duration::from_secs(30)))
                .map_err(|err| format!("E_DIST_RENDEZVOUS_IO: {}", err))?;
            let peer_payload: RendezvousBarrierPayload = read_json_line(&stream)?;
            if peer_payload.step != step
                || peer_payload.phase != phase
                || peer_payload.world_size != config.world_size
            {
                return Err(
                    "E_DIST_RENDEZVOUS_MISMATCH: barrier payload metadata mismatch".to_string(),
                );
            }
            if peer_payload.rank == 0 || peer_payload.rank >= config.world_size {
                return Err(
                    "E_DIST_RENDEZVOUS_MISMATCH: barrier peer rank out of range".to_string(),
                );
            }
            if !peer_ranks.insert(peer_payload.rank) {
                return Err(
                    "E_DIST_RENDEZVOUS_MISMATCH: duplicate barrier peer rank".to_string(),
                );
            }
            if peer_payload.group_id != group_id {
                group_mismatch = true;
            }
            peers.push(stream);
        }
        if group_mismatch {
            return Err(
                "E_DIST_RENDEZVOUS_MISMATCH: barrier payload metadata mismatch".to_string(),
            );
        }
        let ack = RendezvousBarrierAck {
            schema_version: 1,
            step,
            phase: phase.to_string(),
            world_size: config.world_size,
            group_id,
            ok: true,
        };
        for stream in &mut peers {
            write_json_line(stream, &ack)?;
        }
        Ok(())
    } else {
        let mut last_err = None;
        for attempt in 0..=config.dist_retry_budget {
            if attempt == 0 && should_force_rendezvous_retry(config, step, phase) {
                last_err = Some("fault-injected preconnect retry".to_string());
                if attempt < config.dist_retry_budget {
                    *retry_count += 1;
                    std::thread::sleep(std::time::Duration::from_millis(50));
                    continue;
                }
            }
            match TcpStream::connect((host.as_str(), port)) {
                Ok(mut stream) => {
                    stream
                        .set_read_timeout(Some(std::time::Duration::from_secs(30)))
                        .map_err(|err| format!("E_DIST_RENDEZVOUS_IO: {}", err))?;
                    stream
                        .set_write_timeout(Some(std::time::Duration::from_secs(30)))
                        .map_err(|err| format!("E_DIST_RENDEZVOUS_IO: {}", err))?;
                    write_json_line(&mut stream, &payload)?;
                    let ack: RendezvousBarrierAck = read_json_line(&stream)?;
                    if ack.step != step
                        || ack.phase != phase
                        || ack.world_size != config.world_size
                        || ack.group_id != group_id
                        || !ack.ok
                    {
                        return Err("E_DIST_RENDEZVOUS_MISMATCH: invalid barrier ack".to_string());
                    }
                    return Ok(());
                }
                Err(err) => {
                    last_err = Some(err.to_string());
                    if attempt < config.dist_retry_budget {
                        *retry_count += 1;
                        std::thread::sleep(std::time::Duration::from_millis(
                            50 * (attempt as u64 + 1),
                        ));
                        continue;
                    }
                }
            }
        }
        Err(format!(
            "E_DIST_RENDEZVOUS_CONNECT: failed to join barrier after {} retries: {}",
            config.dist_retry_budget,
            last_err.unwrap_or_else(|| "unknown".to_string())
        ))
    }
}

fn write_json_file<T: Serialize>(path: &Path, payload: &T) -> Result<(), String> {
    let text = serde_json::to_string_pretty(payload).map_err(|err| err.to_string())?;
    fs::write(path, text).map_err(|err| format!("E_DIST_SYNC_IO: {}", err))
}

fn read_json_file<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, String> {
    let text = fs::read_to_string(path).map_err(|err| format!("E_DIST_SYNC_IO: {}", err))?;
    serde_json::from_str(&text).map_err(|err| format!("E_DIST_SYNC_CORRUPT: {}", err))
}

fn wait_for_path(path: &Path, timeout_ms: u64) -> Result<(), String> {
    let start = Instant::now();
    while !path.exists() {
        if start.elapsed().as_millis() as u64 > timeout_ms {
            return Err(format!(
                "E_DIST_SYNC_TIMEOUT: timed out waiting for {}",
                path.display()
            ));
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    Ok(())
}

fn average_gradient_vectors(vectors: &[Vec<f32>]) -> Result<Vec<f32>, String> {
    let first = vectors
        .first()
        .ok_or_else(|| "E_DIST_SYNC_EMPTY: no gradient vectors supplied".to_string())?;
    let len = first.len();
    let mut merged = vec![0.0f32; len];
    for vector in vectors {
        if vector.len() != len {
            return Err("E_DIST_SYNC_SHAPE: gradient vector length mismatch".to_string());
        }
        for (dst, src) in merged.iter_mut().zip(vector.iter()) {
            *dst += *src;
        }
    }
    let scale = 1.0 / vectors.len().max(1) as f32;
    for value in &mut merged {
        *value *= scale;
    }
    Ok(merged)
}

fn synchronize_preview_gradients(
    config: &CpuRuntimeConfig,
    step: usize,
    grads: &[f32],
) -> Result<Vec<f32>, String> {
    let group_id = distributed_group_id(config);
    let step_dir = distributed_sync_step_dir(config, step);
    fs::create_dir_all(&step_dir).map_err(|err| format!("E_DIST_SYNC_IO: {}", err))?;
    let rank_path = step_dir.join(format!("rank{:02}.json", config.rank));
    let payload = GradSyncPayload {
        schema_version: 1,
        step,
        rank: config.rank,
        world_size: config.world_size,
        config_hash: group_id.clone(),
        grads: grads.to_vec(),
    };
    write_json_file(&rank_path, &payload)?;
    let merged_path = step_dir.join("merged.json");

    if config.rank == 0 {
        let mut vectors = Vec::with_capacity(config.world_size);
        for rank in 0..config.world_size {
            let path = step_dir.join(format!("rank{:02}.json", rank));
            wait_for_path(&path, 30_000)?;
            let payload: GradSyncPayload = read_json_file(&path)?;
            if payload.config_hash != group_id
                || payload.step != step
                || payload.world_size != config.world_size
            {
                return Err("E_DIST_SYNC_MISMATCH: gradient payload metadata mismatch".to_string());
            }
            vectors.push(payload.grads);
        }
        let merged = average_gradient_vectors(&vectors)?;
        let merged_payload = MergedGradPayload {
            schema_version: 1,
            step,
            world_size: config.world_size,
            config_hash: group_id.clone(),
            grads: merged,
        };
        write_json_file(&merged_path, &merged_payload)?;
    }

    wait_for_path(&merged_path, 30_000)?;
    let merged_payload: MergedGradPayload = read_json_file(&merged_path)?;
    if merged_payload.config_hash != group_id
        || merged_payload.step != step
        || merged_payload.world_size != config.world_size
    {
        return Err("E_DIST_SYNC_MISMATCH: merged gradient payload metadata mismatch".to_string());
    }
    Ok(merged_payload.grads)
}

fn networked_grad_timeout_ms() -> u64 {
    env::var("ENKAI_DIST_GRAD_TIMEOUT_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(30_000)
}

fn networked_grad_fault_mode(config: &CpuRuntimeConfig, step: usize) -> Option<String> {
    let mode = env::var("ENKAI_DIST_GRAD_FAULT_MODE")
        .ok()?
        .to_ascii_lowercase();
    let target_rank = env::var("ENKAI_DIST_GRAD_FAULT_RANK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1);
    let target_step = env::var("ENKAI_DIST_GRAD_FAULT_STEP")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1);
    if config.rank == target_rank && step == target_step {
        Some(mode)
    } else {
        None
    }
}

fn json_payload_bytes<T: Serialize>(payload: &T) -> Result<u64, String> {
    serde_json::to_vec(payload)
        .map(|bytes| bytes.len() as u64)
        .map_err(|err| format!("E_DIST_SYNC_JSON: {}", err))
}

fn connect_networked_grad_peer(host: &str, port: u16, timeout_ms: u64) -> Result<TcpStream, String> {
    let start = Instant::now();
    let mut last_err = None;
    while start.elapsed().as_millis() as u64 <= timeout_ms {
        match TcpStream::connect((host, port)) {
            Ok(stream) => {
                stream
                    .set_read_timeout(Some(std::time::Duration::from_millis(timeout_ms.max(1))))
                    .map_err(|err| format!("E_DIST_SYNC_IO: {}", err))?;
                stream
                    .set_write_timeout(Some(std::time::Duration::from_millis(timeout_ms.max(1))))
                    .map_err(|err| format!("E_DIST_SYNC_IO: {}", err))?;
                return Ok(stream);
            }
            Err(err) => {
                last_err = Some(err.to_string());
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }
    }
    Err(format!(
        "E_DIST_SYNC_TIMEOUT: timed out connecting to gradient exchange after {} ms: {}",
        timeout_ms,
        last_err.unwrap_or_else(|| "unknown".to_string())
    ))
}

fn synchronize_networked_gradients(
    config: &CpuRuntimeConfig,
    step: usize,
    grads: &[f32],
) -> Result<(Vec<f32>, u64), String> {
    let (host, base_port) = parse_tcp_rendezvous_uri(&config.dist_rendezvous)?;
    let port = phase_port(base_port, step, "grad-sync");
    let group_id = distributed_group_id(config);
    let timeout_ms = networked_grad_timeout_ms();
    let mut payload = GradSyncPayload {
        schema_version: 1,
        step,
        rank: config.rank,
        world_size: config.world_size,
        config_hash: group_id.clone(),
        grads: grads.to_vec(),
    };

    if config.rank == 0 {
        let listener = TcpListener::bind((host.as_str(), port))
            .map_err(|err| format!("E_DIST_SYNC_BIND: {}", err))?;
        listener
            .set_nonblocking(true)
            .map_err(|err| format!("E_DIST_SYNC_IO: {}", err))?;
        let mut peers = Vec::with_capacity(config.world_size.saturating_sub(1));
        let mut peer_ranks = BTreeSet::from([0usize]);
        let mut vectors = vec![payload.grads.clone()];
        let mut exchanged_bytes = json_payload_bytes(&payload)?;
        let start = Instant::now();
        while peer_ranks.len() < config.world_size {
            match listener.accept() {
                Ok((stream, _)) => {
                    stream
                        .set_nonblocking(false)
                        .map_err(|err| format!("E_DIST_SYNC_IO: {}", err))?;
                    stream
                        .set_read_timeout(Some(std::time::Duration::from_millis(timeout_ms.max(1))))
                        .map_err(|err| format!("E_DIST_SYNC_IO: {}", err))?;
                    stream
                        .set_write_timeout(Some(std::time::Duration::from_millis(timeout_ms.max(1))))
                        .map_err(|err| format!("E_DIST_SYNC_IO: {}", err))?;
                    let peer_payload: GradSyncPayload = read_json_line(&stream)?;
                    exchanged_bytes = exchanged_bytes.saturating_add(json_payload_bytes(&peer_payload)?);
                    if peer_payload.rank == 0 || peer_payload.rank >= config.world_size {
                        return Err(
                            "E_DIST_SYNC_MISMATCH: gradient peer rank out of range".to_string(),
                        );
                    }
                    if !peer_ranks.insert(peer_payload.rank) {
                        return Err(
                            "E_DIST_SYNC_MISMATCH: duplicate gradient peer rank".to_string(),
                        );
                    }
                    if peer_payload.config_hash != group_id
                        || peer_payload.step != step
                        || peer_payload.world_size != config.world_size
                    {
                        return Err(
                            "E_DIST_SYNC_MISMATCH: gradient payload metadata mismatch".to_string(),
                        );
                    }
                    vectors.push(peer_payload.grads);
                    peers.push(stream);
                }
                Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                    if start.elapsed().as_millis() as u64 > timeout_ms {
                        return Err(format!(
                            "E_DIST_SYNC_TIMEOUT: gradient aggregation timed out waiting for peers after {} ms",
                            timeout_ms
                        ));
                    }
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                Err(err) => return Err(format!("E_DIST_SYNC_ACCEPT: {}", err)),
            }
        }
        let merged = average_gradient_vectors(&vectors)?;
        let merged_payload = MergedGradPayload {
            schema_version: 1,
            step,
            world_size: config.world_size,
            config_hash: group_id,
            grads: merged.clone(),
        };
        let merged_bytes = json_payload_bytes(&merged_payload)?;
        for stream in &mut peers {
            write_json_line(stream, &merged_payload)?;
            exchanged_bytes = exchanged_bytes.saturating_add(merged_bytes);
        }
        Ok((merged, exchanged_bytes))
    } else {
        if let Some(mode) = networked_grad_fault_mode(config, step) {
            match mode.as_str() {
                "timeout" => {
                    return Err(
                        "E_DIST_SYNC_TIMEOUT: fault-injected peer skipped gradient exchange"
                            .to_string(),
                    );
                }
                "stale-step" => {
                    payload.step = payload.step.saturating_sub(1);
                }
                "wrong-tensor-length" => {
                    payload.grads.pop();
                }
                "duplicate-rank" => {
                    payload.rank = 1;
                }
                "peer-disconnect" => {
                    let stream = connect_networked_grad_peer(&host, port, timeout_ms)?;
                    let _ = stream.shutdown(Shutdown::Both);
                    return Err(
                        "E_DIST_SYNC_PEER_DISCONNECT: fault-injected peer disconnected during gradient exchange"
                            .to_string(),
                    );
                }
                _ => {}
            }
        }
        let mut stream = connect_networked_grad_peer(&host, port, timeout_ms)?;
        let mut exchanged_bytes = json_payload_bytes(&payload)?;
        write_json_line(&mut stream, &payload)?;
        let merged_payload: MergedGradPayload = read_json_line(&stream)?;
        exchanged_bytes = exchanged_bytes.saturating_add(json_payload_bytes(&merged_payload)?);
        if merged_payload.config_hash != group_id
            || merged_payload.step != step
            || merged_payload.world_size != config.world_size
        {
            return Err("E_DIST_SYNC_MISMATCH: merged gradient payload metadata mismatch".to_string());
        }
        if merged_payload.grads.len() != grads.len() {
            return Err("E_DIST_SYNC_SHAPE: merged gradient vector length mismatch".to_string());
        }
        Ok((merged_payload.grads, exchanged_bytes))
    }
}

fn as_string(value: Option<&serde_json::Value>, key: &str) -> Result<String, String> {
    value
        .and_then(|value| value.as_str())
        .map(|value| value.to_string())
        .ok_or_else(|| format!("{} must be a string", key))
}

fn as_usize(value: Option<&serde_json::Value>, key: &str) -> Result<usize, String> {
    value
        .and_then(|value| value.as_u64())
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| format!("{} must be a non-negative integer", key))
}

fn as_usize_default(value: Option<&serde_json::Value>, default: usize) -> Result<usize, String> {
    match value {
        Some(value) => as_usize(Some(value), "numeric field"),
        None => Ok(default),
    }
}

fn as_u32_default(value: Option<&serde_json::Value>, default: u32) -> Result<u32, String> {
    match value.and_then(|value| value.as_u64()) {
        Some(value) => u32::try_from(value).map_err(|_| "pad_id out of range".to_string()),
        None => Ok(default),
    }
}

fn as_u64_default(value: Option<&serde_json::Value>, default: u64) -> Result<u64, String> {
    match value.and_then(|value| value.as_u64()) {
        Some(value) => Ok(value),
        None => Ok(default),
    }
}

fn as_f32(value: Option<&serde_json::Value>, key: &str) -> Result<f32, String> {
    value
        .and_then(|value| value.as_f64())
        .map(|value| value as f32)
        .ok_or_else(|| format!("{} must be numeric", key))
}

fn as_bool_default(value: Option<&serde_json::Value>, default: bool) -> bool {
    value.and_then(|value| value.as_bool()).unwrap_or(default)
}

impl CpuAdamW {
    fn new(param_len: usize, lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            m: vec![0.0; param_len],
            v: vec![0.0; param_len],
            beta1_pow: 1.0,
            beta2_pow: 1.0,
        }
    }

    fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        self.beta1_pow *= self.beta1;
        self.beta2_pow *= self.beta2;
        let bias1 = 1.0 - self.beta1_pow;
        let bias2 = 1.0 - self.beta2_pow;
        for i in 0..params.len() {
            let grad = grads[i];
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grad * grad;
            let m_hat = self.m[i] / bias1;
            let v_hat = self.v[i] / bias2;
            params[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

impl CpuAdamWMulti {
    fn new(group_sizes: &[usize], lr: f32) -> Self {
        let groups = group_sizes
            .iter()
            .map(|size| CpuAdamW::new(*size, lr))
            .collect();
        Self { groups }
    }

    fn step(&mut self, params: &mut [f32], grads: &[f32], groups: &[CpuParamGroup]) {
        for (group, opt) in groups.iter().zip(self.groups.iter_mut()) {
            let start = group.start;
            let end = group.start + group.len;
            opt.step(&mut params[start..end], &grads[start..end]);
        }
    }

    fn state_vec(&self) -> Vec<f32> {
        let total: usize = self.groups.iter().map(|group| group.m.len()).sum();
        let mut out = Vec::with_capacity(total * 2);
        for group in &self.groups {
            out.extend_from_slice(&group.m);
        }
        for group in &self.groups {
            out.extend_from_slice(&group.v);
        }
        out
    }

    fn load_state(&mut self, state: &[f32]) -> Result<(), String> {
        let total: usize = self.groups.iter().map(|group| group.m.len()).sum();
        if state.len() != total * 2 {
            return Err("Optimizer state size mismatch".to_string());
        }
        let (m_slice, v_slice) = state.split_at(total);
        let mut offset = 0usize;
        for group in &mut self.groups {
            let len = group.m.len();
            group.m.copy_from_slice(&m_slice[offset..offset + len]);
            group.v.copy_from_slice(&v_slice[offset..offset + len]);
            offset += len;
        }
        Ok(())
    }
}

impl CpuTinyModel {
    fn new(vocab_size: usize, hidden_size: usize, seed: u64) -> Self {
        let embed_len = vocab_size * hidden_size;
        let out_len = hidden_size * vocab_size;
        let bias_len = vocab_size;
        let total = embed_len + out_len + bias_len;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut params = Vec::with_capacity(total);
        for _ in 0..total {
            params.push(rng.gen_range(-0.02f32..0.02f32));
        }
        Self {
            vocab_size,
            hidden_size,
            params,
            embed_offset: 0,
            out_offset: embed_len,
            bias_offset: embed_len + out_len,
        }
    }

    fn param_groups(&self) -> Vec<CpuParamGroup> {
        let embed_len = self.out_offset;
        let out_len = self.bias_offset - self.out_offset;
        vec![
            CpuParamGroup {
                start: self.embed_offset,
                len: embed_len,
            },
            CpuParamGroup {
                start: self.out_offset,
                len: out_len,
            },
            CpuParamGroup {
                start: self.bias_offset,
                len: self.vocab_size,
            },
        ]
    }

    fn params(&self) -> &[f32] {
        &self.params
    }

    fn set_params(&mut self, values: &[f32]) -> Result<(), String> {
        if values.len() != self.params.len() {
            return Err("Checkpoint weight size mismatch".to_string());
        }
        self.params.copy_from_slice(values);
        Ok(())
    }

    fn compute_grads(&self, batch: &Batch) -> (f32, Vec<f32>) {
        let total = batch.input_ids.len();
        if total == 0 {
            return (0.0, vec![0.0f32; self.params.len()]);
        }
        let mut grads = vec![0.0f32; self.params.len()];
        let mut loss = 0.0f32;
        for idx in 0..total {
            if batch.attention_mask[idx] == 0 {
                continue;
            }
            let token = batch.input_ids[idx] as usize;
            let target = batch.target_ids[idx] as usize;
            if token >= self.vocab_size || target >= self.vocab_size {
                continue;
            }
            let emb_start = self.embed_offset + token * self.hidden_size;
            let mut logits = vec![0.0f32; self.vocab_size];
            for (j, logit) in logits.iter_mut().enumerate() {
                let mut sum = self.params[self.bias_offset + j];
                for h in 0..self.hidden_size {
                    let w = self.params[self.out_offset + h * self.vocab_size + j];
                    let e = self.params[emb_start + h];
                    sum += e * w;
                }
                *logit = sum;
            }
            let max = logits.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b));
            let mut exp_sum = 0.0f32;
            for logit in &mut logits {
                *logit = (*logit - max).exp();
                exp_sum += *logit;
            }
            for logit in &mut logits {
                *logit /= exp_sum;
            }
            let prob = logits[target].max(1e-9);
            loss += -prob.ln();
            for (j, prob) in logits.iter().enumerate() {
                let grad_logit = *prob - if j == target { 1.0 } else { 0.0 };
                grads[self.bias_offset + j] += grad_logit;
                for h in 0..self.hidden_size {
                    let out_idx = self.out_offset + h * self.vocab_size + j;
                    grads[out_idx] += self.params[emb_start + h] * grad_logit;
                    grads[emb_start + h] += self.params[out_idx] * grad_logit;
                }
            }
        }
        let denom = total.max(1) as f32;
        loss /= denom;
        for grad in &mut grads {
            *grad /= denom;
        }
        (loss, grads)
    }

    fn apply_grads(&mut self, grads: &[f32], opt: &mut CpuAdamWMulti, groups: &[CpuParamGroup]) {
        opt.step(&mut self.params, grads, groups);
    }

    fn eval_loss(&self, batch: &Batch) -> f32 {
        let total = batch.input_ids.len();
        if total == 0 {
            return 0.0;
        }
        let mut loss = 0.0f32;
        for idx in 0..total {
            if batch.attention_mask[idx] == 0 {
                continue;
            }
            let token = batch.input_ids[idx] as usize;
            let target = batch.target_ids[idx] as usize;
            if token >= self.vocab_size || target >= self.vocab_size {
                continue;
            }
            let emb_start = self.embed_offset + token * self.hidden_size;
            let mut logits = vec![0.0f32; self.vocab_size];
            for (j, logit) in logits.iter_mut().enumerate() {
                let mut sum = self.params[self.bias_offset + j];
                for h in 0..self.hidden_size {
                    let w = self.params[self.out_offset + h * self.vocab_size + j];
                    let e = self.params[emb_start + h];
                    sum += e * w;
                }
                *logit = sum;
            }
            let max = logits.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b));
            let mut exp_sum = 0.0f32;
            for logit in &mut logits {
                *logit = (*logit - max).exp();
                exp_sum += *logit;
            }
            let prob = logits[target] / exp_sum;
            loss += -prob.max(1e-9).ln();
        }
        loss / total.max(1) as f32
    }
}

impl EnkaiAccelTinyModel {
    fn new(vocab_size: usize, hidden_size: usize, seed: u64, shape: &AccelModelShape) -> Self {
        let embed_len = vocab_size * hidden_size;
        let out_len = hidden_size * vocab_size;
        let bias_len = vocab_size;
        let gate_len = shape.n_layers * hidden_size;
        let total = embed_len + out_len + bias_len + gate_len * 2;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut params = Vec::with_capacity(total);
        for _ in 0..total {
            params.push(rng.gen_range(-0.02f32..0.02f32));
        }
        let mut model = Self {
            vocab_size,
            hidden_size,
            n_layers: shape.n_layers,
            n_heads: shape.n_heads,
            ff_mult: shape.ff_mult,
            activation: shape.activation.clone(),
            norm: shape.norm.clone(),
            tie_embeddings: shape.tie_embeddings,
            params,
            embed_offset: 0,
            out_offset: embed_len,
            bias_offset: embed_len + out_len,
            gate_in_offset: embed_len + out_len + bias_len,
            gate_out_offset: embed_len + out_len + bias_len + gate_len,
            out_transposed: vec![0.0f32; out_len],
        };
        model.sync_out_transposed();
        model
    }

    fn sync_out_transposed(&mut self) {
        for h in 0..self.hidden_size {
            for j in 0..self.vocab_size {
                self.out_transposed[j * self.hidden_size + h] =
                    self.params[self.out_offset + h * self.vocab_size + j];
            }
        }
    }

    fn param_groups(&self) -> Vec<CpuParamGroup> {
        let embed_len = self.out_offset;
        let out_len = self.bias_offset - self.out_offset;
        let gate_len = self.n_layers * self.hidden_size;
        vec![
            CpuParamGroup {
                start: self.embed_offset,
                len: embed_len,
            },
            CpuParamGroup {
                start: self.out_offset,
                len: out_len,
            },
            CpuParamGroup {
                start: self.bias_offset,
                len: self.vocab_size,
            },
            CpuParamGroup {
                start: self.gate_in_offset,
                len: gate_len,
            },
            CpuParamGroup {
                start: self.gate_out_offset,
                len: gate_len,
            },
        ]
    }

    fn params(&self) -> &[f32] {
        &self.params
    }

    fn set_params(&mut self, values: &[f32]) -> Result<(), String> {
        if values.len() != self.params.len() {
            return Err("Checkpoint weight size mismatch".to_string());
        }
        self.params.copy_from_slice(values);
        self.sync_out_transposed();
        Ok(())
    }

    fn project_logits_from_hidden(&self, hidden: &[f32], logits: &mut [f32]) {
        logits.copy_from_slice(&self.params[self.bias_offset..self.bias_offset + self.vocab_size]);
        for (j, logit) in logits.iter_mut().enumerate() {
            let row = if self.shape_ties_embeddings() {
                &self.params[self.embed_offset + j * self.hidden_size
                    ..self.embed_offset + (j + 1) * self.hidden_size]
            } else {
                &self.out_transposed[j * self.hidden_size..(j + 1) * self.hidden_size]
            };
            let mut sum = *logit;
            for h in 0..self.hidden_size {
                sum += hidden[h] * row[h];
            }
            *logit = sum;
        }
    }

    fn shape_ties_embeddings(&self) -> bool {
        self.tie_embeddings
    }

    fn activation_value(&self, x: f32) -> f32 {
        match self.activation.as_str() {
            "relu" => x.max(0.0),
            "silu" => {
                let sig = 1.0 / (1.0 + (-x).exp());
                x * sig
            }
            "tanh" => x.tanh(),
            _ => {
                let c = 0.7978846 * (x + 0.044715 * x * x * x);
                0.5 * x * (1.0 + c.tanh())
            }
        }
    }

    fn activation_derivative(&self, x: f32) -> f32 {
        match self.activation.as_str() {
            "relu" => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            "silu" => {
                let sig = 1.0 / (1.0 + (-x).exp());
                sig * (1.0 + x * (1.0 - sig))
            }
            "tanh" => {
                let y = x.tanh();
                1.0 - y * y
            }
            _ => {
                let x2 = x * x;
                let inner = 0.7978846 * (x + 0.044715 * x * x2);
                let tanh_inner = inner.tanh();
                0.5 * (1.0 + tanh_inner)
                    + 0.5 * x * (1.0 - tanh_inner * tanh_inner) * 0.7978846 * (1.0 + 0.134145 * x2)
            }
        }
    }

    fn norm_scale(&self, hidden: &[f32]) -> f32 {
        if self.norm == "none" {
            return 1.0;
        }
        let mut sum_sq = 0.0f32;
        for value in hidden {
            sum_sq += value * value;
        }
        (sum_sq / hidden.len().max(1) as f32 + 1e-6).sqrt()
    }

    fn forward_hidden(&self, emb_start: usize) -> EnkaiAccelForwardCache {
        let mut current = self.params[emb_start..emb_start + self.hidden_size].to_vec();
        let mut residuals = Vec::with_capacity(self.n_layers);
        let mut normalized = Vec::with_capacity(self.n_layers);
        let mut scales = Vec::with_capacity(self.n_layers);
        let head_scale = (self.n_heads as f32).sqrt().max(1.0);
        for layer in 0..self.n_layers {
            residuals.push(current.clone());
            let scale = self.norm_scale(&current);
            scales.push(scale);
            let mut normed = vec![0.0f32; self.hidden_size];
            for h in 0..self.hidden_size {
                let gate_in = self.params[self.gate_in_offset + layer * self.hidden_size + h];
                let gate_out = self.params[self.gate_out_offset + layer * self.hidden_size + h];
                let base = current[h] / scale / head_scale;
                normed[h] = base;
                let activated = self.activation_value(base * gate_in * self.ff_mult);
                current[h] = residuals[layer][h] + activated * gate_out;
            }
            normalized.push(normed);
        }
        EnkaiAccelForwardCache {
            hidden: current,
            residuals,
            normalized,
            scales,
        }
    }

    fn worker_count(total_positions: usize) -> usize {
        let available = std::thread::available_parallelism()
            .map(|value| value.get())
            .unwrap_or(1);
        let suggested = (total_positions / 32).max(1);
        available.min(suggested).max(1)
    }

    fn compute_grads_chunk(
        &self,
        batch: &Batch,
        start: usize,
        end: usize,
    ) -> (f32, Vec<f32>, usize) {
        let mut grads = vec![0.0f32; self.params.len()];
        let mut logits = vec![0.0f32; self.vocab_size];
        let mut loss = 0.0f32;
        let mut active = 0usize;
        for idx in start..end {
            if batch.attention_mask[idx] == 0 {
                continue;
            }
            let token = batch.input_ids[idx] as usize;
            let target = batch.target_ids[idx] as usize;
            if token >= self.vocab_size || target >= self.vocab_size {
                continue;
            }
            let emb_start = self.embed_offset + token * self.hidden_size;
            let cache = self.forward_hidden(emb_start);
            self.project_logits_from_hidden(&cache.hidden, &mut logits);
            let max = logits.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b));
            let mut exp_sum = 0.0f32;
            for logit in &mut logits {
                *logit = (*logit - max).exp();
                exp_sum += *logit;
            }
            for logit in &mut logits {
                *logit /= exp_sum;
            }
            let prob = logits[target].max(1e-9);
            loss += -prob.ln();
            active += 1;
            let mut hidden_grads = vec![0.0f32; self.hidden_size];
            for (j, prob) in logits.iter().enumerate() {
                let grad_logit = *prob - if j == target { 1.0 } else { 0.0 };
                grads[self.bias_offset + j] += grad_logit;
                for h in 0..self.hidden_size {
                    if self.shape_ties_embeddings() {
                        let embed_idx = self.embed_offset + j * self.hidden_size + h;
                        grads[embed_idx] += cache.hidden[h] * grad_logit;
                        hidden_grads[h] += self.params[embed_idx] * grad_logit;
                    } else {
                        grads[self.out_offset + h * self.vocab_size + j] +=
                            cache.hidden[h] * grad_logit;
                        hidden_grads[h] +=
                            self.out_transposed[j * self.hidden_size + h] * grad_logit;
                    }
                }
            }
            let head_scale = (self.n_heads as f32).sqrt().max(1.0);
            for layer in (0..self.n_layers).rev() {
                let mut next_hidden = cache.residuals[layer].clone();
                for h in 0..self.hidden_size {
                    let gate_in_idx = self.gate_in_offset + layer * self.hidden_size + h;
                    let gate_out_idx = self.gate_out_offset + layer * self.hidden_size + h;
                    let gate_in = self.params[gate_in_idx];
                    let gate_out = self.params[gate_out_idx];
                    let normed = cache.normalized[layer][h];
                    let act_input = normed * gate_in * self.ff_mult;
                    let activated = self.activation_value(act_input);
                    let d_out = hidden_grads[h];
                    grads[gate_out_idx] += d_out * activated;
                    let d_activated = d_out * gate_out;
                    let d_act_input = d_activated * self.activation_derivative(act_input);
                    grads[gate_in_idx] += d_act_input * normed * self.ff_mult;
                    next_hidden[h] +=
                        d_act_input * gate_in * self.ff_mult / cache.scales[layer] / head_scale;
                }
                hidden_grads = next_hidden;
            }
            for h in 0..self.hidden_size {
                grads[emb_start + h] += hidden_grads[h];
            }
        }
        (loss, grads, active)
    }

    fn compute_grads(&self, batch: &Batch) -> (f32, Vec<f32>) {
        let total = batch.input_ids.len();
        if total == 0 {
            return (0.0, vec![0.0f32; self.params.len()]);
        }
        let workers = Self::worker_count(total);
        let chunk_len = total.div_ceil(workers);
        let mut partials = Vec::with_capacity(workers);
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(workers);
            for chunk_idx in 0..workers {
                let start = chunk_idx * chunk_len;
                if start >= total {
                    break;
                }
                let end = ((chunk_idx + 1) * chunk_len).min(total);
                handles.push(scope.spawn(move || self.compute_grads_chunk(batch, start, end)));
            }
            for handle in handles {
                partials.push(handle.join().expect("enkai_accel worker thread"));
            }
        });
        let mut grads = vec![0.0f32; self.params.len()];
        let mut loss = 0.0f32;
        let mut active = 0usize;
        for (partial_loss, partial_grads, partial_active) in partials {
            loss += partial_loss;
            active += partial_active;
            for (dst, src) in grads.iter_mut().zip(partial_grads.iter()) {
                *dst += *src;
            }
        }
        let denom = active.max(1) as f32;
        loss /= denom;
        for grad in &mut grads {
            *grad /= denom;
        }
        (loss, grads)
    }

    fn apply_grads(&mut self, grads: &[f32], opt: &mut CpuAdamWMulti, groups: &[CpuParamGroup]) {
        opt.step(&mut self.params, grads, groups);
        self.sync_out_transposed();
    }

    fn eval_loss(&self, batch: &Batch) -> f32 {
        let total = batch.input_ids.len();
        if total == 0 {
            return 0.0;
        }
        let workers = Self::worker_count(total);
        let chunk_len = total.div_ceil(workers);
        let mut partials = Vec::with_capacity(workers);
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(workers);
            for chunk_idx in 0..workers {
                let start = chunk_idx * chunk_len;
                if start >= total {
                    break;
                }
                let end = ((chunk_idx + 1) * chunk_len).min(total);
                handles.push(scope.spawn(move || {
                    let mut logits = vec![0.0f32; self.vocab_size];
                    let mut loss = 0.0f32;
                    let mut active = 0usize;
                    for idx in start..end {
                        if batch.attention_mask[idx] == 0 {
                            continue;
                        }
                        let token = batch.input_ids[idx] as usize;
                        let target = batch.target_ids[idx] as usize;
                        if token >= self.vocab_size || target >= self.vocab_size {
                            continue;
                        }
                        let emb_start = self.embed_offset + token * self.hidden_size;
                        let cache = self.forward_hidden(emb_start);
                        self.project_logits_from_hidden(&cache.hidden, &mut logits);
                        let max = logits.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b));
                        let mut exp_sum = 0.0f32;
                        for logit in &mut logits {
                            *logit = (*logit - max).exp();
                            exp_sum += *logit;
                        }
                        let prob = (logits[target] / exp_sum).max(1e-9);
                        loss += -prob.ln();
                        active += 1;
                    }
                    (loss, active)
                }));
            }
            for handle in handles {
                partials.push(handle.join().expect("enkai_accel eval worker thread"));
            }
        });
        let mut loss = 0.0f32;
        let mut active = 0usize;
        for (partial_loss, partial_active) in partials {
            loss += partial_loss;
            active += partial_active;
        }
        loss / active.max(1) as f32
    }
}

fn clip_grad_norm(grads: &mut [f32], max_norm: f32) -> Option<f32> {
    if !(max_norm.is_finite() && max_norm > 0.0) {
        return None;
    }
    let mut sum_sq = 0.0f32;
    for grad in grads.iter() {
        sum_sq += grad * grad;
    }
    let norm = sum_sq.sqrt();
    if norm > max_norm {
        let scale = max_norm / norm.max(1e-6);
        for grad in grads.iter_mut() {
            *grad *= scale;
        }
    }
    Some(norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn load_train_manifest_round_trips() {
        let manifest = TrainCommandManifest {
            schema_version: 1,
            profile: "train_command_manifest".to_string(),
            command: TrainManifestCommand::Eval,
            config_path: "config/train.enk".to_string(),
            strict_contracts: true,
            evaluated_config: serde_json::json!({"backend": "cpu"}),
        };
        let file = NamedTempFile::new().expect("temp file");
        std::fs::write(
            file.path(),
            serde_json::to_string_pretty(&manifest).expect("serialize"),
        )
        .expect("write manifest");
        let loaded = load_train_command_manifest(file.path()).expect("load manifest");
        assert_eq!(loaded.profile, manifest.profile);
        assert!(matches!(loaded.command, TrainManifestCommand::Eval));
        assert_eq!(loaded.evaluated_config["backend"], "cpu");
    }

    #[test]
    fn native_backend_candidates_include_root_and_deps_paths() {
        let root = PathBuf::from("C:/enkai-native-candidate-root");
        let paths = candidate_tensor_library_paths_from_roots(std::slice::from_ref(&root));
        let file_name = tensor_library_file_name();
        assert_eq!(paths[0], root.join(file_name));
        assert_eq!(paths[1], root.join("deps").join(file_name));
    }

    #[test]
    fn execution_backend_maps_enkai_accel_backend_distinctly() {
        let manifest = TrainCommandManifest {
            schema_version: 1,
            profile: "train_command_manifest".to_string(),
            command: TrainManifestCommand::Train,
            config_path: "config/train.enk".to_string(),
            strict_contracts: true,
            evaluated_config: serde_json::json!({"backend": "enkai_accel"}),
        };
        assert_eq!(
            execution_backend(&manifest),
            TrainExecutionBackend::EnkaiAccel
        );
    }

    #[test]
    fn native_backend_error_mentions_resolved_tensor_library() {
        let mut vars = BTreeMap::new();
        vars.insert(
            "ENKAI_TENSOR_PATH".to_string(),
            "C:\\tensor\\enkai_tensor.dll".to_string(),
        );
        let scope = ScopedEnv {
            vars,
            std_override: None,
        };
        let message = enrich_native_backend_error("torch backend not enabled".to_string(), &scope);
        assert!(message.contains("torch backend not enabled"));
        assert!(message.contains("C:\\tensor\\enkai_tensor.dll"));
    }

    #[test]
    fn oom_budget_guard_uses_deterministic_error_code() {
        let config = CpuRuntimeConfig {
            requested_backend: "enkai_accel".to_string(),
            executed_backend: "enkai_accel".to_string(),
            fallback_reason: None,
            command_name: "train".to_string(),
            suite_id: Some("v3_7_0_ai_runtime_foundation".to_string()),
            vocab_size: 128,
            hidden_size: 32,
            seq_len: 16,
            batch_size: 4,
            lr: 0.001,
            dataset_path: "dataset.txt".to_string(),
            eval_dataset_path: None,
            checkpoint_dir: "checkpoints".to_string(),
            world_size: 1,
            rank: 0,
            max_steps: 8,
            save_every: 8,
            log_every: 1,
            eval_steps: 1,
            drop_remainder: false,
            add_eos: true,
            pad_id: 0,
            seed: 0,
            shuffle: true,
            prefetch_batches: 1,
            oom_budget_bytes: Some(1),
            grad_accum_steps: 1,
            grad_clip_norm: None,
            dist_topology: "single-node".to_string(),
            dist_rendezvous: "env://".to_string(),
            dist_retry_budget: 0,
            dist_preview_mode: "none".to_string(),
            shape: AccelModelShape {
                preset: "enkai_accel_v1".to_string(),
                n_layers: 1,
                n_heads: 1,
                ff_mult: 1.0,
                activation: "gelu".to_string(),
                norm: "rmsnorm".to_string(),
                tie_embeddings: false,
            },
            tokenizer: CpuTokenizerConfig::Train {
                path: PathBuf::from("dataset.txt"),
                vocab_size: 128,
                save_path: None,
            },
            config_hash: "test-hash".to_string(),
        };
        let err = enforce_oom_budget(&config).expect_err("expected oom budget failure");
        assert!(err.starts_with("E_TRAIN_OOM_BUDGET:"));
    }
}
