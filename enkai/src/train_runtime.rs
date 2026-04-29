use std::collections::BTreeMap;
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;
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
    Native,
}

#[derive(Debug, Clone)]
struct CpuRuntimeConfig {
    vocab_size: usize,
    hidden_size: usize,
    seq_len: usize,
    batch_size: usize,
    lr: f32,
    dataset_path: String,
    eval_dataset_path: Option<String>,
    checkpoint_dir: String,
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
    grad_accum_steps: usize,
    grad_clip_norm: Option<f32>,
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

pub(crate) fn execution_backend(manifest: &TrainCommandManifest) -> TrainExecutionBackend {
    match manifest
        .evaluated_config
        .get("backend")
        .and_then(|value| value.as_str())
        .unwrap_or("cpu")
    {
        "native" => TrainExecutionBackend::Native,
        _ => TrainExecutionBackend::Cpu,
    }
}

pub(crate) fn execute_train_command_manifest(
    manifest: &TrainCommandManifest,
) -> Result<(), String> {
    match execution_backend(manifest) {
        TrainExecutionBackend::Cpu => execute_cpu_manifest(manifest),
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
            eprintln!(
                "[train-runtime] native backend unavailable, falling back to cpu execution: {}",
                enrich_native_backend_error(err, &env_scope)
            );
            execute_cpu_manifest(manifest)
        }
        Err(err) => Err(enrich_native_backend_error(err, &env_scope)),
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

fn execute_cpu_manifest(manifest: &TrainCommandManifest) -> Result<(), String> {
    let config = parse_cpu_runtime_config(manifest)?;
    match manifest.command {
        TrainManifestCommand::Train | TrainManifestCommand::Pretrain => execute_cpu_train(&config),
        TrainManifestCommand::Eval => execute_cpu_eval(&config),
    }
}

fn execute_cpu_train(config: &CpuRuntimeConfig) -> Result<(), String> {
    let tokenizer = build_tokenizer(&config.tokenizer)?;
    let dataset_paths = resolve_dataset_paths(&config.dataset_path)?;
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
    if let Some(latest) = latest_checkpoint(checkpoint_root).map_err(|err| err.to_string())? {
        let state = load_checkpoint(&latest).map_err(|err| err.to_string())?;
        model.set_params(&state.weights)?;
        opt.load_state(&state.optimizer)?;
        step = state.meta.step as usize;
        tokens = state.meta.tokens;
    }

    let accum_steps = config.grad_accum_steps.max(1);
    let mut grad_accum = vec![0.0f32; model.params().len()];
    let start = Instant::now();
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
                return Err("non-finite loss detected".to_string());
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
                    world_size: 1,
                    rank: 0,
                    grad_accum_steps: config.grad_accum_steps,
                    grad_clip_norm: config.grad_clip_norm.map(|value| value as f64),
                    amp: None,
                },
            };
            let path = save_checkpoint(checkpoint_root, &state).map_err(|err| err.to_string())?;
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
    }
    Ok(())
}

fn execute_cpu_eval(config: &CpuRuntimeConfig) -> Result<(), String> {
    let tokenizer = build_tokenizer(&config.tokenizer)?;
    let dataset_path = config
        .eval_dataset_path
        .as_ref()
        .unwrap_or(&config.dataset_path);
    let dataset_paths = resolve_dataset_paths(dataset_path)?;
    let mut data_cfg = DatasetConfig::new(config.seq_len, config.batch_size);
    data_cfg.add_eos = config.add_eos;
    data_cfg.drop_remainder = config.drop_remainder;
    data_cfg.pad_id = config.pad_id;
    data_cfg.seed = Some(config.seed);
    data_cfg.prefetch_batches = config.prefetch_batches;
    let mut stream = DatasetStream::new(dataset_paths, tokenizer, data_cfg)?;

    let checkpoint_root = Path::new(&config.checkpoint_dir);
    let latest = latest_checkpoint(checkpoint_root)
        .map_err(|err| err.to_string())?
        .ok_or_else(|| "No checkpoint found".to_string())?;
    let state = load_checkpoint(&latest).map_err(|err| err.to_string())?;
    let mut model = CpuTinyModel::new(config.vocab_size, config.hidden_size, config.seed);
    model.set_params(&state.weights)?;

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
        return Err("No eval batches produced".to_string());
    }
    let avg_loss = total_loss / batches as f32;
    let ppl = avg_loss.exp();
    println!("eval loss {:.4} ppl {:.4}", avg_loss, ppl);
    Ok(())
}

fn parse_cpu_runtime_config(manifest: &TrainCommandManifest) -> Result<CpuRuntimeConfig, String> {
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

    Ok(CpuRuntimeConfig {
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
        grad_accum_steps: as_usize_default(root.get("grad_accum_steps"), 1)?,
        grad_clip_norm: root
            .get("grad_clip_norm")
            .map(|value| as_f32(Some(value), "grad_clip_norm"))
            .transpose()?,
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
        "tiny:{}:{}:{}:{}",
        config.vocab_size, config.hidden_size, config.seq_len, config.batch_size
    )
}

fn hash_manifest_config(value: &serde_json::Value) -> String {
    let encoded = serde_json::to_vec(value).unwrap_or_default();
    let mut hasher = Sha256::new();
    hasher.update(encoded);
    format!("{:x}", hasher.finalize())
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
}
