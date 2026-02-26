use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::backend::Backend;
use crate::checkpoint::{latest_checkpoint, load_checkpoint as load_checkpoint_file};
use crate::dataset::Batch;
use crate::error::RuntimeError;

/// Training configuration provided by the Enkai front‑end.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    pub data: DataConfig,
    pub model: ModelConfig,
    pub optim: OptimConfig,
    pub checkpoint: CheckpointConfig,
    pub log: LogConfig,
    pub world_size: usize,
    pub rank: usize,
    pub grad_accum_steps: usize,
    pub grad_clip_norm: Option<f64>,
    pub amp: AmpConfig,
}

impl TrainConfig {
    pub fn new(data: DataConfig, checkpoint: CheckpointConfig, log: LogConfig) -> Self {
        Self {
            data,
            model: ModelConfig::default(),
            optim: OptimConfig::default(),
            checkpoint,
            log,
            world_size: 1,
            rank: 0,
            grad_accum_steps: 1,
            grad_clip_norm: None,
            amp: AmpConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimConfig {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmpConfig {
    pub enabled: bool,
    pub dtype: String,
    pub init_scale: f64,
    pub growth_factor: f64,
    pub backoff_factor: f64,
    pub growth_interval: i64,
}

impl Default for AmpConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            dtype: "fp16".to_string(),
            init_scale: 65536.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
        }
    }
}

impl Default for OptimConfig {
    fn default() -> Self {
        Self {
            lr: 3e-4,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub seed: u64,
    pub device: String,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            d_model: 512,
            n_layers: 4,
            n_heads: 8,
            seed: 0,
            device: "cpu".to_string(),
        }
    }
}

/// Dataset and batching configuration (front‑end orchestration layer).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    pub train_path: PathBuf,
    pub eval_path: Option<PathBuf>,
    pub seq_len: usize,
    pub batch_size: usize,
}

/// Checkpoint handling configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointConfig {
    pub dir: PathBuf,
    pub keep_last: usize,
}

impl CheckpointConfig {
    pub fn new(dir: PathBuf) -> Self {
        Self { dir, keep_last: 3 }
    }
}

/// Logging configuration for metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogConfig {
    pub log_dir: PathBuf,
    pub log_every: usize,
}

impl LogConfig {
    pub fn new(log_dir: PathBuf) -> Self {
        Self {
            log_dir,
            log_every: 100,
        }
    }
}

/// Minimal metrics reported per step (stubbed for now).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Metrics {
    pub step: u64,
    pub loss: f32,
    pub tokens_per_sec: f32,
    pub step_time_ms: f32,
    pub lr: f32,
    pub gpu_mem_mb: Option<f32>,
    pub gpu_util: Option<f32>,
    pub did_update: bool,
    pub accum_steps: usize,
    pub token_count: usize,
    pub forward_time_ms: f32,
    pub backward_time_ms: f32,
    pub optim_time_ms: f32,
    pub grad_norm: Option<f32>,
    pub found_inf: bool,
}

#[derive(Debug, Clone)]
pub struct Engine {
    cfg: TrainConfig,
    step: u64,
    last_step_started: Instant,
    backend: Backend,
    logger: Option<crate::logging::Logger>,
    params: Vec<i64>,
    opt: Option<i64>,
    accum_steps: usize,
    accum_loss: f32,
    accum_tokens: usize,
    accum_forward_ms: f32,
    accum_backward_ms: f32,
}

impl Engine {
    pub fn new(cfg: TrainConfig) -> Result<Self, RuntimeError> {
        Ok(Self {
            cfg,
            step: 0,
            last_step_started: Instant::now(),
            backend: Backend::new()?,
            logger: None,
            params: Vec::new(),
            opt: None,
            accum_steps: 0,
            accum_loss: 0.0,
            accum_tokens: 0,
            accum_forward_ms: 0.0,
            accum_backward_ms: 0.0,
        })
    }

    /// Provide parameter handles once the model is constructed.
    pub fn set_params(&mut self, params: Vec<i64>) -> Result<(), RuntimeError> {
        self.params = params.clone();
        self.backend.set_params(params)?;
        self.opt = None;
        Ok(())
    }

    fn ensure_logger(&mut self) -> Result<(), RuntimeError> {
        if self.logger.is_none() {
            self.logger = Some(crate::logging::Logger::new(&self.cfg.log.log_dir)?);
        }
        Ok(())
    }
}

/// Initialise the training engine (stub – no GPU work yet).
pub fn init(cfg: TrainConfig) -> Result<Engine, RuntimeError> {
    let mut e = Engine::new(cfg)?;
    if e.cfg.rank >= e.cfg.world_size {
        return Err(RuntimeError::new("rank must be < world_size"));
    }
    if e.cfg.world_size > 1 {
        e.backend
            .configure_dist(e.cfg.world_size as i32, e.cfg.rank as i32)?;
    }
    if e.params.is_empty() {
        let params = e.backend.init_tinylm(
            e.cfg.model.vocab_size as i64,
            e.cfg.data.seq_len as i64,
            e.cfg.model.d_model as i64,
            e.cfg.model.n_layers as i64,
            e.cfg.model.n_heads as i64,
            e.cfg.model.seed as i64,
            &e.cfg.model.device,
        )?;
        e.set_params(params)?;
    }
    Ok(e)
}

/// Perform one training step. Currently returns placeholder metrics.
pub fn train_step(engine: &mut Engine, batch: &Batch) -> Result<Metrics, RuntimeError> {
    engine.ensure_logger()?;
    let target_accum = engine.cfg.grad_accum_steps.max(1);
    if engine.accum_steps == 0 {
        engine.backend.zero_all_grads()?;
        engine.last_step_started = Instant::now();
        engine.accum_loss = 0.0;
        engine.accum_tokens = 0;
        engine.accum_forward_ms = 0.0;
        engine.accum_backward_ms = 0.0;
    }

    let step_result = engine.backend.train_step_on_device(
        engine.cfg.rank,
        batch,
        engine.cfg.model.n_heads,
        &engine.cfg.amp,
    )?;
    engine.accum_loss += step_result.loss;
    engine.accum_tokens += batch.token_count;
    engine.accum_forward_ms += step_result.forward_time_ms;
    engine.accum_backward_ms += step_result.backward_time_ms;
    engine.accum_steps += 1;

    let mut did_update = false;
    let mut grad_norm = None;
    let mut optim_time_ms = 0.0;
    let mut found_inf = false;
    let mut tokens_per_sec = 0.0;
    let mut step_time_ms = 0.0;
    let mut loss = step_result.loss;
    let mut token_count = batch.token_count;
    let mut forward_time_ms = step_result.forward_time_ms;
    let mut backward_time_ms = step_result.backward_time_ms;

    if engine.accum_steps >= target_accum {
        let grads = engine.backend.grad_multi(&engine.params)?;
        if engine.cfg.amp.enabled {
            found_inf = engine.backend.amp_unscale_grads(&grads, &engine.cfg.amp)?;
            engine
                .backend
                .amp_scaler_update(&engine.cfg.amp, found_inf)?;
            if found_inf {
                engine.backend.zero_all_grads()?;
                engine.backend.free_tensors(&grads);
                engine.accum_steps = 0;
                engine.accum_loss = 0.0;
                engine.accum_tokens = 0;
                engine.accum_forward_ms = 0.0;
                engine.accum_backward_ms = 0.0;
                let metrics = Metrics {
                    step: engine.step,
                    loss,
                    tokens_per_sec: 0.0,
                    step_time_ms: 0.0,
                    lr: engine.cfg.optim.lr as f32,
                    gpu_mem_mb: None,
                    gpu_util: None,
                    did_update: false,
                    accum_steps: target_accum,
                    token_count,
                    forward_time_ms,
                    backward_time_ms,
                    optim_time_ms: 0.0,
                    grad_norm: None,
                    found_inf: true,
                };
                maybe_log(engine, &metrics)?;
                return Ok(metrics);
            }
        }
        if engine.cfg.world_size > 1 {
            engine.backend.dist_allreduce_grads(&grads)?;
        }
        if target_accum > 1 {
            engine
                .backend
                .scale_grads(&grads, 1.0 / target_accum as f64)?;
        }
        if let Some(max_norm) = engine.cfg.grad_clip_norm {
            grad_norm = Some(engine.backend.clip_grad_norm(&grads, max_norm)? as f32);
        }
        engine.backend.check_finite_multi(&grads, "grads")?;
        let optim_start = Instant::now();
        engine
            .backend
            .optimizer_step_with_grads(&grads, &engine.cfg.optim)?;
        optim_time_ms = to_ms(optim_start.elapsed());
        engine.backend.free_tensors(&grads);
        did_update = true;
        engine.step += 1;

        let elapsed = engine.last_step_started.elapsed();
        let elapsed_s = elapsed.as_secs_f32().max(1e-6);
        let total_tokens = (engine.accum_tokens as f32) * engine.cfg.world_size as f32;
        tokens_per_sec = total_tokens / elapsed_s;
        step_time_ms = to_ms(elapsed);
        loss = engine.accum_loss / target_accum as f32;
        token_count = engine.accum_tokens;
        forward_time_ms = engine.accum_forward_ms;
        backward_time_ms = engine.accum_backward_ms;

        engine.accum_steps = 0;
        engine.accum_loss = 0.0;
        engine.accum_tokens = 0;
        engine.accum_forward_ms = 0.0;
        engine.accum_backward_ms = 0.0;
    }

    let metrics = Metrics {
        step: engine.step,
        loss,
        tokens_per_sec,
        step_time_ms,
        lr: engine.cfg.optim.lr as f32,
        gpu_mem_mb: None,
        gpu_util: None,
        did_update,
        accum_steps: target_accum,
        token_count,
        forward_time_ms,
        backward_time_ms,
        optim_time_ms,
        grad_norm,
        found_inf,
    };
    maybe_log(engine, &metrics)?;
    Ok(metrics)
}

/// Perform one evaluation step. Currently identical to `train_step`.
pub fn eval_step(engine: &mut Engine, batch: &Batch) -> Result<Metrics, RuntimeError> {
    engine.ensure_logger()?;
    let elapsed = engine.last_step_started.elapsed();
    engine.last_step_started = Instant::now();

    let step_result = engine.backend.eval_step_on_device(
        engine.cfg.rank,
        batch,
        engine.cfg.model.n_heads,
        &engine.cfg.amp,
    )?;
    let loss = step_result.loss;
    let tokens = (batch.token_count as f32) * engine.cfg.world_size as f32;
    let elapsed_s = elapsed.as_secs_f32().max(1e-6);
    let tokens_per_sec = tokens / elapsed_s;
    engine.step += 1;
    let metrics = Metrics {
        step: engine.step,
        loss,
        tokens_per_sec,
        step_time_ms: to_ms(elapsed),
        lr: 0.0,
        gpu_mem_mb: None,
        gpu_util: None,
        did_update: true,
        accum_steps: 1,
        token_count: batch.token_count,
        forward_time_ms: step_result.forward_time_ms,
        backward_time_ms: step_result.backward_time_ms,
        optim_time_ms: 0.0,
        grad_norm: None,
        found_inf: false,
    };
    maybe_log(engine, &metrics)?;
    Ok(metrics)
}

/// Save a lightweight checkpoint marker for crash‑safe resume.
pub fn save_checkpoint(engine: &Engine, dir: impl AsRef<Path>) -> Result<(), RuntimeError> {
    let dir = dir.as_ref();
    fs::create_dir_all(dir).map_err(|e| RuntimeError::new(&e.to_string()))?;
    let tmp = dir.join("step.tmp");
    let final_path = dir.join("step.txt");
    let mut file = File::create(&tmp).map_err(|e| RuntimeError::new(&e.to_string()))?;
    write!(file, "{}", engine.step).map_err(|e| RuntimeError::new(&e.to_string()))?;
    fs::rename(&tmp, &final_path).map_err(|e| RuntimeError::new(&e.to_string()))?;
    Ok(())
}

/// Load a lightweight checkpoint marker if present.
pub fn load_checkpoint(engine: &mut Engine, dir: impl AsRef<Path>) -> Result<(), RuntimeError> {
    let dir = dir.as_ref();
    if let Some(path) =
        latest_checkpoint(dir).map_err(|e| RuntimeError::new(&format!("find checkpoint: {}", e)))?
    {
        let state = load_checkpoint_file(&path)
            .map_err(|e| RuntimeError::new(&format!("load checkpoint: {}", e)))?;
        engine.step = state.meta.step;
    } else {
        let marker = dir.join("step.txt");
        if marker.is_file() {
            let contents =
                fs::read_to_string(&marker).map_err(|e| RuntimeError::new(&e.to_string()))?;
            engine.step = contents.trim().parse().unwrap_or(0);
        }
    }
    Ok(())
}

/// Shut down the engine gracefully (stubbed).
pub fn shutdown(_engine: Engine) -> Result<(), RuntimeError> {
    Ok(())
}

/// Save checkpoint via backend (params/optimizer + meta JSON string).
pub fn checkpoint_save(engine: &Engine, dir: &str, meta_json: &str) -> Result<(), RuntimeError> {
    engine.backend.checkpoint_save(dir, meta_json)?;
    if engine.cfg.world_size > 1 && engine.cfg.rank == 0 {
        write_checkpoint_manifest(Path::new(dir), engine.cfg.world_size)?;
    }
    Ok(())
}

/// Load checkpoint, updating engine params/opt and returning meta JSON.
pub fn checkpoint_load(engine: &mut Engine, dir: &str) -> Result<String, RuntimeError> {
    if engine.cfg.world_size > 1 {
        if let Some(manifest) = load_checkpoint_manifest(Path::new(dir))? {
            if manifest.world_size != engine.cfg.world_size {
                return Err(RuntimeError::new("checkpoint manifest world_size mismatch"));
            }
        }
    }
    let (params, opt, meta) = engine.backend.checkpoint_load(dir)?;
    engine.params = params;
    engine.opt = opt;
    Ok(meta)
}

#[derive(Debug, Serialize, Deserialize)]
struct CheckpointManifest {
    format_version: u32,
    world_size: usize,
    ranks: Vec<usize>,
}

fn write_checkpoint_manifest(dir: &Path, world_size: usize) -> Result<(), RuntimeError> {
    let manifest = CheckpointManifest {
        format_version: 1,
        world_size,
        ranks: (0..world_size).collect(),
    };
    let text = serde_json::to_string_pretty(&manifest)
        .map_err(|err| RuntimeError::new(&err.to_string()))?;
    let path = dir.join("manifest.json");
    fs::write(&path, text).map_err(|err| RuntimeError::new(&err.to_string()))?;
    Ok(())
}

fn load_checkpoint_manifest(dir: &Path) -> Result<Option<CheckpointManifest>, RuntimeError> {
    let path = dir.join("manifest.json");
    if !path.is_file() {
        return Ok(None);
    }
    let text = fs::read_to_string(&path).map_err(|err| RuntimeError::new(&err.to_string()))?;
    let manifest: CheckpointManifest =
        serde_json::from_str(&text).map_err(|err| RuntimeError::new(&err.to_string()))?;
    Ok(Some(manifest))
}

fn to_ms(duration: Duration) -> f32 {
    (duration.as_secs_f64() * 1_000.0) as f32
}

fn maybe_log(engine: &mut Engine, metrics: &Metrics) -> Result<(), RuntimeError> {
    if !metrics.did_update {
        return Ok(());
    }
    if engine.cfg.log.log_every == 0 {
        return Ok(());
    }
    if !metrics.step.is_multiple_of(engine.cfg.log.log_every as u64) {
        return Ok(());
    }
    if let Some(logger) = &engine.logger {
        logger.append(metrics)?;
    }
    crate::logging::print_summary("train", metrics);
    Ok(())
}
