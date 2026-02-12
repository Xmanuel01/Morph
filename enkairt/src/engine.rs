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
    if e.cfg.world_size > 1 {
        e.backend.configure_dist(e.cfg.world_size as i32, 0)?; // rank 0 for single-process launcher
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
    // Lazy-create optimizer once params are known.
    engine.backend.ensure_optimizer(
        engine.cfg.optim.lr,
        engine.cfg.optim.beta1,
        engine.cfg.optim.beta2,
        engine.cfg.optim.eps,
        engine.cfg.optim.weight_decay,
    )?;
    engine.backend.zero_all_grads()?;
    engine.step += 1;
    let elapsed = engine.last_step_started.elapsed();
    engine.last_step_started = Instant::now();

    // Multi-device stub: run per device and average loss.
    let mut total_loss = 0.0;
    for device_idx in 0..engine.cfg.world_size {
        total_loss +=
            engine
                .backend
                .train_step_on_device(device_idx, batch, engine.cfg.model.n_heads)?;
    }
    let loss = total_loss / engine.cfg.world_size as f32;

    let tokens =
        (engine.cfg.data.seq_len * engine.cfg.data.batch_size * engine.cfg.world_size) as f32;
    let elapsed_s = elapsed.as_secs_f32().max(1e-6);
    let tokens_per_sec = tokens / elapsed_s;
    engine.backend.optimizer_step()?;
    let metrics = Metrics {
        step: engine.step,
        loss,
        tokens_per_sec,
        step_time_ms: to_ms(elapsed),
        lr: engine.cfg.optim.lr as f32,
        gpu_mem_mb: None,
    };
    maybe_log(engine, &metrics)?;
    Ok(metrics)
}

/// Perform one evaluation step. Currently identical to `train_step`.
pub fn eval_step(engine: &mut Engine, batch: &Batch) -> Result<Metrics, RuntimeError> {
    engine.ensure_logger()?;
    engine.step += 1;
    let elapsed = engine.last_step_started.elapsed();
    engine.last_step_started = Instant::now();

    let mut total_loss = 0.0;
    for device_idx in 0..engine.cfg.world_size {
        total_loss +=
            engine
                .backend
                .eval_step_on_device(device_idx, batch, engine.cfg.model.n_heads)?;
    }
    let loss = total_loss / engine.cfg.world_size as f32;

    let tokens =
        (engine.cfg.data.seq_len * engine.cfg.data.batch_size * engine.cfg.world_size) as f32;
    let elapsed_s = elapsed.as_secs_f32().max(1e-6);
    let tokens_per_sec = tokens / elapsed_s;
    // No optimizer update on eval.
    let metrics = Metrics {
        step: engine.step,
        loss,
        tokens_per_sec,
        step_time_ms: to_ms(elapsed),
        lr: 0.0,
        gpu_mem_mb: None,
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
    engine.backend.checkpoint_save(dir, meta_json)
}

/// Load checkpoint, updating engine params/opt and returning meta JSON.
pub fn checkpoint_load(engine: &mut Engine, dir: &str) -> Result<String, RuntimeError> {
    let (params, opt, meta) = engine.backend.checkpoint_load(dir)?;
    engine.params = params;
    engine.opt = opt;
    Ok(meta)
}

fn to_ms(duration: Duration) -> f32 {
    (duration.as_secs_f64() * 1_000.0) as f32
}

fn maybe_log(engine: &mut Engine, metrics: &Metrics) -> Result<(), RuntimeError> {
    if engine.cfg.log.log_every == 0 {
        return Ok(());
    }
    if !engine.step.is_multiple_of(engine.cfg.log.log_every as u64) {
        return Ok(());
    }
    if let Some(logger) = &engine.logger {
        logger.append(metrics)?;
    }
    crate::logging::print_summary("train", metrics);
    Ok(())
}
