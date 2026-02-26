use std::collections::hash_map::DefaultHasher;
use std::fs::{self, OpenOptions};
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use enkai_compiler::bytecode::{ByteFunction, Chunk, Constant, Instruction, Program};
use enkai_compiler::compiler::{compile_package, CompileError};
use enkai_compiler::modules::load_package;
use enkai_compiler::{TypeChecker, TypeError};
use enkai_runtime::checkpoint::{
    latest_checkpoint, load_checkpoint, rotate_checkpoints, save_checkpoint, CheckpointAmp,
    CheckpointMeta, CheckpointState,
};
use enkai_runtime::dataset::{resolve_dataset_paths, Batch, DatasetConfig, DatasetStream};
use enkai_runtime::object::Obj;
use enkai_runtime::tokenizer::{Tokenizer, TrainConfig as TokenizerTrainConfig};
use enkai_runtime::{
    engine::{
        checkpoint_load as rt_checkpoint_load, checkpoint_save as rt_checkpoint_save,
        eval_step as rt_eval_step, init, train_step as rt_train_step, AmpConfig as RtAmpConfig,
        CheckpointConfig as RtCkptConfig, DataConfig as RtDataConfig, LogConfig as RtLogConfig,
        ModelConfig as RtModelConfig, OptimConfig as RtOptimConfig, TrainConfig as RtTrainConfig,
    },
    Value, VM,
};

#[derive(Debug, Clone)]
struct TrainConfig {
    config_version: u32,
    backend: String,
    vocab_size: usize,
    hidden_size: usize,
    seq_len: usize,
    batch_size: usize,
    lr: f32,
    model: ModelConfig,
    optim: OptimConfig,
    amp: AmpConfig,
    dataset_path: String,
    eval_dataset_path: Option<String>,
    tokenizer: TokenizerConfig,
    checkpoint_dir: String,
    max_steps: usize,
    save_every: usize,
    log_every: usize,
    eval_steps: usize,
    drop_remainder: bool,
    add_eos: bool,
    pad_id: u32,
    keep_last: usize,
    seed: Option<u64>,
    shuffle: bool,
    prefetch_batches: usize,
    world_size: usize,
    rank: usize,
    grad_accum_steps: usize,
    grad_clip_norm: Option<f32>,
    legacy_config: bool,
}

#[derive(Debug, Clone)]
struct ModelConfig {
    d_model: usize,
    n_layers: usize,
    n_heads: usize,
    device: String,
    dtype: String,
}

#[derive(Debug, Clone)]
struct OptimConfig {
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct AmpConfig {
    enabled: bool,
    dtype: String,
    init_scale: f64,
    growth_factor: f64,
    backoff_factor: f64,
    growth_interval: i64,
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

#[derive(Debug, Clone)]
enum TokenizerConfig {
    Load(String),
    Train {
        path: String,
        vocab_size: usize,
        save_path: Option<String>,
    },
}

#[derive(Debug)]
struct AdamW {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    m: Vec<f32>,
    v: Vec<f32>,
    step: u64,
    beta1_pow: f32,
    beta2_pow: f32,
}

impl AdamW {
    fn new(param_len: usize, lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            m: vec![0.0; param_len],
            v: vec![0.0; param_len],
            step: 0,
            beta1_pow: 1.0,
            beta2_pow: 1.0,
        }
    }

    fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        self.step += 1;
        self.beta1_pow *= self.beta1;
        self.beta2_pow *= self.beta2;
        let bias1 = 1.0 - self.beta1_pow;
        let bias2 = 1.0 - self.beta2_pow;
        for i in 0..params.len() {
            let mut g = grads[i];
            if self.weight_decay != 0.0 {
                g += self.weight_decay * params[i];
            }
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;
            let m_hat = self.m[i] / bias1;
            let v_hat = self.v[i] / bias2;
            params[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ParamGroup {
    start: usize,
    len: usize,
}

#[derive(Debug)]
struct AdamWMulti {
    groups: Vec<AdamW>,
}

impl AdamWMulti {
    fn new(group_sizes: &[usize], lr: f32) -> Self {
        let mut groups = Vec::with_capacity(group_sizes.len());
        for size in group_sizes {
            groups.push(AdamW::new(*size, lr));
        }
        Self { groups }
    }

    fn step(&mut self, params: &mut [f32], grads: &[f32], groups: &[ParamGroup]) {
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

#[derive(Debug)]
struct TinyModel {
    vocab_size: usize,
    hidden_size: usize,
    params: Vec<f32>,
    embed_offset: usize,
    out_offset: usize,
    bias_offset: usize,
}

impl TinyModel {
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

    fn param_groups(&self) -> Vec<ParamGroup> {
        let embed_len = self.out_offset;
        let out_len = self.bias_offset - self.out_offset;
        let bias_len = self.vocab_size;
        vec![
            ParamGroup {
                start: self.embed_offset,
                len: embed_len,
            },
            ParamGroup {
                start: self.out_offset,
                len: out_len,
            },
            ParamGroup {
                start: self.bias_offset,
                len: bias_len,
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
            for logit in logits.iter_mut() {
                *logit = (*logit - max).exp();
                exp_sum += *logit;
            }
            for logit in logits.iter_mut() {
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
        for g in grads.iter_mut() {
            *g /= denom;
        }
        (loss, grads)
    }

    fn apply_grads(&mut self, grads: &[f32], opt: &mut AdamWMulti, groups: &[ParamGroup]) {
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
            for logit in logits.iter_mut() {
                *logit = (*logit - max).exp();
                exp_sum += *logit;
            }
            let prob = logits[target] / exp_sum;
            loss += -prob.max(1e-9).ln();
        }
        loss / total.max(1) as f32
    }
}

#[derive(Serialize)]
struct LogEvent {
    step: usize,
    loss: f32,
    tokens: u64,
    lr: f32,
    elapsed_ms: u128,
    event: String,
    tokens_per_sec: Option<f32>,
    step_time_ms: Option<f32>,
    packing_efficiency: Option<f32>,
    gpu_util: Option<f32>,
    grad_norm: Option<f32>,
    forward_time_ms: Option<f32>,
    backward_time_ms: Option<f32>,
    optim_time_ms: Option<f32>,
    found_inf: Option<bool>,
}

pub fn train(config_path: &Path) -> Result<(), String> {
    let config_value = load_config_value(config_path)?;
    let config = parse_train_config(&config_value)?;
    let tokenizer = build_tokenizer(&config)?;
    let mut dataset_paths = resolve_dataset_paths(&config.dataset_path)?;
    dataset_paths = shard_paths(dataset_paths, config.world_size, config.rank);
    let mut data_cfg = DatasetConfig::new(config.seq_len, config.batch_size);
    data_cfg.add_eos = config.add_eos;
    data_cfg.drop_remainder = config.drop_remainder;
    data_cfg.pad_id = config.pad_id;
    data_cfg.seed = data_seed(&config);
    data_cfg.shuffle = config.shuffle;
    data_cfg.prefetch_batches = config.prefetch_batches;
    let mut stream = DatasetStream::new(dataset_paths, tokenizer.clone(), data_cfg)?;
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
    if config.backend == "native" {
        let mut engine = init(rt_config_from(&config)).map_err(|e| e.to_string())?;
        if let Some(meta) =
            try_load_native_checkpoint(&mut engine, &config, &config.checkpoint_dir)?
        {
            step = meta.step as usize;
            tokens = meta.tokens;
        }
        let mut last_saved_step = step;
        let mut packing_sum = 0.0f32;
        let mut packing_count = 0usize;
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
                    let _ =
                        write_error_checkpoint(&engine, &config, step, tokens, &err.to_string());
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
                    &config,
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
            let avg_packing = if packing_count == 0 {
                0.0
            } else {
                packing_sum / packing_count as f32
            };
            if step == 1 || step.is_multiple_of(config.log_every) {
                let event = LogEvent {
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
                };
                let line = serde_json::to_string(&event).map_err(|err| err.to_string())?;
                writeln!(log_file, "{}", line).map_err(|err| err.to_string())?;
                println!("step {} loss {:.4} tokens {}", step, metrics.loss, tokens);
            }
            if step == config.max_steps || step.is_multiple_of(config.save_every) {
                let meta = native_checkpoint_meta(&config, step as u64, tokens, metrics.loss)?;
                rt_checkpoint_save(&engine, &config.checkpoint_dir, &meta)
                    .map_err(|err| err.to_string())?;
                if step < last_saved_step {
                    return Err("checkpoint step went backwards".to_string());
                }
                last_saved_step = step;
                let event = LogEvent {
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
                };
                let line = serde_json::to_string(&event).map_err(|err| err.to_string())?;
                writeln!(log_file, "{}", line).map_err(|err| err.to_string())?;
                println!("saved checkpoint latest");
            }
            packing_sum = 0.0;
            packing_count = 0;
        }
    } else {
        let seed = config_seed(&config);
        let mut model = TinyModel::new(config.vocab_size, config.hidden_size, seed);
        let groups = model.param_groups();
        let group_sizes: Vec<usize> = groups.iter().map(|group| group.len).collect();
        let mut opt = AdamWMulti::new(&group_sizes, config.lr);
        if let Some(latest) =
            latest_checkpoint(Path::new(&config.checkpoint_dir)).map_err(|err| err.to_string())?
        {
            let state = load_checkpoint(&latest).map_err(|err| err.to_string())?;
            if state.meta.format_version > 1 {
                return Err("unsupported checkpoint format version".to_string());
            }
            let expected_hash = hash_config_hex(&config);
            if !state.meta.config_hash.is_empty() && state.meta.config_hash != expected_hash {
                return Err("checkpoint config hash mismatch".to_string());
            }
            if !state.meta.model_sig.is_empty() && state.meta.model_sig != model_signature(&config)
            {
                return Err("checkpoint model signature mismatch".to_string());
            }
            if !state.meta.dtype.is_empty() && state.meta.dtype != config.model.dtype {
                return Err("checkpoint dtype mismatch".to_string());
            }
            if !state.meta.device.is_empty() && state.meta.device != config.model.device {
                return Err("checkpoint device mismatch".to_string());
            }
            if state.meta.world_size != 0 && state.meta.world_size != config.world_size {
                return Err("checkpoint world_size mismatch".to_string());
            }
            if state.meta.rank != 0 && state.meta.rank != config.rank {
                return Err("checkpoint rank mismatch".to_string());
            }
            if state.meta.grad_accum_steps != 0
                && state.meta.grad_accum_steps != config.grad_accum_steps
            {
                return Err("checkpoint grad_accum_steps mismatch".to_string());
            }
            if let Some(norm) = state.meta.grad_clip_norm {
                if config.grad_clip_norm.map(|v| v as f64) != Some(norm) {
                    return Err("checkpoint grad_clip_norm mismatch".to_string());
                }
            }
            if let Some(amp) = state.meta.amp.as_ref() {
                if *amp != checkpoint_amp_from(&config.amp) {
                    return Err("checkpoint amp config mismatch".to_string());
                }
            }
            model.set_params(&state.weights)?;
            opt.load_state(&state.optimizer)?;
            step = state.meta.step as usize;
            tokens = state.meta.tokens;
        }
        let accum_steps = config.grad_accum_steps.max(1);
        let mut packing_sum = 0.0f32;
        let mut packing_count = 0usize;
        let mut grad_accum = vec![0.0f32; model.params().len()];
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
                packing_sum += batch.packing_efficiency;
                packing_count += 1;
                let (loss, grads) = model.compute_grads(&batch);
                if !loss.is_finite() {
                    return Err("non-finite loss detected".to_string());
                }
                for (acc, g) in grad_accum.iter_mut().zip(grads.iter()) {
                    *acc += *g;
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
            for g in grad_accum.iter_mut() {
                *g *= scale;
            }
            let grad_norm = config
                .grad_clip_norm
                .and_then(|max_norm| clip_grad_norm(&mut grad_accum, max_norm));
            model.apply_grads(&grad_accum, &mut opt, &groups);
            grad_accum.fill(0.0);
            step += 1;
            let avg_loss = step_loss_sum / micro_batches as f32;
            let avg_packing = if packing_count == 0 {
                0.0
            } else {
                packing_sum / packing_count as f32
            };
            let elapsed_s = step_start.elapsed().as_secs_f32().max(1e-6);
            let tokens_per_sec = (step_tokens as f32) / elapsed_s;
            let step_time_ms = (step_start.elapsed().as_secs_f64() * 1_000.0) as f32;
            if step == 1 || step.is_multiple_of(config.log_every) {
                let event = LogEvent {
                    step,
                    loss: avg_loss,
                    tokens,
                    lr: config.lr,
                    elapsed_ms: start.elapsed().as_millis(),
                    event: "step".to_string(),
                    tokens_per_sec: Some(tokens_per_sec),
                    step_time_ms: Some(step_time_ms),
                    packing_efficiency: Some(avg_packing),
                    gpu_util: None,
                    grad_norm,
                    forward_time_ms: None,
                    backward_time_ms: None,
                    optim_time_ms: None,
                    found_inf: Some(false),
                };
                let line = serde_json::to_string(&event).map_err(|err| err.to_string())?;
                writeln!(log_file, "{}", line).map_err(|err| err.to_string())?;
                println!("step {} loss {:.4} tokens {}", step, avg_loss, tokens);
            }
            if step == config.max_steps || step.is_multiple_of(config.save_every) {
                let meta = CheckpointMeta {
                    format_version: 1,
                    step: step as u64,
                    tokens,
                    loss: avg_loss as f64,
                    config_hash: hash_config_hex(&config),
                    model_sig: model_signature(&config),
                    dtype: config.model.dtype.clone(),
                    device: config.model.device.clone(),
                    world_size: config.world_size,
                    rank: config.rank,
                    grad_accum_steps: config.grad_accum_steps,
                    grad_clip_norm: config.grad_clip_norm.map(|v| v as f64),
                    amp: Some(checkpoint_amp_from(&config.amp)),
                };
                let state = CheckpointState {
                    weights: model.params().to_vec(),
                    optimizer: opt.state_vec(),
                    meta,
                };
                let path = save_checkpoint(Path::new(&config.checkpoint_dir), &state)
                    .map_err(|err| err.to_string())?;
                rotate_checkpoints(Path::new(&config.checkpoint_dir), config.keep_last)
                    .map_err(|err| err.to_string())?;
                let event = LogEvent {
                    step,
                    loss: avg_loss,
                    tokens,
                    lr: config.lr,
                    elapsed_ms: start.elapsed().as_millis(),
                    event: "checkpoint".to_string(),
                    tokens_per_sec: None,
                    step_time_ms: None,
                    packing_efficiency: Some(avg_packing),
                    gpu_util: None,
                    grad_norm,
                    forward_time_ms: None,
                    backward_time_ms: None,
                    optim_time_ms: None,
                    found_inf: Some(false),
                };
                let line = serde_json::to_string(&event).map_err(|err| err.to_string())?;
                writeln!(log_file, "{}", line).map_err(|err| err.to_string())?;
                println!("saved checkpoint {}", path.display());
            }
            packing_sum = 0.0;
            packing_count = 0;
        }
    }
    Ok(())
}

pub fn eval(config_path: &Path) -> Result<(), String> {
    let config_value = load_config_value(config_path)?;
    let config = parse_train_config(&config_value)?;
    let tokenizer = build_tokenizer(&config)?;
    let dataset_path = config
        .eval_dataset_path
        .as_ref()
        .unwrap_or(&config.dataset_path);
    let dataset_paths = resolve_dataset_paths(dataset_path)?;
    let mut data_cfg = DatasetConfig::new(config.seq_len, config.batch_size);
    data_cfg.add_eos = config.add_eos;
    data_cfg.drop_remainder = config.drop_remainder;
    data_cfg.pad_id = config.pad_id;
    data_cfg.seed = data_seed(&config);
    data_cfg.prefetch_batches = config.prefetch_batches;
    let mut stream = DatasetStream::new(dataset_paths, tokenizer, data_cfg)?;
    if config.backend == "native" {
        let mut engine = init(rt_config_from(&config)).map_err(|e| e.to_string())?;
        let _meta = try_load_native_checkpoint(&mut engine, &config, &config.checkpoint_dir)?
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
    } else {
        let seed = config_seed(&config);
        let mut model = TinyModel::new(config.vocab_size, config.hidden_size, seed);
        let latest = latest_checkpoint(Path::new(&config.checkpoint_dir))
            .map_err(|err| err.to_string())?
            .ok_or_else(|| "No checkpoint found".to_string())?;
        let state = load_checkpoint(&latest).map_err(|err| err.to_string())?;
        if state.meta.format_version > 1 {
            return Err("unsupported checkpoint format version".to_string());
        }
        let expected_hash = hash_config_hex(&config);
        if !state.meta.config_hash.is_empty() && state.meta.config_hash != expected_hash {
            return Err("checkpoint config hash mismatch".to_string());
        }
        if !state.meta.model_sig.is_empty() && state.meta.model_sig != model_signature(&config) {
            return Err("checkpoint model signature mismatch".to_string());
        }
        if !state.meta.dtype.is_empty() && state.meta.dtype != config.model.dtype {
            return Err("checkpoint dtype mismatch".to_string());
        }
        if !state.meta.device.is_empty() && state.meta.device != config.model.device {
            return Err("checkpoint device mismatch".to_string());
        }
        if state.meta.world_size != 0 && state.meta.world_size != config.world_size {
            return Err("checkpoint world_size mismatch".to_string());
        }
        if state.meta.rank != 0 && state.meta.rank != config.rank {
            return Err("checkpoint rank mismatch".to_string());
        }
        if state.meta.grad_accum_steps != 0
            && state.meta.grad_accum_steps != config.grad_accum_steps
        {
            return Err("checkpoint grad_accum_steps mismatch".to_string());
        }
        if let Some(norm) = state.meta.grad_clip_norm {
            if config.grad_clip_norm.map(|v| v as f64) != Some(norm) {
                return Err("checkpoint grad_clip_norm mismatch".to_string());
            }
        }
        if let Some(amp) = state.meta.amp.as_ref() {
            if *amp != checkpoint_amp_from(&config.amp) {
                return Err("checkpoint amp config mismatch".to_string());
            }
        }
        model.set_params(&state.weights)?;
        let mut total_loss = 0.0f32;
        let mut batches = 0usize;
        while batches < config.eval_steps {
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
}

fn build_tokenizer(config: &TrainConfig) -> Result<Tokenizer, String> {
    match &config.tokenizer {
        TokenizerConfig::Load(path) => Tokenizer::load(Path::new(path)),
        TokenizerConfig::Train {
            path,
            vocab_size,
            save_path,
        } => {
            let mut cfg = TokenizerTrainConfig::default();
            cfg.vocab_size = *vocab_size;
            cfg.seed = config.seed;
            let tok = Tokenizer::train_from_path(Path::new(path), &cfg)?;
            if let Some(save_path) = save_path {
                tok.save(Path::new(save_path))?;
            }
            Ok(tok)
        }
    }
}

fn load_config_value(path: &Path) -> Result<Value, String> {
    let package = load_package(path).map_err(|err| err.to_string())?;
    if let Err(err) = TypeChecker::check_package(&package) {
        return Err(type_error_message(&err));
    }
    let mut program = compile_package(&package).map_err(|err| compile_error_message(&err))?;
    let main_global = format!("{}::main", package.entry.0.join("::"));
    if let Some(main_idx) = program.globals.iter().position(|name| name == &main_global) {
        program = wrap_program_with_main(&program, main_idx as u16);
    }
    let mut vm = VM::new(false, false, false, false);
    let value = vm
        .run(&program)
        .map_err(|err| format!("Runtime error: {}", err))?;
    Ok(value)
}

fn parse_train_config(value: &Value) -> Result<TrainConfig, String> {
    let map_ref = value_as_record(value)?;
    let map = &*map_ref;
    let (config_version, legacy_config) = match map.get("config_version") {
        Some(Value::Int(i)) if *i >= 1 => (*i as u32, false),
        Some(_) => return Err("config_version must be Int >= 1".to_string()),
        None => (1, true),
    };
    if !legacy_config && config_version != 1 {
        return Err(format!(
            "Unsupported config_version {} (expected 1)",
            config_version
        ));
    }
    let model = map.get("model").and_then(|v| as_record(v).ok());
    let model_ref = model.as_ref().map(|m| &**m).unwrap_or(map);
    let backend = if legacy_config {
        map.get("backend")
            .and_then(|v| as_string(v).ok())
            .unwrap_or_else(|| "cpu".to_string())
    } else {
        record_string(map, "backend")?
    };
    validate_backend(&backend)?;
    let vocab_size = record_usize(model_ref, "vocab_size")?;
    let hidden_size = record_usize(model_ref, "hidden_size")?;
    let d_model = record_usize_default(model_ref, "d_model", hidden_size);
    let n_layers = record_usize_default(model_ref, "n_layers", 2);
    let n_heads = record_usize_default(model_ref, "n_heads", 4);
    let mut device = record_string_default(model_ref, "device", "cpu");
    let dtype = record_string_default(model_ref, "dtype", "fp32");
    validate_dtype(&dtype)?;
    let seq_len = record_usize(map, "seq_len")?;
    let batch_size = record_usize(map, "batch_size")?;
    let lr = record_f32(map, "lr")?;
    let optim = map.get("optim").and_then(|v| as_record(v).ok());
    let optim_ref = optim.as_ref().map(|m| &**m).unwrap_or(map);
    let opt_lr = record_f64_default(optim_ref, "lr", lr as f64);
    let opt_beta1 = record_f64_default(optim_ref, "beta1", 0.9);
    let opt_beta2 = record_f64_default(optim_ref, "beta2", 0.999);
    let opt_eps = record_f64_default(optim_ref, "eps", 1e-8);
    let opt_wd = record_f64_default(optim_ref, "weight_decay", 0.01);
    let dataset_path = record_string(map, "dataset_path")?;
    let eval_dataset_path = map.get("eval_dataset_path").and_then(|v| as_string(v).ok());
    let checkpoint_dir = record_string(map, "checkpoint_dir")?;
    let max_steps = record_usize(map, "max_steps")?;
    let save_every = record_usize_default(map, "save_every", 100);
    let log_every = record_usize_default(map, "log_every", 10);
    let eval_steps = record_usize_default(map, "eval_steps", 10);
    let drop_remainder = record_bool_default(map, "drop_remainder", true);
    let add_eos = record_bool_default(map, "add_eos", true);
    let pad_id = record_usize_default(map, "pad_id", 0) as u32;
    let keep_last = record_usize_default(map, "keep_last", 3);
    let seed = match map.get("seed") {
        Some(Value::Int(i)) if *i >= 0 => Some(*i as u64),
        Some(_) => return Err("seed must be Int >= 0".to_string()),
        None => None,
    };
    let shuffle = record_bool_default(map, "shuffle", seed.is_some());
    if shuffle && seed.is_none() {
        return Err("shuffle requires seed".to_string());
    }
    let prefetch_batches = record_usize_default(map, "prefetch_batches", 0);
    let world_size = record_usize_default(map, "world_size", 1);
    let rank = record_usize_default(map, "rank", 0);
    if world_size == 0 {
        return Err("world_size must be >= 1".to_string());
    }
    if rank >= world_size {
        return Err("rank must be < world_size".to_string());
    }
    if backend == "cpu" && world_size > 1 {
        return Err("world_size > 1 requires backend = \"native\"".to_string());
    }
    if backend == "native" && world_size > 1 && device == "cuda" {
        device = format!("cuda:{}", rank);
    }
    validate_device(&device, &backend)?;
    let grad_accum_steps = record_usize_default(map, "grad_accum_steps", 1);
    if grad_accum_steps == 0 {
        return Err("grad_accum_steps must be >= 1".to_string());
    }
    let grad_clip_norm = match map.get("grad_clip_norm") {
        Some(Value::Float(f)) if *f > 0.0 => Some(*f as f32),
        Some(Value::Int(i)) if *i > 0 => Some(*i as f32),
        Some(Value::Float(_)) | Some(Value::Int(_)) => None,
        Some(_) => return Err("grad_clip_norm must be Float".to_string()),
        None => None,
    };
    let amp = parse_amp_config(map, &backend, &device)?;
    let tokenizer = if let Some(value) = map.get("tokenizer_path") {
        TokenizerConfig::Load(as_string(value)?)
    } else if let Some(value) = map.get("tokenizer_train") {
        let tmap_ref = value_as_record(value)?;
        let tmap = &*tmap_ref;
        let path = record_string(tmap, "path")?;
        let vocab_size = record_usize_default(tmap, "vocab_size", vocab_size);
        let save_path = tmap.get("save_path").and_then(|v| as_string(v).ok());
        TokenizerConfig::Train {
            path,
            vocab_size,
            save_path,
        }
    } else {
        return Err("Config missing tokenizer_path or tokenizer_train".to_string());
    };
    Ok(TrainConfig {
        config_version,
        backend,
        vocab_size,
        hidden_size,
        seq_len,
        batch_size,
        lr,
        model: ModelConfig {
            d_model,
            n_layers,
            n_heads,
            device,
            dtype,
        },
        optim: OptimConfig {
            lr: opt_lr,
            beta1: opt_beta1,
            beta2: opt_beta2,
            eps: opt_eps,
            weight_decay: opt_wd,
        },
        amp,
        dataset_path,
        eval_dataset_path,
        tokenizer,
        checkpoint_dir,
        max_steps,
        save_every,
        log_every,
        eval_steps,
        drop_remainder,
        add_eos,
        pad_id,
        keep_last,
        seed,
        shuffle,
        prefetch_batches,
        world_size,
        rank,
        grad_accum_steps,
        grad_clip_norm,
        legacy_config,
    })
}

fn type_error_message(err: &TypeError) -> String {
    if let Some(diag) = err.diagnostic() {
        diag.to_string()
    } else {
        format!("Type error: {}", err.message)
    }
}

fn compile_error_message(err: &CompileError) -> String {
    if let Some(diag) = err.diagnostic() {
        diag.to_string()
    } else {
        format!("Compile error: {}", err.message)
    }
}

fn value_as_record<'a>(
    value: &'a Value,
) -> Result<std::cell::Ref<'a, std::collections::HashMap<String, Value>>, String> {
    match value {
        Value::Obj(obj) => match obj.as_obj() {
            Obj::Record(map) => Ok(map.borrow()),
            _ => Err("Expected record config".to_string()),
        },
        _ => Err("Expected record config".to_string()),
    }
}

fn as_record<'a>(
    value: &'a Value,
) -> Result<std::cell::Ref<'a, std::collections::HashMap<String, Value>>, String> {
    match value {
        Value::Obj(obj) => match obj.as_obj() {
            Obj::Record(map) => Ok(map.borrow()),
            _ => Err("Expected record".to_string()),
        },
        _ => Err("Expected record".to_string()),
    }
}

fn as_string(value: &Value) -> Result<String, String> {
    match value {
        Value::Obj(obj) => match obj.as_obj() {
            Obj::String(s) => Ok(s.clone()),
            _ => Err("Expected String".to_string()),
        },
        _ => Err("Expected String".to_string()),
    }
}

fn record_string(
    map: &std::collections::HashMap<String, Value>,
    key: &str,
) -> Result<String, String> {
    let value = map
        .get(key)
        .ok_or_else(|| format!("Config missing {}", key))?;
    as_string(value)
}

fn record_usize(
    map: &std::collections::HashMap<String, Value>,
    key: &str,
) -> Result<usize, String> {
    let value = map
        .get(key)
        .ok_or_else(|| format!("Config missing {}", key))?;
    match value {
        Value::Int(i) if *i > 0 => Ok(*i as usize),
        _ => Err(format!("Config {} must be Int > 0", key)),
    }
}

fn record_usize_default(
    map: &std::collections::HashMap<String, Value>,
    key: &str,
    default: usize,
) -> usize {
    match map.get(key) {
        Some(Value::Int(i)) if *i > 0 => *i as usize,
        _ => default,
    }
}

fn record_string_default(
    map: &std::collections::HashMap<String, Value>,
    key: &str,
    default: &str,
) -> String {
    match map.get(key) {
        Some(Value::Obj(obj)) => match obj.as_obj() {
            Obj::String(s) => s.clone(),
            _ => default.to_string(),
        },
        _ => default.to_string(),
    }
}

fn record_f32(map: &std::collections::HashMap<String, Value>, key: &str) -> Result<f32, String> {
    let value = map
        .get(key)
        .ok_or_else(|| format!("Config missing {}", key))?;
    match value {
        Value::Float(f) => Ok(*f as f32),
        Value::Int(i) => Ok(*i as f32),
        _ => Err(format!("Config {} must be Float", key)),
    }
}

fn record_f64_default(
    map: &std::collections::HashMap<String, Value>,
    key: &str,
    default: f64,
) -> f64 {
    match map.get(key) {
        Some(Value::Float(f)) => *f,
        Some(Value::Int(i)) => *i as f64,
        _ => default,
    }
}

fn record_bool_default(
    map: &std::collections::HashMap<String, Value>,
    key: &str,
    default: bool,
) -> bool {
    match map.get(key) {
        Some(Value::Bool(b)) => *b,
        _ => default,
    }
}

fn parse_amp_config(
    map: &std::collections::HashMap<String, Value>,
    backend: &str,
    device: &str,
) -> Result<AmpConfig, String> {
    let mut amp = AmpConfig::default();
    let Some(value) = map.get("amp") else {
        return Ok(amp);
    };
    let amap_ref = value_as_record(value)?;
    let amap = &*amap_ref;
    amp.enabled = record_bool_default(amap, "enabled", amp.enabled);
    let dtype = record_string_default(amap, "dtype", &amp.dtype);
    let dtype_norm = match dtype.as_str() {
        "fp16" | "f16" => "fp16",
        "bf16" => "bf16",
        other => {
            return Err(format!(
                "Unsupported amp dtype {} (expected fp16/f16 or bf16)",
                other
            ))
        }
    };
    amp.dtype = dtype_norm.to_string();
    amp.init_scale = record_f64_default(amap, "init_scale", amp.init_scale);
    amp.growth_factor = record_f64_default(amap, "growth_factor", amp.growth_factor);
    amp.backoff_factor = record_f64_default(amap, "backoff_factor", amp.backoff_factor);
    amp.growth_interval =
        record_f64_default(amap, "growth_interval", amp.growth_interval as f64) as i64;
    if amp.enabled {
        if backend != "native" {
            return Err("amp is only supported with backend = \"native\"".to_string());
        }
        if device == "cpu" {
            return Err(
                "amp requires a CUDA device (device must be \"cuda\" or \"cuda:<idx>\")"
                    .to_string(),
            );
        }
        if amp.init_scale <= 0.0 || !amp.init_scale.is_finite() {
            return Err("amp.init_scale must be finite and > 0".to_string());
        }
        if amp.growth_factor <= 1.0 || !amp.growth_factor.is_finite() {
            return Err("amp.growth_factor must be > 1.0".to_string());
        }
        if amp.backoff_factor <= 0.0 || amp.backoff_factor >= 1.0 || !amp.backoff_factor.is_finite()
        {
            return Err("amp.backoff_factor must be finite and in (0, 1)".to_string());
        }
        if amp.growth_interval <= 0 {
            return Err("amp.growth_interval must be > 0".to_string());
        }
    }
    Ok(amp)
}

fn validate_backend(backend: &str) -> Result<(), String> {
    match backend {
        "cpu" | "native" => Ok(()),
        _ => Err(format!(
            "Unsupported backend {} (expected \"cpu\" or \"native\")",
            backend
        )),
    }
}

fn validate_dtype(dtype: &str) -> Result<(), String> {
    match dtype {
        "fp16" | "f16" | "bf16" | "fp32" | "f32" | "int64" | "i64" => Ok(()),
        _ => Err(format!(
            "Unsupported dtype {} (expected fp16/f16, bf16, fp32/f32, or int64/i64)",
            dtype
        )),
    }
}

fn validate_device(device: &str, backend: &str) -> Result<(), String> {
    if backend == "cpu" {
        if device != "cpu" {
            return Err("device must be \"cpu\" when backend is \"cpu\"".to_string());
        }
        return Ok(());
    }
    if device == "cpu" || device == "cuda" {
        return Ok(());
    }
    if let Some(rest) = device.strip_prefix("cuda:") {
        if rest.parse::<usize>().is_ok() {
            return Ok(());
        }
    }
    Err("device must be \"cpu\", \"cuda\", or \"cuda:<index>\"".to_string())
}

fn hash_config(config: &TrainConfig) -> u64 {
    let mut hasher = DefaultHasher::new();
    if config.legacy_config {
        return hash_config_legacy(config);
    }
    config.config_version.hash(&mut hasher);
    config.backend.hash(&mut hasher);
    config.vocab_size.hash(&mut hasher);
    config.hidden_size.hash(&mut hasher);
    config.model.d_model.hash(&mut hasher);
    config.model.n_layers.hash(&mut hasher);
    config.model.n_heads.hash(&mut hasher);
    config.model.device.hash(&mut hasher);
    config.model.dtype.hash(&mut hasher);
    config.seq_len.hash(&mut hasher);
    config.batch_size.hash(&mut hasher);
    config.dataset_path.hash(&mut hasher);
    config.checkpoint_dir.hash(&mut hasher);
    config.lr.to_bits().hash(&mut hasher);
    config.optim.lr.to_bits().hash(&mut hasher);
    config.optim.beta1.to_bits().hash(&mut hasher);
    config.optim.beta2.to_bits().hash(&mut hasher);
    config.optim.eps.to_bits().hash(&mut hasher);
    config.optim.weight_decay.to_bits().hash(&mut hasher);
    config.seed.is_some().hash(&mut hasher);
    if let Some(seed) = config.seed {
        seed.hash(&mut hasher);
    }
    hasher.finish()
}

fn hash_config_legacy(config: &TrainConfig) -> u64 {
    let mut hasher = DefaultHasher::new();
    config.backend.hash(&mut hasher);
    config.vocab_size.hash(&mut hasher);
    config.hidden_size.hash(&mut hasher);
    config.model.d_model.hash(&mut hasher);
    config.model.n_layers.hash(&mut hasher);
    config.model.n_heads.hash(&mut hasher);
    config.model.device.hash(&mut hasher);
    config.model.dtype.hash(&mut hasher);
    config.seq_len.hash(&mut hasher);
    config.batch_size.hash(&mut hasher);
    config.dataset_path.hash(&mut hasher);
    config.checkpoint_dir.hash(&mut hasher);
    config.lr.to_bits().hash(&mut hasher);
    config.optim.lr.to_bits().hash(&mut hasher);
    config.optim.beta1.to_bits().hash(&mut hasher);
    config.optim.beta2.to_bits().hash(&mut hasher);
    config.optim.eps.to_bits().hash(&mut hasher);
    config.optim.weight_decay.to_bits().hash(&mut hasher);
    hasher.finish()
}

fn hash_config_hex(config: &TrainConfig) -> String {
    format!("{:x}", hash_config(config))
}

fn config_seed(config: &TrainConfig) -> u64 {
    config.seed.unwrap_or_else(|| hash_config(config))
}

fn data_seed(config: &TrainConfig) -> Option<u64> {
    config.seed.map(|seed| {
        if config.world_size > 1 {
            seed.wrapping_add(config.rank as u64)
        } else {
            seed
        }
    })
}

fn checkpoint_amp_from(config: &AmpConfig) -> CheckpointAmp {
    CheckpointAmp {
        enabled: config.enabled,
        dtype: config.dtype.clone(),
        init_scale: config.init_scale,
        growth_factor: config.growth_factor,
        backoff_factor: config.backoff_factor,
        growth_interval: config.growth_interval,
    }
}

fn clip_grad_norm(grads: &mut [f32], max_norm: f32) -> Option<f32> {
    if max_norm <= 0.0 || !max_norm.is_finite() {
        return None;
    }
    let mut total_sq = 0.0f64;
    for g in grads.iter() {
        if !g.is_finite() {
            return None;
        }
        let g64 = *g as f64;
        total_sq += g64 * g64;
    }
    let norm = total_sq.sqrt() as f32;
    if norm > max_norm {
        let scale = max_norm / (norm + 1e-6);
        for g in grads.iter_mut() {
            *g *= scale;
        }
    }
    Some(norm)
}

fn shard_paths(paths: Vec<PathBuf>, world_size: usize, rank: usize) -> Vec<PathBuf> {
    if world_size <= 1 || paths.is_empty() {
        return paths;
    }
    let original = paths.clone();
    let out: Vec<PathBuf> = paths
        .into_iter()
        .enumerate()
        .filter_map(|(idx, path)| {
            if idx % world_size == rank {
                Some(path)
            } else {
                None
            }
        })
        .collect();
    if out.is_empty() {
        // Fall back to full list if sharding produced no files.
        return original;
    }
    out
}

#[derive(Serialize, Deserialize)]
struct NativeCheckpointMeta {
    #[serde(default)]
    format_version: u32,
    step: u64,
    tokens: u64,
    loss: f32,
    #[serde(default)]
    config_hash: String,
    #[serde(default)]
    model_sig: String,
    #[serde(default)]
    dtype: String,
    #[serde(default)]
    device: String,
    #[serde(default)]
    world_size: usize,
    #[serde(default)]
    rank: usize,
    #[serde(default)]
    grad_accum_steps: usize,
    #[serde(default)]
    grad_clip_norm: Option<f32>,
    #[serde(default)]
    amp: Option<AmpConfig>,
}

fn model_signature(config: &TrainConfig) -> String {
    format!(
        "vocab={};seq={};d_model={};layers={};heads={}",
        config.vocab_size,
        config.seq_len,
        config.model.d_model,
        config.model.n_layers,
        config.model.n_heads
    )
}

fn native_checkpoint_meta(
    config: &TrainConfig,
    step: u64,
    tokens: u64,
    loss: f32,
) -> Result<String, String> {
    let meta = NativeCheckpointMeta {
        format_version: 1,
        step,
        tokens,
        loss,
        config_hash: hash_config_hex(config),
        model_sig: model_signature(config),
        dtype: config.model.dtype.clone(),
        device: config.model.device.clone(),
        world_size: config.world_size,
        rank: config.rank,
        grad_accum_steps: config.grad_accum_steps,
        grad_clip_norm: config.grad_clip_norm,
        amp: Some(config.amp.clone()),
    };
    serde_json::to_string(&meta).map_err(|err| err.to_string())
}

fn rt_config_from(config: &TrainConfig) -> RtTrainConfig {
    RtTrainConfig {
        data: RtDataConfig {
            train_path: Path::new(&config.dataset_path).to_path_buf(),
            eval_path: config
                .eval_dataset_path
                .as_ref()
                .map(|p| Path::new(p).to_path_buf()),
            seq_len: config.seq_len,
            batch_size: config.batch_size,
        },
        model: RtModelConfig {
            vocab_size: config.vocab_size,
            d_model: config.model.d_model,
            n_layers: config.model.n_layers,
            n_heads: config.model.n_heads,
            seed: config_seed(config),
            device: config.model.device.clone(),
        },
        optim: RtOptimConfig {
            lr: config.optim.lr,
            beta1: config.optim.beta1,
            beta2: config.optim.beta2,
            eps: config.optim.eps,
            weight_decay: config.optim.weight_decay,
        },
        checkpoint: RtCkptConfig::new(Path::new(&config.checkpoint_dir).to_path_buf()),
        log: RtLogConfig::new(Path::new(&config.checkpoint_dir).to_path_buf()),
        world_size: config.world_size,
        rank: config.rank,
        grad_accum_steps: config.grad_accum_steps,
        grad_clip_norm: config.grad_clip_norm.map(|v| v as f64),
        amp: RtAmpConfig {
            enabled: config.amp.enabled,
            dtype: config.amp.dtype.clone(),
            init_scale: config.amp.init_scale,
            growth_factor: config.amp.growth_factor,
            backoff_factor: config.amp.backoff_factor,
            growth_interval: config.amp.growth_interval,
        },
    }
}

fn try_load_native_checkpoint(
    engine: &mut enkai_runtime::engine::Engine,
    config: &TrainConfig,
    dir: &str,
) -> Result<Option<NativeCheckpointMeta>, String> {
    let path = Path::new(dir).join("latest");
    if !path.is_dir() {
        return Ok(None);
    }
    let meta_json = rt_checkpoint_load(engine, dir).map_err(|err| err.to_string())?;
    let meta: NativeCheckpointMeta =
        serde_json::from_str(&meta_json).map_err(|err| err.to_string())?;
    if meta.format_version > 1 {
        return Err("unsupported checkpoint format version".to_string());
    }
    let expected_hash = hash_config_hex(config);
    if !meta.config_hash.is_empty() && meta.config_hash != expected_hash {
        return Err("checkpoint config hash mismatch".to_string());
    }
    if !meta.model_sig.is_empty() && meta.model_sig != model_signature(config) {
        return Err("checkpoint model signature mismatch".to_string());
    }
    if !meta.dtype.is_empty() && meta.dtype != config.model.dtype {
        return Err("checkpoint dtype mismatch".to_string());
    }
    if !meta.device.is_empty() && meta.device != config.model.device {
        return Err("checkpoint device mismatch".to_string());
    }
    if meta.world_size != 0 {
        if meta.world_size != config.world_size {
            return Err("checkpoint world_size mismatch".to_string());
        }
        if meta.rank != config.rank {
            return Err("checkpoint rank mismatch".to_string());
        }
    }
    if meta.grad_accum_steps != 0 && meta.grad_accum_steps != config.grad_accum_steps {
        return Err("checkpoint grad_accum_steps mismatch".to_string());
    }
    if let Some(norm) = meta.grad_clip_norm {
        if config.grad_clip_norm != Some(norm) {
            return Err("checkpoint grad_clip_norm mismatch".to_string());
        }
    }
    if let Some(amp) = meta.amp.as_ref() {
        if *amp != config.amp {
            return Err("checkpoint amp config mismatch".to_string());
        }
    }
    Ok(Some(meta))
}

fn write_error_checkpoint(
    engine: &enkai_runtime::engine::Engine,
    config: &TrainConfig,
    step: usize,
    tokens: u64,
    error: &str,
) -> Result<(), String> {
    let meta = serde_json::json!({
        "format_version": 1,
        "step": step as u64,
        "tokens": tokens,
        "loss": "nan",
        "config_hash": hash_config_hex(config),
        "model_sig": model_signature(config),
        "dtype": config.model.dtype,
        "device": config.model.device,
        "world_size": config.world_size,
        "rank": config.rank,
        "grad_accum_steps": config.grad_accum_steps,
        "grad_clip_norm": config.grad_clip_norm,
        "amp": config.amp.clone(),
        "error": error
    });
    let text = serde_json::to_string(&meta).map_err(|err| err.to_string())?;
    rt_checkpoint_save(engine, &config.checkpoint_dir, &text).map_err(|err| err.to_string())
}

fn wrap_program_with_main(program: &Program, main_global: u16) -> Program {
    let mut program = program.clone();
    let mut chunk = Chunk::new();
    let const_idx = chunk.add_constant(Constant::Function(program.main));
    chunk.write(Instruction::Const(const_idx), 0);
    chunk.write(Instruction::Call(0), 0);
    chunk.write(Instruction::LoadGlobal(main_global), 0);
    chunk.write(Instruction::Call(0), 0);
    chunk.write(Instruction::Return, 0);
    let wrapper = ByteFunction {
        name: Some("<config_main>".to_string()),
        arity: 0,
        chunk,
        source_name: None,
    };
    let idx = program.functions.len() as u16;
    program.functions.push(wrapper);
    program.main = idx;
    program
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn escape_enkai_string(input: &str) -> String {
        input.replace('\\', "\\\\").replace('\"', "\\\"")
    }

    fn write_config(path: &Path, json: &serde_json::Value) -> Result<(), String> {
        let text = json.to_string();
        let escaped = escape_enkai_string(&text);
        let source = format!("fn main() ::\n    return json.parse(\"{}\")\n::\n", escaped);
        fs::write(path, source).map_err(|err| err.to_string())
    }

    #[test]
    fn train_runs_and_resumes() {
        let dir = tempdir().expect("tempdir");
        let data = dir.path().join("data.txt");
        fs::write(&data, "alpha beta gamma\ndelta epsilon").unwrap();
        let ckpt = dir.path().join("ckpt");
        let config_path = dir.path().join("config.enk");
        let config = serde_json::json!({
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
            "drop_remainder": false,
            "tokenizer_train": { "path": data.to_string_lossy(), "vocab_size": 8 }
        });
        write_config(&config_path, &config).expect("config");
        train(&config_path).expect("train");
        let latest = latest_checkpoint(&ckpt)
            .expect("latest")
            .expect("checkpoint");
        assert!(latest.ends_with("step_00000001"));

        let config2 = serde_json::json!({
            "config_version": 1,
            "backend": "cpu",
            "vocab_size": 8,
            "hidden_size": 4,
            "seq_len": 4,
            "batch_size": 2,
            "lr": 0.1,
            "dataset_path": data.to_string_lossy(),
            "checkpoint_dir": ckpt.to_string_lossy(),
            "max_steps": 2,
            "save_every": 1,
            "log_every": 1,
            "drop_remainder": false,
            "tokenizer_train": { "path": data.to_string_lossy(), "vocab_size": 8 }
        });
        write_config(&config_path, &config2).expect("config2");
        train(&config_path).expect("train resume");
        let latest = latest_checkpoint(&ckpt)
            .expect("latest")
            .expect("checkpoint");
        assert!(latest.ends_with("step_00000002"));
    }

    #[test]
    fn eval_runs() {
        let dir = tempdir().expect("tempdir");
        let data = dir.path().join("data.txt");
        fs::write(&data, "alpha beta gamma\ndelta epsilon").unwrap();
        let ckpt = dir.path().join("ckpt");
        let config_path = dir.path().join("config.enk");
        let config = serde_json::json!({
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
        write_config(&config_path, &config).expect("config");
        train(&config_path).expect("train");
        eval(&config_path).expect("eval");
    }

    #[test]
    fn train_runs_and_resumes_native_backend() {
        if std::env::var("ENKAI_TORCH").ok().as_deref() != Some("1") {
            return;
        }
        let dir = tempdir().expect("tempdir");
        let data = dir.path().join("data.txt");
        fs::write(&data, "alpha beta gamma\ndelta epsilon").unwrap();
        let ckpt = dir.path().join("ckpt_native");
        let config_path = dir.path().join("config_native.enk");
        let config = serde_json::json!({
            "config_version": 1,
            "backend": "native",
            "vocab_size": 32,
            "hidden_size": 32,
            "seq_len": 4,
            "batch_size": 2,
            "lr": 0.001,
            "dataset_path": data.to_string_lossy(),
            "checkpoint_dir": ckpt.to_string_lossy(),
            "max_steps": 2,
            "save_every": 1,
            "log_every": 1,
            "drop_remainder": false,
            "model": {
                "d_model": 32,
                "n_layers": 2,
                "n_heads": 4,
                "device": "cpu",
                "dtype": "fp32",
                "vocab_size": 32,
                "hidden_size": 32
            },
            "optim": {
                "lr": 0.001,
                "beta1": 0.9,
                "beta2": 0.999,
                "eps": 1e-8,
                "weight_decay": 0.0
            },
            "tokenizer_train": { "path": data.to_string_lossy(), "vocab_size": 32 }
        });
        write_config(&config_path, &config).expect("config");
        train(&config_path).expect("train native");
        let latest = Path::new(&ckpt).join("latest");
        assert!(latest.is_dir());
        train(&config_path).expect("train native resume");
    }
}
