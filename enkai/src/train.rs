use std::collections::hash_map::DefaultHasher;
use std::fs::{self, OpenOptions};
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;

use enkai_compiler::bytecode::{ByteFunction, Chunk, Constant, Instruction, Program};
use enkai_compiler::compiler::{compile_package, CompileError};
use enkai_compiler::modules::load_package;
use enkai_compiler::{TypeChecker, TypeError};
use enkai_runtime::checkpoint::{
    latest_checkpoint, load_checkpoint, rotate_checkpoints, save_checkpoint, CheckpointMeta,
    CheckpointState,
};
use enkai_runtime::dataset::{resolve_dataset_paths, Batch, DatasetConfig, DatasetStream};
use enkai_runtime::object::Obj;
use enkai_runtime::tokenizer::{Tokenizer, TrainConfig as TokenizerTrainConfig};
use enkai_runtime::{
    engine::{
        checkpoint_load as rt_checkpoint_load, checkpoint_save as rt_checkpoint_save,
        eval_step as rt_eval_step, init, train_step as rt_train_step,
        CheckpointConfig as RtCkptConfig, DataConfig as RtDataConfig, LogConfig as RtLogConfig,
        ModelConfig as RtModelConfig, OptimConfig as RtOptimConfig, TrainConfig as RtTrainConfig,
    },
    Value, VM,
};

#[derive(Debug, Clone)]
struct TrainConfig {
    backend: String,
    vocab_size: usize,
    hidden_size: usize,
    seq_len: usize,
    batch_size: usize,
    lr: f32,
    model: ModelConfig,
    optim: OptimConfig,
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

    fn train_step(&mut self, batch: &Batch, opt: &mut AdamWMulti, groups: &[ParamGroup]) -> f32 {
        let total = batch.input_ids.len();
        if total == 0 {
            return 0.0;
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
        opt.step(&mut self.params, &grads, groups);
        loss
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
}

pub fn train(config_path: &Path) -> Result<(), String> {
    let config_value = load_config_value(config_path)?;
    let config = parse_train_config(&config_value)?;
    let tokenizer = build_tokenizer(&config)?;
    let dataset_paths = resolve_dataset_paths(&config.dataset_path)?;
    let mut data_cfg = DatasetConfig::new(config.seq_len, config.batch_size);
    data_cfg.add_eos = config.add_eos;
    data_cfg.drop_remainder = config.drop_remainder;
    data_cfg.pad_id = config.pad_id;
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
        while step < config.max_steps {
            let batch = match stream.next_batch()? {
                Some(batch) => batch,
                None => break,
            };
            let metrics = match rt_train_step(&mut engine, &batch) {
                Ok(m) => m,
                Err(err) => {
                    let _ =
                        write_error_checkpoint(&engine, &config, step, tokens, &err.to_string());
                    return Err(format!("Train error: {}", err));
                }
            };
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
            tokens += batch.input_ids.len() as u64;
            step += 1;
            if step == 1 || step.is_multiple_of(config.log_every) {
                let event = LogEvent {
                    step,
                    loss: metrics.loss,
                    tokens,
                    lr: config.optim.lr as f32,
                    elapsed_ms: start.elapsed().as_millis(),
                    event: "step".to_string(),
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
                };
                let line = serde_json::to_string(&event).map_err(|err| err.to_string())?;
                writeln!(log_file, "{}", line).map_err(|err| err.to_string())?;
                println!("saved checkpoint latest");
            }
        }
    } else {
        let seed = hash_config(&config);
        let mut model = TinyModel::new(config.vocab_size, config.hidden_size, seed);
        let groups = model.param_groups();
        let group_sizes: Vec<usize> = groups.iter().map(|group| group.len).collect();
        let mut opt = AdamWMulti::new(&group_sizes, config.lr);
        if let Some(latest) =
            latest_checkpoint(Path::new(&config.checkpoint_dir)).map_err(|err| err.to_string())?
        {
            let state = load_checkpoint(&latest).map_err(|err| err.to_string())?;
            if state.meta.config_hash != hash_config_hex(&config) {
                return Err("checkpoint config hash mismatch".to_string());
            }
            model.set_params(&state.weights)?;
            opt.load_state(&state.optimizer)?;
            step = state.meta.step as usize;
            tokens = state.meta.tokens;
        }
        while step < config.max_steps {
            let batch = match stream.next_batch()? {
                Some(batch) => batch,
                None => break,
            };
            let loss = model.train_step(&batch, &mut opt, &groups);
            if !loss.is_finite() {
                return Err("non-finite loss detected".to_string());
            }
            tokens += batch.input_ids.len() as u64;
            step += 1;
            if step == 1 || step.is_multiple_of(config.log_every) {
                let event = LogEvent {
                    step,
                    loss,
                    tokens,
                    lr: config.lr,
                    elapsed_ms: start.elapsed().as_millis(),
                    event: "step".to_string(),
                };
                let line = serde_json::to_string(&event).map_err(|err| err.to_string())?;
                writeln!(log_file, "{}", line).map_err(|err| err.to_string())?;
                println!("step {} loss {:.4} tokens {}", step, loss, tokens);
            }
            if step == config.max_steps || step.is_multiple_of(config.save_every) {
                let meta = CheckpointMeta {
                    step: step as u64,
                    tokens,
                    loss: loss as f64,
                    config_hash: hash_config_hex(&config),
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
                    loss,
                    tokens,
                    lr: config.lr,
                    elapsed_ms: start.elapsed().as_millis(),
                    event: "checkpoint".to_string(),
                };
                let line = serde_json::to_string(&event).map_err(|err| err.to_string())?;
                writeln!(log_file, "{}", line).map_err(|err| err.to_string())?;
                println!("saved checkpoint {}", path.display());
            }
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
        let seed = hash_config(&config);
        let mut model = TinyModel::new(config.vocab_size, config.hidden_size, seed);
        let latest = latest_checkpoint(Path::new(&config.checkpoint_dir))
            .map_err(|err| err.to_string())?
            .ok_or_else(|| "No checkpoint found".to_string())?;
        let state = load_checkpoint(&latest).map_err(|err| err.to_string())?;
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
            let cfg = TokenizerTrainConfig {
                vocab_size: *vocab_size,
                ..TokenizerTrainConfig::default()
            };
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
    let map = value_as_record(value)?;
    let model = map.get("model").and_then(|v| as_record(v).ok());
    let backend = map
        .get("backend")
        .and_then(|v| as_string(v).ok())
        .unwrap_or_else(|| "cpu".to_string());
    let vocab_size = record_usize(model.unwrap_or(map), "vocab_size")?;
    let hidden_size = record_usize(model.unwrap_or(map), "hidden_size")?;
    let d_model = record_usize_default(model.unwrap_or(map), "d_model", hidden_size);
    let n_layers = record_usize_default(model.unwrap_or(map), "n_layers", 2);
    let n_heads = record_usize_default(model.unwrap_or(map), "n_heads", 4);
    let device = record_string_default(model.unwrap_or(map), "device", "cpu");
    let dtype = record_string_default(model.unwrap_or(map), "dtype", "fp32");
    let seq_len = record_usize(map, "seq_len")?;
    let batch_size = record_usize(map, "batch_size")?;
    let lr = record_f32(map, "lr")?;
    let optim = map.get("optim").and_then(|v| as_record(v).ok());
    let opt_lr = record_f64_default(optim.unwrap_or(map), "lr", lr as f64);
    let opt_beta1 = record_f64_default(optim.unwrap_or(map), "beta1", 0.9);
    let opt_beta2 = record_f64_default(optim.unwrap_or(map), "beta2", 0.999);
    let opt_eps = record_f64_default(optim.unwrap_or(map), "eps", 1e-8);
    let opt_wd = record_f64_default(optim.unwrap_or(map), "weight_decay", 0.01);
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
    let tokenizer = if let Some(value) = map.get("tokenizer_path") {
        TokenizerConfig::Load(as_string(value)?)
    } else if let Some(value) = map.get("tokenizer_train") {
        let tmap = value_as_record(value)?;
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

fn value_as_record(value: &Value) -> Result<&std::collections::HashMap<String, Value>, String> {
    match value {
        Value::Obj(obj) => match obj.as_obj() {
            Obj::Record(map) => Ok(map),
            _ => Err("Expected record config".to_string()),
        },
        _ => Err("Expected record config".to_string()),
    }
}

fn as_record(value: &Value) -> Result<&std::collections::HashMap<String, Value>, String> {
    match value {
        Value::Obj(obj) => match obj.as_obj() {
            Obj::Record(map) => Ok(map),
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

fn hash_config(config: &TrainConfig) -> u64 {
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

#[derive(Serialize, serde::Deserialize)]
struct NativeCheckpointMeta {
    step: u64,
    tokens: u64,
    loss: f32,
    config_hash: String,
    model_sig: String,
    dtype: String,
    device: String,
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
        step,
        tokens,
        loss,
        config_hash: hash_config_hex(config),
        model_sig: model_signature(config),
        dtype: config.model.dtype.clone(),
        device: config.model.device.clone(),
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
            seed: hash_config(config),
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
        world_size: 1,
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
    if meta.config_hash != hash_config_hex(config) {
        return Err("checkpoint config hash mismatch".to_string());
    }
    if meta.model_sig != model_signature(config) {
        return Err("checkpoint model signature mismatch".to_string());
    }
    if meta.dtype != config.model.dtype {
        return Err("checkpoint dtype mismatch".to_string());
    }
    if meta.device != config.model.device {
        return Err("checkpoint device mismatch".to_string());
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
        "step": step as u64,
        "tokens": tokens,
        "loss": "nan",
        "config_hash": hash_config_hex(config),
        "model_sig": model_signature(config),
        "dtype": config.model.dtype,
        "device": config.model.device,
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
