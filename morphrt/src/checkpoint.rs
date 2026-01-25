use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMeta {
    pub step: u64,
    pub tokens: u64,
    pub loss: f64,
    pub config_hash: String,
}

#[derive(Debug, Clone)]
pub struct CheckpointState {
    pub weights: Vec<f32>,
    pub optimizer: Vec<f32>,
    pub meta: CheckpointMeta,
}

pub fn save_checkpoint(dir: &Path, state: &CheckpointState) -> Result<PathBuf, String> {
    fs::create_dir_all(dir)
        .map_err(|err| format!("Failed to create {}: {}", dir.display(), err))?;
    let checkpoint_dir = dir.join(format!("step_{:08}", state.meta.step));
    let tmp_dir = dir.join(format!(".tmp_step_{:08}", state.meta.step));
    if tmp_dir.exists() {
        let _ = fs::remove_dir_all(&tmp_dir);
    }
    fs::create_dir_all(&tmp_dir)
        .map_err(|err| format!("Failed to create {}: {}", tmp_dir.display(), err))?;
    let meta_path = tmp_dir.join("meta.json");
    let weights_path = tmp_dir.join("weights.bin");
    let opt_path = tmp_dir.join("optimizer.bin");
    let meta_text = serde_json::to_string_pretty(&state.meta).map_err(|err| err.to_string())?;
    write_atomic(&meta_path, meta_text.as_bytes())?;
    write_f32_atomic(&weights_path, &state.weights)?;
    write_f32_atomic(&opt_path, &state.optimizer)?;
    if checkpoint_dir.exists() {
        fs::remove_dir_all(&checkpoint_dir)
            .map_err(|err| format!("Failed to remove {}: {}", checkpoint_dir.display(), err))?;
    }
    fs::rename(&tmp_dir, &checkpoint_dir)
        .map_err(|err| format!("Failed to move checkpoint: {}", err))?;
    Ok(checkpoint_dir)
}

pub fn load_checkpoint(path: &Path) -> Result<CheckpointState, String> {
    let meta_path = path.join("meta.json");
    let weights_path = path.join("weights.bin");
    let opt_path = path.join("optimizer.bin");
    let meta_text = fs::read_to_string(&meta_path)
        .map_err(|err| format!("Failed to read {}: {}", meta_path.display(), err))?;
    let meta: CheckpointMeta = serde_json::from_str(&meta_text).map_err(|err| err.to_string())?;
    let weights = read_f32(&weights_path)?;
    let optimizer = read_f32(&opt_path)?;
    Ok(CheckpointState {
        weights,
        optimizer,
        meta,
    })
}

pub fn latest_checkpoint(dir: &Path) -> Result<Option<PathBuf>, String> {
    if !dir.exists() {
        return Ok(None);
    }
    let mut best: Option<(u64, PathBuf)> = None;
    for entry in
        fs::read_dir(dir).map_err(|err| format!("Failed to read {}: {}", dir.display(), err))?
    {
        let entry = entry.map_err(|err| err.to_string())?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        if let Some(step) = parse_step_dir(&path) {
            match best {
                Some((best_step, _)) if step <= best_step => {}
                _ => best = Some((step, path)),
            }
        }
    }
    Ok(best.map(|(_, path)| path))
}

pub fn rotate_checkpoints(dir: &Path, keep_last: usize) -> Result<(), String> {
    if keep_last == 0 {
        return Ok(());
    }
    let mut checkpoints: Vec<(u64, PathBuf)> = Vec::new();
    if !dir.exists() {
        return Ok(());
    }
    for entry in
        fs::read_dir(dir).map_err(|err| format!("Failed to read {}: {}", dir.display(), err))?
    {
        let entry = entry.map_err(|err| err.to_string())?;
        let path = entry.path();
        if let Some(step) = parse_step_dir(&path) {
            checkpoints.push((step, path));
        }
    }
    checkpoints.sort_by(|a, b| b.0.cmp(&a.0));
    for (_, path) in checkpoints.into_iter().skip(keep_last) {
        fs::remove_dir_all(&path)
            .map_err(|err| format!("Failed to remove {}: {}", path.display(), err))?;
    }
    Ok(())
}

fn parse_step_dir(path: &Path) -> Option<u64> {
    let name = path.file_name()?.to_string_lossy();
    let stripped = name.strip_prefix("step_")?;
    stripped.parse::<u64>().ok()
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, bytes).map_err(|err| format!("Failed to write {}: {}", tmp.display(), err))?;
    fs::rename(&tmp, path)
        .map_err(|err| format!("Failed to rename {}: {}", path.display(), err))?;
    Ok(())
}

fn write_f32_atomic(path: &Path, values: &[f32]) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    write_atomic(path, &bytes)
}

fn read_f32(path: &Path) -> Result<Vec<f32>, String> {
    let bytes =
        fs::read(path).map_err(|err| format!("Failed to read {}: {}", path.display(), err))?;
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "Invalid checkpoint data size in {}",
            path.display()
        ));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}
