use std::fs::{create_dir_all, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::engine::Metrics;
use crate::error::RuntimeError;

#[derive(Debug, Clone)]
pub struct Logger {
    log_path: PathBuf,
}

impl Logger {
    pub fn new(dir: impl AsRef<Path>) -> Result<Self, RuntimeError> {
        let dir = dir.as_ref();
        create_dir_all(dir).map_err(|e| RuntimeError::new(&e.to_string()))?;
        let log_path = dir.join("train.jsonl");
        Ok(Self { log_path })
    }

    pub fn append(&self, entry: &Metrics) -> Result<(), RuntimeError> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_path)
            .map_err(|e| RuntimeError::new(&e.to_string()))?;
        let line =
            serde_json::to_string(entry).map_err(|e| RuntimeError::new(&e.to_string()))? + "\n";
        file.write_all(line.as_bytes())
            .map_err(|e| RuntimeError::new(&e.to_string()))
    }
}

/// Convenience helper for console summaries.
pub fn print_summary(prefix: &str, metrics: &Metrics) {
    println!(
        "{prefix} step={} loss={:.4} tok/s={:.1} time_ms={:.1} lr={:.6}",
        metrics.step, metrics.loss, metrics.tokens_per_sec, metrics.step_time_ms, metrics.lr
    );
}
