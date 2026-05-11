use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread;

use crate::tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};
use sha1::{Digest, Sha1};

#[derive(Debug, Clone)]
pub struct DatasetConfig {
    pub seq_len: usize,
    pub batch_size: usize,
    pub add_eos: bool,
    pub drop_remainder: bool,
    pub pad_id: u32,
    pub seed: Option<u64>,
    pub shuffle: bool,
    pub prefetch_batches: usize,
}

impl DatasetConfig {
    pub fn new(seq_len: usize, batch_size: usize) -> Self {
        Self {
            seq_len,
            batch_size,
            add_eos: true,
            drop_remainder: true,
            pad_id: 0,
            seed: None,
            shuffle: false,
            prefetch_batches: 0,
        }
    }
}

#[derive(Debug)]
pub struct Batch {
    pub input_ids: Vec<u32>,
    pub target_ids: Vec<u32>,
    pub attention_mask: Vec<u8>,
    pub batch_size: usize,
    pub seq_len: usize,
    pub token_count: usize,
    pub packing_efficiency: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DatasetCursor {
    pub current_file_index: usize,
    pub current_line: u64,
    pub token_buffer: Vec<u32>,
    pub exhausted: bool,
    pub emitted_batches: u64,
}

#[derive(Debug)]
pub struct DatasetStream {
    inner: Option<DatasetStreamInner>,
    prefetch: Option<PrefetchState>,
}

#[derive(Debug)]
struct DatasetStreamInner {
    files: Vec<PathBuf>,
    current: usize,
    reader: Option<BufReader<File>>,
    tokenizer: Tokenizer,
    config: DatasetConfig,
    token_buffer: Vec<u32>,
    exhausted: bool,
    current_line: u64,
    emitted_batches: u64,
}

struct PrefetchState {
    rx: mpsc::Receiver<Result<Option<Batch>, String>>,
    handle: thread::JoinHandle<()>,
}

impl std::fmt::Debug for PrefetchState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrefetchState").finish()
    }
}

impl DatasetStream {
    pub fn new(
        files: Vec<PathBuf>,
        tokenizer: Tokenizer,
        config: DatasetConfig,
    ) -> Result<Self, String> {
        let inner = DatasetStreamInner::new(files, tokenizer, config.clone())?;
        if config.prefetch_batches > 0 {
            let (tx, rx) = mpsc::sync_channel(config.prefetch_batches);
            let handle = thread::spawn(move || {
                let mut inner = inner;
                loop {
                    let batch = inner.next_batch();
                    let done = matches!(batch, Ok(None));
                    if tx.send(batch).is_err() {
                        break;
                    }
                    if done {
                        break;
                    }
                }
            });
            Ok(Self {
                inner: None,
                prefetch: Some(PrefetchState { rx, handle }),
            })
        } else {
            Ok(Self {
                inner: Some(inner),
                prefetch: None,
            })
        }
    }

    pub fn next_batch(&mut self) -> Result<Option<Batch>, String> {
        if let Some(prefetch) = &self.prefetch {
            match prefetch.rx.recv() {
                Ok(batch) => batch,
                Err(_) => Ok(None),
            }
        } else if let Some(inner) = &mut self.inner {
            inner.next_batch()
        } else {
            Ok(None)
        }
    }

    pub fn cursor(&self) -> Result<DatasetCursor, String> {
        if self.prefetch.is_some() {
            return Err(
                "Dataset cursor snapshots require prefetch_batches = 0 for deterministic replay"
                    .to_string(),
            );
        }
        self.inner
            .as_ref()
            .map(DatasetStreamInner::cursor)
            .ok_or_else(|| "Dataset stream is not cursor-addressable".to_string())
    }

    pub fn restore_cursor(&mut self, cursor: DatasetCursor) -> Result<(), String> {
        if self.prefetch.is_some() {
            return Err(
                "Dataset cursor restore requires prefetch_batches = 0 for deterministic replay"
                    .to_string(),
            );
        }
        self.inner
            .as_mut()
            .ok_or_else(|| "Dataset stream is not cursor-addressable".to_string())?
            .restore_cursor(cursor)
    }
}

impl Drop for DatasetStream {
    fn drop(&mut self) {
        if let Some(prefetch) = self.prefetch.take() {
            drop(prefetch.rx);
            let _ = prefetch.handle.join();
        }
    }
}

impl DatasetStreamInner {
    fn new(
        files: Vec<PathBuf>,
        tokenizer: Tokenizer,
        config: DatasetConfig,
    ) -> Result<Self, String> {
        if config.seq_len == 0 || config.batch_size == 0 {
            return Err("Dataset config requires seq_len and batch_size > 0".to_string());
        }
        let mut files = files;
        if config.shuffle {
            let seed = config
                .seed
                .ok_or_else(|| "Dataset shuffle requires seed".to_string())?;
            shuffle_paths(&mut files, seed);
        }
        Ok(Self {
            files,
            current: 0,
            reader: None,
            tokenizer,
            config,
            token_buffer: Vec::new(),
            exhausted: false,
            current_line: 0,
            emitted_batches: 0,
        })
    }

    fn next_batch(&mut self) -> Result<Option<Batch>, String> {
        if self.exhausted && self.token_buffer.is_empty() {
            return Ok(None);
        }
        let mut sequences: Vec<(Vec<u32>, Vec<u8>)> = Vec::new();
        while sequences.len() < self.config.batch_size {
            if self.token_buffer.len() < self.config.seq_len && !self.fill_buffer()? {
                self.exhausted = true;
            }
            if self.token_buffer.len() >= self.config.seq_len {
                let seq: Vec<u32> = self.token_buffer.drain(..self.config.seq_len).collect();
                sequences.push((seq, vec![1u8; self.config.seq_len]));
                continue;
            }
            if self.token_buffer.is_empty() {
                break;
            }
            if self.config.drop_remainder {
                self.token_buffer.clear();
                break;
            }
            let mut seq: Vec<u32> = self.token_buffer.drain(..).collect();
            let real = seq.len();
            seq.resize(self.config.seq_len, self.config.pad_id);
            let mut mask = vec![1u8; real];
            mask.resize(self.config.seq_len, 0u8);
            sequences.push((seq, mask));
            break;
        }
        if sequences.is_empty() {
            return Ok(None);
        }
        if sequences.len() < self.config.batch_size {
            if self.config.drop_remainder {
                return Ok(None);
            }
            while sequences.len() < self.config.batch_size {
                sequences.push((
                    vec![self.config.pad_id; self.config.seq_len],
                    vec![0u8; self.config.seq_len],
                ));
            }
        }
        let total = self.config.batch_size * self.config.seq_len;
        let mut input_ids = Vec::with_capacity(total);
        let mut target_ids = Vec::with_capacity(total);
        let mut attention_mask = Vec::with_capacity(total);
        let mut token_count = 0usize;
        for (seq, mask) in sequences {
            for idx in 0..self.config.seq_len {
                let token = seq[idx];
                input_ids.push(token);
                let target = if idx + 1 < self.config.seq_len {
                    seq[idx + 1]
                } else {
                    self.config.pad_id
                };
                target_ids.push(target);
                attention_mask.push(mask[idx]);
                if mask[idx] == 1 {
                    token_count += 1;
                }
            }
        }
        let packing_efficiency = if total == 0 {
            0.0
        } else {
            token_count as f32 / total as f32
        };
        self.emitted_batches += 1;
        Ok(Some(Batch {
            input_ids,
            target_ids,
            attention_mask,
            batch_size: self.config.batch_size,
            seq_len: self.config.seq_len,
            token_count,
            packing_efficiency,
        }))
    }

    fn fill_buffer(&mut self) -> Result<bool, String> {
        while self.token_buffer.len() < self.config.seq_len {
            let line = match self.next_line()? {
                Some(line) => line,
                None => return Ok(false),
            };
            if line.trim().is_empty() {
                continue;
            }
            let ids = self.tokenizer.encode(&line, self.config.add_eos);
            self.token_buffer.extend_from_slice(&ids);
        }
        Ok(true)
    }

    fn next_line(&mut self) -> Result<Option<String>, String> {
        loop {
            if self.reader.is_none() {
                if self.current >= self.files.len() {
                    return Ok(None);
                }
                let file = File::open(&self.files[self.current]).map_err(|err| {
                    format!(
                        "Failed to open {}: {}",
                        self.files[self.current].display(),
                        err
                    )
                })?;
                self.reader = Some(BufReader::new(file));
                self.current_line = 0;
            }
            let mut line = String::new();
            let read = match self.reader.as_mut() {
                Some(reader) => reader
                    .read_line(&mut line)
                    .map_err(|err| format!("Failed to read dataset: {}", err))?,
                None => 0,
            };
            if read == 0 {
                self.reader = None;
                self.current += 1;
                self.current_line = 0;
                continue;
            }
            self.current_line += 1;
            if line.ends_with('\n') {
                line.pop();
                if line.ends_with('\r') {
                    line.pop();
                }
            }
            return Ok(Some(line));
        }
    }

    fn cursor(&self) -> DatasetCursor {
        DatasetCursor {
            current_file_index: self.current,
            current_line: self.current_line,
            token_buffer: self.token_buffer.clone(),
            exhausted: self.exhausted,
            emitted_batches: self.emitted_batches,
        }
    }

    fn restore_cursor(&mut self, cursor: DatasetCursor) -> Result<(), String> {
        if cursor.current_file_index > self.files.len() {
            return Err("Dataset cursor current_file_index is out of range".to_string());
        }
        self.current = cursor.current_file_index;
        self.current_line = cursor.current_line;
        self.token_buffer = cursor.token_buffer;
        self.exhausted = cursor.exhausted;
        self.emitted_batches = cursor.emitted_batches;
        self.reader = None;
        if self.current < self.files.len() {
            self.open_reader_at_current_line()?;
        }
        Ok(())
    }

    fn open_reader_at_current_line(&mut self) -> Result<(), String> {
        let file = File::open(&self.files[self.current]).map_err(|err| {
            format!(
                "Failed to open {}: {}",
                self.files[self.current].display(),
                err
            )
        })?;
        let mut reader = BufReader::new(file);
        for _ in 0..self.current_line {
            let mut discarded = String::new();
            let read = reader
                .read_line(&mut discarded)
                .map_err(|err| format!("Failed to restore dataset cursor: {}", err))?;
            if read == 0 {
                return Err("Dataset cursor line offset is beyond end of file".to_string());
            }
        }
        self.reader = Some(reader);
        Ok(())
    }
}

pub fn resolve_dataset_paths(path: &str) -> Result<Vec<PathBuf>, String> {
    if path.contains('*') || path.contains('?') || path.contains('[') {
        let mut matches = Vec::new();
        for entry in glob::glob(path).map_err(|err| err.to_string())? {
            let entry = entry.map_err(|err| err.to_string())?;
            if entry.is_file() {
                matches.push(entry);
            }
        }
        matches.sort();
        if matches.is_empty() {
            return Err(format!("Dataset glob matched no files: {}", path));
        }
        return Ok(matches);
    }
    let path = Path::new(path);
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    if path.is_dir() {
        let mut files = Vec::new();
        collect_files(path, &mut files)?;
        files.sort();
        if files.is_empty() {
            return Err(format!(
                "Dataset directory has no files: {}",
                path.display()
            ));
        }
        return Ok(files);
    }
    Err(format!("Dataset path not found: {}", path.display()))
}

fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries =
        fs::read_dir(dir).map_err(|err| format!("Failed to read {}: {}", dir.display(), err))?;
    for entry in entries {
        let entry = entry.map_err(|err| err.to_string())?;
        let path = entry.path();
        if path.is_dir() {
            collect_files(&path, out)?;
        } else if path.is_file() {
            out.push(path);
        }
    }
    Ok(())
}

pub fn dataset_pipeline_manifest(
    files: &[PathBuf],
    tokenizer: &Tokenizer,
    config: &DatasetConfig,
) -> Result<serde_json::Value, String> {
    let mut file_entries = Vec::new();
    let mut dataset_hasher = Sha1::new();
    for path in files {
        let bytes =
            fs::read(path).map_err(|err| format!("Failed to read {}: {}", path.display(), err))?;
        let file_sha1 = format!("{:x}", Sha1::digest(&bytes));
        dataset_hasher.update(path.to_string_lossy().as_bytes());
        dataset_hasher.update((bytes.len() as u64).to_le_bytes());
        dataset_hasher.update(file_sha1.as_bytes());
        file_entries.push(serde_json::json!({
            "path": path.to_string_lossy(),
            "bytes": bytes.len(),
            "sha1": file_sha1
        }));
    }
    Ok(serde_json::json!({
        "schema_version": 1,
        "pipeline": "enkai_dataset_stream_v1",
        "dataset_sha1": format!("{:x}", dataset_hasher.finalize()),
        "tokenizer_sha1": tokenizer.fingerprint(),
        "files": file_entries,
        "config": {
            "seq_len": config.seq_len,
            "batch_size": config.batch_size,
            "add_eos": config.add_eos,
            "drop_remainder": config.drop_remainder,
            "pad_id": config.pad_id,
            "seed": config.seed,
            "shuffle": config.shuffle,
            "prefetch_batches": config.prefetch_batches
        },
        "features": {
            "streaming_reader": true,
            "deterministic_shuffle": config.shuffle,
            "packed_sequences": true,
            "checkpointable_cursor": config.prefetch_batches == 0,
            "tokenizer_fingerprint_required": true,
            "deterministic_replay": true
        },
        "deterministic_errors": [
            "Dataset shuffle requires seed",
            "Dataset cursor snapshots require prefetch_batches = 0 for deterministic replay",
            "Dataset cursor line offset is beyond end of file"
        ]
    }))
}

fn shuffle_paths(paths: &mut [PathBuf], seed: u64) {
    if paths.len() <= 1 {
        return;
    }
    let mut state = seed ^ 0x9e3779b97f4a7c15;
    for i in (1..paths.len()).rev() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (state % ((i + 1) as u64)) as usize;
        paths.swap(i, j);
    }
}
