use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use crate::tokenizer::Tokenizer;

#[derive(Debug, Clone)]
pub struct DatasetConfig {
    pub seq_len: usize,
    pub batch_size: usize,
    pub add_eos: bool,
    pub drop_remainder: bool,
    pub pad_id: u32,
    pub seed: Option<u64>,
    pub shuffle: bool,
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
}

#[derive(Debug)]
pub struct DatasetStream {
    files: Vec<PathBuf>,
    current: usize,
    reader: Option<BufReader<File>>,
    tokenizer: Tokenizer,
    config: DatasetConfig,
    token_buffer: Vec<u32>,
    exhausted: bool,
}

impl DatasetStream {
    pub fn new(
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
        })
    }

    pub fn next_batch(&mut self) -> Result<Option<Batch>, String> {
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
            }
        }
        Ok(Some(Batch {
            input_ids,
            target_ids,
            attention_mask,
            batch_size: self.config.batch_size,
            seq_len: self.config.seq_len,
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
                continue;
            }
            if line.ends_with('\n') {
                line.pop();
                if line.ends_with('\r') {
                    line.pop();
                }
            }
            return Ok(Some(line));
        }
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
    let entries = std::fs::read_dir(dir)
        .map_err(|err| format!("Failed to read {}: {}", dir.display(), err))?;
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

fn shuffle_paths(paths: &mut [PathBuf], seed: u64) {
    if paths.len() <= 1 {
        return;
    }
    let mut state = seed ^ 0x9e3779b97f4a7c15;
    for i in (1..paths.len()).rev() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        let j = (state % ((i + 1) as u64)) as usize;
        paths.swap(i, j);
    }
}
