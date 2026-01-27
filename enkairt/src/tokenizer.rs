use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct Tokenizer {
    vocab: Vec<String>,
    map: HashMap<String, u32>,
    unk_id: u32,
    eos_id: u32,
}

#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub vocab_size: usize,
    pub lowercase: bool,
    pub min_freq: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            lowercase: false,
            min_freq: 1,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct TokenizerFile {
    vocab: Vec<String>,
    unk_id: u32,
    eos_id: u32,
}

impl Tokenizer {
    pub fn train_from_path(path: &Path, config: &TrainConfig) -> Result<Self, String> {
        if config.vocab_size < 2 {
            return Err("Tokenizer vocab_size must be at least 2".to_string());
        }
        let files = collect_files(path)?;
        if files.is_empty() {
            return Err("Tokenizer training path has no files".to_string());
        }
        let mut counts: HashMap<String, usize> = HashMap::new();
        for file in files {
            let contents = fs::read_to_string(&file)
                .map_err(|err| format!("Failed to read {}: {}", file.display(), err))?;
            for mut token in contents.split_whitespace().map(|s| s.to_string()) {
                if config.lowercase {
                    token = token.to_lowercase();
                }
                *counts.entry(token).or_insert(0) += 1;
            }
        }
        let mut items: Vec<(String, usize)> = counts
            .into_iter()
            .filter(|(_, freq)| *freq >= config.min_freq)
            .collect();
        items.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let mut vocab = Vec::with_capacity(config.vocab_size);
        vocab.push("<unk>".to_string());
        vocab.push("<eos>".to_string());
        let max = config.vocab_size.saturating_sub(2);
        for (token, _) in items.into_iter().take(max) {
            vocab.push(token);
        }
        Self::from_vocab(vocab, 0, 1)
    }

    pub fn from_vocab(vocab: Vec<String>, unk_id: u32, eos_id: u32) -> Result<Self, String> {
        if vocab.is_empty() {
            return Err("Tokenizer vocab is empty".to_string());
        }
        let mut map = HashMap::new();
        for (idx, token) in vocab.iter().enumerate() {
            map.insert(token.clone(), idx as u32);
        }
        if unk_id as usize >= vocab.len() || eos_id as usize >= vocab.len() {
            return Err("Tokenizer special token ids out of range".to_string());
        }
        Ok(Self {
            vocab,
            map,
            unk_id,
            eos_id,
        })
    }

    pub fn encode(&self, text: &str, add_eos: bool) -> Vec<u32> {
        let mut out = Vec::new();
        for token in text.split_whitespace() {
            let id = self.map.get(token).copied().unwrap_or(self.unk_id);
            out.push(id);
        }
        if add_eos {
            out.push(self.eos_id);
        }
        out
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            let token = self
                .vocab
                .get(*id as usize)
                .cloned()
                .unwrap_or_else(|| self.vocab[self.unk_id as usize].clone());
            out.push(token);
        }
        out.join(" ")
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        let file = TokenizerFile {
            vocab: self.vocab.clone(),
            unk_id: self.unk_id,
            eos_id: self.eos_id,
        };
        let text = serde_json::to_string_pretty(&file)
            .map_err(|err| format!("Failed to serialize tokenizer: {}", err))?;
        fs::write(path, text)
            .map_err(|err| format!("Failed to write {}: {}", path.display(), err))?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let contents = fs::read_to_string(path)
            .map_err(|err| format!("Failed to read {}: {}", path.display(), err))?;
        let file: TokenizerFile = serde_json::from_str(&contents)
            .map_err(|err| format!("Failed to parse tokenizer: {}", err))?;
        Self::from_vocab(file.vocab, file.unk_id, file.eos_id)
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

pub fn ids_to_bytes(ids: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(ids.len() * 4);
    for id in ids {
        out.extend_from_slice(&id.to_le_bytes());
    }
    out
}

pub fn bytes_to_ids(bytes: &[u8]) -> Result<Vec<u32>, String> {
    if !bytes.len().is_multiple_of(4) {
        return Err("Token buffer length must be a multiple of 4".to_string());
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

fn collect_files(path: &Path) -> Result<Vec<PathBuf>, String> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    if !path.is_dir() {
        return Err(format!("Tokenizer path not found: {}", path.display()));
    }
    let mut files = Vec::new();
    collect_files_in_dir(path, &mut files)?;
    Ok(files)
}

fn collect_files_in_dir(dir: &Path, files: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries =
        fs::read_dir(dir).map_err(|err| format!("Failed to read {}: {}", dir.display(), err))?;
    for entry in entries {
        let entry = entry.map_err(|err| err.to_string())?;
        let path = entry.path();
        if path.is_dir() {
            collect_files_in_dir(&path, files)?;
        } else {
            files.push(path);
        }
    }
    Ok(())
}
