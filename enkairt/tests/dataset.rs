use std::fs;

use tempfile::tempdir;

use enkairt::dataset::{resolve_dataset_paths, DatasetConfig, DatasetStream};
use enkairt::tokenizer::{Tokenizer, TrainConfig};

#[test]
fn dataset_resolves_directory_files() {
    let dir = tempdir().expect("tempdir");
    fs::write(dir.path().join("a.txt"), "one two").unwrap();
    fs::write(dir.path().join("b.txt"), "three four").unwrap();
    let paths = resolve_dataset_paths(dir.path().to_string_lossy().as_ref()).expect("paths");
    assert_eq!(paths.len(), 2);
}

#[test]
fn dataset_produces_batches() {
    let dir = tempdir().expect("tempdir");
    let file = dir.path().join("data.txt");
    fs::write(&file, "alpha beta gamma\ndelta epsilon").unwrap();
    let mut cfg = TrainConfig::default();
    cfg.vocab_size = 16;
    let tokenizer = Tokenizer::train_from_path(&file, &cfg).expect("tokenizer");
    let mut data_cfg = DatasetConfig::new(4, 2);
    data_cfg.drop_remainder = false;
    let files = resolve_dataset_paths(file.to_string_lossy().as_ref()).expect("paths");
    let mut stream = DatasetStream::new(files, tokenizer, data_cfg).expect("stream");
    let batch = stream.next_batch().expect("batch").expect("some batch");
    assert_eq!(batch.input_ids.len(), 8);
    assert_eq!(batch.target_ids.len(), 8);
    assert_eq!(batch.attention_mask.len(), 8);
}

#[test]
fn dataset_shuffle_is_deterministic() {
    let dir = tempdir().expect("tempdir");
    fs::write(dir.path().join("a.txt"), "alpha").unwrap();
    fs::write(dir.path().join("b.txt"), "beta").unwrap();
    fs::write(dir.path().join("c.txt"), "gamma").unwrap();
    let mut cfg = TrainConfig::default();
    cfg.vocab_size = 16;
    let tokenizer = Tokenizer::train_from_path(dir.path(), &cfg).expect("tokenizer");
    let files = resolve_dataset_paths(dir.path().to_string_lossy().as_ref()).expect("paths");
    let mut data_cfg = DatasetConfig::new(1, 1);
    data_cfg.shuffle = true;
    data_cfg.seed = Some(7);
    let mut stream1 =
        DatasetStream::new(files.clone(), tokenizer.clone(), data_cfg.clone()).expect("stream1");
    let mut stream2 = DatasetStream::new(files, tokenizer, data_cfg).expect("stream2");
    let batch1 = stream1.next_batch().expect("batch1").expect("some batch1");
    let batch2 = stream2.next_batch().expect("batch2").expect("some batch2");
    assert_eq!(batch1.input_ids, batch2.input_ids);
    assert_eq!(batch1.target_ids, batch2.target_ids);
}
