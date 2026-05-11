use std::fs;

use tempfile::tempdir;

use enkairt::dataset::{
    dataset_pipeline_manifest, resolve_dataset_paths, DatasetConfig, DatasetStream,
};
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
    let cfg = TrainConfig {
        vocab_size: 16,
        ..TrainConfig::default()
    };
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
    let cfg = TrainConfig {
        vocab_size: 16,
        ..TrainConfig::default()
    };
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

#[test]
fn dataset_cursor_replays_same_next_batch() {
    let dir = tempdir().expect("tempdir");
    let file = dir.path().join("data.txt");
    fs::write(
        &file,
        "alpha beta gamma\ndelta epsilon zeta\neta theta iota",
    )
    .unwrap();
    let cfg = TrainConfig {
        vocab_size: 32,
        seed: Some(11),
        ..TrainConfig::default()
    };
    let tokenizer = Tokenizer::train_from_path(&file, &cfg).expect("tokenizer");
    let mut data_cfg = DatasetConfig::new(3, 1);
    data_cfg.drop_remainder = false;
    let files = resolve_dataset_paths(file.to_string_lossy().as_ref()).expect("paths");
    let mut stream =
        DatasetStream::new(files.clone(), tokenizer.clone(), data_cfg.clone()).expect("stream");
    let _first = stream.next_batch().expect("first").expect("some first");
    let cursor = stream.cursor().expect("cursor");
    let expected = stream.next_batch().expect("second").expect("some second");

    let mut replay = DatasetStream::new(files, tokenizer, data_cfg).expect("replay");
    replay.restore_cursor(cursor).expect("restore");
    let actual = replay
        .next_batch()
        .expect("replayed")
        .expect("some replayed");
    assert_eq!(actual.input_ids, expected.input_ids);
    assert_eq!(actual.target_ids, expected.target_ids);
    assert_eq!(actual.attention_mask, expected.attention_mask);
}

#[test]
fn dataset_manifest_archives_replay_and_tokenizer_provenance() {
    let dir = tempdir().expect("tempdir");
    let file = dir.path().join("data.txt");
    fs::write(&file, "alpha beta gamma").unwrap();
    let cfg = TrainConfig {
        vocab_size: 16,
        seed: Some(9),
        ..TrainConfig::default()
    };
    let tokenizer = Tokenizer::train_from_path(&file, &cfg).expect("tokenizer");
    let data_cfg = DatasetConfig::new(4, 2);
    let files = resolve_dataset_paths(file.to_string_lossy().as_ref()).expect("paths");
    let manifest =
        dataset_pipeline_manifest(&files, &tokenizer, &data_cfg).expect("pipeline manifest");
    assert_eq!(manifest["pipeline"], "enkai_dataset_stream_v1");
    assert_eq!(manifest["features"]["checkpointable_cursor"], true);
    assert_eq!(
        manifest["tokenizer_sha1"].as_str().unwrap(),
        tokenizer.fingerprint()
    );
    assert!(manifest["dataset_sha1"].as_str().unwrap().len() >= 40);
}
