use std::fs;

use tempfile::tempdir;

use enkairt::tokenizer::{bytes_to_ids, ids_to_bytes, Tokenizer, TrainConfig};

#[test]
fn trains_and_roundtrips_tokens() {
    let dir = tempdir().expect("tempdir");
    let file = dir.path().join("data.txt");
    fs::write(&file, "hello world hello").expect("write");
    let mut cfg = TrainConfig::default();
    cfg.vocab_size = 8;
    let tok = Tokenizer::train_from_path(&file, &cfg).expect("train");
    let ids = tok.encode("hello world", false);
    assert_eq!(ids, vec![2, 3]);
    let text = tok.decode(&ids);
    assert_eq!(text, "hello world");
}

#[test]
fn saves_and_loads_tokenizer() {
    let dir = tempdir().expect("tempdir");
    let file = dir.path().join("data.txt");
    fs::write(&file, "alpha beta beta").expect("write");
    let mut cfg = TrainConfig::default();
    cfg.vocab_size = 8;
    let tok = Tokenizer::train_from_path(&file, &cfg).expect("train");
    let out = dir.path().join("tok.json");
    tok.save(&out).expect("save");
    let loaded = Tokenizer::load(&out).expect("load");
    let ids = loaded.encode("alpha beta", false);
    let text = loaded.decode(&ids);
    assert_eq!(text, "alpha beta");
}

#[test]
fn token_buffer_roundtrip() {
    let ids = vec![1u32, 42u32, 7u32];
    let bytes = ids_to_bytes(&ids);
    let decoded = bytes_to_ids(&bytes).expect("decode");
    assert_eq!(decoded, ids);
}

#[test]
fn tokenizer_seed_is_deterministic() {
    let dir = tempdir().expect("tempdir");
    let file = dir.path().join("data.txt");
    fs::write(&file, "alpha beta gamma delta").expect("write");
    let mut cfg = TrainConfig::default();
    cfg.vocab_size = 6;
    cfg.seed = Some(42);
    let tok1 = Tokenizer::train_from_path(&file, &cfg).expect("train1");
    let tok2 = Tokenizer::train_from_path(&file, &cfg).expect("train2");
    let ids: Vec<u32> = (0..tok1.vocab_size() as u32).collect();
    assert_eq!(tok1.decode(&ids), tok2.decode(&ids));
}
