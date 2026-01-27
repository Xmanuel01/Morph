use tempfile::tempdir;

use enkairt::checkpoint::{
    latest_checkpoint, load_checkpoint, rotate_checkpoints, save_checkpoint, CheckpointMeta,
    CheckpointState,
};

#[test]
fn checkpoint_save_and_load_roundtrip() {
    let dir = tempdir().expect("tempdir");
    let state = CheckpointState {
        weights: vec![1.0, 2.0, 3.5],
        optimizer: vec![0.1, 0.2],
        meta: CheckpointMeta {
            step: 1,
            tokens: 100,
            loss: 1.234,
            config_hash: "abc".to_string(),
        },
    };
    let path = save_checkpoint(dir.path(), &state).expect("save");
    let loaded = load_checkpoint(&path).expect("load");
    assert_eq!(loaded.weights, state.weights);
    assert_eq!(loaded.optimizer, state.optimizer);
    assert_eq!(loaded.meta.step, state.meta.step);
    assert_eq!(loaded.meta.tokens, state.meta.tokens);
    assert_eq!(loaded.meta.loss, state.meta.loss);
    assert_eq!(loaded.meta.config_hash, state.meta.config_hash);
}

#[test]
fn checkpoint_latest_and_rotate() {
    let dir = tempdir().expect("tempdir");
    for step in 1..=3 {
        let state = CheckpointState {
            weights: vec![step as f32],
            optimizer: vec![],
            meta: CheckpointMeta {
                step,
                tokens: 0,
                loss: 0.0,
                config_hash: "".to_string(),
            },
        };
        save_checkpoint(dir.path(), &state).expect("save");
    }
    let latest = latest_checkpoint(dir.path())
        .expect("latest")
        .expect("some");
    assert!(latest.ends_with("step_00000003"));
    rotate_checkpoints(dir.path(), 2).expect("rotate");
    let remaining = std::fs::read_dir(dir.path()).expect("read dir").count();
    assert_eq!(remaining, 2);
}
