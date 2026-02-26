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
            format_version: 1,
            step: 1,
            tokens: 100,
            loss: 1.234,
            config_hash: "abc".to_string(),
            model_sig: "sig".to_string(),
            dtype: "fp32".to_string(),
            device: "cpu".to_string(),
            world_size: 1,
            rank: 0,
            grad_accum_steps: 1,
            grad_clip_norm: None,
            amp: None,
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
    assert_eq!(loaded.meta.format_version, state.meta.format_version);
    assert_eq!(loaded.meta.model_sig, state.meta.model_sig);
    assert_eq!(loaded.meta.dtype, state.meta.dtype);
    assert_eq!(loaded.meta.device, state.meta.device);
}

#[test]
fn checkpoint_latest_and_rotate() {
    let dir = tempdir().expect("tempdir");
    for step in 1..=3 {
        let state = CheckpointState {
            weights: vec![step as f32],
            optimizer: vec![],
            meta: CheckpointMeta {
                format_version: 1,
                step,
                tokens: 0,
                loss: 0.0,
                config_hash: "".to_string(),
                model_sig: "".to_string(),
                dtype: "".to_string(),
                device: "".to_string(),
                world_size: 1,
                rank: 0,
                grad_accum_steps: 1,
                grad_clip_norm: None,
                amp: None,
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
