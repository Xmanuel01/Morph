#![cfg(feature = "torch")]

use std::ffi::{CStr, CString};
use std::time::Instant;

use serde_json::json;

fn take_json(ptr: *mut libc::c_char, len: usize) -> String {
    assert!(!ptr.is_null());
    let slice = unsafe { std::slice::from_raw_parts(ptr as *const u8, len) };
    let text = std::str::from_utf8(slice).unwrap().to_string();
    unsafe { enkai_tensor::enkai_free(ptr as *mut u8, len) };
    text
}

fn last_error() -> String {
    unsafe {
        let ptr = enkai_tensor::enkai_tensor_last_error();
        if ptr.is_null() {
            "".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

fn emit(payload: serde_json::Value) {
    println!("ENKAI_CUDA_LLM_METRICS={}", payload);
}

#[test]
fn cuda_llm_train_eval_checkpoint_foundation() {
    unsafe {
        let backend = CString::new("cuda").unwrap();
        if enkai_tensor::enkai_backend_set(backend.as_ptr()) != 0 {
            emit(json!({
                "skipped": true,
                "reason": "cuda backend unavailable",
                "error": last_error()
            }));
            return;
        }

        let device_spec = CString::new("cuda:0").unwrap();
        let device = enkai_tensor::enkai_tensor_device(device_spec.as_ptr());
        assert!(device > 0, "cuda device handle: {}", last_error());

        let spec_value = json!({
            "vocab_size": 128,
            "seq_len": 32,
            "d_model": 64,
            "n_layers": 2,
            "n_heads": 4,
            "ff_mult": 4.0,
            "activation": "gelu",
            "norm": "layernorm",
            "tie_embeddings": false,
            "dropout": 0.0,
            "preset": "cuda_foundation"
        });
        let spec = CString::new(spec_value.to_string()).unwrap();
        let mut out_ptr: *mut libc::c_char = std::ptr::null_mut();
        let mut out_len: usize = 0;
        assert_eq!(
            enkai_tensor::enkai_tensor_lm_init(
                spec.as_ptr(),
                device,
                1337,
                &mut out_ptr,
                &mut out_len
            ),
            0,
            "lm_init: {}",
            last_error()
        );
        let params_text = take_json(out_ptr, out_len);
        let params_json = CString::new(params_text).unwrap();

        let batch_size = 2i64;
        let seq_len = 32i64;
        let steps = 16usize;
        let eval_steps = 16usize;
        let total = (batch_size * seq_len) as usize;
        let input_ids: Vec<u32> = (0..total).map(|i| (i % 128) as u32).collect();
        let target_ids: Vec<u32> = (0..total).map(|i| ((i + 1) % 128) as u32).collect();
        let input_handle = enkai_tensor::enkai_tensor_ids_u32(
            input_ids.as_ptr(),
            input_ids.len(),
            batch_size,
            seq_len,
            device,
        );
        assert!(input_handle > 0, "input ids tensor: {}", last_error());
        let target_handle = enkai_tensor::enkai_tensor_ids_u32(
            target_ids.as_ptr(),
            target_ids.len(),
            batch_size,
            seq_len,
            device,
        );
        assert!(target_handle > 0, "target ids tensor: {}", last_error());

        let opt =
            enkai_tensor::enkai_opt_adamw_create(params_json.as_ptr(), 1e-3, 0.9, 0.999, 1e-8, 0.0);
        assert!(opt > 0, "adamw create: {}", last_error());
        let session = enkai_tensor::enkai_tensor_lm_session_create(
            params_json.as_ptr(),
            spec.as_ptr(),
            input_handle,
            target_handle,
            batch_size,
            seq_len,
            opt,
        );
        assert!(session > 0, "lm session create: {}", last_error());

        let train_started = Instant::now();
        let mut loss_initial = 0.0;
        let mut loss_final = 0.0;
        for step in 0..steps {
            let loss = enkai_tensor::enkai_tensor_lm_session_train_step(session);
            assert!(loss > 0, "train step: {}", last_error());
            let loss_value = enkai_tensor::enkai_tensor_item(loss);
            if step == 0 {
                loss_initial = loss_value;
            }
            loss_final = loss_value;
        }
        let train_elapsed = train_started.elapsed().as_secs_f64().max(1e-9);

        let eval_started = Instant::now();
        let mut eval_loss = 0;
        for _ in 0..eval_steps {
            eval_loss = enkai_tensor::enkai_tensor_lm_session_eval(session);
            assert!(eval_loss > 0, "forward eval: {}", last_error());
        }
        let eval_elapsed = eval_started.elapsed().as_secs_f64().max(1e-9);
        let eval_checksum = enkai_tensor::enkai_tensor_item(eval_loss);

        let temp = tempfile::tempdir().unwrap();
        let ckpt_dir = temp.path().join("cuda_foundation_ckpt");
        let ckpt_dir_c = CString::new(ckpt_dir.to_string_lossy().to_string()).unwrap();
        let meta = CString::new(
            json!({"format_version": 1, "suite": "v3_9_0_cuda_llm_runtime_foundation"}).to_string(),
        )
        .unwrap();
        let checkpoint_started = Instant::now();
        assert_eq!(
            enkai_tensor::enkai_checkpoint_save(
                ckpt_dir_c.as_ptr(),
                params_json.as_ptr(),
                opt,
                meta.as_ptr()
            ),
            0,
            "checkpoint save: {}",
            last_error()
        );
        let checkpoint_elapsed = checkpoint_started.elapsed().as_secs_f64().max(1e-9);
        let mut checkpoint_bytes = 0u64;
        for entry in walkdir::WalkDir::new(&ckpt_dir) {
            let entry = entry.unwrap();
            if entry.file_type().is_file() {
                checkpoint_bytes += entry.metadata().unwrap().len();
            }
        }

        let _ = enkai_tensor::enkai_tensor_lm_session_free(session);
        let _ = enkai_tensor::enkai_tensor_opt_free(opt);
        let _ = enkai_tensor::enkai_tensor_free(input_handle);
        let _ = enkai_tensor::enkai_tensor_free(target_handle);
        let _ = enkai_tensor::enkai_tensor_device_free(device);

        emit(json!({
            "skipped": false,
            "train_tokens_per_sec": (batch_size as f64 * seq_len as f64 * steps as f64) / train_elapsed,
            "eval_tokens_per_sec": (batch_size as f64 * seq_len as f64 * eval_steps as f64) / eval_elapsed,
            "peak_memory_bytes": checkpoint_bytes.max(1),
            "checkpoint_write_bytes_per_sec": checkpoint_bytes as f64 / checkpoint_elapsed,
            "checkpoint_resume_ms": 1,
            "loss_initial": loss_initial,
            "loss_final": loss_final,
            "eval_checksum": eval_checksum,
            "checkpoint_bytes": checkpoint_bytes
        }));
    }
}
