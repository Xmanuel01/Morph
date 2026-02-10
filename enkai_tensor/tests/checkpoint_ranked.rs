#![cfg(feature = "torch")]

use serde_json::json;
use std::ffi::CString;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn tmp_dir(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let mut dir = std::env::temp_dir();
    dir.push(format!("{}_{}", prefix, nanos));
    dir
}

#[test]
fn ranked_checkpoint_roundtrip_cpu() {
    // Use CPU so the test can run on non-CUDA hosts.
    let rc = enkai_tensor::enkai_backend_set(CString::new("cpu").unwrap().as_ptr());
    assert_eq!(rc, 0);

    let dev = enkai_tensor::enkai_tensor_device(CString::new("cpu").unwrap().as_ptr());
    assert!(dev > 0);

    let tensor = enkai_tensor::enkai_tensor_randn(
        CString::new("[2,2]").unwrap().as_ptr(),
        CString::new("fp32").unwrap().as_ptr(),
        dev,
    );
    assert!(tensor > 0);

    // Capture shape to compare after load.
    let mut shape_json: *mut i8 = std::ptr::null_mut();
    let mut shape_len: usize = 0;
    let rc = enkai_tensor::enkai_tensor_shape(tensor, &mut shape_json, &mut shape_len);
    assert_eq!(rc, 0);
    let shape_slice = unsafe { std::slice::from_raw_parts(shape_json as *const u8, shape_len) };
    let shape_before = std::str::from_utf8(shape_slice).unwrap().to_string();
    unsafe { enkai_tensor::enkai_free(shape_json as *mut u8, shape_len) };

    let params_json = json!([tensor]).to_string();
    let meta_json = json!({"tag": "test"}).to_string();
    let dir = tmp_dir("enkai_ranked_ckpt");

    let save_rc = enkai_tensor::enkai_checkpoint_save_ranked(
        CString::new(dir.to_str().unwrap()).unwrap().as_ptr(),
        0,
        1,
        CString::new(params_json).unwrap().as_ptr(),
        0,
        CString::new(meta_json).unwrap().as_ptr(),
    );
    assert_eq!(save_rc, 0, "save should succeed");

    // free original tensor so we know load returns fresh handles.
    assert_eq!(enkai_tensor::enkai_tensor_free(tensor), 0);

    // Load back.
    let mut out_params: *mut i8 = std::ptr::null_mut();
    let mut out_params_len: usize = 0;
    let mut out_opt: i64 = -1;
    let mut out_meta: *mut i8 = std::ptr::null_mut();
    let mut out_meta_len: usize = 0;
    let load_rc = enkai_tensor::enkai_checkpoint_load_ranked(
        CString::new(dir.to_str().unwrap()).unwrap().as_ptr(),
        0,
        &mut out_params,
        &mut out_params_len,
        &mut out_opt,
        &mut out_meta,
        &mut out_meta_len,
    );
    assert_eq!(load_rc, 0, "load should succeed");
    assert_eq!(out_opt, 0, "no optimizer state expected");

    let params_slice =
        unsafe { std::slice::from_raw_parts(out_params as *const u8, out_params_len) };
    let params_text = std::str::from_utf8(params_slice).unwrap();
    let handles: Vec<i64> = serde_json::from_str(params_text).unwrap();
    assert_eq!(handles.len(), 1);
    let loaded = handles[0];
    assert!(loaded > 0);

    // Compare shape.
    let mut loaded_shape: *mut i8 = std::ptr::null_mut();
    let mut loaded_shape_len: usize = 0;
    let rc = enkai_tensor::enkai_tensor_shape(loaded, &mut loaded_shape, &mut loaded_shape_len);
    assert_eq!(rc, 0);
    let slice = unsafe { std::slice::from_raw_parts(loaded_shape as *const u8, loaded_shape_len) };
    let shape_after = std::str::from_utf8(slice).unwrap().to_string();
    unsafe { enkai_tensor::enkai_free(loaded_shape as *mut u8, loaded_shape_len) };
    assert_eq!(shape_before, shape_after);

    // Check integrity file exists.
    let integrity_path = dir.join("rank0").join("integrity_rank0.json");
    assert!(
        integrity_path.exists(),
        "integrity file should be written: {:?}",
        integrity_path
    );

    // cleanup
    unsafe {
        enkai_tensor::enkai_free(out_params as *mut u8, out_params_len);
        enkai_tensor::enkai_free(out_meta as *mut u8, out_meta_len);
    }
    assert_eq!(enkai_tensor::enkai_tensor_free(loaded), 0);
    assert_eq!(enkai_tensor::enkai_tensor_device_free(dev), 0);
    let _ = fs::remove_dir_all(&dir);
}
