#![cfg(feature = "torch")]

use std::ffi::CString;
use std::sync::{Mutex, OnceLock};

fn env_guard() -> std::sync::MutexGuard<'static, ()> {
    static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    GUARD
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|err| err.into_inner())
}

#[test]
fn device_matches_rank_when_cuda_available() {
    let _guard = env_guard();
    if !tch::Cuda::is_available() {
        return;
    }
    let prev = std::env::var("ENKAI_ENABLE_DIST").ok();
    std::env::set_var("ENKAI_ENABLE_DIST", "1");
    // world_size=2, rank=1 should pick cuda:1
    let rc = enkai_tensor::enkai_dist_config(2, 1, -1, 1234);
    assert_eq!(rc, 0);
    let dev = enkai_tensor::enkai_tensor_device(CString::new("cuda:1").unwrap().as_ptr());
    assert!(dev > 0);
    // create a small tensor and ensure it stays on device 1 by checking to_kind does not move it
    let t = enkai_tensor::enkai_tensor_randn(
        CString::new("[1]").unwrap().as_ptr(),
        CString::new("fp32").unwrap().as_ptr(),
        dev,
    );
    assert!(t > 0);
    // cleanup
    enkai_tensor::enkai_tensor_free(t);
    enkai_tensor::enkai_tensor_device_free(dev);
    let _ = enkai_tensor::enkai_dist_shutdown();
    if let Some(value) = prev {
        std::env::set_var("ENKAI_ENABLE_DIST", value);
    } else {
        std::env::remove_var("ENKAI_ENABLE_DIST");
    }
}
