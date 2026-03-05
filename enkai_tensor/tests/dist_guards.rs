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

fn last_error() -> String {
    let ptr = enkai_tensor::enkai_tensor_last_error();
    if ptr.is_null() {
        return String::new();
    }
    unsafe { std::ffi::CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

#[test]
fn dist_init_requires_explicit_opt_in() {
    let _guard = env_guard();
    let prev = std::env::var("ENKAI_ENABLE_DIST").ok();
    std::env::remove_var("ENKAI_ENABLE_DIST");

    let rc = enkai_tensor::enkai_dist_init(2, 0);
    assert_ne!(rc, 0);
    assert!(last_error().contains("ENKAI_ENABLE_DIST=1"));

    if let Some(value) = prev {
        std::env::set_var("ENKAI_ENABLE_DIST", value);
    } else {
        std::env::remove_var("ENKAI_ENABLE_DIST");
    }
}

#[cfg(not(feature = "dist"))]
#[test]
fn dist_init_reports_missing_dist_feature() {
    let _guard = env_guard();
    let prev = std::env::var("ENKAI_ENABLE_DIST").ok();
    std::env::set_var("ENKAI_ENABLE_DIST", "1");

    let rc = enkai_tensor::enkai_dist_init(2, 0);
    assert_ne!(rc, 0);
    assert!(last_error().contains("features \"torch,dist\""));

    if let Some(value) = prev {
        std::env::set_var("ENKAI_ENABLE_DIST", value);
    } else {
        std::env::remove_var("ENKAI_ENABLE_DIST");
    }
}

#[cfg(feature = "dist")]
#[test]
fn dist_allreduce_requires_initialized_context() {
    let _guard = env_guard();
    let prev = std::env::var("ENKAI_ENABLE_DIST").ok();
    std::env::set_var("ENKAI_ENABLE_DIST", "1");
    let _ = enkai_tensor::enkai_dist_shutdown();

    let payload = CString::new("[1]").expect("json");
    let rc = enkai_tensor::enkai_dist_allreduce_sum_multi(payload.as_ptr());
    assert_ne!(rc, 0);
    assert!(last_error().contains("dist not initialized"));

    if let Some(value) = prev {
        std::env::set_var("ENKAI_ENABLE_DIST", value);
    } else {
        std::env::remove_var("ENKAI_ENABLE_DIST");
    }
}
