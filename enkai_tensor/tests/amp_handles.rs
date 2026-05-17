#![cfg(feature = "torch")]

use std::ffi::CString;

#[test]
fn amp_scaler_retain_and_double_free() {
    // Ensure backend is torch (CPU path is fine for handle ops).
    let rc = enkai_tensor::enkai_backend_set(CString::new("torch").unwrap().as_ptr());
    assert_eq!(rc, 0);

    let scaler = enkai_tensor::enkai_amp_scaler_create(65536.0, 2.0, 0.5, 2000);
    assert!(scaler > 0, "scaler handle should be non-zero");

    // Retain should succeed.
    assert_eq!(enkai_tensor::enkai_amp_scaler_retain(scaler), 0);

    // The first free releases the retained reference, the second releases the
    // original handle, and any later free must be rejected as stale.
    assert_eq!(enkai_tensor::enkai_amp_scaler_free(scaler), 0);
    assert_eq!(enkai_tensor::enkai_amp_scaler_free(scaler), 0);
    assert_ne!(
        enkai_tensor::enkai_amp_scaler_free(scaler),
        0,
        "free after final release must error"
    );
}

#[test]
fn amp_scaler_rejects_invalid_policy_values() {
    let scaler = enkai_tensor::enkai_amp_scaler_create(f64::INFINITY, 2.0, 0.5, 2000);
    assert_eq!(scaler, 0);
    let err = unsafe {
        let ptr = enkai_tensor::enkai_tensor_last_error();
        assert!(!ptr.is_null());
        std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
    };
    assert!(err.contains("E_AMP_INVALID_SCALE"));

    let scaler = enkai_tensor::enkai_amp_scaler_create(65536.0, 1.0, 0.5, 2000);
    assert_eq!(scaler, 0);
    let err = unsafe {
        let ptr = enkai_tensor::enkai_tensor_last_error();
        assert!(!ptr.is_null());
        std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
    };
    assert!(err.contains("E_AMP_INVALID_GROWTH_FACTOR"));
}
