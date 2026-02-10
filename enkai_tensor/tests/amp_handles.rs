#![cfg(feature = "torch")]

use std::ffi::CString;

#[test]
fn amp_scaler_retain_and_double_free() {
    // Ensure backend is torch (CPU path is fine for handle ops).
    let rc = enkai_tensor::enkai_backend_set(CString::new("torch").unwrap().as_ptr());
    assert_eq!(rc, 0);

    let scaler = enkai_tensor::enkai_amp_scaler_create();
    assert!(scaler > 0, "scaler handle should be non-zero");

    // Retain should succeed.
    assert_eq!(enkai_tensor::enkai_amp_scaler_retain(scaler), 0);

    // Free twice should flag double-free on second call.
    assert_eq!(enkai_tensor::enkai_amp_scaler_free(scaler), 0);
    assert_ne!(
        enkai_tensor::enkai_amp_scaler_free(scaler),
        0,
        "second free must error"
    );
}
