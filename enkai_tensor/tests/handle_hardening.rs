#![cfg(feature = "torch")]

use std::ffi::{CStr, CString};

fn last_error() -> String {
    let ptr = enkai_tensor::enkai_tensor_last_error();
    if ptr.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() }
}

#[test]
fn ffi_handles_are_opaque_typed_and_stale_checked() {
    assert_eq!(
        enkai_tensor::enkai_backend_set(CString::new("torch").unwrap().as_ptr()),
        0
    );

    let device = enkai_tensor::enkai_tensor_device(CString::new("cpu").unwrap().as_ptr());
    assert!(device > 0, "device allocation failed: {}", last_error());

    let shape = CString::new("[1]").unwrap();
    let dtype = CString::new("fp32").unwrap();
    let tensor = enkai_tensor::enkai_tensor_zeros(shape.as_ptr(), dtype.as_ptr(), device);
    assert!(tensor > 0, "tensor allocation failed: {}", last_error());

    assert_ne!(device, 1, "device handles must not be sequential raw ids");
    assert_ne!(tensor, 2, "tensor handles must not be sequential raw ids");
    assert_ne!(
        (device as u64) >> 56,
        (tensor as u64) >> 56,
        "handle tags must encode the registry kind"
    );

    assert_ne!(enkai_tensor::enkai_tensor_device_retain(tensor), 0);
    assert!(
        last_error().contains("Invalid device handle kind"),
        "wrong-kind device retain should be explicit, got: {}",
        last_error()
    );

    assert_ne!(enkai_tensor::enkai_tensor_retain(device), 0);
    assert!(
        last_error().contains("Invalid tensor handle kind"),
        "wrong-kind tensor retain should be explicit, got: {}",
        last_error()
    );

    assert_ne!(enkai_tensor::enkai_tensor_retain(42), 0);
    assert!(
        last_error().contains("opaque handle tag missing"),
        "untagged handles should be rejected before registry lookup, got: {}",
        last_error()
    );

    assert_eq!(enkai_tensor::enkai_tensor_free(tensor), 0);
    assert_ne!(enkai_tensor::enkai_tensor_retain(tensor), 0);
    assert!(
        last_error().contains("Stale tensor handle"),
        "freed tensor handles must be reported as stale, got: {}",
        last_error()
    );

    assert_eq!(enkai_tensor::enkai_tensor_device_free(device), 0);
    assert_ne!(enkai_tensor::enkai_tensor_device_retain(device), 0);
    assert!(
        last_error().contains("Stale device handle"),
        "freed device handles must be reported as stale, got: {}",
        last_error()
    );
}
