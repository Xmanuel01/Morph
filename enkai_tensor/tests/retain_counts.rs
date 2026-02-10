#![cfg(feature = "torch")]

use std::ffi::CString;

#[test]
fn retain_and_free_counts() {
    let dev = enkai_tensor::enkai_tensor_device(CString::new("cpu").unwrap().as_ptr());
    assert!(dev > 0);
    let shape = CString::new("[1,1]").unwrap();
    let dtype = CString::new("fp32").unwrap();
    let t = enkai_tensor::enkai_tensor_randn(shape.as_ptr(), dtype.as_ptr(), dev);
    assert!(t > 0);

    // Initial live counts
    let tensors_before = enkai_tensor::enkai_tensor_live_tensors();
    let devices_before = enkai_tensor::enkai_tensor_live_devices();
    assert!(tensors_before >= 1);
    assert!(devices_before >= 1);

    // Retain both handles
    assert_eq!(enkai_tensor::enkai_tensor_retain(t), 0);
    assert_eq!(enkai_tensor::enkai_tensor_device_retain(dev), 0);

    // Free once: should still be alive because of retain
    assert_eq!(enkai_tensor::enkai_tensor_free(t), 0);
    assert_eq!(enkai_tensor::enkai_tensor_device_free(dev), 0);

    // Live counts should remain >= previous
    let tensors_mid = enkai_tensor::enkai_tensor_live_tensors();
    let devices_mid = enkai_tensor::enkai_tensor_live_devices();
    assert!(tensors_mid >= tensors_before);
    assert!(devices_mid >= devices_before);

    // Final free drops to zero additional refs
    assert_eq!(enkai_tensor::enkai_tensor_free(t), 0);
    assert_eq!(enkai_tensor::enkai_tensor_device_free(dev), 0);

    // Live counts should not increase after full release
    let tensors_after = enkai_tensor::enkai_tensor_live_tensors();
    let devices_after = enkai_tensor::enkai_tensor_live_devices();
    assert!(tensors_after <= tensors_mid);
    assert!(devices_after <= devices_mid);
}
