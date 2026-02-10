#![cfg(feature = "torch")]

use std::ffi::CString;

#[test]
fn cpu_backend_basic_ops() {
    assert_eq!(
        enkai_tensor::enkai_backend_set(CString::new("cpu").unwrap().as_ptr()),
        0
    );
    let dev = enkai_tensor::enkai_tensor_device(CString::new("cpu").unwrap().as_ptr());
    assert!(dev > 0);
    let t = enkai_tensor::enkai_tensor_zeros(
        CString::new("[2,2]").unwrap().as_ptr(),
        CString::new("fp32").unwrap().as_ptr(),
        dev,
    );
    assert!(t > 0);
    let s = enkai_tensor::enkai_tensor_sum(t, -1, 0);
    assert!(s > 0);
    assert_eq!(enkai_tensor::enkai_tensor_free(s), 0);
    assert_eq!(enkai_tensor::enkai_tensor_free(t), 0);
    assert_eq!(enkai_tensor::enkai_tensor_device_free(dev), 0);
}
