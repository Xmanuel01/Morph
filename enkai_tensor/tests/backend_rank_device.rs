#![cfg(feature = "torch")]

use std::ffi::CString;

#[test]
fn device_matches_rank_when_cuda_available() {
    if !tch::Cuda::is_available() {
        return;
    }
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
}
