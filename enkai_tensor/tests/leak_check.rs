#![cfg(feature = "torch")]

use std::ffi::CString;

/// Ensures that typical tensor + optimizer usage (with retain/free and grad queries)
/// returns live counts back to baseline.
#[test]
fn leak_free_happy_path() {
    let baseline_tensors = enkai_tensor::enkai_tensor_live_tensors();
    let baseline_devices = enkai_tensor::enkai_tensor_live_devices();
    let baseline_opts = enkai_tensor::enkai_tensor_live_opts();

    let dev = enkai_tensor::enkai_tensor_device(CString::new("cpu").unwrap().as_ptr());
    assert!(dev > 0);
    assert_eq!(
        enkai_tensor::enkai_tensor_live_devices(),
        baseline_devices + 1
    );
    assert_eq!(enkai_tensor::enkai_tensor_live_tensors(), baseline_tensors);
    let shape = CString::new("[2,2]").unwrap();
    let dtype = CString::new("fp32").unwrap();

    let p = enkai_tensor::enkai_tensor_randn(shape.as_ptr(), dtype.as_ptr(), dev);
    let g = enkai_tensor::enkai_tensor_zeros(shape.as_ptr(), dtype.as_ptr(), dev);
    assert!(p > 0 && g > 0);
    assert_eq!(
        enkai_tensor::enkai_tensor_live_tensors(),
        baseline_tensors + 2
    );

    // Retain handles to exercise refcount paths.
    assert_eq!(enkai_tensor::enkai_tensor_retain(p), 0);
    assert_eq!(enkai_tensor::enkai_tensor_device_retain(dev), 0);
    // Retain should not change live counts.
    assert_eq!(
        enkai_tensor::enkai_tensor_live_tensors(),
        baseline_tensors + 2
    );
    assert_eq!(
        enkai_tensor::enkai_tensor_live_devices(),
        baseline_devices + 1
    );

    // single-param adamw
    let params = CString::new(format!("[{}]", p)).unwrap();
    let grads = CString::new(format!("[{}]", g)).unwrap();
    let state = enkai_tensor::enkai_tensor_adamw_step_multi(
        params.as_ptr(),
        grads.as_ptr(),
        0,
        1e-3,
        0.9,
        0.999,
        1e-8,
        0.0,
    );
    assert!(state > 0);
    assert_eq!(enkai_tensor::enkai_tensor_live_opts(), baseline_opts + 1);
    // Optimizer retain/free cycle
    assert_eq!(enkai_tensor::enkai_opt_retain(state), 0);
    // Release the retained ref; one more free later for the original.
    assert_eq!(enkai_tensor::enkai_tensor_opt_free(state), 0);
    assert_eq!(enkai_tensor::enkai_tensor_live_opts(), baseline_opts + 1);

    // Touch grads to ensure grad handles are created and freed
    let grad_handle = enkai_tensor::enkai_tensor_grad(p);
    if grad_handle > 0 {
        assert_eq!(enkai_tensor::enkai_tensor_free(grad_handle), 0);
    }

    // clean up
    assert_eq!(enkai_tensor::enkai_tensor_opt_free(state), 0);
    assert_eq!(enkai_tensor::enkai_tensor_free(p), 0);
    assert_eq!(enkai_tensor::enkai_tensor_free(p), 0); // drop retained ref
    assert_eq!(enkai_tensor::enkai_tensor_free(g), 0);
    assert_eq!(enkai_tensor::enkai_tensor_device_free(dev), 0);
    assert_eq!(enkai_tensor::enkai_tensor_device_free(dev), 0); // drop retained ref

    // Counts must return exactly to baseline.
    assert_eq!(enkai_tensor::enkai_tensor_live_tensors(), baseline_tensors);
    assert_eq!(enkai_tensor::enkai_tensor_live_devices(), baseline_devices);
    assert_eq!(enkai_tensor::enkai_tensor_live_opts(), baseline_opts);
}
