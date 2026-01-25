#[cfg(feature = "torch")]
#[test]
fn creates_tensor_and_shape() {
    use std::ffi::CString;

    let dev = morph_tensor::morph_tensor_device(CString::new("cpu").unwrap().as_ptr());
    assert!(dev > 0);
    let shape = CString::new("[2,3]").unwrap();
    let dtype = CString::new("fp32").unwrap();
    let t = morph_tensor::morph_tensor_randn(shape.as_ptr(), dtype.as_ptr(), dev);
    assert!(t > 0);
    let shape_slice = morph_tensor::morph_tensor_shape(t);
    assert!(!shape_slice.ptr.is_null());
    assert!(shape_slice.len > 0);
    unsafe {
        morph_tensor::morph_free(shape_slice.ptr, shape_slice.len);
    }
    morph_tensor::morph_tensor_free(t);
    morph_tensor::morph_tensor_device_free(dev);
}

#[cfg(feature = "torch")]
#[test]
fn reshape_transpose_concat_sum_mean() {
    use std::ffi::CString;

    let dev = morph_tensor::morph_tensor_device(CString::new("cpu").unwrap().as_ptr());
    assert!(dev > 0);
    let shape = CString::new("[2,3]").unwrap();
    let dtype = CString::new("fp32").unwrap();
    let t = morph_tensor::morph_tensor_zeros(shape.as_ptr(), dtype.as_ptr(), dev);
    assert!(t > 0);

    let reshaped = morph_tensor::morph_tensor_reshape(t, CString::new("[3,2]").unwrap().as_ptr());
    assert!(reshaped > 0);
    let transposed = morph_tensor::morph_tensor_transpose(reshaped, 0, 1);
    assert!(transposed > 0);

    let handles = CString::new(format!("[{},{}]", transposed, transposed)).unwrap();
    let cat = morph_tensor::morph_tensor_concat(handles.as_ptr(), 0);
    assert!(cat > 0);

    let sum_all = morph_tensor::morph_tensor_sum(cat, -1, 0);
    assert!(sum_all > 0);
    let mean_dim = morph_tensor::morph_tensor_mean(cat, 1, 1);
    assert!(mean_dim > 0);

    morph_tensor::morph_tensor_free(sum_all);
    morph_tensor::morph_tensor_free(mean_dim);
    morph_tensor::morph_tensor_free(cat);
    morph_tensor::morph_tensor_free(transposed);
    morph_tensor::morph_tensor_free(reshaped);
    morph_tensor::morph_tensor_free(t);
    morph_tensor::morph_tensor_device_free(dev);
}

#[cfg(feature = "torch")]
#[test]
fn adamw_step_multi_updates() {
    use std::ffi::CString;

    let dev = morph_tensor::morph_tensor_device(CString::new("cpu").unwrap().as_ptr());
    assert!(dev > 0);
    let shape = CString::new("[2,2]").unwrap();
    let dtype = CString::new("fp32").unwrap();
    let p1 = morph_tensor::morph_tensor_randn(shape.as_ptr(), dtype.as_ptr(), dev);
    let p2 = morph_tensor::morph_tensor_randn(shape.as_ptr(), dtype.as_ptr(), dev);
    let g1 = morph_tensor::morph_tensor_zeros(shape.as_ptr(), dtype.as_ptr(), dev);
    let g2 = morph_tensor::morph_tensor_zeros(shape.as_ptr(), dtype.as_ptr(), dev);
    assert!(p1 > 0 && p2 > 0 && g1 > 0 && g2 > 0);
    let params = CString::new(format!("[{},{}]", p1, p2)).unwrap();
    let grads = CString::new(format!("[{},{}]", g1, g2)).unwrap();
    let state = morph_tensor::morph_tensor_adamw_step_multi(
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
    morph_tensor::morph_tensor_opt_free(state);
    morph_tensor::morph_tensor_free(p1);
    morph_tensor::morph_tensor_free(p2);
    morph_tensor::morph_tensor_free(g1);
    morph_tensor::morph_tensor_free(g2);
    morph_tensor::morph_tensor_device_free(dev);
}

#[cfg(feature = "torch")]
#[test]
fn relu_sigmoid_dropout_slice_view() {
    use std::ffi::CString;

    let dev = morph_tensor::morph_tensor_device(CString::new("cpu").unwrap().as_ptr());
    assert!(dev > 0);
    let shape = CString::new("[2,4]").unwrap();
    let dtype = CString::new("fp32").unwrap();
    let t = morph_tensor::morph_tensor_randn(shape.as_ptr(), dtype.as_ptr(), dev);
    assert!(t > 0);

    let relu = morph_tensor::morph_tensor_relu(t);
    assert!(relu > 0);
    let sig = morph_tensor::morph_tensor_sigmoid(relu);
    assert!(sig > 0);
    let drop = morph_tensor::morph_tensor_dropout(sig, 0.5, 1);
    assert!(drop > 0);
    let slice = morph_tensor::morph_tensor_slice(drop, 1, 0, 2, 1);
    assert!(slice > 0);
    let view = morph_tensor::morph_tensor_view(slice, CString::new("[4]").unwrap().as_ptr());
    assert!(view > 0);

    morph_tensor::morph_tensor_free(view);
    morph_tensor::morph_tensor_free(slice);
    morph_tensor::morph_tensor_free(drop);
    morph_tensor::morph_tensor_free(sig);
    morph_tensor::morph_tensor_free(relu);
    morph_tensor::morph_tensor_free(t);
    morph_tensor::morph_tensor_device_free(dev);
}

#[cfg(feature = "torch")]
#[test]
fn masked_softmax_and_layernorm_backward() {
    use std::ffi::{CStr, CString};

    let dev = morph_tensor::morph_tensor_device(CString::new("cpu").unwrap().as_ptr());
    assert!(dev > 0);
    let shape = CString::new("[2,2]").unwrap();
    let dtype = CString::new("fp32").unwrap();
    let x = morph_tensor::morph_tensor_randn(shape.as_ptr(), dtype.as_ptr(), dev);
    let mask = morph_tensor::morph_tensor_zeros(shape.as_ptr(), dtype.as_ptr(), dev);
    assert!(x > 0 && mask > 0);
    let masked = morph_tensor::morph_tensor_masked_softmax(x, mask, 1, 0);
    assert!(masked > 0);
    let grad = morph_tensor::morph_tensor_zeros(shape.as_ptr(), dtype.as_ptr(), dev);
    let backward = morph_tensor::morph_tensor_masked_softmax_backward(grad, masked, mask, 1);
    assert!(backward > 0);

    let w = morph_tensor::morph_tensor_randn(
        CString::new("[2]").unwrap().as_ptr(),
        dtype.as_ptr(),
        dev,
    );
    let b = morph_tensor::morph_tensor_randn(
        CString::new("[2]").unwrap().as_ptr(),
        dtype.as_ptr(),
        dev,
    );
    let grad_out = morph_tensor::morph_tensor_randn(shape.as_ptr(), dtype.as_ptr(), dev);
    let grads_slice = morph_tensor::morph_tensor_layernorm_backward(x, w, b, 1e-5, grad_out);
    assert!(!grads_slice.ptr.is_null());
    let grads_json = unsafe { std::slice::from_raw_parts(grads_slice.ptr, grads_slice.len) };
    let grads_json = std::str::from_utf8(grads_json).expect("grads json");
    let handles: Vec<i64> = serde_json::from_str(grads_json).expect("parse grads");
    assert_eq!(handles.len(), 3);
    for handle in handles {
        assert!(handle > 0);
        morph_tensor::morph_tensor_free(handle);
    }
    unsafe {
        morph_tensor::morph_free(grads_slice.ptr, grads_slice.len);
    }

    morph_tensor::morph_tensor_free(backward);
    morph_tensor::morph_tensor_free(grad);
    morph_tensor::morph_tensor_free(masked);
    morph_tensor::morph_tensor_free(mask);
    morph_tensor::morph_tensor_free(x);
    morph_tensor::morph_tensor_free(w);
    morph_tensor::morph_tensor_free(b);
    morph_tensor::morph_tensor_free(grad_out);
    morph_tensor::morph_tensor_device_free(dev);
}
