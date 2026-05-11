#![cfg(feature = "torch")]

use std::ffi::CString;

#[test]
fn backend_list_and_set() {
    let mut out: *mut i8 = std::ptr::null_mut();
    let mut len: usize = 0;
    let rc = unsafe { enkai_tensor::enkai_backend_list(&mut out, &mut len) };
    assert_eq!(rc, 0);
    assert!(len > 0);
    let slice = unsafe { std::slice::from_raw_parts(out as *const u8, len) };
    let text = std::str::from_utf8(slice).unwrap();
    assert!(text.contains("torch"));
    assert!(text.contains("cpu"));
    unsafe { enkai_tensor::enkai_free(out as *mut u8, len) };

    let mut catalog_out: *mut i8 = std::ptr::null_mut();
    let mut catalog_len: usize = 0;
    let rc = unsafe { enkai_tensor::enkai_backend_catalog(&mut catalog_out, &mut catalog_len) };
    assert_eq!(rc, 0);
    let slice = unsafe { std::slice::from_raw_parts(catalog_out as *const u8, catalog_len) };
    let catalog = std::str::from_utf8(slice).unwrap();
    assert!(catalog.contains("\"name\":\"cuda\""));
    assert!(catalog.contains("\"name\":\"rocm\""));
    assert!(catalog.contains("\"name\":\"metal\""));
    unsafe { enkai_tensor::enkai_free(catalog_out as *mut u8, catalog_len) };

    let rc = enkai_tensor::enkai_backend_set(CString::new("torch").unwrap().as_ptr());
    assert_eq!(rc, 0);

    let mut cur_out: *mut i8 = std::ptr::null_mut();
    let mut cur_len: usize = 0;
    let rc = unsafe { enkai_tensor::enkai_backend_current(&mut cur_out, &mut cur_len) };
    assert_eq!(rc, 0);
    let slice = unsafe { std::slice::from_raw_parts(cur_out as *const u8, cur_len) };
    let text = std::str::from_utf8(slice).unwrap();
    assert_eq!(text, "torch");
    unsafe { enkai_tensor::enkai_free(cur_out as *mut u8, cur_len) };

    // Switch to CPU backend and run a simple op.
    let rc = enkai_tensor::enkai_backend_set(CString::new("cpu").unwrap().as_ptr());
    assert_eq!(rc, 0);
    let dev = enkai_tensor::enkai_tensor_device(CString::new("cpu").unwrap().as_ptr());
    assert!(dev > 0);
    let t = enkai_tensor::enkai_tensor_randn(
        CString::new("[1,1]").unwrap().as_ptr(),
        CString::new("fp32").unwrap().as_ptr(),
        dev,
    );
    assert!(t > 0);
    assert_eq!(enkai_tensor::enkai_tensor_free(t), 0);
    assert_eq!(enkai_tensor::enkai_tensor_device_free(dev), 0);
}

#[test]
fn unsupported_backend_errors_are_deterministic() {
    let rc = enkai_tensor::enkai_backend_set(CString::new("metal").unwrap().as_ptr());
    assert_eq!(rc, 1);
    let err = unsafe {
        let ptr = enkai_tensor::enkai_tensor_last_error();
        assert!(!ptr.is_null());
        std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
    };
    assert!(err.contains("E_BACKEND_FEATURE_MISSING") || err.contains("E_BACKEND_UNAVAILABLE"));

    let rc = enkai_tensor::enkai_backend_set(CString::new("rocm").unwrap().as_ptr());
    assert_eq!(rc, 1);
    let err = unsafe {
        let ptr = enkai_tensor::enkai_tensor_last_error();
        assert!(!ptr.is_null());
        std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
    };
    assert!(err.contains("E_BACKEND_FEATURE_MISSING") || err.contains("E_BACKEND_UNAVAILABLE"));

    let rc = enkai_tensor::enkai_backend_set(CString::new("not_real").unwrap().as_ptr());
    assert_eq!(rc, 1);
    let err = unsafe {
        let ptr = enkai_tensor::enkai_tensor_last_error();
        assert!(!ptr.is_null());
        std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
    };
    assert!(err.contains("E_BACKEND_UNKNOWN"));
}
