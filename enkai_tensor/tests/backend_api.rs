#![cfg(feature = "torch")]

use std::ffi::CString;

#[test]
fn backend_list_and_set() {
    let mut out: *mut i8 = std::ptr::null_mut();
    let mut len: usize = 0;
    let rc = enkai_tensor::enkai_backend_list(&mut out, &mut len);
    assert_eq!(rc, 0);
    assert!(len > 0);
    let slice = unsafe { std::slice::from_raw_parts(out as *const u8, len) };
    let text = std::str::from_utf8(slice).unwrap();
    assert!(text.contains("torch"));
    assert!(text.contains("cpu"));
    unsafe { enkai_tensor::enkai_free(out as *mut u8, len) };

    let rc = enkai_tensor::enkai_backend_set(CString::new("torch").unwrap().as_ptr());
    assert_eq!(rc, 0);

    let mut cur_out: *mut i8 = std::ptr::null_mut();
    let mut cur_len: usize = 0;
    let rc = enkai_tensor::enkai_backend_current(&mut cur_out, &mut cur_len);
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
