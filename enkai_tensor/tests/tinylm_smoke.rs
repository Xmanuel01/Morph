#![cfg(feature = "torch")]

use std::ffi::CString;

use libc;
use serde_json;

#[test]
fn tinylm_loss_decreases_on_toy_data() {
    unsafe {
        let mut out_ptr: *mut libc::c_char = std::ptr::null_mut();
        let mut out_len: usize = 0;
        let dev = CString::new("cpu").unwrap();
        let device = enkai_tensor::enkai_tensor_device(dev.as_ptr());
        assert!(device != 0, "device handle");

        let vocab = 16i64;
        let seq_len = 8i64;
        let d_model = 64i64;
        let n_layers = 2i64;
        let n_heads = 4i64;
        let seed = 1234i64;

        let rc = enkai_tensor::enkai_tensor_tinylm_init(
            vocab,
            seq_len,
            d_model,
            n_layers,
            n_heads,
            device,
            seed,
            &mut out_ptr,
            &mut out_len,
        );
        assert_eq!(rc, 0);
        assert!(!out_ptr.is_null());

        let slice = std::slice::from_raw_parts(out_ptr as *const u8, out_len);
        let json = std::str::from_utf8(slice).unwrap();
        let params: Vec<i64> = serde_json::from_str(json).unwrap();
        let _ = CString::from_raw(out_ptr);

        let params_json = CString::new(serde_json::to_string(&params).unwrap()).unwrap();

        let batch_size = 2i64;
        let total = (batch_size * seq_len) as usize;
        let mut input_ids = Vec::with_capacity(total);
        let mut target_ids = Vec::with_capacity(total);
        for i in 0..total {
            let token = (i % vocab as usize) as u32;
            input_ids.push(token);
            target_ids.push(((i + 1) % vocab as usize) as u32);
        }

        let opt =
            enkai_tensor::enkai_opt_adamw_create(params_json.as_ptr(), 1e-2, 0.9, 0.999, 1e-8, 0.0);
        assert!(opt != 0);

        let mut loss_first = None;
        let mut loss_prev = f64::MAX;
        for _ in 0..8 {
            let loss_handle = enkai_tensor::enkai_tensor_forward_tinylm(
                params_json.as_ptr(),
                input_ids.as_ptr(),
                input_ids.len(),
                target_ids.as_ptr(),
                target_ids.len(),
                batch_size,
                seq_len,
                n_heads,
            );
            assert!(loss_handle != 0);
            let rc = enkai_tensor::enkai_tensor_backward(loss_handle);
            assert_eq!(rc, 0);

            let grads_slice = enkai_tensor::enkai_tensor_grad_multi(
                params_json.as_ptr(),
                &mut out_ptr,
                &mut out_len,
            );
            assert_eq!(grads_slice, 0);
            let grads = if out_ptr.is_null() || out_len == 0 {
                vec![]
            } else {
                let slice = std::slice::from_raw_parts(out_ptr as *const u8, out_len);
                let json = std::str::from_utf8(slice).unwrap();
                let g: Vec<i64> = serde_json::from_str(json).unwrap();
                let _ = CString::from_raw(out_ptr);
                g
            };
            assert_eq!(grads.len(), params.len());

            let rc = enkai_tensor::enkai_opt_adamw_step(opt);
            assert_eq!(rc, 0);
            let rc = enkai_tensor::enkai_tensor_zero_grad_multi(params_json.as_ptr());
            assert_eq!(rc, 0);

            let loss_val = enkai_tensor::enkai_tensor_item(loss_handle);
            if loss_first.is_none() {
                loss_first = Some(loss_val);
            }
            if loss_prev != f64::MAX {
                assert!(
                    loss_val <= loss_prev * 1.1,
                    "loss should not explode on toy data"
                );
            }
            loss_prev = loss_val;
        }

        let _ = enkai_tensor::enkai_tensor_opt_free(opt);
        let first = loss_first.unwrap_or(loss_prev);
        assert!(
            loss_prev <= first,
            "final loss should be <= initial loss on toy data"
        );
        let _ = enkai_tensor::enkai_tensor_device_free(device);
    }
}
