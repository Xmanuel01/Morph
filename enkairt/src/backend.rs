use std::path::PathBuf;

use libloading::Library;
use std::ffi::CString;
use std::os::raw::{c_char, c_int};
use std::sync::Arc;
use std::time::Instant;

use crate::dataset::Batch;
use crate::engine::AmpConfig;
use crate::error::RuntimeError;

#[derive(Debug, Clone, Copy)]
pub struct StepResult {
    pub loss: f32,
    pub forward_time_ms: f32,
    pub backward_time_ms: f32,
}

/// Minimal native backend wrapper (FFI). Uses enkai_tensor (libtorch) under the hood.
#[derive(Debug, Clone)]
pub struct Backend {
    _lib: Arc<Library>,
    symbols: Symbols,
    params: Vec<i64>,
    opt: Option<i64>,
    world_size: i32,
    rank: i32,
    device: Option<i64>,
    amp_scaler: Option<i64>,
}

#[derive(Debug, Clone)]
struct Symbols {
    opt_create: unsafe extern "C" fn(*const std::os::raw::c_char, f64, f64, f64, f64, f64) -> i64,
    opt_step: unsafe extern "C" fn(i64) -> c_int,
    opt_free: unsafe extern "C" fn(i64) -> c_int,
    grad_multi:
        unsafe extern "C" fn(*const std::os::raw::c_char, *mut *mut c_char, *mut usize) -> c_int,
    zero_grad_multi: unsafe extern "C" fn(*const std::os::raw::c_char) -> c_int,
    tinylm_init: unsafe extern "C" fn(
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        *mut *mut c_char,
        *mut usize,
    ) -> c_int,
    forward_tinylm: unsafe extern "C" fn(
        *const c_char,
        *const u32,
        usize,
        *const u32,
        usize,
        i64,
        i64,
        i64,
    ) -> i64,
    backward: unsafe extern "C" fn(i64) -> c_int,
    item: unsafe extern "C" fn(i64) -> f64,
    device: unsafe extern "C" fn(*const c_char) -> i64,
    device_free: unsafe extern "C" fn(i64) -> c_int,
    check_finite_multi: unsafe extern "C" fn(*const c_char) -> c_int,
    tensor_free: Option<unsafe extern "C" fn(i64) -> c_int>,
    grad_scale_multi: Option<unsafe extern "C" fn(*const c_char, f64) -> c_int>,
    grad_clip_norm_multi: Option<unsafe extern "C" fn(*const c_char, f64, f64, *mut f64) -> c_int>,
    adamw_step_multi: Option<
        unsafe extern "C" fn(*const c_char, *const c_char, i64, f64, f64, f64, f64, f64) -> i64,
    >,
    amp_scaler_create: Option<unsafe extern "C" fn(f64, f64, f64, i64) -> i64>,
    amp_scaler_free: Option<unsafe extern "C" fn(i64) -> c_int>,
    amp_scale_loss: Option<unsafe extern "C" fn(i64, i64) -> i64>,
    amp_unscale_grads: Option<unsafe extern "C" fn(*const c_char, i64, *mut c_int) -> c_int>,
    amp_scaler_update: Option<unsafe extern "C" fn(i64, c_int) -> c_int>,
    autocast_enter: Option<unsafe extern "C" fn() -> c_int>,
    autocast_exit: Option<unsafe extern "C" fn() -> c_int>,
    dist_init: Option<unsafe extern "C" fn(i32, i32) -> c_int>,
    dist_allreduce: Option<unsafe extern "C" fn(*const c_char) -> c_int>,
    dist_shutdown: Option<unsafe extern "C" fn() -> c_int>,
    ckpt_save: unsafe extern "C" fn(*const c_char, *const c_char, i64, *const c_char) -> c_int,
    ckpt_load: unsafe extern "C" fn(
        *const c_char,
        *mut *mut c_char,
        *mut usize,
        *mut i64,
        *mut *mut c_char,
        *mut usize,
    ) -> c_int,
    ckpt_save_ranked: Option<
        unsafe extern "C" fn(*const c_char, i64, i64, *const c_char, i64, *const c_char) -> c_int,
    >,
    ckpt_load_ranked: Option<
        unsafe extern "C" fn(
            *const c_char,
            i64,
            *mut *mut c_char,
            *mut usize,
            *mut i64,
            *mut *mut c_char,
            *mut usize,
        ) -> c_int,
    >,
    last_error: unsafe extern "C" fn() -> *const c_char,
}

impl Backend {
    pub fn new() -> Result<Self, RuntimeError> {
        let lib = load_library()?;
        let symbols = unsafe { load_symbols(&lib)? };
        Ok(Self {
            _lib: Arc::new(lib),
            symbols,
            params: Vec::new(),
            opt: None,
            world_size: 1,
            rank: 0,
            device: None,
            amp_scaler: None,
        })
    }

    pub fn opt_create(
        &self,
        params: &[i64],
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
    ) -> Result<i64, RuntimeError> {
        let json = serde_json::to_string(params).map_err(|e| RuntimeError::new(&e.to_string()))?;
        let c = CString::new(json).unwrap();
        let handle =
            unsafe { (self.symbols.opt_create)(c.as_ptr(), lr, beta1, beta2, eps, weight_decay) };
        if handle == 0 {
            Err(self.last_error())
        } else {
            Ok(handle)
        }
    }

    pub fn opt_step(&self, handle: i64) -> Result<(), RuntimeError> {
        let rc = unsafe { (self.symbols.opt_step)(handle) };
        if rc == 0 {
            Ok(())
        } else {
            Err(self.last_error())
        }
    }

    pub fn opt_free(&self, handle: i64) {
        unsafe {
            let _ = (self.symbols.opt_free)(handle);
        }
    }

    pub fn zero_grad_multi(&self, params: &[i64]) -> Result<(), RuntimeError> {
        let json = serde_json::to_string(params).map_err(|e| RuntimeError::new(&e.to_string()))?;
        let c = CString::new(json).unwrap();
        let rc = unsafe { (self.symbols.zero_grad_multi)(c.as_ptr()) };
        if rc == 0 {
            Ok(())
        } else {
            Err(self.last_error())
        }
    }

    pub fn grad_multi(&self, params: &[i64]) -> Result<Vec<i64>, RuntimeError> {
        let json = serde_json::to_string(params).map_err(|e| RuntimeError::new(&e.to_string()))?;
        let c = CString::new(json).unwrap();
        let mut out_len: usize = 0;
        let mut out_ptr: *mut c_char = std::ptr::null_mut();
        let rc = unsafe { (self.symbols.grad_multi)(c.as_ptr(), &mut out_ptr, &mut out_len) };
        if rc != 0 {
            return Err(self.last_error());
        }
        if out_ptr.is_null() || out_len == 0 {
            return Ok(vec![]);
        }
        let slice = unsafe { std::slice::from_raw_parts(out_ptr as *const u8, out_len) };
        let json_str = std::str::from_utf8(slice).map_err(|e| RuntimeError::new(&e.to_string()))?;
        let handles: Vec<i64> =
            serde_json::from_str(json_str).map_err(|e| RuntimeError::new(&e.to_string()))?;
        unsafe {
            let _ = CString::from_raw(out_ptr);
        }
        Ok(handles)
    }

    pub fn train_step_on_device(
        &mut self,
        _device_idx: usize,
        batch: &Batch,
        n_heads: usize,
        amp: &AmpConfig,
    ) -> Result<StepResult, RuntimeError> {
        if self.params.is_empty() {
            return Err(RuntimeError::new("No parameters registered for training"));
        }
        // Forward (TinyLM)
        let params_json =
            serde_json::to_string(&self.params).map_err(|e| RuntimeError::new(&e.to_string()))?;
        let c = CString::new(params_json).unwrap();
        let forward_start = Instant::now();
        if amp.enabled {
            self.autocast_enter()?;
        }
        let loss_handle = unsafe {
            (self.symbols.forward_tinylm)(
                c.as_ptr(),
                batch.input_ids.as_ptr(),
                batch.input_ids.len(),
                batch.target_ids.as_ptr(),
                batch.target_ids.len(),
                batch.batch_size as i64,
                batch.seq_len as i64,
                n_heads as i64,
            )
        };
        if amp.enabled {
            self.autocast_exit()?;
        }
        let forward_time_ms = to_ms(forward_start.elapsed());
        if loss_handle == 0 {
            return Err(self.last_error());
        }
        let loss_val = unsafe { (self.symbols.item)(loss_handle) } as f32;
        if !loss_val.is_finite() {
            return Err(RuntimeError::new("non-finite loss detected"));
        }
        // Backward
        let backward_start = Instant::now();
        let rc = if amp.enabled {
            let scaled = self.amp_scale_loss(loss_handle, amp)?;
            unsafe { (self.symbols.backward)(scaled) }
        } else {
            unsafe { (self.symbols.backward)(loss_handle) }
        };
        let backward_time_ms = to_ms(backward_start.elapsed());
        if rc != 0 {
            return Err(self.last_error());
        }
        Ok(StepResult {
            loss: loss_val,
            forward_time_ms,
            backward_time_ms,
        })
    }

    pub fn eval_step_on_device(
        &mut self,
        _device_idx: usize,
        batch: &Batch,
        n_heads: usize,
        amp: &AmpConfig,
    ) -> Result<StepResult, RuntimeError> {
        let params_json =
            serde_json::to_string(&self.params).map_err(|e| RuntimeError::new(&e.to_string()))?;
        let c = CString::new(params_json).unwrap();
        let forward_start = Instant::now();
        if amp.enabled {
            self.autocast_enter()?;
        }
        let loss_handle = unsafe {
            (self.symbols.forward_tinylm)(
                c.as_ptr(),
                batch.input_ids.as_ptr(),
                batch.input_ids.len(),
                batch.target_ids.as_ptr(),
                batch.target_ids.len(),
                batch.batch_size as i64,
                batch.seq_len as i64,
                n_heads as i64,
            )
        };
        if amp.enabled {
            self.autocast_exit()?;
        }
        let forward_time_ms = to_ms(forward_start.elapsed());
        if loss_handle == 0 {
            return Err(self.last_error());
        }
        let loss_val = unsafe { (self.symbols.item)(loss_handle) } as f32;
        if !loss_val.is_finite() {
            return Err(RuntimeError::new("non-finite loss detected"));
        }
        Ok(StepResult {
            loss: loss_val,
            forward_time_ms,
            backward_time_ms: 0.0,
        })
    }

    pub fn set_params(&mut self, params: Vec<i64>) -> Result<(), RuntimeError> {
        self.params = params;
        if self.opt.is_some() {
            self.backend_opt_free();
        }
        self.opt = None;
        // Re-init dist rank/world_size if needed; keep current values.
        if self.world_size > 1 {
            self.dist_init(self.world_size, self.rank)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn init_tinylm(
        &mut self,
        vocab_size: i64,
        seq_len: i64,
        d_model: i64,
        n_layers: i64,
        n_heads: i64,
        seed: i64,
        device_spec: &str,
    ) -> Result<Vec<i64>, RuntimeError> {
        let device = self.device_from_spec(device_spec)?;
        let mut out_len: usize = 0;
        let mut out_ptr: *mut c_char = std::ptr::null_mut();
        let rc = unsafe {
            (self.symbols.tinylm_init)(
                vocab_size,
                seq_len,
                d_model,
                n_layers,
                n_heads,
                device,
                seed,
                &mut out_ptr,
                &mut out_len,
            )
        };
        if rc != 0 {
            return Err(self.last_error());
        }
        if out_ptr.is_null() || out_len == 0 {
            return Err(RuntimeError::new("tinylm_init returned empty params"));
        }
        let slice = unsafe { std::slice::from_raw_parts(out_ptr as *const u8, out_len) };
        let json_str = std::str::from_utf8(slice).map_err(|e| RuntimeError::new(&e.to_string()))?;
        let handles: Vec<i64> =
            serde_json::from_str(json_str).map_err(|e| RuntimeError::new(&e.to_string()))?;
        unsafe {
            let _ = CString::from_raw(out_ptr);
        }
        self.device = Some(device);
        Ok(handles)
    }

    pub fn device_from_spec(&self, spec: &str) -> Result<i64, RuntimeError> {
        let c = CString::new(spec).unwrap();
        let handle = unsafe { (self.symbols.device)(c.as_ptr()) };
        if handle == 0 {
            Err(self.last_error())
        } else {
            Ok(handle)
        }
    }

    pub fn device_free(&self, handle: i64) {
        unsafe {
            let _ = (self.symbols.device_free)(handle);
        }
    }

    pub fn check_finite_multi(&self, handles: &[i64], label: &str) -> Result<(), RuntimeError> {
        let json = serde_json::to_string(handles).map_err(|e| RuntimeError::new(&e.to_string()))?;
        let c = CString::new(json).unwrap();
        let rc = unsafe { (self.symbols.check_finite_multi)(c.as_ptr()) };
        if rc == 0 {
            Ok(())
        } else {
            Err(RuntimeError::new(&format!(
                "non-finite values in {} (see enkai_tensor_last_error)",
                label
            )))
        }
    }

    pub fn free_tensors(&self, handles: &[i64]) {
        let Some(free) = self.symbols.tensor_free else {
            return;
        };
        for handle in handles {
            if *handle != 0 {
                unsafe {
                    let _ = free(*handle);
                }
            }
        }
    }

    pub fn scale_grads(&self, grads: &[i64], scale: f64) -> Result<(), RuntimeError> {
        if grads.is_empty() {
            return Ok(());
        }
        if (scale - 1.0).abs() < f64::EPSILON {
            return Ok(());
        }
        let Some(scale_fn) = self.symbols.grad_scale_multi else {
            return Err(RuntimeError::new(
                "grad scaling unavailable: enkai_tensor_scale_grads_multi missing",
            ));
        };
        let json = serde_json::to_string(grads).map_err(|e| RuntimeError::new(&e.to_string()))?;
        let c = CString::new(json).unwrap();
        let rc = unsafe { scale_fn(c.as_ptr(), scale) };
        if rc == 0 {
            Ok(())
        } else {
            Err(self.last_error())
        }
    }

    pub fn clip_grad_norm(&self, grads: &[i64], max_norm: f64) -> Result<f64, RuntimeError> {
        if grads.is_empty() {
            return Ok(0.0);
        }
        let Some(clip_fn) = self.symbols.grad_clip_norm_multi else {
            return Err(RuntimeError::new(
                "grad clipping unavailable: enkai_tensor_clip_grad_norm_multi missing",
            ));
        };
        let json = serde_json::to_string(grads).map_err(|e| RuntimeError::new(&e.to_string()))?;
        let c = CString::new(json).unwrap();
        let mut out_norm: f64 = 0.0;
        let rc = unsafe { clip_fn(c.as_ptr(), max_norm, 2.0, &mut out_norm as *mut f64) };
        if rc == 0 {
            Ok(out_norm)
        } else {
            Err(self.last_error())
        }
    }

    pub fn optimizer_step_with_grads(
        &mut self,
        grads: &[i64],
        hyper: &crate::engine::OptimConfig,
    ) -> Result<(), RuntimeError> {
        if self.params.is_empty() {
            return Ok(());
        }
        let Some(step_fn) = self.symbols.adamw_step_multi else {
            return Err(RuntimeError::new(
                "optimizer step unavailable: enkai_tensor_adamw_step_multi missing",
            ));
        };
        let params_json =
            serde_json::to_string(&self.params).map_err(|e| RuntimeError::new(&e.to_string()))?;
        let grads_json =
            serde_json::to_string(grads).map_err(|e| RuntimeError::new(&e.to_string()))?;
        let params_c = CString::new(params_json).unwrap();
        let grads_c = CString::new(grads_json).unwrap();
        let state = self.opt.unwrap_or(0);
        let handle = unsafe {
            step_fn(
                params_c.as_ptr(),
                grads_c.as_ptr(),
                state,
                hyper.lr,
                hyper.beta1,
                hyper.beta2,
                hyper.eps,
                hyper.weight_decay,
            )
        };
        if handle == 0 {
            return Err(self.last_error());
        }
        self.opt = Some(handle);
        Ok(())
    }

    fn ensure_amp_scaler(&mut self, amp: &AmpConfig) -> Result<i64, RuntimeError> {
        if let Some(handle) = self.amp_scaler {
            return Ok(handle);
        }
        let Some(create) = self.symbols.amp_scaler_create else {
            return Err(RuntimeError::new(
                "AMP unavailable: enkai_amp_scaler_create missing",
            ));
        };
        let handle = unsafe {
            create(
                amp.init_scale,
                amp.growth_factor,
                amp.backoff_factor,
                amp.growth_interval,
            )
        };
        if handle == 0 {
            return Err(self.last_error());
        }
        self.amp_scaler = Some(handle);
        Ok(handle)
    }

    fn amp_scale_loss(&mut self, loss: i64, amp: &AmpConfig) -> Result<i64, RuntimeError> {
        let scaler = self.ensure_amp_scaler(amp)?;
        let Some(scale_fn) = self.symbols.amp_scale_loss else {
            return Err(RuntimeError::new(
                "AMP unavailable: enkai_amp_scale_loss missing",
            ));
        };
        let handle = unsafe { scale_fn(loss, scaler) };
        if handle == 0 {
            Err(self.last_error())
        } else {
            Ok(handle)
        }
    }

    pub fn amp_unscale_grads(
        &mut self,
        grads: &[i64],
        amp: &AmpConfig,
    ) -> Result<bool, RuntimeError> {
        if grads.is_empty() {
            return Ok(false);
        }
        let scaler = self.ensure_amp_scaler(amp)?;
        let Some(unscale_fn) = self.symbols.amp_unscale_grads else {
            return Err(RuntimeError::new(
                "AMP unavailable: enkai_amp_unscale_grads missing",
            ));
        };
        let json = serde_json::to_string(grads).map_err(|e| RuntimeError::new(&e.to_string()))?;
        let c = CString::new(json).unwrap();
        let mut found_inf: c_int = 0;
        let rc = unsafe { unscale_fn(c.as_ptr(), scaler, &mut found_inf as *mut c_int) };
        if rc == 0 {
            Ok(found_inf != 0)
        } else {
            Err(self.last_error())
        }
    }

    pub fn amp_scaler_update(
        &mut self,
        amp: &AmpConfig,
        found_inf: bool,
    ) -> Result<(), RuntimeError> {
        let scaler = self.ensure_amp_scaler(amp)?;
        let Some(update_fn) = self.symbols.amp_scaler_update else {
            return Err(RuntimeError::new(
                "AMP unavailable: enkai_amp_scaler_update missing",
            ));
        };
        let rc = unsafe { update_fn(scaler, if found_inf { 1 } else { 0 }) };
        if rc == 0 {
            Ok(())
        } else {
            Err(self.last_error())
        }
    }

    fn autocast_enter(&self) -> Result<(), RuntimeError> {
        let Some(enter) = self.symbols.autocast_enter else {
            return Err(RuntimeError::new(
                "AMP unavailable: enkai_autocast_enter missing",
            ));
        };
        let rc = unsafe { enter() };
        if rc == 0 {
            Ok(())
        } else {
            Err(self.last_error())
        }
    }

    fn autocast_exit(&self) -> Result<(), RuntimeError> {
        let Some(exit) = self.symbols.autocast_exit else {
            return Err(RuntimeError::new(
                "AMP unavailable: enkai_autocast_exit missing",
            ));
        };
        let rc = unsafe { exit() };
        if rc == 0 {
            Ok(())
        } else {
            Err(self.last_error())
        }
    }

    pub fn ensure_optimizer(
        &mut self,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
    ) -> Result<(), RuntimeError> {
        if self.opt.is_none() && !self.params.is_empty() {
            let opt = self.opt_create(&self.params, lr, beta1, beta2, eps, weight_decay)?;
            self.opt = Some(opt);
        }
        Ok(())
    }

    pub fn zero_all_grads(&self) -> Result<(), RuntimeError> {
        if self.params.is_empty() {
            return Ok(());
        }
        self.zero_grad_multi(&self.params)
    }

    pub fn optimizer_step(&self) -> Result<(), RuntimeError> {
        if let Some(opt) = self.opt {
            self.opt_step(opt)
        } else {
            Ok(())
        }
    }

    pub fn backend_opt_free(&mut self) {
        if let Some(opt) = self.opt.take() {
            self.opt_free(opt);
        }
    }

    pub fn checkpoint_save(&self, dir: &str, meta_json: &str) -> Result<(), RuntimeError> {
        let params_json =
            serde_json::to_string(&self.params).map_err(|e| RuntimeError::new(&e.to_string()))?;
        let dir_c = CString::new(dir).unwrap();
        let params_c = CString::new(params_json).unwrap();
        let meta_c = CString::new(meta_json).unwrap();
        let opt = self.opt.unwrap_or(0);
        let rc = if self.world_size > 1 {
            let Some(save_ranked) = self.symbols.ckpt_save_ranked else {
                return Err(RuntimeError::new(
                    "ranked checkpoint save unavailable: enkai_checkpoint_save_ranked missing",
                ));
            };
            unsafe {
                save_ranked(
                    dir_c.as_ptr(),
                    self.rank as i64,
                    self.world_size as i64,
                    params_c.as_ptr(),
                    opt,
                    meta_c.as_ptr(),
                )
            }
        } else {
            unsafe {
                (self.symbols.ckpt_save)(dir_c.as_ptr(), params_c.as_ptr(), opt, meta_c.as_ptr())
            }
        };
        if rc == 0 {
            Ok(())
        } else {
            Err(self.last_error())
        }
    }

    pub fn checkpoint_load(
        &mut self,
        dir: &str,
    ) -> Result<(Vec<i64>, Option<i64>, String), RuntimeError> {
        let dir_c = CString::new(dir).unwrap();
        let mut out_params_len: usize = 0;
        let mut out_meta_len: usize = 0;
        let mut out_params_ptr: *mut c_char = std::ptr::null_mut();
        let mut out_meta_ptr: *mut c_char = std::ptr::null_mut();
        let mut out_opt: i64 = 0;
        let rc = if self.world_size > 1 {
            let Some(load_ranked) = self.symbols.ckpt_load_ranked else {
                return Err(RuntimeError::new(
                    "ranked checkpoint load unavailable: enkai_checkpoint_load_ranked missing",
                ));
            };
            unsafe {
                load_ranked(
                    dir_c.as_ptr(),
                    self.rank as i64,
                    &mut out_params_ptr,
                    &mut out_params_len,
                    &mut out_opt,
                    &mut out_meta_ptr,
                    &mut out_meta_len,
                )
            }
        } else {
            unsafe {
                (self.symbols.ckpt_load)(
                    dir_c.as_ptr(),
                    &mut out_params_ptr,
                    &mut out_params_len,
                    &mut out_opt,
                    &mut out_meta_ptr,
                    &mut out_meta_len,
                )
            }
        };
        if rc != 0 {
            return Err(self.last_error());
        }
        let params_json = unsafe {
            let slice = std::slice::from_raw_parts(out_params_ptr as *const u8, out_params_len);
            let s = std::str::from_utf8(slice).map_err(|e| RuntimeError::new(&e.to_string()))?;
            let _ = CString::from_raw(out_params_ptr);
            s.to_string()
        };
        let meta_json = unsafe {
            let slice = std::slice::from_raw_parts(out_meta_ptr as *const u8, out_meta_len);
            let s = std::str::from_utf8(slice).map_err(|e| RuntimeError::new(&e.to_string()))?;
            let _ = CString::from_raw(out_meta_ptr);
            s.to_string()
        };
        let params: Vec<i64> =
            serde_json::from_str(&params_json).map_err(|e| RuntimeError::new(&e.to_string()))?;
        self.params = params.clone();
        self.opt = if out_opt != 0 { Some(out_opt) } else { None };
        Ok((params, self.opt, meta_json))
    }

    pub fn dist_init(&self, world_size: i32, rank: i32) -> Result<(), RuntimeError> {
        let Some(dist_init) = self.symbols.dist_init else {
            return Err(RuntimeError::new(
                "distributed backend symbols unavailable: enkai_dist_init missing",
            ));
        };
        let rc = unsafe { dist_init(world_size, rank) };
        if rc == 0 {
            Ok(())
        } else {
            Err(self.last_error())
        }
    }

    pub fn dist_allreduce_grads(&self, grads: &[i64]) -> Result<(), RuntimeError> {
        if grads.is_empty() {
            return Ok(());
        }
        let Some(dist_allreduce) = self.symbols.dist_allreduce else {
            return Err(RuntimeError::new(
                "distributed backend symbols unavailable: enkai_dist_allreduce_sum_multi missing",
            ));
        };
        let json = serde_json::to_string(grads).map_err(|e| RuntimeError::new(&e.to_string()))?;
        let c = CString::new(json).unwrap();
        let rc = unsafe { dist_allreduce(c.as_ptr()) };
        if rc == 0 {
            Ok(())
        } else {
            Err(self.last_error())
        }
    }

    pub fn dist_shutdown(&self) {
        if let Some(dist_shutdown) = self.symbols.dist_shutdown {
            unsafe {
                let _ = dist_shutdown();
            }
        }
    }

    pub fn configure_dist(&mut self, world_size: i32, rank: i32) -> Result<(), RuntimeError> {
        self.world_size = world_size;
        self.rank = rank;
        if world_size > 1 {
            self.dist_init(world_size, rank)?;
        }
        Ok(())
    }

    fn last_error(&self) -> RuntimeError {
        let ptr = unsafe { (self.symbols.last_error)() };
        if ptr.is_null() {
            return RuntimeError::new("native backend error");
        }
        let msg = unsafe { std::ffi::CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned();
        if msg.is_empty() {
            RuntimeError::new("native backend error")
        } else {
            RuntimeError::new(&msg)
        }
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        if let Some(handle) = self.amp_scaler.take() {
            if let Some(free) = self.symbols.amp_scaler_free {
                unsafe {
                    let _ = free(handle);
                }
            }
        }
        if let Some(opt) = self.opt.take() {
            unsafe {
                let _ = (self.symbols.opt_free)(opt);
            }
        }
        if !self.params.is_empty() {
            self.free_tensors(&self.params);
            self.params.clear();
        }
        if let Some(device) = self.device.take() {
            self.device_free(device);
        }
        if self.world_size > 1 {
            self.dist_shutdown();
        }
    }
}

fn load_library() -> Result<Library, RuntimeError> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(path) = std::env::var("ENKAI_TENSOR_PATH") {
        candidates.push(PathBuf::from(path));
    }
    candidates.push(PathBuf::from(lib_name("enkai_tensor")));
    for candidate in candidates {
        if let Ok(lib) = unsafe { Library::new(&candidate) } {
            return Ok(lib);
        }
    }
    Err(RuntimeError::new(
        "Failed to load enkai_tensor (libtorch backend)",
    ))
}

fn lib_name(base: &str) -> String {
    if cfg!(target_os = "windows") {
        format!("{base}.dll")
    } else if cfg!(target_os = "macos") {
        format!("lib{base}.dylib")
    } else {
        format!("lib{base}.so")
    }
}

fn to_ms(duration: std::time::Duration) -> f32 {
    (duration.as_secs_f64() * 1_000.0) as f32
}
unsafe fn load_symbols(lib: &Library) -> Result<Symbols, RuntimeError> {
    macro_rules! sym {
        ($name:literal, $ty:ty) => {{
            let s: libloading::Symbol<$ty> = lib
                .get($name)
                .map_err(|e| RuntimeError::new(&format!("load symbol {:?}: {}", $name, e)))?;
            *s
        }};
    }
    macro_rules! sym_opt {
        ($name:literal, $ty:ty) => {{
            lib.get::<$ty>($name).ok().map(|s| *s)
        }};
    }
    Ok(Symbols {
        opt_create: sym!(
            b"enkai_opt_adamw_create\0",
            unsafe extern "C" fn(*const c_char, f64, f64, f64, f64, f64) -> i64
        ),
        opt_step: sym!(
            b"enkai_opt_adamw_step\0",
            unsafe extern "C" fn(i64) -> c_int
        ),
        opt_free: sym!(
            b"enkai_tensor_opt_free\0",
            unsafe extern "C" fn(i64) -> c_int
        ),
        grad_multi: sym!(
            b"enkai_tensor_grad_multi\0",
            unsafe extern "C" fn(*const c_char, *mut *mut c_char, *mut usize) -> c_int
        ),
        zero_grad_multi: sym!(
            b"enkai_tensor_zero_grad_multi\0",
            unsafe extern "C" fn(*const c_char) -> c_int
        ),
        tinylm_init: sym!(
            b"enkai_tensor_tinylm_init\0",
            unsafe extern "C" fn(
                i64,
                i64,
                i64,
                i64,
                i64,
                i64,
                i64,
                *mut *mut c_char,
                *mut usize,
            ) -> c_int
        ),
        forward_tinylm: sym!(
            b"enkai_tensor_forward_tinylm\0",
            unsafe extern "C" fn(
                *const c_char,
                *const u32,
                usize,
                *const u32,
                usize,
                i64,
                i64,
                i64,
            ) -> i64
        ),
        backward: sym!(
            b"enkai_tensor_backward\0",
            unsafe extern "C" fn(i64) -> c_int
        ),
        item: sym!(b"enkai_tensor_item\0", unsafe extern "C" fn(i64) -> f64),
        device: sym!(
            b"enkai_tensor_device\0",
            unsafe extern "C" fn(*const c_char) -> i64
        ),
        device_free: sym!(
            b"enkai_tensor_device_free\0",
            unsafe extern "C" fn(i64) -> c_int
        ),
        check_finite_multi: sym!(
            b"enkai_tensor_check_finite_multi\0",
            unsafe extern "C" fn(*const c_char) -> c_int
        ),
        tensor_free: sym_opt!(b"enkai_tensor_free\0", unsafe extern "C" fn(i64) -> c_int),
        grad_scale_multi: sym_opt!(
            b"enkai_tensor_scale_grads_multi\0",
            unsafe extern "C" fn(*const c_char, f64) -> c_int
        ),
        grad_clip_norm_multi: sym_opt!(
            b"enkai_tensor_clip_grad_norm_multi\0",
            unsafe extern "C" fn(*const c_char, f64, f64, *mut f64) -> c_int
        ),
        adamw_step_multi: sym_opt!(
            b"enkai_tensor_adamw_step_multi\0",
            unsafe extern "C" fn(*const c_char, *const c_char, i64, f64, f64, f64, f64, f64) -> i64
        ),
        amp_scaler_create: sym_opt!(
            b"enkai_amp_scaler_create\0",
            unsafe extern "C" fn(f64, f64, f64, i64) -> i64
        ),
        amp_scaler_free: sym_opt!(
            b"enkai_amp_scaler_free\0",
            unsafe extern "C" fn(i64) -> c_int
        ),
        amp_scale_loss: sym_opt!(
            b"enkai_amp_scale_loss\0",
            unsafe extern "C" fn(i64, i64) -> i64
        ),
        amp_unscale_grads: sym_opt!(
            b"enkai_amp_unscale_grads\0",
            unsafe extern "C" fn(*const c_char, i64, *mut c_int) -> c_int
        ),
        amp_scaler_update: sym_opt!(
            b"enkai_amp_scaler_update\0",
            unsafe extern "C" fn(i64, c_int) -> c_int
        ),
        autocast_enter: sym_opt!(b"enkai_autocast_enter\0", unsafe extern "C" fn() -> c_int),
        autocast_exit: sym_opt!(b"enkai_autocast_exit\0", unsafe extern "C" fn() -> c_int),
        dist_init: lib
            .get::<unsafe extern "C" fn(i32, i32) -> c_int>(b"enkai_dist_init\0")
            .ok()
            .map(|s| *s),
        dist_allreduce: lib
            .get::<unsafe extern "C" fn(*const c_char) -> c_int>(
                b"enkai_dist_allreduce_sum_multi\0",
            )
            .ok()
            .map(|s| *s),
        dist_shutdown: lib
            .get::<unsafe extern "C" fn() -> c_int>(b"enkai_dist_shutdown\0")
            .ok()
            .map(|s| *s),
        ckpt_save: sym!(
            b"enkai_checkpoint_save\0",
            unsafe extern "C" fn(*const c_char, *const c_char, i64, *const c_char) -> c_int
        ),
        ckpt_load: sym!(
            b"enkai_checkpoint_load\0",
            unsafe extern "C" fn(
                *const c_char,
                *mut *mut c_char,
                *mut usize,
                *mut i64,
                *mut *mut c_char,
                *mut usize,
            ) -> c_int
        ),
        ckpt_save_ranked: sym_opt!(
            b"enkai_checkpoint_save_ranked\0",
            unsafe extern "C" fn(
                *const c_char,
                i64,
                i64,
                *const c_char,
                i64,
                *const c_char,
            ) -> c_int
        ),
        ckpt_load_ranked: sym_opt!(
            b"enkai_checkpoint_load_ranked\0",
            unsafe extern "C" fn(
                *const c_char,
                i64,
                *mut *mut c_char,
                *mut usize,
                *mut i64,
                *mut *mut c_char,
                *mut usize,
            ) -> c_int
        ),
        last_error: sym!(
            b"enkai_tensor_last_error\0",
            unsafe extern "C" fn() -> *const c_char
        ),
    })
}
