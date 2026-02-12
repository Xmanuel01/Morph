use std::path::PathBuf;

use libloading::Library;
use std::ffi::CString;
use std::os::raw::{c_char, c_int};
use std::sync::Arc;

use crate::dataset::Batch;
use crate::error::RuntimeError;

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
    ) -> Result<f32, RuntimeError> {
        if self.params.is_empty() {
            return Err(RuntimeError::new("No parameters registered for training"));
        }
        // Forward (TinyLM)
        let params_json =
            serde_json::to_string(&self.params).map_err(|e| RuntimeError::new(&e.to_string()))?;
        let c = CString::new(params_json).unwrap();
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
        if loss_handle == 0 {
            return Err(self.last_error());
        }
        let loss_val = unsafe { (self.symbols.item)(loss_handle) } as f32;
        if !loss_val.is_finite() {
            return Err(RuntimeError::new("non-finite loss detected"));
        }
        // Backward
        let rc = unsafe { (self.symbols.backward)(loss_handle) };
        if rc != 0 {
            return Err(self.last_error());
        }
        // Allreduce grads if distributed.
        if self.world_size > 1 {
            let grads = self.grad_multi(&self.params)?;
            self.dist_allreduce_grads(&grads)?;
        }
        // Validate grads
        let grads = self.grad_multi(&self.params)?;
        if !grads.is_empty() {
            self.check_finite_multi(&grads, "grads")?;
        }
        Ok(loss_val)
    }

    pub fn eval_step_on_device(
        &mut self,
        device_idx: usize,
        batch: &Batch,
        n_heads: usize,
    ) -> Result<f32, RuntimeError> {
        self.train_step_on_device(device_idx, batch, n_heads)
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
        let rc = unsafe {
            (self.symbols.ckpt_save)(dir_c.as_ptr(), params_c.as_ptr(), opt, meta_c.as_ptr())
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
        let rc = unsafe {
            (self.symbols.ckpt_load)(
                dir_c.as_ptr(),
                &mut out_params_ptr,
                &mut out_params_len,
                &mut out_opt,
                &mut out_meta_ptr,
                &mut out_meta_len,
            )
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
unsafe fn load_symbols(lib: &Library) -> Result<Symbols, RuntimeError> {
    macro_rules! sym {
        ($name:literal, $ty:ty) => {{
            let s: libloading::Symbol<$ty> = lib
                .get($name)
                .map_err(|e| RuntimeError::new(&format!("load symbol {:?}: {}", $name, e)))?;
            *s
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
        last_error: sym!(
            b"enkai_tensor_last_error\0",
            unsafe extern "C" fn() -> *const c_char
        ),
    })
}
