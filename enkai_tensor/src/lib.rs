use libc::{c_char, c_int};
#[cfg(feature = "torch")]
use once_cell::sync::Lazy;
#[cfg(feature = "torch")]
use sha2::{Digest, Sha256};
use std::cell::RefCell;
#[cfg(feature = "torch")]
use std::collections::{HashMap, HashSet};
use std::ffi::{CStr, CString};
use std::fs;
#[cfg(feature = "torch")]
use std::fs::OpenOptions;
use std::panic::{self, AssertUnwindSafe};
#[cfg(feature = "torch")]
use std::path::Path;
use std::ptr;
#[cfg(feature = "torch")]
use std::sync::atomic::{AtomicI64, Ordering};
#[cfg(feature = "torch")]
use std::sync::Mutex;

mod backend;
#[cfg_attr(not(feature = "torch"), allow(unused_imports))]
use backend::{backend_is_cpu, backend_is_torch, current_backend, set_backend, BackendKind};
#[cfg(feature = "torch")]
use tch::{Device, Kind, Tensor};

#[cfg(all(feature = "torch", feature = "dist"))]
pub mod dist;

#[cfg(feature = "torch")]
#[derive(Debug)]
struct TensorEntry {
    tensor: Tensor,
    refcount: u32,
}

#[cfg(feature = "torch")]
#[derive(Debug, Clone, Copy)]
struct DeviceEntry {
    device: Device,
    refcount: u32,
}

#[cfg(feature = "torch")]
#[derive(Debug)]
struct OptEntry {
    state: AdamWState,
    refcount: u32,
}

#[repr(C)]
pub struct FfiSlice {
    pub ptr: *mut u8,
    pub len: usize,
}

fn make_slice(mut bytes: Vec<u8>) -> FfiSlice {
    let len = bytes.len();
    let ptr = bytes.as_mut_ptr();
    std::mem::forget(bytes);
    FfiSlice { ptr, len }
}

fn null_slice() -> FfiSlice {
    FfiSlice {
        ptr: ptr::null_mut(),
        len: 0,
    }
}

#[cfg(feature = "torch")]
fn fsync_file(path: &str) -> Result<(), String> {
    let file = OpenOptions::new()
        .read(true)
        .open(path)
        .map_err(|err| format!("fsync open {}: {}", path, err))?;
    file.sync_all()
        .map_err(|err| format!("fsync {}: {}", path, err))
}

#[cfg(feature = "torch")]
fn fsync_dir(path: &str) -> Result<(), String> {
    let dir = OpenOptions::new()
        .read(true)
        .open(path)
        .map_err(|err| format!("fsync dir open {}: {}", path, err))?;
    dir.sync_all()
        .map_err(|err| format!("fsync dir {}: {}", path, err))
}

#[cfg(feature = "torch")]
#[derive(Debug)]
struct AdamWSlot {
    m: Tensor,
    v: Tensor,
}

#[cfg(feature = "torch")]
#[derive(Debug)]
struct AdamWState {
    slots: HashMap<i64, AdamWSlot>,
    step: i64,
}

#[cfg(feature = "torch")]
#[derive(Debug, Clone, Copy)]
struct AdamWHyper {
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
}

#[cfg(feature = "torch")]
#[derive(Debug, Clone)]
struct GradScalerState {
    scale: f64,
    growth_factor: f64,
    backoff_factor: f64,
    growth_interval: i64,
    growth_tracker: i64,
    refcount: u32,
}

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_error(message: impl Into<String>) {
    let msg = message.into();
    let cstr = CString::new(msg).unwrap_or_else(|_| CString::new("Unknown error").unwrap());
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = Some(cstr);
    });
}

fn clear_error() {
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = None;
    });
}

#[cfg(feature = "torch")]
fn require_tch_backend() -> bool {
    if backend_is_torch() || backend_is_cpu() {
        true
    } else {
        set_error("unknown backend selected");
        false
    }
}

#[cfg(feature = "torch")]
macro_rules! guard_backend {
    ($default:expr) => {
        if !require_tch_backend() {
            return $default;
        }
    };
}

fn ffi_guard<T, F>(default: T, f: F) -> T
where
    F: FnOnce() -> T + std::panic::UnwindSafe,
{
    match panic::catch_unwind(AssertUnwindSafe(f)) {
        Ok(v) => v,
        Err(payload) => {
            let msg = if let Some(s) = payload.downcast_ref::<&str>() {
                *s
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.as_str()
            } else {
                "panic across FFI boundary"
            };
            set_error(format!("panic: {}", msg));
            default
        }
    }
}

#[cfg(feature = "torch")]
fn ensure_requires_grad(t: Tensor) -> Tensor {
    if t.requires_grad() {
        t
    } else {
        t.set_requires_grad(true)
    }
}

#[no_mangle]
/// # Safety
/// `out_json` and `out_len` must be valid writable pointers for this call.
pub unsafe extern "C" fn enkai_backend_list(
    out_json: *mut *mut c_char,
    out_len: *mut usize,
) -> c_int {
    ffi_guard(1, || {
        clear_error();
        let list = vec!["torch", "cpu"];
        let json = match serde_json::to_string(&list) {
            Ok(s) => s,
            Err(err) => {
                set_error(err.to_string());
                return 1;
            }
        };
        unsafe {
            *out_len = json.len();
            match CString::new(json) {
                Ok(c) => {
                    *out_json = c.into_raw();
                    0
                }
                Err(_) => {
                    set_error("backend list contained null byte");
                    1
                }
            }
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_backend_set(name: *const c_char) -> c_int {
    ffi_guard(1, || {
        clear_error();
        let name = match cstr_to_string(name) {
            Ok(s) => s,
            Err(err) => {
                set_error(err);
                return 1;
            }
        };
        if name == "torch" {
            set_backend(BackendKind::Torch);
            return 0;
        }
        if name == "cpu" {
            set_backend(BackendKind::Cpu);
            return 0;
        }
        set_error("unknown backend");
        1
    })
}

#[no_mangle]
/// # Safety
/// `out_json` and `out_len` must be valid writable pointers for this call.
pub unsafe extern "C" fn enkai_backend_current(
    out_json: *mut *mut c_char,
    out_len: *mut usize,
) -> c_int {
    ffi_guard(1, || {
        clear_error();
        let name = match current_backend() {
            BackendKind::Torch => "torch",
            BackendKind::Cpu => "cpu",
        };
        let s = name.to_string();
        unsafe {
            *out_len = s.len();
            match CString::new(s) {
                Ok(c) => {
                    *out_json = c.into_raw();
                    0
                }
                Err(_) => {
                    set_error("backend name contained null byte");
                    1
                }
            }
        }
    })
}

#[no_mangle]
/// Configure distributed context (stub; no NCCL init yet).
pub extern "C" fn enkai_dist_config(world_size: i64, rank: i64, device: i64, seed: i64) -> c_int {
    ffi_guard(1, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            if world_size <= 0 || rank < 0 || rank >= world_size {
                set_error("invalid world_size or rank");
                return 1;
            }
            DIST_WORLD.store(world_size, Ordering::SeqCst);
            DIST_RANK.store(rank, Ordering::SeqCst);
            let dev_to_use = if device >= 0 {
                device
            } else {
                if backend_is_torch() {
                    if tch::Cuda::is_available() {
                        rank
                    } else {
                        -1
                    }
                } else {
                    -1
                }
            };
            DIST_DEVICE.store(dev_to_use, Ordering::SeqCst);
            DIST_SEED.store(seed, Ordering::SeqCst);
            0
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = world_size;
            let _ = rank;
            let _ = device;
            let _ = seed;
            set_error("torch backend not enabled");
            1
        }
    })
}

#[no_mangle]
/// Stub all-reduce (sum) that currently errors until NCCL is added.
pub extern "C" fn enkai_dist_allreduce_sum(_params_json: *const c_char) -> c_int {
    ffi_guard(1, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            let params_json = match cstr_to_string(_params_json) {
                Ok(v) => v,
                Err(err) => {
                    set_error(err);
                    return 1;
                }
            };
            let handles = match parse_handle_list(&params_json) {
                Ok(list) => list,
                Err(err) => {
                    set_error(err);
                    return 1;
                }
            };
            let world = DIST_WORLD.load(Ordering::SeqCst);
            if world > 1 {
                set_error("multi-rank allreduce requires NCCL; not implemented");
                return 1;
            }
            for h in handles {
                if h == 0 {
                    continue;
                }
                if get_tensor(h).is_err() {
                    set_error("Invalid tensor handle in allreduce");
                    return 1;
                }
            }
            0
        }
        #[cfg(not(feature = "torch"))]
        {
            set_error("torch backend not enabled");
            1
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_last_error() -> *const c_char {
    ffi_guard(ptr::null(), || {
        LAST_ERROR.with(|cell| match &*cell.borrow() {
            Some(msg) => msg.as_ptr(),
            None => ptr::null(),
        })
    })
}

fn cstr_to_string(ptr: *const c_char) -> Result<String, String> {
    if ptr.is_null() {
        return Err("Null string pointer".to_string());
    }
    let cstr = unsafe { CStr::from_ptr(ptr) };
    cstr.to_str()
        .map(|s| s.to_string())
        .map_err(|_| "Invalid UTF-8 string".to_string())
}

#[cfg(feature = "torch")]
static NEXT_HANDLE: AtomicI64 = AtomicI64::new(1);

#[cfg(feature = "torch")]
static TENSORS: Lazy<Mutex<HashMap<i64, TensorEntry>>> = Lazy::new(|| Mutex::new(HashMap::new()));
#[cfg(feature = "torch")]
static TENSOR_FREED: Lazy<Mutex<HashSet<i64>>> = Lazy::new(|| Mutex::new(HashSet::new()));

#[cfg(feature = "torch")]
static DEVICES: Lazy<Mutex<HashMap<i64, DeviceEntry>>> = Lazy::new(|| Mutex::new(HashMap::new()));
#[cfg(feature = "torch")]
static DEVICE_FREED: Lazy<Mutex<HashSet<i64>>> = Lazy::new(|| Mutex::new(HashSet::new()));

#[cfg(feature = "torch")]
static OPT_STATES: Lazy<Mutex<HashMap<i64, OptEntry>>> = Lazy::new(|| Mutex::new(HashMap::new()));
#[cfg(feature = "torch")]
static OPT_PARAMS: Lazy<Mutex<HashMap<i64, Vec<i64>>>> = Lazy::new(|| Mutex::new(HashMap::new()));
#[cfg(feature = "torch")]
static OPT_HYPER: Lazy<Mutex<HashMap<i64, AdamWHyper>>> = Lazy::new(|| Mutex::new(HashMap::new()));
#[cfg(feature = "torch")]
static OPT_FREED: Lazy<Mutex<HashSet<i64>>> = Lazy::new(|| Mutex::new(HashSet::new()));
#[cfg(feature = "torch")]
static SCALERS: Lazy<Mutex<HashMap<i64, GradScalerState>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));
#[cfg(feature = "torch")]
static SCALER_FREED: Lazy<Mutex<HashSet<i64>>> = Lazy::new(|| Mutex::new(HashSet::new()));
#[cfg(feature = "torch")]
static DIST_RANK: AtomicI64 = AtomicI64::new(0);
#[cfg(feature = "torch")]
static DIST_WORLD: AtomicI64 = AtomicI64::new(1);
#[cfg(feature = "torch")]
static DIST_DEVICE: AtomicI64 = AtomicI64::new(-1);
#[cfg(feature = "torch")]
static DIST_SEED: AtomicI64 = AtomicI64::new(0);

#[cfg(feature = "torch")]
fn file_sha256(path: &str) -> Result<String, String> {
    let data = fs::read(path).map_err(|e| e.to_string())?;
    let mut hasher = Sha256::new();
    hasher.update(data);
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(feature = "torch")]
fn next_handle() -> i64 {
    NEXT_HANDLE.fetch_add(1, Ordering::SeqCst)
}

#[cfg(feature = "torch")]
fn register_tensor(t: Tensor) -> i64 {
    if !require_tch_backend() {
        return 0;
    }
    match TENSORS.lock() {
        Ok(mut guard) => {
            let id = next_handle();
            if let Ok(mut freed) = TENSOR_FREED.lock() {
                freed.remove(&id);
            }
            guard.insert(
                id,
                TensorEntry {
                    tensor: t,
                    refcount: 1,
                },
            );
            id
        }
        Err(_) => {
            set_error("tensor registry poisoned");
            0
        }
    }
}

#[cfg(feature = "torch")]
fn register_device(d: Device) -> i64 {
    if !require_tch_backend() {
        return 0;
    }
    match DEVICES.lock() {
        Ok(mut guard) => {
            let id = next_handle();
            if let Ok(mut freed) = DEVICE_FREED.lock() {
                freed.remove(&id);
            }
            guard.insert(
                id,
                DeviceEntry {
                    device: d,
                    refcount: 1,
                },
            );
            id
        }
        Err(_) => {
            set_error("device registry poisoned");
            0
        }
    }
}

#[cfg(feature = "torch")]
fn register_opt(state: AdamWState) -> i64 {
    if !require_tch_backend() {
        return 0;
    }
    match OPT_STATES.lock() {
        Ok(mut guard) => {
            let id = next_handle();
            if let Ok(mut freed) = OPT_FREED.lock() {
                freed.remove(&id);
            }
            guard.insert(id, OptEntry { state, refcount: 1 });
            id
        }
        Err(_) => {
            set_error("optimizer registry poisoned");
            0
        }
    }
}

#[cfg(feature = "torch")]
fn register_scaler(state: GradScalerState) -> i64 {
    if !require_tch_backend() {
        return 0;
    }
    match SCALERS.lock() {
        Ok(mut guard) => {
            let id = next_handle();
            if let Ok(mut freed) = SCALER_FREED.lock() {
                freed.remove(&id);
            }
            guard.insert(id, state);
            id
        }
        Err(_) => {
            set_error("scaler registry poisoned");
            0
        }
    }
}

#[cfg(feature = "torch")]
fn get_tensor(handle: i64) -> Result<Tensor, String> {
    if !require_tch_backend() {
        return Err("backend not supported".to_string());
    }
    if let Ok(freed) = TENSOR_FREED.lock() {
        if freed.contains(&handle) {
            return Err("Stale tensor handle (freed)".to_string());
        }
    }
    let guard = TENSORS
        .lock()
        .map_err(|_| "tensor registry poisoned".to_string())?;
    guard
        .get(&handle)
        .map(|entry| entry.tensor.shallow_clone())
        .ok_or_else(|| "Invalid tensor handle".to_string())
}

#[cfg(feature = "torch")]
fn get_device(handle: i64) -> Result<Device, String> {
    if !require_tch_backend() {
        return Err("backend not supported".to_string());
    }
    if let Ok(freed) = DEVICE_FREED.lock() {
        if freed.contains(&handle) {
            return Err("Stale device handle (freed)".to_string());
        }
    }
    let guard = DEVICES
        .lock()
        .map_err(|_| "device registry poisoned".to_string())?;
    guard
        .get(&handle)
        .map(|entry| entry.device)
        .ok_or_else(|| "Invalid device handle".to_string())
}

#[cfg(feature = "torch")]
fn get_opt_mut(handle: i64) -> Result<AdamWState, String> {
    if !require_tch_backend() {
        return Err("backend not supported".to_string());
    }
    if let Ok(freed) = OPT_FREED.lock() {
        if freed.contains(&handle) {
            return Err("Stale optimizer handle (freed)".to_string());
        }
    }
    let guard = OPT_STATES
        .lock()
        .map_err(|_| "optimizer registry poisoned".to_string())?;
    guard
        .get(&handle)
        .map(|entry| clone_opt_state(&entry.state))
        .ok_or_else(|| "Invalid optimizer handle".to_string())
}

#[cfg(feature = "torch")]
fn update_tensor(handle: i64, tensor: Tensor) {
    if let Ok(mut guard) = TENSORS.lock() {
        if let Some(entry) = guard.get_mut(&handle) {
            entry.tensor = tensor;
        } else {
            set_error("Invalid tensor handle");
        }
    } else {
        set_error("tensor registry poisoned");
    }
}

#[cfg(feature = "torch")]
fn update_opt(handle: i64, state: AdamWState) {
    if let Ok(mut guard) = OPT_STATES.lock() {
        if let Some(entry) = guard.get_mut(&handle) {
            entry.state = state;
        } else {
            set_error("Invalid optimizer handle");
        }
    } else {
        set_error("optimizer registry poisoned");
    }
}

#[cfg(feature = "torch")]
fn set_opt_params(handle: i64, params: Vec<i64>) {
    if let Ok(mut guard) = OPT_PARAMS.lock() {
        guard.insert(handle, params);
    } else {
        set_error("optimizer params registry poisoned");
    }
}

#[cfg(feature = "torch")]
fn get_opt_params(handle: i64) -> Result<Vec<i64>, String> {
    if let Ok(freed) = OPT_FREED.lock() {
        if freed.contains(&handle) {
            return Err("Stale optimizer handle (freed)".to_string());
        }
    }
    OPT_PARAMS
        .lock()
        .map_err(|_| "optimizer params registry poisoned".to_string())?
        .get(&handle)
        .cloned()
        .ok_or_else(|| "Invalid optimizer handle (params)".to_string())
}

#[cfg(feature = "torch")]
fn set_opt_hyper(handle: i64, hyper: AdamWHyper) {
    if let Ok(mut guard) = OPT_HYPER.lock() {
        guard.insert(handle, hyper);
    } else {
        set_error("optimizer hyper registry poisoned");
    }
}

#[cfg(feature = "torch")]
fn get_opt_hyper(handle: i64) -> Result<AdamWHyper, String> {
    if let Ok(freed) = OPT_FREED.lock() {
        if freed.contains(&handle) {
            return Err("Stale optimizer handle (freed)".to_string());
        }
    }
    OPT_HYPER
        .lock()
        .map_err(|_| "optimizer hyper registry poisoned".to_string())?
        .get(&handle)
        .copied()
        .ok_or_else(|| "Invalid optimizer handle (hyper)".to_string())
}

#[cfg(feature = "torch")]
fn get_scaler_mut(handle: i64) -> Result<GradScalerState, String> {
    if let Ok(freed) = SCALER_FREED.lock() {
        if freed.contains(&handle) {
            return Err("Stale scaler handle (freed)".to_string());
        }
    }
    SCALERS
        .lock()
        .map_err(|_| "scaler registry poisoned".to_string())?
        .get(&handle)
        .cloned()
        .ok_or_else(|| "Invalid scaler handle".to_string())
}

#[cfg(feature = "torch")]
fn update_scaler(handle: i64, state: GradScalerState) {
    if let Ok(mut guard) = SCALERS.lock() {
        guard.insert(handle, state);
    } else {
        set_error("scaler registry poisoned");
    }
}

#[cfg(feature = "torch")]
fn clone_opt_state(state: &AdamWState) -> AdamWState {
    let mut slots = HashMap::with_capacity(state.slots.len());
    for (key, slot) in &state.slots {
        slots.insert(
            *key,
            AdamWSlot {
                m: slot.m.shallow_clone(),
                v: slot.v.shallow_clone(),
            },
        );
    }
    AdamWState {
        slots,
        step: state.step,
    }
}

#[cfg(feature = "torch")]
fn clone_slot(slot: &AdamWSlot) -> AdamWSlot {
    AdamWSlot {
        m: slot.m.shallow_clone(),
        v: slot.v.shallow_clone(),
    }
}

#[cfg(feature = "torch")]
fn save_opt_state(dir: &str, state: &AdamWState, params: &[i64]) -> Result<(), String> {
    let opt_path = format!("{}/optim_rank0.bin", dir);
    let opt_meta_path = format!("{}/optim_meta.json", dir);
    let mut tensors = Vec::new();
    let mut slots_meta = Vec::new();
    for (idx, param_handle) in params.iter().enumerate() {
        let slot = state
            .slots
            .get(param_handle)
            .ok_or_else(|| format!("Optimizer state missing for param {}", param_handle))?;
        let m_name = format!("m_{}", idx);
        let v_name = format!("v_{}", idx);
        tensors.push((m_name.clone(), slot.m.shallow_clone()));
        tensors.push((v_name.clone(), slot.v.shallow_clone()));
        slots_meta.push(serde_json::json!({
            "index": idx,
            "m": m_name,
            "v": v_name
        }));
    }
    if !tensors.is_empty() {
        Tensor::save_multi(&tensors, opt_path).map_err(|err| err.to_string())?;
    }
    let opt_meta = serde_json::json!({
        "step": state.step,
        "param_count": params.len(),
        "slots": slots_meta
    });
    fs::write(opt_meta_path, opt_meta.to_string()).map_err(|err| err.to_string())?;
    Ok(())
}

#[cfg(feature = "torch")]
fn load_opt_state(dir: &str) -> Result<AdamWState, String> {
    let opt_meta_path = format!("{}/optim_meta.json", dir);
    let meta_text = fs::read_to_string(opt_meta_path).map_err(|err| err.to_string())?;
    let meta_val: serde_json::Value = serde_json::from_str(&meta_text)
        .map_err(|_| "Invalid optimizer metadata JSON".to_string())?;
    let step = meta_val.get("step").and_then(|v| v.as_i64()).unwrap_or(0);
    let slots_val = meta_val
        .get("slots")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "Optimizer metadata missing slots".to_string())?;
    if slots_val.is_empty() {
        return Ok(AdamWState {
            slots: HashMap::new(),
            step,
        });
    }
    let opt_path = format!("{}/optim_rank0.bin", dir);
    let list = Tensor::load_multi(opt_path).map_err(|err| err.to_string())?;
    let mut tensor_map = HashMap::with_capacity(list.len());
    for (name, tensor) in list {
        tensor_map.insert(name, tensor);
    }
    let mut slots = HashMap::new();
    for slot_val in slots_val {
        let index = slot_val
            .get("index")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| "Optimizer metadata missing index".to_string())?;
        let m_name = slot_val
            .get("m")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Optimizer metadata missing m".to_string())?;
        let v_name = slot_val
            .get("v")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Optimizer metadata missing v".to_string())?;
        let m_tensor = tensor_map
            .remove(m_name)
            .ok_or_else(|| "Optimizer tensor m missing".to_string())?;
        let v_tensor = tensor_map
            .remove(v_name)
            .ok_or_else(|| "Optimizer tensor v missing".to_string())?;
        slots.insert(
            index,
            AdamWSlot {
                m: m_tensor,
                v: v_tensor,
            },
        );
    }
    Ok(AdamWState { slots, step })
}

#[cfg(feature = "torch")]
fn bind_opt_state(state: AdamWState, params: &[i64]) -> Result<AdamWState, String> {
    if state.slots.is_empty() {
        return Ok(AdamWState {
            slots: HashMap::new(),
            step: state.step,
        });
    }
    if state.slots.len() != params.len() {
        return Err("Optimizer state size does not match params list".to_string());
    }
    let mut slots = HashMap::with_capacity(params.len());
    for (idx, param_handle) in params.iter().enumerate() {
        let slot = state
            .slots
            .get(&(idx as i64))
            .ok_or_else(|| "Optimizer slot missing for index".to_string())?;
        slots.insert(*param_handle, clone_slot(slot));
    }
    Ok(AdamWState {
        slots,
        step: state.step,
    })
}

#[cfg(feature = "torch")]
fn parse_device(spec: &str) -> Result<Device, String> {
    if spec == "cpu" {
        return Ok(Device::Cpu);
    }
    if let Some(rest) = spec.strip_prefix("cuda:") {
        let idx: i64 = rest
            .parse()
            .map_err(|_| "Invalid cuda device index".to_string())?;
        let idx: usize = idx
            .try_into()
            .map_err(|_| "CUDA device index out of range".to_string())?;
        return Ok(Device::Cuda(idx));
    }
    if spec == "cuda" {
        return Ok(Device::Cuda(0));
    }
    Err("Unknown device spec".to_string())
}

#[cfg(feature = "torch")]
fn parse_dtype(dtype: &str) -> Result<Kind, String> {
    match dtype {
        "fp16" | "f16" => Ok(Kind::Half),
        "bf16" => Ok(Kind::BFloat16),
        "fp32" | "f32" => Ok(Kind::Float),
        "int64" | "i64" => Ok(Kind::Int64),
        _ => Err("Unsupported dtype".to_string()),
    }
}

#[cfg(feature = "torch")]
fn parse_shape(json: &str) -> Result<Vec<i64>, String> {
    let value: serde_json::Value =
        serde_json::from_str(json).map_err(|_| "Invalid shape JSON".to_string())?;
    let arr = value
        .as_array()
        .ok_or_else(|| "Shape must be a JSON array".to_string())?;
    let mut out = Vec::with_capacity(arr.len());
    for v in arr {
        let n = v
            .as_i64()
            .ok_or_else(|| "Shape values must be integers".to_string())?;
        out.push(n);
    }
    Ok(out)
}

#[no_mangle]
/// # Safety
/// Caller must pass a pointer that was allocated by `enkai_tensor_*` APIs returning
/// a heap-allocated C string, and must not use the pointer after freeing.
pub unsafe extern "C" fn enkai_tensor_string_free(ptr: *mut c_char) {
    ffi_guard((), || {
        if ptr.is_null() {
            return;
        }
        let _ = CString::from_raw(ptr);
    })
}

#[no_mangle]
/// # Safety
/// The caller must pass a pointer and length originally allocated by `enkai_tensor`
/// and must not free the buffer more than once.
pub unsafe extern "C" fn enkai_free(ptr: *mut u8, len: usize) {
    ffi_guard((), || {
        if ptr.is_null() {
            return;
        }
        let _ = Vec::from_raw_parts(ptr, len, len);
    })
}

#[cfg(feature = "torch")]
fn parse_handle_list(json: &str) -> Result<Vec<i64>, String> {
    let value: serde_json::Value =
        serde_json::from_str(json).map_err(|_| "Invalid handle list JSON".to_string())?;
    let arr = value
        .as_array()
        .ok_or_else(|| "Handle list must be a JSON array".to_string())?;
    let mut out = Vec::with_capacity(arr.len());
    for v in arr {
        let n = v
            .as_i64()
            .ok_or_else(|| "Handle list values must be integers".to_string())?;
        out.push(n);
    }
    Ok(out)
}

#[no_mangle]
pub extern "C" fn enkai_tensor_device(spec: *const c_char) -> i64 {
    ffi_guard(0, || {
        clear_error();
        let spec = match cstr_to_string(spec) {
            Ok(v) => v,
            Err(err) => {
                set_error(err);
                return 0;
            }
        };
        #[cfg(feature = "torch")]
        {
            if !require_tch_backend() {
                return 0;
            }
            let device = match parse_device(&spec) {
                Ok(d) => d,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            return register_device(device);
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = spec;
            set_error("torch backend not enabled");
            0
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_free(handle: i64) -> c_int {
    ffi_guard(1, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            guard_backend!(1);
            if let Ok(freed) = TENSOR_FREED.lock() {
                if freed.contains(&handle) {
                    set_error("tensor handle already freed");
                    return 1;
                }
            }
            match TENSORS.lock() {
                Ok(mut guard) => match guard.get_mut(&handle) {
                    Some(entry) => {
                        if entry.refcount == 0 {
                            set_error("tensor handle already freed");
                            return 1;
                        }
                        entry.refcount -= 1;
                        if entry.refcount == 0 {
                            guard.remove(&handle);
                            if let Ok(mut freed) = TENSOR_FREED.lock() {
                                freed.insert(handle);
                            }
                        }
                        return 0;
                    }
                    None => {
                        set_error("Invalid tensor handle");
                        return 1;
                    }
                },
                Err(_) => {
                    set_error("tensor registry poisoned");
                    return 1;
                }
            }
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = handle;
            set_error("torch backend not enabled");
            1
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_retain(handle: i64) -> c_int {
    ffi_guard(1, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            if let Ok(freed) = TENSOR_FREED.lock() {
                if freed.contains(&handle) {
                    set_error("tensor handle already freed");
                    return 1;
                }
            }
            match TENSORS.lock() {
                Ok(mut guard) => match guard.get_mut(&handle) {
                    Some(entry) => {
                        entry.refcount = entry.refcount.saturating_add(1);
                        return 0;
                    }
                    None => {
                        set_error("Invalid tensor handle");
                        return 1;
                    }
                },
                Err(_) => {
                    set_error("tensor registry poisoned");
                    return 1;
                }
            }
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = handle;
            set_error("torch backend not enabled");
            1
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_item(handle: i64) -> f64 {
    ffi_guard(0.0, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            let tensor = match get_tensor(handle) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return 0.0;
                }
            };
            match tensor.double_value(&[]) {
                val => val,
            }
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = handle;
            set_error("torch backend not enabled");
            0.0
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_require_grad(handle: i64) -> i64 {
    ffi_guard(0, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            if let Ok(freed) = TENSOR_FREED.lock() {
                if freed.contains(&handle) {
                    set_error("Stale tensor handle (freed)");
                    return 0;
                }
            }
            let mut guard = match TENSORS.lock() {
                Ok(g) => g,
                Err(_) => {
                    set_error("tensor registry poisoned");
                    return 0;
                }
            };
            let entry = match guard.get_mut(&handle) {
                Some(e) => e,
                None => {
                    set_error("Invalid tensor handle");
                    return 0;
                }
            };
            if !entry.tensor.requires_grad() {
                entry.tensor = entry.tensor.set_requires_grad(true);
            }
            handle
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = handle;
            set_error("torch backend not enabled");
            0
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_grad(handle: i64) -> i64 {
    ffi_guard(0, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            let tensor = match get_tensor(handle) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            let grad = tensor.grad();
            if grad.defined() {
                return register_tensor(grad);
            }
            0
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = handle;
            set_error("torch backend not enabled");
            0
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_zero_grad(handle: i64) -> c_int {
    ffi_guard(1, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            let tensor = match get_tensor(handle) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return 1;
                }
            };
            let mut grad = tensor.grad();
            if grad.defined() {
                let _ = grad.zero_();
            }
            return 0;
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = handle;
            set_error("torch backend not enabled");
            1
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_zero_grad_multi(handles_json: *const c_char) -> c_int {
    ffi_guard(1, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            let handles_json = match cstr_to_string(handles_json) {
                Ok(v) => v,
                Err(err) => {
                    set_error(err);
                    return 1;
                }
            };
            let handles = match parse_handle_list(&handles_json) {
                Ok(list) => list,
                Err(err) => {
                    set_error(err);
                    return 1;
                }
            };
            for h in handles {
                let tensor = match get_tensor(h) {
                    Ok(t) => t,
                    Err(err) => {
                        set_error(err);
                        return 1;
                    }
                };
                let mut grad = tensor.grad();
                if grad.defined() {
                    let _ = grad.zero_();
                }
            }
            return 0;
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = handles_json;
            set_error("torch backend not enabled");
            1
        }
    })
}

#[no_mangle]
/// Returns a JSON array of grad tensor handles for the provided tensor handles.
pub extern "C" fn enkai_tensor_grad_multi(
    handles_json: *const c_char,
    out_json: *mut *mut c_char,
    out_len: *mut usize,
) -> c_int {
    ffi_guard(1, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            let handles_json = match cstr_to_string(handles_json) {
                Ok(v) => v,
                Err(err) => {
                    set_error(err);
                    return 1;
                }
            };
            let handles = match parse_handle_list(&handles_json) {
                Ok(list) => list,
                Err(err) => {
                    set_error(err);
                    return 1;
                }
            };
            let mut grads: Vec<i64> = Vec::with_capacity(handles.len());
            for h in handles {
                let tensor = match get_tensor(h) {
                    Ok(t) => t,
                    Err(err) => {
                        set_error(err);
                        return 1;
                    }
                };
                let grad = tensor.grad();
                if grad.defined() {
                    grads.push(register_tensor(grad));
                } else {
                    grads.push(0);
                }
            }
            let json = match serde_json::to_string(&grads) {
                Ok(s) => s,
                Err(err) => {
                    set_error(err.to_string());
                    return 1;
                }
            };
            unsafe {
                let bytes = json.into_bytes();
                *out_len = bytes.len();
                let cstr = match CString::new(bytes) {
                    Ok(s) => s,
                    Err(_) => {
                        set_error("Grad list contained interior null byte");
                        return 1;
                    }
                };
                let ptr = cstr.into_raw();
                *out_json = ptr;
            }
            return 0;
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = handles_json;
            let _ = out_json;
            let _ = out_len;
            set_error("torch backend not enabled");
            1
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_device_free(handle: i64) -> c_int {
    ffi_guard(1, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            guard_backend!(1);
            if let Ok(freed) = DEVICE_FREED.lock() {
                if freed.contains(&handle) {
                    set_error("device handle already freed");
                    return 1;
                }
            }
            match DEVICES.lock() {
                Ok(mut guard) => match guard.get_mut(&handle) {
                    Some(entry) => {
                        if entry.refcount == 0 {
                            set_error("device handle already freed");
                            return 1;
                        }
                        entry.refcount -= 1;
                        if entry.refcount == 0 {
                            guard.remove(&handle);
                            if let Ok(mut freed) = DEVICE_FREED.lock() {
                                freed.insert(handle);
                            }
                        }
                        return 0;
                    }
                    None => {
                        set_error("Invalid device handle");
                        return 1;
                    }
                },
                Err(_) => {
                    set_error("device registry poisoned");
                    return 1;
                }
            }
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = handle;
            set_error("torch backend not enabled");
            1
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_randn(
    shape_json: *const c_char,
    dtype: *const c_char,
    device_handle: i64,
) -> i64 {
    ffi_guard(0, || {
        clear_error();
        let shape_json = match cstr_to_string(shape_json) {
            Ok(v) => v,
            Err(err) => {
                set_error(err);
                return 0;
            }
        };
        let dtype = match cstr_to_string(dtype) {
            Ok(v) => v,
            Err(err) => {
                set_error(err);
                return 0;
            }
        };
        #[cfg(feature = "torch")]
        {
            guard_backend!(0);
            let shape = match parse_shape(&shape_json) {
                Ok(v) => v,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            let device = match get_device(device_handle) {
                Ok(d) => d,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            let kind = match parse_dtype(&dtype) {
                Ok(k) => k,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            return register_tensor(Tensor::randn(shape.as_slice(), (kind, device)));
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = shape_json;
            let _ = dtype;
            let _ = device_handle;
            set_error("torch backend not enabled");
            0
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_zeros(
    shape_json: *const c_char,
    dtype: *const c_char,
    device_handle: i64,
) -> i64 {
    ffi_guard(0, || {
        clear_error();
        let shape_json = match cstr_to_string(shape_json) {
            Ok(v) => v,
            Err(err) => {
                set_error(err);
                return 0;
            }
        };
        let dtype = match cstr_to_string(dtype) {
            Ok(v) => v,
            Err(err) => {
                set_error(err);
                return 0;
            }
        };
        #[cfg(feature = "torch")]
        {
            guard_backend!(0);
            let shape = match parse_shape(&shape_json) {
                Ok(v) => v,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            let device = match get_device(device_handle) {
                Ok(d) => d,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            let kind = match parse_dtype(&dtype) {
                Ok(k) => k,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            return register_tensor(Tensor::zeros(shape.as_slice(), (kind, device)));
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = shape_json;
            let _ = dtype;
            let _ = device_handle;
            set_error("torch backend not enabled");
            0
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_matmul(a: i64, b: i64) -> i64 {
    ffi_guard(0, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            let a = match get_tensor(a) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            let b = match get_tensor(b) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            return register_tensor(a.matmul(&b));
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = a;
            let _ = b;
            set_error("torch backend not enabled");
            0
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_add(a: i64, b: i64) -> i64 {
    ffi_guard(0, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            let a = match get_tensor(a) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            let b = match get_tensor(b) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            return register_tensor(a + b);
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = a;
            let _ = b;
            set_error("torch backend not enabled");
            0
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_mul(a: i64, b: i64) -> i64 {
    ffi_guard(0, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            let a = match get_tensor(a) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            let b = match get_tensor(b) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            return register_tensor(a * b);
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = a;
            let _ = b;
            set_error("torch backend not enabled");
            0
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_reshape(x: i64, shape_json: *const c_char) -> i64 {
    ffi_guard(0, || {
        clear_error();
        let shape_json = match cstr_to_string(shape_json) {
            Ok(v) => v,
            Err(err) => {
                set_error(err);
                return 0;
            }
        };
        #[cfg(feature = "torch")]
        {
            let x = match get_tensor(x) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            let shape = match parse_shape(&shape_json) {
                Ok(s) => s,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            return register_tensor(x.reshape(shape.as_slice()));
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = x;
            let _ = shape_json;
            set_error("torch backend not enabled");
            0
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_transpose(x: i64, dim0: i64, dim1: i64) -> i64 {
    ffi_guard(0, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            let x = match get_tensor(x) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            return register_tensor(x.transpose(dim0, dim1));
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = x;
            let _ = dim0;
            let _ = dim1;
            set_error("torch backend not enabled");
            0
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_concat(handles_json: *const c_char, dim: i64) -> i64 {
    ffi_guard(0, || {
        clear_error();
        let handles_json = match cstr_to_string(handles_json) {
            Ok(v) => v,
            Err(err) => {
                set_error(err);
                return 0;
            }
        };
        #[cfg(feature = "torch")]
        {
            let handles = match parse_handle_list(&handles_json) {
                Ok(list) => list,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            if handles.is_empty() {
                set_error("Concat requires at least one tensor");
                return 0;
            }
            let mut tensors = Vec::with_capacity(handles.len());
            for handle in handles {
                match get_tensor(handle) {
                    Ok(t) => tensors.push(t),
                    Err(err) => {
                        set_error(err);
                        return 0;
                    }
                }
            }
            return register_tensor(Tensor::cat(tensors.as_slice(), dim));
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = handles_json;
            let _ = dim;
            set_error("torch backend not enabled");
            0
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_sum(x: i64, dim: i64, keepdim: c_int) -> i64 {
    ffi_guard(0, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            let x = match get_tensor(x) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            let keep = keepdim != 0;
            let out = if dim < 0 {
                x.sum(Kind::Float)
            } else {
                x.sum_dim_intlist(Some([dim].as_slice()), keep, Kind::Float)
            };
            return register_tensor(out);
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = x;
            let _ = dim;
            let _ = keepdim;
            set_error("torch backend not enabled");
            0
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_mean(x: i64, dim: i64, keepdim: c_int) -> i64 {
    ffi_guard(0, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            let x = match get_tensor(x) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            let keep = keepdim != 0;
            let out = if dim < 0 {
                x.mean(Kind::Float)
            } else {
                x.mean_dim(Some([dim].as_slice()), keep, Kind::Float)
            };
            return register_tensor(out);
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = x;
            let _ = dim;
            let _ = keepdim;
            set_error("torch backend not enabled");
            0
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_softmax(x: i64, dim: i64) -> i64 {
    ffi_guard(0, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            let x = match get_tensor(x) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            return register_tensor(x.softmax(dim, Kind::Float));
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = x;
            let _ = dim;
            set_error("torch backend not enabled");
            0
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_masked_softmax(
    x: i64,
    mask: i64,
    dim: i64,
    mask_type: c_int,
) -> i64 {
    ffi_guard(0, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            let x = match get_tensor(x) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            let mask = match get_tensor(mask) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            return register_tensor(x.internal_masked_softmax(&mask, dim, mask_type as i64));
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = x;
            let _ = mask;
            let _ = dim;
            let _ = mask_type;
            set_error("torch backend not enabled");
            0
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_masked_softmax_backward(
    grad_output: i64,
    output: i64,
    mask: i64,
    dim: i64,
) -> i64 {
    ffi_guard(0, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            let grad_output = match get_tensor(grad_output) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            let output = match get_tensor(output) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            let mask = match get_tensor(mask) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            return register_tensor(Tensor::internal_masked_softmax_backward(
                &grad_output,
                &output,
                &mask,
                dim,
            ));
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = grad_output;
            let _ = output;
            let _ = mask;
            let _ = dim;
            set_error("torch backend not enabled");
            0
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_relu(x: i64) -> i64 {
    ffi_guard(0, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            let x = match get_tensor(x) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            return register_tensor(x.relu());
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = x;
            set_error("torch backend not enabled");
            0
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_sigmoid(x: i64) -> i64 {
    ffi_guard(0, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            let x = match get_tensor(x) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            return register_tensor(x.sigmoid());
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = x;
            set_error("torch backend not enabled");
            0
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_dropout(x: i64, p: f64, train: c_int) -> i64 {
    ffi_guard(0, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            let x = match get_tensor(x) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            let train = train != 0;
            return register_tensor(x.dropout(p, train));
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = x;
            let _ = p;
            let _ = train;
            set_error("torch backend not enabled");
            0
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_slice(x: i64, dim: i64, start: i64, end: i64, step: i64) -> i64 {
    ffi_guard(0, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            let x = match get_tensor(x) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            let start = if start < 0 { None } else { Some(start) };
            let end = if end < 0 { None } else { Some(end) };
            return register_tensor(x.slice(dim, start, end, step));
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = x;
            let _ = dim;
            let _ = start;
            let _ = end;
            let _ = step;
            set_error("torch backend not enabled");
            0
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_view(x: i64, shape_json: *const c_char) -> i64 {
    ffi_guard(0, || {
        clear_error();
        let shape_json = match cstr_to_string(shape_json) {
            Ok(v) => v,
            Err(err) => {
                set_error(err);
                return 0;
            }
        };
        #[cfg(feature = "torch")]
        {
            let x = match get_tensor(x) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            let shape = match parse_shape(&shape_json) {
                Ok(s) => s,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            return register_tensor(x.view(shape.as_slice()));
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = x;
            let _ = shape_json;
            set_error("torch backend not enabled");
            0
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_gelu(x: i64) -> i64 {
    ffi_guard(0, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            let x = match get_tensor(x) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            return register_tensor(x.gelu("none"));
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = x;
            set_error("torch backend not enabled");
            0
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_layernorm(x: i64, w: i64, b: i64, eps: f64) -> i64 {
    ffi_guard(0, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            let x = match get_tensor(x) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            let w = match get_tensor(w) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            let b = match get_tensor(b) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            let shape = w.size();
            return register_tensor(x.layer_norm(&shape, Some(&w), Some(&b), eps, true));
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = x;
            let _ = w;
            let _ = b;
            let _ = eps;
            set_error("torch backend not enabled");
            0
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_layernorm_backward(
    x: i64,
    w: i64,
    b: i64,
    eps: f64,
    grad_out: i64,
) -> FfiSlice {
    ffi_guard(null_slice(), || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            let x = match get_tensor(x) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return null_slice();
                }
            };
            let w = match get_tensor(w) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return null_slice();
                }
            };
            let b = match get_tensor(b) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return null_slice();
                }
            };
            let grad_out = match get_tensor(grad_out) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return null_slice();
                }
            };
            let x_req = x.set_requires_grad(true);
            let w_req = w.set_requires_grad(true);
            let b_req = b.set_requires_grad(true);
            let shape = w_req.size();
            let out = match x_req.f_layer_norm(&shape, Some(&w_req), Some(&b_req), eps, true) {
                Ok(out) => out,
                Err(err) => {
                    set_error(err.to_string());
                    return null_slice();
                }
            };
            let loss = (&out * grad_out).sum(Kind::Float);
            if let Err(err) = loss.f_backward() {
                set_error(err.to_string());
                return null_slice();
            }
            let dx = x_req.grad();
            let dw = w_req.grad();
            let db = b_req.grad();
            let handles = vec![
                register_tensor(dx),
                register_tensor(dw),
                register_tensor(db),
            ];
            match serde_json::to_string(&handles) {
                Ok(text) => make_slice(text.into_bytes()),
                Err(_) => {
                    set_error("Failed to serialize layernorm backward handles");
                    null_slice()
                }
            }
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = x;
            let _ = w;
            let _ = b;
            let _ = eps;
            let _ = grad_out;
            set_error("torch backend not enabled");
            null_slice()
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_embedding(w: i64, ids: i64) -> i64 {
    ffi_guard(0, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            let w = match get_tensor(w) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            let ids = match get_tensor(ids) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            return register_tensor(Tensor::embedding(&w, &ids, -1, false, false));
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = w;
            let _ = ids;
            set_error("torch backend not enabled");
            0
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_linear(x: i64, w: i64, b: i64) -> i64 {
    ffi_guard(0, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            let x = match get_tensor(x) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            let w = match get_tensor(w) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            let b = match get_tensor(b) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            return register_tensor(x.matmul(&w) + b);
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = x;
            let _ = w;
            let _ = b;
            set_error("torch backend not enabled");
            0
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_to_device(x: i64, device: i64) -> i64 {
    ffi_guard(0, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            let x = match get_tensor(x) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            let device = match get_device(device) {
                Ok(d) => d,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            return register_tensor(x.to_device(device));
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = x;
            let _ = device;
            set_error("torch backend not enabled");
            0
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_to_dtype(x: i64, dtype: *const c_char) -> i64 {
    ffi_guard(0, || {
        clear_error();
        let dtype = match cstr_to_string(dtype) {
            Ok(v) => v,
            Err(err) => {
                set_error(err);
                return 0;
            }
        };
        #[cfg(feature = "torch")]
        {
            let x = match get_tensor(x) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            let kind = match parse_dtype(&dtype) {
                Ok(k) => k,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            return register_tensor(x.to_kind(kind));
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = x;
            let _ = dtype;
            set_error("torch backend not enabled");
            0
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_shape(x: i64) -> FfiSlice {
    ffi_guard(null_slice(), || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            let x = match get_tensor(x) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return null_slice();
                }
            };
            let shape = x.size();
            return match serde_json::to_string(&shape) {
                Ok(text) => make_slice(text.into_bytes()),
                Err(_) => {
                    set_error("Failed to serialize shape");
                    null_slice()
                }
            };
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = x;
            set_error("torch backend not enabled");
            null_slice()
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_cross_entropy(logits: i64, targets: i64) -> i64 {
    ffi_guard(0, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            let logits = match get_tensor(logits) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            let targets = match get_tensor(targets) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            return register_tensor(logits.cross_entropy_for_logits(&targets));
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = logits;
            let _ = targets;
            set_error("torch backend not enabled");
            0
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_backward(loss: i64) -> c_int {
    ffi_guard(1, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            let loss = match get_tensor(loss) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return 1;
                }
            };
            loss.backward();
            return 0;
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = loss;
            set_error("torch backend not enabled");
            1
        }
    })
}

/// Check that all tensors in the handle list are finite (no NaN/Inf).
#[no_mangle]
pub extern "C" fn enkai_tensor_check_finite_multi(handles_json: *const c_char) -> c_int {
    ffi_guard(1, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            let handles_json = match cstr_to_string(handles_json) {
                Ok(v) => v,
                Err(err) => {
                    set_error(err);
                    return 1;
                }
            };
            let handles = match parse_handle_list(&handles_json) {
                Ok(list) => list,
                Err(err) => {
                    set_error(err);
                    return 1;
                }
            };
            for h in handles {
                if h == 0 {
                    continue;
                }
                let t = match get_tensor(h) {
                    Ok(t) => t,
                    Err(err) => {
                        set_error(err);
                        return 1;
                    }
                };
                let ok = t.isfinite().all().int64_value(&[]) != 0;
                if !ok {
                    set_error(format!("non-finite tensor handle {}", h));
                    return 1;
                }
            }
            0
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = handles_json;
            set_error("torch backend not enabled");
            1
        }
    })
}

/// Initialize a tiny transformer language model and return parameter handles as JSON.
///
/// Layout (handles list):
/// 0: tok_embed [vocab, d_model]
/// 1: pos_embed [seq_len, d_model]
/// Per-layer (12 params):
///   ln1_w, ln1_b, qkv_w, qkv_b, proj_w, proj_b, ln2_w, ln2_b, mlp_w1, mlp_b1, mlp_w2, mlp_b2
/// Final:
///   ln_f_w, ln_f_b, head_w, head_b
#[no_mangle]
/// # Safety
/// `out_json` and `out_len` must be valid writable pointers for this call.
pub unsafe extern "C" fn enkai_tensor_tinylm_init(
    vocab_size: i64,
    seq_len: i64,
    d_model: i64,
    n_layers: i64,
    n_heads: i64,
    device_handle: i64,
    seed: i64,
    out_json: *mut *mut c_char,
    out_len: *mut usize,
) -> c_int {
    ffi_guard(1, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            if vocab_size <= 0 || seq_len <= 0 || d_model <= 0 || n_layers <= 0 || n_heads <= 0 {
                set_error("tinylm_init: invalid model dimensions");
                return 1;
            }
            if d_model % n_heads != 0 {
                set_error("tinylm_init: d_model must be divisible by n_heads");
                return 1;
            }
            let device = match get_device(device_handle) {
                Ok(d) => d,
                Err(err) => {
                    set_error(err);
                    return 1;
                }
            };
            tch::manual_seed(seed);
            let init_scale = 0.02;
            let mut handles: Vec<i64> = Vec::new();

            let tok_embed =
                Tensor::randn([vocab_size, d_model], (Kind::Float, device)) * init_scale;
            handles.push(register_tensor(tok_embed.set_requires_grad(true)));
            let pos_embed = Tensor::randn([seq_len, d_model], (Kind::Float, device)) * init_scale;
            handles.push(register_tensor(pos_embed.set_requires_grad(true)));

            for _ in 0..n_layers {
                let ln1_w = Tensor::ones([d_model], (Kind::Float, device));
                let ln1_b = Tensor::zeros([d_model], (Kind::Float, device));
                handles.push(register_tensor(ln1_w.set_requires_grad(true)));
                handles.push(register_tensor(ln1_b.set_requires_grad(true)));

                let qkv_w =
                    Tensor::randn([d_model, 3 * d_model], (Kind::Float, device)) * init_scale;
                let qkv_b = Tensor::zeros([3 * d_model], (Kind::Float, device));
                handles.push(register_tensor(qkv_w.set_requires_grad(true)));
                handles.push(register_tensor(qkv_b.set_requires_grad(true)));

                let proj_w = Tensor::randn([d_model, d_model], (Kind::Float, device)) * init_scale;
                let proj_b = Tensor::zeros([d_model], (Kind::Float, device));
                handles.push(register_tensor(proj_w.set_requires_grad(true)));
                handles.push(register_tensor(proj_b.set_requires_grad(true)));

                let ln2_w = Tensor::ones([d_model], (Kind::Float, device));
                let ln2_b = Tensor::zeros([d_model], (Kind::Float, device));
                handles.push(register_tensor(ln2_w.set_requires_grad(true)));
                handles.push(register_tensor(ln2_b.set_requires_grad(true)));

                let mlp_w1 =
                    Tensor::randn([d_model, 4 * d_model], (Kind::Float, device)) * init_scale;
                let mlp_b1 = Tensor::zeros([4 * d_model], (Kind::Float, device));
                let mlp_w2 =
                    Tensor::randn([4 * d_model, d_model], (Kind::Float, device)) * init_scale;
                let mlp_b2 = Tensor::zeros([d_model], (Kind::Float, device));
                handles.push(register_tensor(mlp_w1.set_requires_grad(true)));
                handles.push(register_tensor(mlp_b1.set_requires_grad(true)));
                handles.push(register_tensor(mlp_w2.set_requires_grad(true)));
                handles.push(register_tensor(mlp_b2.set_requires_grad(true)));
            }

            let ln_f_w = Tensor::ones([d_model], (Kind::Float, device));
            let ln_f_b = Tensor::zeros([d_model], (Kind::Float, device));
            handles.push(register_tensor(ln_f_w.set_requires_grad(true)));
            handles.push(register_tensor(ln_f_b.set_requires_grad(true)));

            let head_w = Tensor::randn([d_model, vocab_size], (Kind::Float, device)) * init_scale;
            let head_b = Tensor::zeros([vocab_size], (Kind::Float, device));
            handles.push(register_tensor(head_w.set_requires_grad(true)));
            handles.push(register_tensor(head_b.set_requires_grad(true)));

            let json = match serde_json::to_string(&handles) {
                Ok(s) => s,
                Err(err) => {
                    set_error(err.to_string());
                    return 1;
                }
            };
            *out_len = json.len();
            match CString::new(json) {
                Ok(c) => {
                    *out_json = c.into_raw();
                    0
                }
                Err(_) => {
                    set_error("tinylm_init: json contained null byte");
                    1
                }
            }
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = vocab_size;
            let _ = seq_len;
            let _ = d_model;
            let _ = n_layers;
            let _ = n_heads;
            let _ = device_handle;
            let _ = seed;
            let _ = out_json;
            let _ = out_len;
            set_error("torch backend not enabled");
            1
        }
    })
}

/// Forward pass for TinyLM with causal self-attention and cross-entropy loss.
#[no_mangle]
/// # Safety
/// `input_ptr`/`target_ptr` must be valid for `input_len`/`target_len` elements.
pub unsafe extern "C" fn enkai_tensor_forward_tinylm(
    params_json: *const c_char,
    input_ptr: *const u32,
    input_len: usize,
    target_ptr: *const u32,
    target_len: usize,
    batch_size: i64,
    seq_len: i64,
    n_heads: i64,
) -> i64 {
    ffi_guard(0, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            let params_json = match cstr_to_string(params_json) {
                Ok(v) => v,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            let handles = match parse_handle_list(&params_json) {
                Ok(list) => list,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            if handles.len() < 6 {
                set_error("tinylm_forward: params list too short");
                return 0;
            }
            if batch_size <= 0 || seq_len <= 0 || n_heads <= 0 {
                set_error("tinylm_forward: invalid batch dimensions");
                return 0;
            }
            let expected = (batch_size * seq_len) as usize;
            if input_len != expected || target_len != expected {
                set_error("tinylm_forward: input/target length mismatch");
                return 0;
            }

            let tok_embed = match get_tensor(handles[0]) {
                Ok(t) => ensure_requires_grad(t),
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            let pos_embed = match get_tensor(handles[1]) {
                Ok(t) => ensure_requires_grad(t),
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            let d_model = tok_embed.size()[1];
            if d_model % n_heads != 0 {
                set_error("tinylm_forward: d_model not divisible by n_heads");
                return 0;
            }
            let head_dim = d_model / n_heads;
            let per_layer = 12usize;
            let tail = handles.len() - 6;
            if tail % per_layer != 0 {
                set_error("tinylm_forward: params list not aligned to layers");
                return 0;
            }
            let n_layers = tail / per_layer;
            let ln_f_w = match get_tensor(handles[handles.len() - 4]) {
                Ok(t) => ensure_requires_grad(t),
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            let ln_f_b = match get_tensor(handles[handles.len() - 3]) {
                Ok(t) => ensure_requires_grad(t),
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            let head_w = match get_tensor(handles[handles.len() - 2]) {
                Ok(t) => ensure_requires_grad(t),
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            let head_b = match get_tensor(handles[handles.len() - 1]) {
                Ok(t) => ensure_requires_grad(t),
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };

            let device = tok_embed.device();
            let input_slice = std::slice::from_raw_parts(input_ptr, input_len);
            let target_slice = std::slice::from_raw_parts(target_ptr, target_len);
            let input_i64: Vec<i64> = input_slice.iter().map(|v| *v as i64).collect();
            let target_i64: Vec<i64> = target_slice.iter().map(|v| *v as i64).collect();
            let input = match Tensor::f_from_slice(&input_i64) {
                Ok(t) => t.to_device(device).view([batch_size, seq_len]),
                Err(err) => {
                    set_error(err.to_string());
                    return 0;
                }
            };
            let targets = match Tensor::f_from_slice(&target_i64) {
                Ok(t) => t.to_device(device).view([batch_size, seq_len]),
                Err(err) => {
                    set_error(err.to_string());
                    return 0;
                }
            };

            let tok = Tensor::embedding(&tok_embed, &input, -1, false, false);
            let pos = pos_embed
                .unsqueeze(0)
                .expand(&[batch_size, seq_len, d_model], true);
            let mut x = tok + pos;

            let scale = 1.0 / (head_dim as f64).sqrt();
            let causal_mask = Tensor::ones([seq_len, seq_len], (Kind::Bool, device))
                .tril(0)
                .unsqueeze(0)
                .unsqueeze(0);

            for layer_idx in 0..n_layers {
                let base = 2 + layer_idx * per_layer;
                let ln1_w = match get_tensor(handles[base]) {
                    Ok(t) => ensure_requires_grad(t),
                    Err(err) => {
                        set_error(err);
                        return 0;
                    }
                };
                let ln1_b = match get_tensor(handles[base + 1]) {
                    Ok(t) => ensure_requires_grad(t),
                    Err(err) => {
                        set_error(err);
                        return 0;
                    }
                };
                let qkv_w = match get_tensor(handles[base + 2]) {
                    Ok(t) => ensure_requires_grad(t),
                    Err(err) => {
                        set_error(err);
                        return 0;
                    }
                };
                let qkv_b = match get_tensor(handles[base + 3]) {
                    Ok(t) => ensure_requires_grad(t),
                    Err(err) => {
                        set_error(err);
                        return 0;
                    }
                };
                let proj_w = match get_tensor(handles[base + 4]) {
                    Ok(t) => ensure_requires_grad(t),
                    Err(err) => {
                        set_error(err);
                        return 0;
                    }
                };
                let proj_b = match get_tensor(handles[base + 5]) {
                    Ok(t) => ensure_requires_grad(t),
                    Err(err) => {
                        set_error(err);
                        return 0;
                    }
                };
                let ln2_w = match get_tensor(handles[base + 6]) {
                    Ok(t) => ensure_requires_grad(t),
                    Err(err) => {
                        set_error(err);
                        return 0;
                    }
                };
                let ln2_b = match get_tensor(handles[base + 7]) {
                    Ok(t) => ensure_requires_grad(t),
                    Err(err) => {
                        set_error(err);
                        return 0;
                    }
                };
                let mlp_w1 = match get_tensor(handles[base + 8]) {
                    Ok(t) => ensure_requires_grad(t),
                    Err(err) => {
                        set_error(err);
                        return 0;
                    }
                };
                let mlp_b1 = match get_tensor(handles[base + 9]) {
                    Ok(t) => ensure_requires_grad(t),
                    Err(err) => {
                        set_error(err);
                        return 0;
                    }
                };
                let mlp_w2 = match get_tensor(handles[base + 10]) {
                    Ok(t) => ensure_requires_grad(t),
                    Err(err) => {
                        set_error(err);
                        return 0;
                    }
                };
                let mlp_b2 = match get_tensor(handles[base + 11]) {
                    Ok(t) => ensure_requires_grad(t),
                    Err(err) => {
                        set_error(err);
                        return 0;
                    }
                };

                let x_norm = x.layer_norm(&[d_model], Some(&ln1_w), Some(&ln1_b), 1e-5, true);
                let qkv = x_norm.matmul(&qkv_w) + qkv_b;
                let qkv = qkv
                    .view([batch_size, seq_len, 3, n_heads, head_dim])
                    .permute([2, 0, 3, 1, 4]);
                let q = qkv.select(0, 0);
                let k = qkv.select(0, 1);
                let v = qkv.select(0, 2);
                let att = (q.matmul(&k.transpose(-2, -1)) * scale)
                    .masked_fill(&causal_mask.logical_not(), f64::NEG_INFINITY)
                    .softmax(-1, Kind::Float);
                let ctx = att.matmul(&v);
                let ctx = ctx
                    .transpose(1, 2)
                    .contiguous()
                    .view([batch_size, seq_len, d_model]);
                let attn_out = ctx.matmul(&proj_w) + proj_b;
                x = x + attn_out;

                let x_norm2 = x.layer_norm(&[d_model], Some(&ln2_w), Some(&ln2_b), 1e-5, true);
                let mut mlp = x_norm2.matmul(&mlp_w1) + mlp_b1;
                mlp = mlp.gelu("none");
                let mlp = mlp.matmul(&mlp_w2) + mlp_b2;
                x = x + mlp;
            }

            let x = x.layer_norm(&[d_model], Some(&ln_f_w), Some(&ln_f_b), 1e-5, true);
            let logits = x.matmul(&head_w) + head_b;
            let logits_finite = logits.isfinite().all().int64_value(&[]) != 0;
            if !logits_finite {
                set_error("tinylm_forward: logits contain NaN/Inf");
                return 0;
            }
            let logits = logits.view([batch_size * seq_len, -1]);
            let targets = targets.view([batch_size * seq_len]);
            let loss = logits.cross_entropy_for_logits(&targets);
            let loss_finite = loss.isfinite().all().int64_value(&[]) != 0;
            if !loss_finite {
                set_error("tinylm_forward: loss is NaN/Inf");
                return 0;
            }
            return register_tensor(loss);
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = params_json;
            let _ = input_ptr;
            let _ = input_len;
            let _ = target_ptr;
            let _ = target_len;
            let _ = batch_size;
            let _ = seq_len;
            let _ = n_heads;
            set_error("torch backend not enabled");
            0
        }
    })
}

/// Simple L2 loss over parameter tensors: mean(sum_i param_i^2).
#[no_mangle]
pub extern "C" fn enkai_tensor_forward_l2(params_json: *const c_char) -> i64 {
    ffi_guard(0, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            let params_json = match cstr_to_string(params_json) {
                Ok(v) => v,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            let handles = match parse_handle_list(&params_json) {
                Ok(list) => list,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            if handles.is_empty() {
                set_error("forward_l2: param list is empty");
                return 0;
            }
            let mut losses = Vec::with_capacity(handles.len());
            for h in handles {
                let t = match get_tensor(h) {
                    Ok(t) => t,
                    Err(err) => {
                        set_error(err);
                        return 0;
                    }
                };
                // Ensure grads are tracked.
                let t = if t.requires_grad() {
                    t
                } else {
                    t.set_requires_grad(true)
                };
                let l = t.pow_tensor_scalar(2.0).mean(Kind::Float);
                losses.push(l);
            }
            let stacked = Tensor::stack(&losses, 0);
            let loss = stacked.mean(Kind::Float);
            return register_tensor(loss);
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = params_json;
            set_error("torch backend not enabled");
            0
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_adamw_step(
    param: i64,
    grad: i64,
    state: i64,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
) -> i64 {
    ffi_guard(0, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            let param_tensor = match get_tensor(param) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            let grad_tensor = match get_tensor(grad) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            let mut state_obj = if state == 0 {
                AdamWState {
                    slots: HashMap::new(),
                    step: 0,
                }
            } else {
                match get_opt_mut(state) {
                    Ok(s) => s,
                    Err(err) => {
                        set_error(err);
                        return 0;
                    }
                }
            };
            state_obj.step += 1;
            let mut slot = match state_obj.slots.remove(&param) {
                Some(existing) => existing,
                None => {
                    if state_obj.slots.len() == 1 && state_obj.slots.contains_key(&0) {
                        match state_obj.slots.remove(&0) {
                            Some(s) => s,
                            None => {
                                set_error("Optimizer state slot missing (index 0)");
                                return 0;
                            }
                        }
                    } else if state_obj.slots.len() > 1 && state_obj.slots.contains_key(&0) {
                        set_error("Optimizer state is indexed; use enkai_tensor_load_sharded_opt_multi to bind params");
                        return 0;
                    } else {
                        let zeros = Tensor::zeros_like(&param_tensor);
                        AdamWSlot {
                            m: zeros.shallow_clone(),
                            v: zeros,
                        }
                    }
                }
            };
            let beta1 = beta1 as f64;
            let beta2 = beta2 as f64;
            let lr = lr as f64;
            let eps = eps as f64;
            let weight_decay = weight_decay as f64;
            slot.m = &slot.m * beta1 + grad_tensor.shallow_clone() * (1.0 - beta1);
            slot.v = &slot.v * beta2 + grad_tensor.pow_tensor_scalar(2.0) * (1.0 - beta2);
            let bias1 = 1.0 - beta1.powi(state_obj.step as i32);
            let bias2 = 1.0 - beta2.powi(state_obj.step as i32);
            let m_hat = &slot.m / bias1;
            let v_hat = &slot.v / bias2;
            let update =
                &m_hat / (v_hat.sqrt() + eps) + param_tensor.shallow_clone() * weight_decay;
            let new_param = param_tensor - update * lr;
            update_tensor(param, new_param);
            state_obj.slots.insert(param, slot);
            let handle = if state == 0 {
                register_opt(state_obj)
            } else {
                update_opt(state, state_obj);
                state
            };
            return handle;
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = param;
            let _ = grad;
            let _ = state;
            let _ = lr;
            let _ = beta1;
            let _ = beta2;
            let _ = eps;
            let _ = weight_decay;
            set_error("torch backend not enabled");
            0
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_adamw_step_multi(
    params_json: *const c_char,
    grads_json: *const c_char,
    state: i64,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
) -> i64 {
    clear_error();
    let params_json = match cstr_to_string(params_json) {
        Ok(v) => v,
        Err(err) => {
            set_error(err);
            return 0;
        }
    };
    let grads_json = match cstr_to_string(grads_json) {
        Ok(v) => v,
        Err(err) => {
            set_error(err);
            return 0;
        }
    };
    #[cfg(feature = "torch")]
    {
        let params = match parse_handle_list(&params_json) {
            Ok(list) => list,
            Err(err) => {
                set_error(err);
                return 0;
            }
        };
        let grads = match parse_handle_list(&grads_json) {
            Ok(list) => list,
            Err(err) => {
                set_error(err);
                return 0;
            }
        };
        if params.len() != grads.len() {
            set_error("Params and grads length mismatch");
            return 0;
        }
        if params.is_empty() {
            set_error("Params list is empty");
            return 0;
        }
        let mut state_obj = if state == 0 {
            AdamWState {
                slots: HashMap::new(),
                step: 0,
            }
        } else {
            match get_opt_mut(state) {
                Ok(s) => s,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            }
        };
        state_obj.step += 1;
        let beta1 = beta1 as f64;
        let beta2 = beta2 as f64;
        let lr = lr as f64;
        let eps = eps as f64;
        let weight_decay = weight_decay as f64;
        for (param_handle, grad_handle) in params.iter().zip(grads.iter()) {
            let param_tensor = match get_tensor(*param_handle) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            let grad_tensor = match get_tensor(*grad_handle) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            let mut slot = match state_obj.slots.remove(param_handle) {
                Some(existing) => existing,
                None => {
                    let zeros = Tensor::zeros_like(&param_tensor);
                    AdamWSlot {
                        m: zeros.shallow_clone(),
                        v: zeros,
                    }
                }
            };
            slot.m = &slot.m * beta1 + grad_tensor.shallow_clone() * (1.0 - beta1);
            slot.v = &slot.v * beta2 + grad_tensor.pow_tensor_scalar(2.0) * (1.0 - beta2);
            let bias1 = 1.0 - beta1.powi(state_obj.step as i32);
            let bias2 = 1.0 - beta2.powi(state_obj.step as i32);
            let m_hat = &slot.m / bias1;
            let v_hat = &slot.v / bias2;
            let update =
                &m_hat / (v_hat.sqrt() + eps) + param_tensor.shallow_clone() * weight_decay;
            let new_param = param_tensor - update * lr;
            update_tensor(*param_handle, new_param);
            state_obj.slots.insert(*param_handle, slot);
        }
        let handle = if state == 0 {
            register_opt(state_obj)
        } else {
            update_opt(state, state_obj);
            state
        };
        return handle;
    }
    #[cfg(not(feature = "torch"))]
    {
        let _ = params_json;
        let _ = grads_json;
        let _ = state;
        let _ = lr;
        let _ = beta1;
        let _ = beta2;
        let _ = eps;
        let _ = weight_decay;
        set_error("torch backend not enabled");
        0
    }
}

#[no_mangle]
pub extern "C" fn enkai_tensor_opt_free(handle: i64) -> c_int {
    ffi_guard(1, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            if !backend_is_torch() {
                set_error("backend 'torch' not selected");
                return 1;
            }
            if let Ok(freed) = OPT_FREED.lock() {
                if freed.contains(&handle) {
                    set_error("optimizer handle already freed");
                    return 1;
                }
            }
            let mut poisoned = false;
            let mut removed = false;
            if let Ok(mut guard) = OPT_STATES.lock() {
                match guard.get_mut(&handle) {
                    Some(entry) => {
                        if entry.refcount == 0 {
                            set_error("optimizer handle already freed");
                            return 1;
                        }
                        entry.refcount -= 1;
                        if entry.refcount == 0 {
                            guard.remove(&handle);
                            removed = true;
                        }
                    }
                    None => {
                        set_error("Invalid optimizer handle");
                        return 1;
                    }
                }
            } else {
                poisoned = true;
            }
            if poisoned {
                set_error("optimizer registry poisoned");
                return 1;
            }
            if removed {
                if let Ok(mut guard) = OPT_PARAMS.lock() {
                    guard.remove(&handle);
                }
                if let Ok(mut guard) = OPT_HYPER.lock() {
                    guard.remove(&handle);
                }
                if let Ok(mut freed) = OPT_FREED.lock() {
                    freed.insert(handle);
                }
            }
            return 0;
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = handle;
            set_error("torch backend not enabled");
            1
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_amp_scaler_create(
    initial_scale: f64,
    growth_factor: f64,
    backoff_factor: f64,
    growth_interval: i64,
) -> i64 {
    ffi_guard(0, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            let state = GradScalerState {
                scale: initial_scale,
                growth_factor,
                backoff_factor,
                growth_interval,
                growth_tracker: 0,
                refcount: 1,
            };
            return register_scaler(state);
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = initial_scale;
            let _ = growth_factor;
            let _ = backoff_factor;
            let _ = growth_interval;
            set_error("torch backend not enabled");
            0
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_amp_scaler_retain(handle: i64) -> c_int {
    ffi_guard(1, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            if let Ok(mut guard) = SCALERS.lock() {
                match guard.get_mut(&handle) {
                    Some(state) => {
                        state.refcount = state.refcount.saturating_add(1);
                        return 0;
                    }
                    None => {
                        set_error("Invalid scaler handle");
                        return 1;
                    }
                }
            }
            set_error("scaler registry poisoned");
            1
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = handle;
            set_error("torch backend not enabled");
            1
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_amp_scaler_free(handle: i64) -> c_int {
    ffi_guard(1, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            if let Ok(mut guard) = SCALERS.lock() {
                match guard.get_mut(&handle) {
                    Some(state) => {
                        if state.refcount == 0 {
                            set_error("scaler already freed");
                            return 1;
                        }
                        state.refcount -= 1;
                        if state.refcount == 0 {
                            guard.remove(&handle);
                            if let Ok(mut freed) = SCALER_FREED.lock() {
                                freed.insert(handle);
                            }
                        }
                        return 0;
                    }
                    None => {
                        set_error("Invalid scaler handle");
                        return 1;
                    }
                }
            }
            set_error("scaler registry poisoned");
            1
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = handle;
            set_error("torch backend not enabled");
            1
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_amp_scale_loss(loss: i64, scaler: i64) -> i64 {
    ffi_guard(0, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            let loss_t = match get_tensor(loss) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            let state = match get_scaler_mut(scaler) {
                Ok(s) => s,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            let scaled = loss_t * state.scale;
            return register_tensor(scaled);
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = loss;
            let _ = scaler;
            set_error("torch backend not enabled");
            0
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_amp_unscale_grads(
    grads_json: *const c_char,
    scaler: i64,
    out_found_inf: *mut c_int,
) -> c_int {
    ffi_guard(1, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            let grads_json = match cstr_to_string(grads_json) {
                Ok(v) => v,
                Err(err) => {
                    set_error(err);
                    return 1;
                }
            };
            let handles = match parse_handle_list(&grads_json) {
                Ok(list) => list,
                Err(err) => {
                    set_error(err);
                    return 1;
                }
            };
            let state = match get_scaler_mut(scaler) {
                Ok(s) => s,
                Err(err) => {
                    set_error(err);
                    return 1;
                }
            };
            let mut found_inf = false;
            for h in handles {
                if h == 0 {
                    continue;
                }
                let grad = match get_tensor(h) {
                    Ok(t) => t,
                    Err(err) => {
                        set_error(err);
                        return 1;
                    }
                };
                if grad.isfinite().all().int64_value(&[]) == 0 {
                    found_inf = true;
                }
                let unscaled = &grad / state.scale;
                update_tensor(h, unscaled);
            }
            unsafe {
                *out_found_inf = if found_inf { 1 } else { 0 };
            }
            update_scaler(scaler, state);
            0
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = grads_json;
            let _ = scaler;
            let _ = out_found_inf;
            set_error("torch backend not enabled");
            1
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_amp_scaler_update(scaler: i64, found_inf: c_int) -> c_int {
    ffi_guard(1, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            let mut state = match get_scaler_mut(scaler) {
                Ok(s) => s,
                Err(err) => {
                    set_error(err);
                    return 1;
                }
            };
            if found_inf != 0 {
                state.scale *= state.backoff_factor;
                state.growth_tracker = 0;
            } else {
                state.growth_tracker += 1;
                if state.growth_tracker >= state.growth_interval {
                    state.scale *= state.growth_factor;
                    state.growth_tracker = 0;
                }
            }
            update_scaler(scaler, state);
            0
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = scaler;
            let _ = found_inf;
            set_error("torch backend not enabled");
            1
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_autocast_enter() -> c_int {
    ffi_guard(1, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            let result = tch::autocast::<i32, _>(true, || 0);
            result
        }
        #[cfg(not(feature = "torch"))]
        {
            set_error("torch backend not enabled");
            1
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_autocast_exit() -> c_int {
    ffi_guard(1, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            let result = tch::autocast::<i32, _>(false, || 0);
            result
        }
        #[cfg(not(feature = "torch"))]
        {
            set_error("torch backend not enabled");
            1
        }
    })
}

#[no_mangle]
/// Convenience: scale loss, backward, unscale grads, update scaler. Returns 0 on success, nonzero on error or found inf.
pub extern "C" fn enkai_amp_step(
    loss: i64,
    scaler: i64,
    params_json: *const c_char,
    grads_out_json: *mut *mut c_char,
    grads_out_len: *mut usize,
) -> c_int {
    ffi_guard(1, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            // Scale loss
            let scaled = enkai_amp_scale_loss(loss, scaler);
            if scaled == 0 {
                return 1;
            }
            // Backward
            if enkai_tensor_backward(scaled) != 0 {
                return 1;
            }
            // Unscale grads
            let params_json = match cstr_to_string(params_json) {
                Ok(v) => v,
                Err(err) => {
                    set_error(err);
                    return 1;
                }
            };
            let mut found_inf: c_int = 0;
            if enkai_amp_unscale_grads(
                params_json.as_ptr() as *const c_char,
                scaler,
                &mut found_inf,
            ) != 0
            {
                return 1;
            }
            // Collect grads
            if enkai_tensor_grad_multi(
                params_json.as_ptr() as *const c_char,
                grads_out_json,
                grads_out_len,
            ) != 0
            {
                return 1;
            }
            // Update scaler
            if enkai_amp_scaler_update(scaler, found_inf) != 0 {
                return 1;
            }
            0
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = loss;
            let _ = scaler;
            let _ = params_json;
            let _ = grads_out_json;
            let _ = grads_out_len;
            set_error("torch backend not enabled");
            1
        }
    })
}
#[no_mangle]
pub extern "C" fn enkai_opt_adamw_create(
    params_json: *const c_char,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
) -> i64 {
    ffi_guard(0, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            let params_json = match cstr_to_string(params_json) {
                Ok(v) => v,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            let params = match parse_handle_list(&params_json) {
                Ok(list) => list,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            let hyper = AdamWHyper {
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
            };
            let mut state = AdamWState {
                slots: HashMap::new(),
                step: 0,
            };
            for p in params.iter() {
                let tensor = match get_tensor(*p) {
                    Ok(t) => t,
                    Err(err) => {
                        set_error(err);
                        return 0;
                    }
                };
                let zeros = Tensor::zeros_like(&tensor);
                state.slots.insert(
                    *p,
                    AdamWSlot {
                        m: zeros.shallow_clone(),
                        v: zeros,
                    },
                );
            }
            let handle = register_opt(state);
            set_opt_params(handle, params);
            set_opt_hyper(handle, hyper);
            return handle;
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = params_json;
            let _ = lr;
            let _ = beta1;
            let _ = beta2;
            let _ = eps;
            let _ = weight_decay;
            set_error("torch backend not enabled");
            0
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_opt_adamw_step(opt: i64) -> c_int {
    ffi_guard(1, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            let params = match get_opt_params(opt) {
                Ok(p) => p,
                Err(err) => {
                    set_error(err);
                    return 1;
                }
            };
            let hyper = match get_opt_hyper(opt) {
                Ok(h) => h,
                Err(err) => {
                    set_error(err);
                    return 1;
                }
            };
            let params_json = match serde_json::to_string(&params) {
                Ok(s) => s,
                Err(err) => {
                    set_error(err.to_string());
                    return 1;
                }
            };
            let grads: Vec<i64> = params.iter().map(|p| enkai_tensor_grad(*p)).collect();
            let grads_json = match serde_json::to_string(&grads) {
                Ok(s) => s,
                Err(err) => {
                    set_error(err.to_string());
                    return 1;
                }
            };
            let params_c = match CString::new(params_json) {
                Ok(s) => s,
                Err(_) => {
                    set_error("params JSON contained interior null byte");
                    return 1;
                }
            };
            let grads_c = match CString::new(grads_json) {
                Ok(s) => s,
                Err(_) => {
                    set_error("grads JSON contained interior null byte");
                    return 1;
                }
            };
            let handle = enkai_tensor_adamw_step_multi(
                params_c.as_ptr(),
                grads_c.as_ptr(),
                opt,
                hyper.lr,
                hyper.beta1,
                hyper.beta2,
                hyper.eps,
                hyper.weight_decay,
            );
            if handle == 0 {
                return 1;
            }
            return 0;
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = opt;
            set_error("torch backend not enabled");
            1
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_save_sharded(
    dir: *const c_char,
    param: i64,
    opt_state: i64,
    meta_json: *const c_char,
) -> c_int {
    ffi_guard(1, || {
        clear_error();
        let dir = match cstr_to_string(dir) {
            Ok(v) => v,
            Err(err) => {
                set_error(err);
                return 1;
            }
        };
        let meta_json = match cstr_to_string(meta_json) {
            Ok(v) => v,
            Err(err) => {
                set_error(err);
                return 1;
            }
        };
        #[cfg(feature = "torch")]
        {
            let tensor = match get_tensor(param) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return 1;
                }
            };
            if let Err(err) = fs::create_dir_all(&dir) {
                set_error(err.to_string());
                return 1;
            }
            let model_path = format!("{}/model_rank0.bin", dir);
            if let Err(err) = tensor.save(model_path) {
                set_error(err.to_string());
                return 1;
            }
            if opt_state != 0 {
                if let Ok(state) = get_opt_mut(opt_state) {
                    if let Err(err) = save_opt_state(&dir, &state, &[param]) {
                        set_error(err);
                        return 1;
                    }
                }
            }
            let meta_path = format!("{}/meta.json", dir);
            if let Err(err) = fs::write(meta_path, meta_json.as_bytes()) {
                set_error(err.to_string());
                return 1;
            }
            return 0;
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = dir;
            let _ = param;
            let _ = opt_state;
            let _ = meta_json;
            set_error("torch backend not enabled");
            1
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_opt_retain(handle: i64) -> c_int {
    ffi_guard(1, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            if let Ok(freed) = OPT_FREED.lock() {
                if freed.contains(&handle) {
                    set_error("optimizer handle already freed");
                    return 1;
                }
            }
            match OPT_STATES.lock() {
                Ok(mut guard) => match guard.get_mut(&handle) {
                    Some(entry) => {
                        entry.refcount = entry.refcount.saturating_add(1);
                        return 0;
                    }
                    None => {
                        set_error("Invalid optimizer handle");
                        return 1;
                    }
                },
                Err(_) => {
                    set_error("optimizer registry poisoned");
                    return 1;
                }
            }
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = handle;
            set_error("torch backend not enabled");
            1
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_device_retain(handle: i64) -> c_int {
    ffi_guard(1, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            if let Ok(freed) = DEVICE_FREED.lock() {
                if freed.contains(&handle) {
                    set_error("device handle already freed");
                    return 1;
                }
            }
            match DEVICES.lock() {
                Ok(mut guard) => match guard.get_mut(&handle) {
                    Some(entry) => {
                        entry.refcount = entry.refcount.saturating_add(1);
                        return 0;
                    }
                    None => {
                        set_error("Invalid device handle");
                        return 1;
                    }
                },
                Err(_) => {
                    set_error("device registry poisoned");
                    return 1;
                }
            }
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = handle;
            set_error("torch backend not enabled");
            1
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_checkpoint_save(
    dir: *const c_char,
    params_json: *const c_char,
    opt_state: i64,
    meta_json: *const c_char,
) -> c_int {
    ffi_guard(1, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            guard_backend!(1);
            let dir = match cstr_to_string(dir) {
                Ok(v) => v,
                Err(err) => {
                    set_error(err);
                    return 1;
                }
            };
            let params_json = match cstr_to_string(params_json) {
                Ok(v) => v,
                Err(err) => {
                    set_error(err);
                    return 1;
                }
            };
            let meta_json = match cstr_to_string(meta_json) {
                Ok(v) => v,
                Err(err) => {
                    set_error(err);
                    return 1;
                }
            };
            let params = match parse_handle_list(&params_json) {
                Ok(list) => list,
                Err(err) => {
                    set_error(err);
                    return 1;
                }
            };
            if params.is_empty() {
                set_error("checkpoint_save: params empty");
                return 1;
            }
            if let Err(err) = fs::create_dir_all(&dir) {
                set_error(err.to_string());
                return 1;
            }
            let tmp_dir = format!("{}/.tmp", dir);
            let _ = fs::remove_dir_all(&tmp_dir);
            if let Err(err) = fs::create_dir_all(&tmp_dir) {
                set_error(err.to_string());
                return 1;
            }
            // Save params
            let mut tensors = Vec::new();
            for (idx, h) in params.iter().enumerate() {
                let t = match get_tensor(*h) {
                    Ok(t) => t,
                    Err(err) => {
                        set_error(err);
                        return 1;
                    }
                };
                tensors.push((format!("param_{}", idx), t.shallow_clone()));
            }
            let params_path = format!("{}/params.bin", tmp_dir);
            if let Err(err) = Tensor::save_multi(&tensors, &params_path) {
                set_error(err.to_string());
                return 1;
            }
            // Save optimizer
            let mut opt_written = false;
            if opt_state != 0 {
                if let Ok(state) = get_opt_mut(opt_state) {
                    if let Err(err) = save_opt_state(&tmp_dir, &state, &params) {
                        set_error(err);
                        return 1;
                    }
                    opt_written = true;
                }
            }
            let meta_path = format!("{}/meta.json", tmp_dir);
            if let Err(err) = fs::write(&meta_path, meta_json.as_bytes()) {
                set_error(err.to_string());
                return 1;
            }
            // Integrity metadata
            let params_hash = match file_sha256(&params_path) {
                Ok(h) => h,
                Err(err) => {
                    set_error(format!("checksum params failed: {}", err));
                    return 1;
                }
            };
            let meta_hash = match file_sha256(&meta_path) {
                Ok(h) => h,
                Err(err) => {
                    set_error(format!("checksum meta failed: {}", err));
                    return 1;
                }
            };
            let mut integrity = serde_json::json!({
                "version": 1,
                "params_sha256": params_hash,
                "meta_sha256": meta_hash
            });
            if opt_written {
                let opt_hash = match file_sha256(&format!("{}/optim_rank0.bin", tmp_dir)) {
                    Ok(h) => h,
                    Err(err) => {
                        set_error(format!("checksum optimizer failed: {}", err));
                        return 1;
                    }
                };
                let opt_meta_hash = match file_sha256(&format!("{}/optim_meta.json", tmp_dir)) {
                    Ok(h) => h,
                    Err(err) => {
                        set_error(format!("checksum optimizer meta failed: {}", err));
                        return 1;
                    }
                };
                integrity.as_object_mut().unwrap().insert(
                    "optimizer_sha256".to_string(),
                    serde_json::Value::String(opt_hash),
                );
                integrity.as_object_mut().unwrap().insert(
                    "optimizer_meta_sha256".to_string(),
                    serde_json::Value::String(opt_meta_hash),
                );
            }
            let integrity_path = format!("{}/integrity.json", tmp_dir);
            if let Err(err) = fs::write(&integrity_path, integrity.to_string()) {
                set_error(err.to_string());
                return 1;
            }
            if let Err(err) = fsync_file(&params_path) {
                set_error(err);
                return 1;
            }
            if let Err(err) = fsync_file(&meta_path) {
                set_error(err);
                return 1;
            }
            if let Err(err) = fsync_file(&integrity_path) {
                set_error(err);
                return 1;
            }
            if opt_written {
                let opt_path = format!("{}/optim_rank0.bin", tmp_dir);
                let opt_meta_path = format!("{}/optim_meta.json", tmp_dir);
                if let Err(err) = fsync_file(&opt_path) {
                    set_error(err);
                    return 1;
                }
                if let Err(err) = fsync_file(&opt_meta_path) {
                    set_error(err);
                    return 1;
                }
            }
            if let Err(err) = fsync_dir(&tmp_dir) {
                set_error(err);
                return 1;
            }
            // Atomic rename
            let final_dir = format!("{}/latest", dir);
            let _ = fs::remove_dir_all(&final_dir);
            if let Err(err) = fs::rename(&tmp_dir, &final_dir) {
                set_error(err.to_string());
                return 1;
            }
            if let Err(err) = fsync_dir(&dir) {
                set_error(err);
                return 1;
            }
            return 0;
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = dir;
            let _ = params_json;
            let _ = opt_state;
            let _ = meta_json;
            set_error("torch backend not enabled");
            1
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_checkpoint_save_ranked(
    dir: *const c_char,
    rank: i64,
    world_size: i64,
    params_json: *const c_char,
    opt_state: i64,
    meta_json: *const c_char,
) -> c_int {
    ffi_guard(1, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            if world_size <= 0 || rank < 0 || rank >= world_size {
                set_error("invalid world_size or rank");
                return 1;
            }
            let dir = match cstr_to_string(dir) {
                Ok(v) => v,
                Err(err) => {
                    set_error(err);
                    return 1;
                }
            };
            let params_json = match cstr_to_string(params_json) {
                Ok(v) => v,
                Err(err) => {
                    set_error(err);
                    return 1;
                }
            };
            let meta_json = match cstr_to_string(meta_json) {
                Ok(v) => v,
                Err(err) => {
                    set_error(err);
                    return 1;
                }
            };
            let params = match parse_handle_list(&params_json) {
                Ok(list) => list,
                Err(err) => {
                    set_error(err);
                    return 1;
                }
            };
            if params.is_empty() {
                set_error("checkpoint_save_ranked: params empty");
                return 1;
            }
            if let Err(err) = fs::create_dir_all(&dir) {
                set_error(err.to_string());
                return 1;
            }
            let tmp_dir = format!("{}/.tmp_rank{}", dir, rank);
            let _ = fs::remove_dir_all(&tmp_dir);
            if let Err(err) = fs::create_dir_all(&tmp_dir) {
                set_error(err.to_string());
                return 1;
            }
            // Save params
            let mut tensors = Vec::new();
            for (idx, h) in params.iter().enumerate() {
                let t = match get_tensor(*h) {
                    Ok(t) => t,
                    Err(err) => {
                        set_error(err);
                        return 1;
                    }
                };
                tensors.push((format!("param_{}", idx), t.shallow_clone()));
            }
            let params_path = format!("{}/params_rank{}.bin", tmp_dir, rank);
            if let Err(err) = Tensor::save_multi(&tensors, &params_path) {
                set_error(err.to_string());
                return 1;
            }
            // Save optimizer
            let mut opt_written = false;
            if opt_state != 0 {
                if let Ok(state) = get_opt_mut(opt_state) {
                    if let Err(err) = save_opt_state(&tmp_dir, &state, &params) {
                        set_error(err);
                        return 1;
                    }
                    opt_written = true;
                }
            }
            let meta_path = format!("{}/meta_rank{}.json", tmp_dir, rank);
            if let Err(err) = fs::write(&meta_path, meta_json.as_bytes()) {
                set_error(err.to_string());
                return 1;
            }
            // Integrity metadata per rank
            let params_hash = match file_sha256(&params_path) {
                Ok(h) => h,
                Err(err) => {
                    set_error(format!("checksum params failed: {}", err));
                    return 1;
                }
            };
            let meta_hash = match file_sha256(&meta_path) {
                Ok(h) => h,
                Err(err) => {
                    set_error(format!("checksum meta failed: {}", err));
                    return 1;
                }
            };
            let mut integrity = serde_json::json!({
                "version": 1,
                "rank": rank,
                "world_size": world_size,
                "params_sha256": params_hash,
                "meta_sha256": meta_hash
            });
            if opt_written {
                let opt_hash = match file_sha256(&format!("{}/optim_rank0.bin", tmp_dir)) {
                    Ok(h) => h,
                    Err(err) => {
                        set_error(format!("checksum optimizer failed: {}", err));
                        return 1;
                    }
                };
                let opt_meta_hash = match file_sha256(&format!("{}/optim_meta.json", tmp_dir)) {
                    Ok(h) => h,
                    Err(err) => {
                        set_error(format!("checksum optimizer meta failed: {}", err));
                        return 1;
                    }
                };
                integrity.as_object_mut().unwrap().insert(
                    "optimizer_sha256".to_string(),
                    serde_json::Value::String(opt_hash),
                );
                integrity.as_object_mut().unwrap().insert(
                    "optimizer_meta_sha256".to_string(),
                    serde_json::Value::String(opt_meta_hash),
                );
            }
            let integrity_path = format!("{}/integrity_rank{}.json", tmp_dir, rank);
            if let Err(err) = fs::write(&integrity_path, integrity.to_string()) {
                set_error(err.to_string());
                return 1;
            }
            if let Err(err) = fsync_file(&params_path) {
                set_error(err);
                return 1;
            }
            if let Err(err) = fsync_file(&meta_path) {
                set_error(err);
                return 1;
            }
            if let Err(err) = fsync_file(&integrity_path) {
                set_error(err);
                return 1;
            }
            if opt_written {
                let opt_path = format!("{}/optim_rank0.bin", tmp_dir);
                let opt_meta_path = format!("{}/optim_meta.json", tmp_dir);
                if let Err(err) = fsync_file(&opt_path) {
                    set_error(err);
                    return 1;
                }
                if let Err(err) = fsync_file(&opt_meta_path) {
                    set_error(err);
                    return 1;
                }
            }
            if let Err(err) = fsync_dir(&tmp_dir) {
                set_error(err);
                return 1;
            }
            // Atomic move into rank dir
            let final_dir = format!("{}/rank{}", dir, rank);
            let _ = fs::remove_dir_all(&final_dir);
            if let Err(err) = fs::rename(&tmp_dir, &final_dir) {
                set_error(err.to_string());
                return 1;
            }
            if let Err(err) = fsync_dir(&dir) {
                set_error(err);
                return 1;
            }
            0
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = dir;
            let _ = rank;
            let _ = world_size;
            let _ = params_json;
            let _ = opt_state;
            let _ = meta_json;
            set_error("torch backend not enabled");
            1
        }
    })
}
#[no_mangle]
pub extern "C" fn enkai_checkpoint_load(
    dir: *const c_char,
    out_params_json: *mut *mut c_char,
    out_params_len: *mut usize,
    out_opt_handle: *mut i64,
    out_meta_json: *mut *mut c_char,
    out_meta_len: *mut usize,
) -> c_int {
    ffi_guard(1, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            guard_backend!(1);
            let dir = match cstr_to_string(dir) {
                Ok(v) => v,
                Err(err) => {
                    set_error(err);
                    return 1;
                }
            };
            let base = format!("{}/latest", dir);
            let params_path = format!("{}/params.bin", base);
            let meta_path = format!("{}/meta.json", base);
            if !Path::new(&params_path).exists() {
                set_error("checkpoint_load: params missing");
                return 1;
            }
            // Integrity (optional)
            let integrity_path = format!("{}/integrity.json", base);
            if Path::new(&integrity_path).exists() {
                let text = match fs::read_to_string(&integrity_path) {
                    Ok(s) => s,
                    Err(err) => {
                        set_error(err.to_string());
                        return 1;
                    }
                };
                let val: serde_json::Value = match serde_json::from_str(&text) {
                    Ok(v) => v,
                    Err(err) => {
                        set_error(format!("integrity parse failed: {}", err));
                        return 1;
                    }
                };
                let ver = val.get("version").and_then(|v| v.as_u64()).unwrap_or(1);
                if ver != 1 {
                    set_error("integrity version unsupported");
                    return 1;
                }
                let params_hash = match val.get("params_sha256").and_then(|v| v.as_str()) {
                    Some(h) => h,
                    None => {
                        set_error("integrity missing params_sha256");
                        return 1;
                    }
                };
                let meta_hash = match val.get("meta_sha256").and_then(|v| v.as_str()) {
                    Some(h) => h,
                    None => {
                        set_error("integrity missing meta_sha256");
                        return 1;
                    }
                };
                match file_sha256(&params_path) {
                    Ok(h) if h == params_hash => {}
                    Ok(_) => {
                        set_error("params checksum mismatch");
                        return 1;
                    }
                    Err(err) => {
                        set_error(err);
                        return 1;
                    }
                }
                match file_sha256(&meta_path) {
                    Ok(h) if h == meta_hash => {}
                    Ok(_) => {
                        set_error("meta checksum mismatch");
                        return 1;
                    }
                    Err(err) => {
                        set_error(err);
                        return 1;
                    }
                }
                if let Some(opt_hash) = val.get("optimizer_sha256").and_then(|v| v.as_str()) {
                    let opt_path = format!("{}/optim_rank0.bin", base);
                    match file_sha256(&opt_path) {
                        Ok(h) if h == opt_hash => {}
                        Ok(_) => {
                            set_error("optimizer checksum mismatch");
                            return 1;
                        }
                        Err(err) => {
                            set_error(err);
                            return 1;
                        }
                    }
                }
                if let Some(opt_meta_hash) =
                    val.get("optimizer_meta_sha256").and_then(|v| v.as_str())
                {
                    let opt_meta_path = format!("{}/optim_meta.json", base);
                    match file_sha256(&opt_meta_path) {
                        Ok(h) if h == opt_meta_hash => {}
                        Ok(_) => {
                            set_error("optimizer meta checksum mismatch");
                            return 1;
                        }
                        Err(err) => {
                            set_error(err);
                            return 1;
                        }
                    }
                }
            }
            let list = match Tensor::load_multi(&params_path) {
                Ok(l) => l,
                Err(err) => {
                    set_error(err.to_string());
                    return 1;
                }
            };
            let mut handles = Vec::new();
            for (_, t) in list {
                handles.push(register_tensor(t));
            }
            let params_json = match serde_json::to_string(&handles) {
                Ok(s) => s,
                Err(err) => {
                    set_error(err.to_string());
                    return 1;
                }
            };
            unsafe {
                *out_params_len = params_json.len();
                let cstr = match CString::new(params_json) {
                    Ok(s) => s,
                    Err(_) => {
                        set_error("params JSON contained interior null byte");
                        return 1;
                    }
                };
                *out_params_json = cstr.into_raw();
            }
            // Optimizer (optional)
            let opt_meta_path = format!("{}/optim_meta.json", base);
            let mut opt_handle: i64 = 0;
            if Path::new(&opt_meta_path).exists() {
                match load_opt_state(&base) {
                    Ok(state) => {
                        let bound = match bind_opt_state(state, &handles) {
                            Ok(s) => s,
                            Err(err) => {
                                set_error(err);
                                return 1;
                            }
                        };
                        opt_handle = register_opt(bound);
                    }
                    Err(err) => {
                        set_error(err);
                        return 1;
                    }
                }
            }
            unsafe {
                *out_opt_handle = opt_handle;
            }
            let meta_text = match fs::read_to_string(&meta_path) {
                Ok(s) => s,
                Err(err) => {
                    set_error(err.to_string());
                    return 1;
                }
            };
            unsafe {
                *out_meta_len = meta_text.len();
                let cstr = match CString::new(meta_text) {
                    Ok(s) => s,
                    Err(_) => {
                        set_error("meta JSON contained interior null byte");
                        return 1;
                    }
                };
                *out_meta_json = cstr.into_raw();
            }
            return 0;
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = dir;
            let _ = out_params_json;
            let _ = out_params_len;
            let _ = out_opt_handle;
            let _ = out_meta_json;
            let _ = out_meta_len;
            set_error("torch backend not enabled");
            1
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_checkpoint_load_ranked(
    dir: *const c_char,
    rank: i64,
    out_params_json: *mut *mut c_char,
    out_params_len: *mut usize,
    out_opt_handle: *mut i64,
    out_meta_json: *mut *mut c_char,
    out_meta_len: *mut usize,
) -> c_int {
    ffi_guard(1, || {
        clear_error();
        #[cfg(feature = "torch")]
        {
            let dir = match cstr_to_string(dir) {
                Ok(v) => v,
                Err(err) => {
                    set_error(err);
                    return 1;
                }
            };
            let base = format!("{}/rank{}", dir, rank);
            let params_path = format!("{}/params_rank{}.bin", base, rank);
            let meta_path = format!("{}/meta_rank{}.json", base, rank);
            if !Path::new(&params_path).exists() {
                set_error("checkpoint_load_ranked: params missing");
                return 1;
            }
            // Integrity if present
            let integrity_path = format!("{}/integrity_rank{}.json", base, rank);
            if Path::new(&integrity_path).exists() {
                let text = match fs::read_to_string(&integrity_path) {
                    Ok(s) => s,
                    Err(err) => {
                        set_error(err.to_string());
                        return 1;
                    }
                };
                let val: serde_json::Value = match serde_json::from_str(&text) {
                    Ok(v) => v,
                    Err(err) => {
                        set_error(format!("integrity parse failed: {}", err));
                        return 1;
                    }
                };
                let params_hash = match val.get("params_sha256").and_then(|v| v.as_str()) {
                    Some(h) => h,
                    None => {
                        set_error("integrity missing params_sha256");
                        return 1;
                    }
                };
                let meta_hash = match val.get("meta_sha256").and_then(|v| v.as_str()) {
                    Some(h) => h,
                    None => {
                        set_error("integrity missing meta_sha256");
                        return 1;
                    }
                };
                match file_sha256(&params_path) {
                    Ok(h) if h == params_hash => {}
                    Ok(_) => {
                        set_error("params checksum mismatch");
                        return 1;
                    }
                    Err(err) => {
                        set_error(err);
                        return 1;
                    }
                }
                match file_sha256(&meta_path) {
                    Ok(h) if h == meta_hash => {}
                    Ok(_) => {
                        set_error("meta checksum mismatch");
                        return 1;
                    }
                    Err(err) => {
                        set_error(err);
                        return 1;
                    }
                }
            }
            let list = match Tensor::load_multi(&params_path) {
                Ok(l) => l,
                Err(err) => {
                    set_error(err.to_string());
                    return 1;
                }
            };
            let mut handles = Vec::new();
            for (_, t) in list {
                handles.push(register_tensor(t));
            }
            let params_json = match serde_json::to_string(&handles) {
                Ok(s) => s,
                Err(err) => {
                    set_error(err.to_string());
                    return 1;
                }
            };
            unsafe {
                *out_params_len = params_json.len();
                let cstr = match CString::new(params_json) {
                    Ok(s) => s,
                    Err(_) => {
                        set_error("params JSON contained interior null byte");
                        return 1;
                    }
                };
                *out_params_json = cstr.into_raw();
            }
            // Optimizer (optional)
            let opt_meta_path = format!("{}/optim_meta.json", base);
            let mut opt_handle: i64 = 0;
            if Path::new(&opt_meta_path).exists() {
                match load_opt_state(&base) {
                    Ok(state) => {
                        let bound = match bind_opt_state(state, &handles) {
                            Ok(s) => s,
                            Err(err) => {
                                set_error(err);
                                return 1;
                            }
                        };
                        opt_handle = register_opt(bound);
                    }
                    Err(err) => {
                        set_error(err);
                        return 1;
                    }
                }
            }
            unsafe {
                *out_opt_handle = opt_handle;
            }
            let meta_text = match fs::read_to_string(&meta_path) {
                Ok(s) => s,
                Err(err) => {
                    set_error(err.to_string());
                    return 1;
                }
            };
            unsafe {
                *out_meta_len = meta_text.len();
                let cstr = match CString::new(meta_text) {
                    Ok(s) => s,
                    Err(_) => {
                        set_error("meta JSON contained interior null byte");
                        return 1;
                    }
                };
                *out_meta_json = cstr.into_raw();
            }
            0
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = dir;
            let _ = rank;
            let _ = out_params_json;
            let _ = out_params_len;
            let _ = out_opt_handle;
            let _ = out_meta_json;
            let _ = out_meta_len;
            set_error("torch backend not enabled");
            1
        }
    })
}
#[no_mangle]
pub extern "C" fn enkai_tensor_save_sharded_multi(
    dir: *const c_char,
    params_json: *const c_char,
    opt_state: i64,
    meta_json: *const c_char,
) -> c_int {
    ffi_guard(1, || {
        clear_error();
        let dir = match cstr_to_string(dir) {
            Ok(v) => v,
            Err(err) => {
                set_error(err);
                return 1;
            }
        };
        let params_json = match cstr_to_string(params_json) {
            Ok(v) => v,
            Err(err) => {
                set_error(err);
                return 1;
            }
        };
        let meta_json = match cstr_to_string(meta_json) {
            Ok(v) => v,
            Err(err) => {
                set_error(err);
                return 1;
            }
        };
        #[cfg(feature = "torch")]
        {
            if !backend_is_torch() {
                set_error("backend 'torch' not selected");
                return 1;
            }
            let params = match parse_handle_list(&params_json) {
                Ok(list) => list,
                Err(err) => {
                    set_error(err);
                    return 1;
                }
            };
            if params.is_empty() {
                set_error("No params provided");
                return 1;
            }
            if let Err(err) = fs::create_dir_all(&dir) {
                set_error(err.to_string());
                return 1;
            }
            let mut tensors = Vec::with_capacity(params.len());
            for (idx, handle) in params.iter().enumerate() {
                let tensor = match get_tensor(*handle) {
                    Ok(t) => t,
                    Err(err) => {
                        set_error(err);
                        return 1;
                    }
                };
                tensors.push((format!("param_{}", idx), tensor));
            }
            let model_path = format!("{}/model_rank0.bin", dir);
            if let Err(err) = Tensor::save_multi(&tensors, model_path) {
                set_error(err.to_string());
                return 1;
            }
            if opt_state != 0 {
                if let Ok(state) = get_opt_mut(opt_state) {
                    if let Err(err) = save_opt_state(&dir, &state, &params) {
                        set_error(err);
                        return 1;
                    }
                }
            }
            let meta_path = format!("{}/meta.json", dir);
            if let Err(err) = fs::write(meta_path, meta_json.as_bytes()) {
                set_error(err.to_string());
                return 1;
            }
            return 0;
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = dir;
            let _ = params_json;
            let _ = opt_state;
            let _ = meta_json;
            set_error("torch backend not enabled");
            1
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_load_sharded(dir: *const c_char) -> i64 {
    ffi_guard(0, || {
        clear_error();
        let dir = match cstr_to_string(dir) {
            Ok(v) => v,
            Err(err) => {
                set_error(err);
                return 0;
            }
        };
        #[cfg(feature = "torch")]
        {
            let model_path = format!("{}/model_rank0.bin", dir);
            match Tensor::load(model_path) {
                Ok(tensor) => register_tensor(tensor),
                Err(err) => {
                    set_error(err.to_string());
                    0
                }
            }
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = dir;
            set_error("torch backend not enabled");
            0
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_load_sharded_params(dir: *const c_char) -> FfiSlice {
    ffi_guard(null_slice(), || {
        clear_error();
        let dir = match cstr_to_string(dir) {
            Ok(v) => v,
            Err(err) => {
                set_error(err);
                return null_slice();
            }
        };
        #[cfg(feature = "torch")]
        {
            let model_path = format!("{}/model_rank0.bin", dir);
            let list = match Tensor::load_multi(model_path) {
                Ok(list) => list,
                Err(err) => {
                    set_error(err.to_string());
                    return null_slice();
                }
            };
            let mut params = Vec::with_capacity(list.len());
            for (name, tensor) in list {
                let idx = name
                    .strip_prefix("param_")
                    .and_then(|rest| rest.parse::<usize>().ok())
                    .ok_or_else(|| "Invalid param name in checkpoint".to_string());
                let idx = match idx {
                    Ok(v) => v,
                    Err(err) => {
                        set_error(err);
                        return null_slice();
                    }
                };
                let handle = register_tensor(tensor);
                params.push((idx, handle));
            }
            params.sort_by_key(|(idx, _)| *idx);
            let handles: Vec<i64> = params.into_iter().map(|(_, h)| h).collect();
            let json = match serde_json::to_string(&handles) {
                Ok(text) => text,
                Err(_) => {
                    set_error("Failed to serialize params list");
                    return null_slice();
                }
            };
            make_slice(json.into_bytes())
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = dir;
            set_error("torch backend not enabled");
            null_slice()
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_load_sharded_opt(dir: *const c_char) -> i64 {
    ffi_guard(0, || {
        clear_error();
        let dir = match cstr_to_string(dir) {
            Ok(v) => v,
            Err(err) => {
                set_error(err);
                return 0;
            }
        };
        #[cfg(feature = "torch")]
        {
            match load_opt_state(&dir) {
                Ok(state) => register_opt(state),
                Err(err) => {
                    set_error(err);
                    0
                }
            }
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = dir;
            set_error("torch backend not enabled");
            0
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_load_sharded_opt_multi(
    dir: *const c_char,
    params_json: *const c_char,
) -> i64 {
    ffi_guard(0, || {
        clear_error();
        let dir = match cstr_to_string(dir) {
            Ok(v) => v,
            Err(err) => {
                set_error(err);
                return 0;
            }
        };
        let params_json = match cstr_to_string(params_json) {
            Ok(v) => v,
            Err(err) => {
                set_error(err);
                return 0;
            }
        };
        #[cfg(feature = "torch")]
        {
            let params = match parse_handle_list(&params_json) {
                Ok(list) => list,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            let state = match load_opt_state(&dir) {
                Ok(state) => state,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            let bound = match bind_opt_state(state, &params) {
                Ok(state) => state,
                Err(err) => {
                    set_error(err);
                    return 0;
                }
            };
            register_opt(bound)
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = dir;
            let _ = params_json;
            set_error("torch backend not enabled");
            0
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_live_tensors() -> i64 {
    ffi_guard(0, || {
        #[cfg(feature = "torch")]
        {
            if !backend_is_torch() {
                set_error("backend 'torch' not selected");
                return 0;
            }
            match TENSORS.lock() {
                Ok(guard) => guard.len() as i64,
                Err(_) => {
                    set_error("tensor registry poisoned");
                    0
                }
            }
        }
        #[cfg(not(feature = "torch"))]
        {
            set_error("torch backend not enabled");
            0
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_live_devices() -> i64 {
    ffi_guard(0, || {
        #[cfg(feature = "torch")]
        {
            if !backend_is_torch() {
                set_error("backend 'torch' not selected");
                return 0;
            }
            match DEVICES.lock() {
                Ok(guard) => guard.len() as i64,
                Err(_) => {
                    set_error("device registry poisoned");
                    0
                }
            }
        }
        #[cfg(not(feature = "torch"))]
        {
            set_error("torch backend not enabled");
            0
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_live_opts() -> i64 {
    ffi_guard(0, || {
        #[cfg(feature = "torch")]
        {
            if !backend_is_torch() {
                set_error("backend 'torch' not selected");
                return 0;
            }
            match OPT_STATES.lock() {
                Ok(guard) => guard.len() as i64,
                Err(_) => {
                    set_error("optimizer registry poisoned");
                    0
                }
            }
        }
        #[cfg(not(feature = "torch"))]
        {
            set_error("torch backend not enabled");
            0
        }
    })
}

#[no_mangle]
pub extern "C" fn enkai_tensor_load_sharded_meta(dir: *const c_char) -> FfiSlice {
    ffi_guard(null_slice(), || {
        clear_error();
        let dir = match cstr_to_string(dir) {
            Ok(v) => v,
            Err(err) => {
                set_error(err);
                return null_slice();
            }
        };
        let meta_path = format!("{}/meta.json", dir);
        match fs::read_to_string(meta_path) {
            Ok(text) => make_slice(text.into_bytes()),
            Err(err) => {
                set_error(err.to_string());
                null_slice()
            }
        }
    })
}
