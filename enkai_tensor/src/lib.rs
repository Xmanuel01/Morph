use libc::{c_char, c_int};
#[cfg(feature = "torch")]
use once_cell::sync::Lazy;
use std::cell::RefCell;
#[cfg(feature = "torch")]
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::fs;
use std::ptr;
#[cfg(feature = "torch")]
use std::sync::atomic::{AtomicI64, Ordering};
#[cfg(feature = "torch")]
use std::sync::Mutex;

#[cfg(feature = "torch")]
use tch::{Device, Kind, Tensor};

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

#[no_mangle]
pub extern "C" fn enkai_tensor_last_error() -> *const c_char {
    LAST_ERROR.with(|cell| match &*cell.borrow() {
        Some(msg) => msg.as_ptr(),
        None => ptr::null(),
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
static TENSORS: Lazy<Mutex<HashMap<i64, Tensor>>> = Lazy::new(|| Mutex::new(HashMap::new()));

#[cfg(feature = "torch")]
static DEVICES: Lazy<Mutex<HashMap<i64, Device>>> = Lazy::new(|| Mutex::new(HashMap::new()));

#[cfg(feature = "torch")]
static OPT_STATES: Lazy<Mutex<HashMap<i64, AdamWState>>> = Lazy::new(|| Mutex::new(HashMap::new()));

#[cfg(feature = "torch")]
fn next_handle() -> i64 {
    NEXT_HANDLE.fetch_add(1, Ordering::Relaxed)
}

#[cfg(feature = "torch")]
fn register_tensor(t: Tensor) -> i64 {
    let id = next_handle();
    TENSORS.lock().unwrap().insert(id, t);
    id
}

#[cfg(feature = "torch")]
fn register_device(d: Device) -> i64 {
    let id = next_handle();
    DEVICES.lock().unwrap().insert(id, d);
    id
}

#[cfg(feature = "torch")]
fn register_opt(state: AdamWState) -> i64 {
    let id = next_handle();
    OPT_STATES.lock().unwrap().insert(id, state);
    id
}

#[cfg(feature = "torch")]
fn get_tensor(handle: i64) -> Result<Tensor, String> {
    let guard = TENSORS.lock().unwrap();
    guard
        .get(&handle)
        .map(|t| t.shallow_clone())
        .ok_or_else(|| "Invalid tensor handle".to_string())
}

#[cfg(feature = "torch")]
fn get_device(handle: i64) -> Result<Device, String> {
    let guard = DEVICES.lock().unwrap();
    guard
        .get(&handle)
        .cloned()
        .ok_or_else(|| "Invalid device handle".to_string())
}

#[cfg(feature = "torch")]
fn get_opt_mut(handle: i64) -> Result<AdamWState, String> {
    let guard = OPT_STATES.lock().unwrap();
    guard
        .get(&handle)
        .map(|state| clone_opt_state(state))
        .ok_or_else(|| "Invalid optimizer handle".to_string())
}

#[cfg(feature = "torch")]
fn update_tensor(handle: i64, tensor: Tensor) {
    TENSORS.lock().unwrap().insert(handle, tensor);
}

#[cfg(feature = "torch")]
fn update_opt(handle: i64, state: AdamWState) {
    OPT_STATES.lock().unwrap().insert(handle, state);
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
    if ptr.is_null() {
        return;
    }
    let _ = CString::from_raw(ptr);
}

#[no_mangle]
/// # Safety
/// The caller must pass a pointer and length originally allocated by `enkai_tensor`
/// and must not free the buffer more than once.
pub unsafe extern "C" fn enkai_free(ptr: *mut u8, len: usize) {
    if ptr.is_null() {
        return;
    }
    let _ = Vec::from_raw_parts(ptr, len, len);
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
}

#[no_mangle]
pub extern "C" fn enkai_tensor_free(handle: i64) -> c_int {
    clear_error();
    #[cfg(feature = "torch")]
    {
        let mut guard = TENSORS.lock().unwrap();
        guard.remove(&handle);
        return 0;
    }
    #[cfg(not(feature = "torch"))]
    {
        let _ = handle;
        set_error("torch backend not enabled");
        1
    }
}

#[no_mangle]
pub extern "C" fn enkai_tensor_device_free(handle: i64) -> c_int {
    clear_error();
    #[cfg(feature = "torch")]
    {
        let mut guard = DEVICES.lock().unwrap();
        guard.remove(&handle);
        return 0;
    }
    #[cfg(not(feature = "torch"))]
    {
        let _ = handle;
        set_error("torch backend not enabled");
        1
    }
}

#[no_mangle]
pub extern "C" fn enkai_tensor_randn(
    shape_json: *const c_char,
    dtype: *const c_char,
    device_handle: i64,
) -> i64 {
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
}

#[no_mangle]
pub extern "C" fn enkai_tensor_zeros(
    shape_json: *const c_char,
    dtype: *const c_char,
    device_handle: i64,
) -> i64 {
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
}

#[no_mangle]
pub extern "C" fn enkai_tensor_matmul(a: i64, b: i64) -> i64 {
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
}

#[no_mangle]
pub extern "C" fn enkai_tensor_add(a: i64, b: i64) -> i64 {
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
}

#[no_mangle]
pub extern "C" fn enkai_tensor_mul(a: i64, b: i64) -> i64 {
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
}

#[no_mangle]
pub extern "C" fn enkai_tensor_reshape(x: i64, shape_json: *const c_char) -> i64 {
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
}

#[no_mangle]
pub extern "C" fn enkai_tensor_transpose(x: i64, dim0: i64, dim1: i64) -> i64 {
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
}

#[no_mangle]
pub extern "C" fn enkai_tensor_concat(handles_json: *const c_char, dim: i64) -> i64 {
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
}

#[no_mangle]
pub extern "C" fn enkai_tensor_sum(x: i64, dim: i64, keepdim: c_int) -> i64 {
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
}

#[no_mangle]
pub extern "C" fn enkai_tensor_mean(x: i64, dim: i64, keepdim: c_int) -> i64 {
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
}

#[no_mangle]
pub extern "C" fn enkai_tensor_softmax(x: i64, dim: i64) -> i64 {
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
}

#[no_mangle]
pub extern "C" fn enkai_tensor_masked_softmax(
    x: i64,
    mask: i64,
    dim: i64,
    mask_type: c_int,
) -> i64 {
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
}

#[no_mangle]
pub extern "C" fn enkai_tensor_masked_softmax_backward(
    grad_output: i64,
    output: i64,
    mask: i64,
    dim: i64,
) -> i64 {
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
}

#[no_mangle]
pub extern "C" fn enkai_tensor_relu(x: i64) -> i64 {
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
}

#[no_mangle]
pub extern "C" fn enkai_tensor_sigmoid(x: i64) -> i64 {
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
}

#[no_mangle]
pub extern "C" fn enkai_tensor_dropout(x: i64, p: f64, train: c_int) -> i64 {
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
}

#[no_mangle]
pub extern "C" fn enkai_tensor_slice(x: i64, dim: i64, start: i64, end: i64, step: i64) -> i64 {
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
}

#[no_mangle]
pub extern "C" fn enkai_tensor_view(x: i64, shape_json: *const c_char) -> i64 {
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
}

#[no_mangle]
pub extern "C" fn enkai_tensor_gelu(x: i64) -> i64 {
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
}

#[no_mangle]
pub extern "C" fn enkai_tensor_layernorm(x: i64, w: i64, b: i64, eps: f64) -> i64 {
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
}

#[no_mangle]
pub extern "C" fn enkai_tensor_layernorm_backward(
    x: i64,
    w: i64,
    b: i64,
    eps: f64,
    grad_out: i64,
) -> FfiSlice {
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
}

#[no_mangle]
pub extern "C" fn enkai_tensor_embedding(w: i64, ids: i64) -> i64 {
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
}

#[no_mangle]
pub extern "C" fn enkai_tensor_linear(x: i64, w: i64, b: i64) -> i64 {
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
}

#[no_mangle]
pub extern "C" fn enkai_tensor_to_device(x: i64, device: i64) -> i64 {
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
}

#[no_mangle]
pub extern "C" fn enkai_tensor_to_dtype(x: i64, dtype: *const c_char) -> i64 {
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
}

#[no_mangle]
pub extern "C" fn enkai_tensor_shape(x: i64) -> FfiSlice {
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
}

#[no_mangle]
pub extern "C" fn enkai_tensor_cross_entropy(logits: i64, targets: i64) -> i64 {
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
}

#[no_mangle]
pub extern "C" fn enkai_tensor_backward(loss: i64) -> c_int {
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
                    state_obj.slots.remove(&0).unwrap()
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
        let update = &m_hat / (v_hat.sqrt() + eps) + param_tensor.shallow_clone() * weight_decay;
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
    clear_error();
    #[cfg(feature = "torch")]
    {
        let mut guard = OPT_STATES.lock().unwrap();
        guard.remove(&handle);
        return 0;
    }
    #[cfg(not(feature = "torch"))]
    {
        let _ = handle;
        set_error("torch backend not enabled");
        1
    }
}

#[no_mangle]
pub extern "C" fn enkai_tensor_save_sharded(
    dir: *const c_char,
    param: i64,
    opt_state: i64,
    meta_json: *const c_char,
) -> c_int {
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
}

#[no_mangle]
pub extern "C" fn enkai_tensor_save_sharded_multi(
    dir: *const c_char,
    params_json: *const c_char,
    opt_state: i64,
    meta_json: *const c_char,
) -> c_int {
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
}

#[no_mangle]
pub extern "C" fn enkai_tensor_load_sharded(dir: *const c_char) -> i64 {
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
}

#[no_mangle]
pub extern "C" fn enkai_tensor_load_sharded_params(dir: *const c_char) -> FfiSlice {
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
}

#[no_mangle]
pub extern "C" fn enkai_tensor_load_sharded_opt(dir: *const c_char) -> i64 {
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
}

#[no_mangle]
pub extern "C" fn enkai_tensor_load_sharded_opt_multi(
    dir: *const c_char,
    params_json: *const c_char,
) -> i64 {
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
}

#[no_mangle]
pub extern "C" fn enkai_tensor_load_sharded_meta(dir: *const c_char) -> FfiSlice {
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
}
