use libc::c_int;
#[cfg(feature = "torch")]
use once_cell::sync::Lazy;
#[cfg(feature = "torch")]
use std::sync::Mutex;

#[cfg(feature = "torch")]
use tch::nn::VarStore;
#[cfg(feature = "torch")]
use tch::Device;

use crate::set_error;

#[cfg(feature = "torch")]
static DIST_STATE: Lazy<Mutex<Option<DistCtx>>> = Lazy::new(|| Mutex::new(None));

#[cfg(feature = "torch")]
#[derive(Clone, Debug)]
struct DistCtx {
    world_size: i32,
    rank: i32,
}

#[no_mangle]
pub extern "C" fn enkai_dist_init(world_size: i32, rank: i32) -> c_int {
    #[cfg(feature = "torch")]
    {
        if world_size <= 0 || rank < 0 || rank >= world_size {
            set_error("invalid dist init arguments");
            return 1;
        }
        // For single-node, libtorch sets up NCCL automatically when tensors are on CUDA.
        let mut guard = DIST_STATE.lock().unwrap();
        *guard = Some(DistCtx { world_size, rank });
        return 0;
    }
    #[cfg(not(feature = "torch"))]
    {
        let _ = world_size;
        let _ = rank;
        set_error("torch backend not enabled");
        1
    }
}

#[no_mangle]
pub extern "C" fn enkai_dist_allreduce_sum_multi(handles_json: *const libc::c_char) -> c_int {
    #[cfg(feature = "torch")]
    {
        use crate::{cstr_to_string, get_tensor, parse_handle_list};
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
        if handles.is_empty() {
            return 0;
        }
        let guard = DIST_STATE.lock().unwrap();
        let ctx = match &*guard {
            Some(c) => c.clone(),
            None => {
                set_error("dist not initialized");
                return 1;
            }
        };
        drop(guard);
        for h in handles {
            let t = match get_tensor(h) {
                Ok(t) => t,
                Err(err) => {
                    set_error(err);
                    return 1;
                }
            };
            // Allreduce sum; divide by world_size to keep average.
            let t = t.all_reduce("sum");
            let t = t / ctx.world_size as f64;
            crate::update_tensor(h, t);
        }
        return 0;
    }
    #[cfg(not(feature = "torch"))]
    {
        let _ = handles_json;
        set_error("torch backend not enabled");
        1
    }
}

#[no_mangle]
pub extern "C" fn enkai_dist_shutdown() -> c_int {
    #[cfg(feature = "torch")]
    {
        let mut guard = DIST_STATE.lock().unwrap();
        *guard = None;
        return 0;
    }
    #[cfg(not(feature = "torch"))]
    {
        set_error("torch backend not enabled");
        1
    }
}
