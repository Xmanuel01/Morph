use libc::c_int;
use once_cell::sync::Lazy;
use std::sync::Mutex;
use tch::Device;

use crate::{
    clear_error, cstr_to_string, env_flag_enabled, get_tensor, parse_handle_list, set_error,
    update_tensor, DIST_DEVICE, DIST_SEED,
};
use std::sync::atomic::Ordering;

static DIST_STATE: Lazy<Mutex<Option<DistCtx>>> = Lazy::new(|| Mutex::new(None));

#[derive(Clone, Debug, PartialEq, Eq)]
struct DistCtx {
    world_size: i32,
    rank: i32,
    device: i32,
}

#[no_mangle]
pub extern "C" fn enkai_dist_init(world_size: i32, rank: i32) -> c_int {
    crate::ffi_guard(1, || {
        clear_error();
        if world_size <= 1 {
            return set_state(None);
        }
        if rank < 0 || rank >= world_size {
            set_error(format!(
                "invalid distributed init args: world_size={}, rank={}",
                world_size, rank
            ));
            return 1;
        }
        if !env_flag_enabled("ENKAI_ENABLE_DIST") {
            set_error("distributed init blocked: set ENKAI_ENABLE_DIST=1");
            return 1;
        }
        if !tch::Cuda::is_available() {
            set_error("distributed init requires CUDA; CUDA runtime is not available");
            return 1;
        }
        let cuda_count = tch::Cuda::device_count() as i32;
        if cuda_count < world_size {
            set_error(format!(
                "distributed init requires at least {} CUDA devices; found {}",
                world_size, cuda_count
            ));
            return 1;
        }

        if let Err(err) = check_env_rank_world(world_size, rank) {
            set_error(err);
            return 1;
        }

        let requested_device = DIST_DEVICE.load(Ordering::SeqCst);
        let device = if requested_device >= 0 {
            requested_device as i32
        } else {
            rank
        };
        if device < 0 || device >= cuda_count {
            set_error(format!(
                "distributed init invalid device mapping: device={} (available CUDA devices={})",
                device, cuda_count
            ));
            return 1;
        }
        if device != rank {
            set_error(format!(
                "distributed init rank/device mismatch: rank={} must map to cuda:{} (got cuda:{})",
                rank, rank, device
            ));
            return 1;
        }

        let target = DistCtx {
            world_size,
            rank,
            device,
        };
        let mut guard = match DIST_STATE.lock() {
            Ok(guard) => guard,
            Err(_) => {
                set_error("distributed state lock poisoned");
                return 1;
            }
        };
        if let Some(existing) = guard.as_ref() {
            if existing == &target {
                return 0;
            }
            set_error(format!(
                "distributed context already initialized (existing world_size={}, rank={}, device=cuda:{}); call enkai_dist_shutdown before reconfiguring",
                existing.world_size, existing.rank, existing.device
            ));
            return 1;
        }
        *guard = Some(target);
        let seed = DIST_SEED.load(Ordering::SeqCst).wrapping_add(rank as i64);
        tch::manual_seed(seed);
        0
    })
}

#[no_mangle]
pub extern "C" fn enkai_dist_allreduce_sum_multi(handles_json: *const libc::c_char) -> c_int {
    crate::ffi_guard(1, || {
        clear_error();
        let ctx = match current_ctx() {
            Ok(ctx) => ctx,
            Err(err) => {
                set_error(err);
                return 1;
            }
        };
        if !env_flag_enabled("ENKAI_ENABLE_DIST") {
            set_error("distributed allreduce blocked: set ENKAI_ENABLE_DIST=1");
            return 1;
        }
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

        for (idx, handle) in handles.into_iter().enumerate() {
            if handle == 0 {
                continue;
            }
            let tensor = match get_tensor(handle) {
                Ok(tensor) => tensor,
                Err(err) => {
                    set_error(format!(
                        "distributed allreduce invalid tensor handle at index {}: {}",
                        idx, err
                    ));
                    return 1;
                }
            };
            match tensor.device() {
                Device::Cuda(dev) if dev as i32 == ctx.device => {}
                Device::Cuda(dev) => {
                    set_error(format!(
                        "distributed allreduce rank/device mismatch for handle {}: rank {} expected cuda:{}, got cuda:{}",
                        handle, ctx.rank, ctx.device, dev
                    ));
                    return 1;
                }
                other => {
                    set_error(format!(
                        "distributed allreduce requires CUDA tensors in multi-rank mode (handle {} on {:?})",
                        handle, other
                    ));
                    return 1;
                }
            }
            let reduced = tensor.all_reduce("sum") / (ctx.world_size as f64);
            update_tensor(handle, reduced);
        }
        0
    })
}

#[no_mangle]
pub extern "C" fn enkai_dist_shutdown() -> c_int {
    crate::ffi_guard(1, || {
        clear_error();
        set_state(None)
    })
}

fn set_state(next: Option<DistCtx>) -> c_int {
    let mut guard = match DIST_STATE.lock() {
        Ok(guard) => guard,
        Err(_) => {
            set_error("distributed state lock poisoned");
            return 1;
        }
    };
    *guard = next;
    0
}

fn current_ctx() -> Result<DistCtx, String> {
    let guard = DIST_STATE
        .lock()
        .map_err(|_| "distributed state lock poisoned".to_string())?;
    guard
        .as_ref()
        .cloned()
        .ok_or_else(|| "dist not initialized".to_string())
}

fn check_env_rank_world(world_size: i32, rank: i32) -> Result<(), String> {
    if let Some(env_world) = env_i32("WORLD_SIZE")? {
        if env_world != world_size {
            return Err(format!(
                "distributed init mismatch: WORLD_SIZE={} but config world_size={}",
                env_world, world_size
            ));
        }
    }
    if let Some(env_rank) = env_i32("RANK")? {
        if env_rank != rank {
            return Err(format!(
                "distributed init mismatch: RANK={} but config rank={}",
                env_rank, rank
            ));
        }
    }
    Ok(())
}

fn env_i32(name: &str) -> Result<Option<i32>, String> {
    let value = match std::env::var(name) {
        Ok(value) => value,
        Err(std::env::VarError::NotPresent) => return Ok(None),
        Err(_) => return Err(format!("failed to read environment variable {}", name)),
    };
    let parsed = value.trim().parse::<i32>().map_err(|_| {
        format!(
            "invalid {} value {:?}: expected signed integer",
            name, value
        )
    })?;
    Ok(Some(parsed))
}
