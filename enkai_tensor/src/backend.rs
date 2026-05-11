#![allow(dead_code)]

use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BackendKind {
    Torch = 1,
    Cpu = 2,
    Cuda = 3,
    Rocm = 4,
    Metal = 5,
}

static BACKEND: AtomicU8 = AtomicU8::new(BackendKind::Torch as u8);

pub fn backend_is_torch() -> bool {
    matches!(
        current_backend(),
        BackendKind::Torch | BackendKind::Cuda | BackendKind::Rocm | BackendKind::Metal
    )
}

pub fn backend_is_cpu() -> bool {
    BACKEND.load(Ordering::SeqCst) == BackendKind::Cpu as u8
}

pub fn set_backend(kind: BackendKind) {
    BACKEND.store(kind as u8, Ordering::SeqCst);
}

pub fn current_backend() -> BackendKind {
    match BACKEND.load(Ordering::SeqCst) {
        x if x == BackendKind::Torch as u8 => BackendKind::Torch,
        x if x == BackendKind::Cpu as u8 => BackendKind::Cpu,
        x if x == BackendKind::Cuda as u8 => BackendKind::Cuda,
        x if x == BackendKind::Rocm as u8 => BackendKind::Rocm,
        x if x == BackendKind::Metal as u8 => BackendKind::Metal,
        _ => BackendKind::Torch,
    }
}
