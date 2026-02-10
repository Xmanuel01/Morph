#![allow(dead_code)]

use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BackendKind {
    Torch = 1,
    Cpu = 2,
}

static BACKEND: AtomicU8 = AtomicU8::new(BackendKind::Torch as u8);

pub fn backend_is_torch() -> bool {
    BACKEND.load(Ordering::SeqCst) == BackendKind::Torch as u8
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
        _ => BackendKind::Torch,
    }
}
