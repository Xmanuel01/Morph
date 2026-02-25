use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use libloading::Library;

use enkaic::bytecode::NativeFunctionDecl;

use crate::error::RuntimeError;
use crate::ffi::native_fn::{requires_free, FfiFunction};

pub struct FfiLoader {
    libraries: HashMap<String, Arc<Library>>,
    free_symbols: HashMap<String, *const c_void>,
}

impl FfiLoader {
    pub fn new() -> Self {
        Self {
            libraries: HashMap::new(),
            free_symbols: HashMap::new(),
        }
    }

    pub fn bind(&mut self, decl: &NativeFunctionDecl) -> Result<FfiFunction, RuntimeError> {
        let lib = self.load_library(&decl.library)?;
        let symbol = self.load_symbol(&decl.library, &decl.name)?;
        let free = if requires_free(&decl.signature) {
            let free_symbol = self.load_free_symbol(&decl.library)?;
            Some(free_symbol)
        } else {
            None
        };
        FfiFunction::new(decl.name.clone(), decl.signature.clone(), lib, symbol, free)
    }

    fn load_library(&mut self, name: &str) -> Result<Arc<Library>, RuntimeError> {
        if let Some(lib) = self.libraries.get(name) {
            return Ok(lib.clone());
        }
        let mut last_err: Option<String> = None;
        for path in library_candidates(name) {
            match unsafe { Library::new(&path) } {
                Ok(lib) => {
                    let lib = Arc::new(lib);
                    self.libraries.insert(name.to_string(), lib.clone());
                    return Ok(lib);
                }
                Err(err) => last_err = Some(err.to_string()),
            }
        }
        let reason = last_err.unwrap_or_else(|| "not found".to_string());
        Err(RuntimeError::new(&format!(
            "Failed to load library '{}': {}",
            name, reason
        )))
    }

    fn load_symbol(&mut self, lib_name: &str, symbol: &str) -> Result<*const c_void, RuntimeError> {
        let lib = self.load_library(lib_name)?;
        let symbol = unsafe {
            lib.get::<*const c_void>(symbol.as_bytes()).map_err(|err| {
                RuntimeError::new(&format!("Failed to resolve symbol '{}': {}", symbol, err))
            })?
        };
        Ok(*symbol)
    }

    fn load_free_symbol(&mut self, lib_name: &str) -> Result<*const c_void, RuntimeError> {
        if let Some(cached) = self.free_symbols.get(lib_name) {
            return Ok(*cached);
        }
        let mut last_err: Option<String> = None;
        for symbol_name in ["enkai_free"] {
            match self.load_symbol(lib_name, symbol_name) {
                Ok(symbol) => {
                    self.free_symbols.insert(lib_name.to_string(), symbol);
                    return Ok(symbol);
                }
                Err(err) => last_err = Some(err.message),
            }
        }
        let reason = last_err.unwrap_or_else(|| "symbol not found".to_string());
        Err(RuntimeError::new(&format!(
            "enkai_free symbol not found for buffer/string return type: {}",
            reason
        )))
    }
}

impl Default for FfiLoader {
    fn default() -> Self {
        Self::new()
    }
}

fn library_candidates(name: &str) -> Vec<PathBuf> {
    let path = Path::new(name);
    if path.is_absolute() || name.contains('\\') || name.contains('/') {
        return vec![path.to_path_buf()];
    }
    let mut names = Vec::new();
    names.push(name.to_string());
    let ext = if cfg!(target_os = "windows") {
        "dll"
    } else if cfg!(target_os = "macos") {
        "dylib"
    } else {
        "so"
    };
    if path.extension().is_none() {
        names.push(format!("{}.{}", name, ext));
        if !cfg!(target_os = "windows") && !name.starts_with("lib") {
            names.push(format!("lib{}.{}", name, ext));
        }
    }
    let mut unique_names = HashSet::new();
    names.retain(|entry| unique_names.insert(entry.clone()));
    let mut dirs = Vec::new();
    let mut dir_set = HashSet::new();
    if let Ok(cwd) = std::env::current_dir() {
        if dir_set.insert(cwd.clone()) {
            dirs.push(cwd);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let dir_path = dir.to_path_buf();
            if dir_set.insert(dir_path.clone()) {
                dirs.push(dir_path);
            }
            if let Some(parent) = dir.parent() {
                let parent_path = parent.to_path_buf();
                if dir_set.insert(parent_path.clone()) {
                    dirs.push(parent_path);
                }
            }
        }
    }
    let mut out = Vec::new();
    let mut out_set = HashSet::new();
    let root = PathBuf::from(name);
    out_set.insert(root.clone());
    out.push(root);
    for dir in dirs {
        for name in &names {
            let candidate = dir.join(name);
            if out_set.insert(candidate.clone()) {
                out.push(candidate);
            }
        }
    }
    out
}
