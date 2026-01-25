use std::collections::HashMap;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use libloading::Library;

use morphc::bytecode::NativeFunctionDecl;

use crate::error::RuntimeError;
use crate::ffi::native_fn::{requires_free, FfiFunction};

pub struct FfiLoader {
    libraries: HashMap<String, Arc<Library>>,
    free_symbols: HashMap<String, Option<*const c_void>>,
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
            return cached.ok_or_else(|| {
                RuntimeError::new("morph_free symbol not found for buffer/string return type")
            });
        }
        let symbol = self.load_symbol(lib_name, "morph_free").map_err(|_| {
            RuntimeError::new("morph_free symbol not found for buffer/string return type")
        })?;
        self.free_symbols.insert(lib_name.to_string(), Some(symbol));
        Ok(symbol)
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
    let mut dirs = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        dirs.push(cwd);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            dirs.push(dir.to_path_buf());
            if let Some(parent) = dir.parent() {
                dirs.push(parent.to_path_buf());
            }
        }
    }
    let mut out = Vec::new();
    out.push(PathBuf::from(name));
    for dir in dirs {
        for name in &names {
            out.push(dir.join(name));
        }
    }
    out
}
