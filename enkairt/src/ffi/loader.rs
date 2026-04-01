use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use libloading::Library;
use serde::Deserialize;

use enkaic::bytecode::NativeFunctionDecl;

use crate::error::RuntimeError;
use crate::ffi::native_fn::{requires_free, requires_handle_free, FfiFunction};

const FFI_ABI_VERSION: i64 = 1;
const E_FFI_LIBRARY_LOAD: &str = "E_FFI_LIBRARY_LOAD";
const E_FFI_SYMBOL_MISSING: &str = "E_FFI_SYMBOL_MISSING";
const E_FFI_ABI_VERSION: &str = "E_FFI_ABI_VERSION";
const E_FFI_SYMBOL_TABLE: &str = "E_FFI_SYMBOL_TABLE";
const E_FFI_FREE_MISSING: &str = "E_FFI_FREE_MISSING";
const E_FFI_HANDLE_FREE_MISSING: &str = "E_FFI_HANDLE_FREE_MISSING";

#[repr(C)]
struct FfiSlice {
    ptr: *mut u8,
    len: usize,
}

#[derive(Clone, Debug)]
struct LibraryAbiPolicy {
    abi_version: i64,
    exported_symbols: Option<HashSet<String>>,
}

#[derive(Debug, Deserialize)]
struct SymbolTableManifest {
    abi_version: i64,
    exports: Vec<String>,
}

pub struct FfiLoader {
    libraries: HashMap<String, Arc<Library>>,
    free_symbols: HashMap<String, *const c_void>,
    handle_free_symbols: HashMap<String, *const c_void>,
    abi_policies: HashMap<String, LibraryAbiPolicy>,
}

impl FfiLoader {
    pub fn new() -> Self {
        Self {
            libraries: HashMap::new(),
            free_symbols: HashMap::new(),
            handle_free_symbols: HashMap::new(),
            abi_policies: HashMap::new(),
        }
    }

    pub fn bind(&mut self, decl: &NativeFunctionDecl) -> Result<FfiFunction, RuntimeError> {
        self.ensure_abi_policy(&decl.library)?;
        let lib = self.load_library(&decl.library)?;
        let symbol = self.load_symbol(&decl.library, &decl.name)?;
        let free = if requires_free(&decl.signature) {
            let free_symbol = self.load_free_symbol(&decl.library)?;
            Some(free_symbol)
        } else {
            None
        };
        let handle_free = if requires_handle_free(&decl.signature) {
            Some(self.load_handle_free_symbol(&decl.library)?)
        } else {
            None
        };
        FfiFunction::new(
            decl.name.clone(),
            decl.signature.clone(),
            lib,
            symbol,
            free,
            handle_free,
        )
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
        Err(RuntimeError::with_code(
            E_FFI_LIBRARY_LOAD,
            &format!("Failed to load library '{}': {}", name, reason),
        ))
    }

    fn load_symbol(&mut self, lib_name: &str, symbol: &str) -> Result<*const c_void, RuntimeError> {
        self.ensure_declared_symbol(lib_name, symbol)?;
        let lib = self.load_library(lib_name)?;
        let symbol = unsafe {
            lib.get::<*const c_void>(symbol.as_bytes()).map_err(|err| {
                RuntimeError::with_code(
                    E_FFI_SYMBOL_MISSING,
                    &format!("Failed to resolve symbol '{}': {}", symbol, err),
                )
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
        Err(RuntimeError::with_code(
            E_FFI_FREE_MISSING,
            &format!(
                "enkai_free symbol not found for buffer/string return type: {}",
                reason
            ),
        ))
    }

    fn load_handle_free_symbol(&mut self, lib_name: &str) -> Result<*const c_void, RuntimeError> {
        if let Some(cached) = self.handle_free_symbols.get(lib_name) {
            return Ok(*cached);
        }
        let symbol = self
            .load_symbol(lib_name, "enkai_handle_free")
            .map_err(|err| {
                RuntimeError::with_code(
                    E_FFI_HANDLE_FREE_MISSING,
                    &format!(
                        "enkai_handle_free symbol missing for handle return type: {}",
                        err.message
                    ),
                )
            })?;
        self.handle_free_symbols
            .insert(lib_name.to_string(), symbol);
        Ok(symbol)
    }

    fn ensure_abi_policy(&mut self, lib_name: &str) -> Result<(), RuntimeError> {
        if self.abi_policies.contains_key(lib_name) {
            return Ok(());
        }
        let lib = self.load_library(lib_name)?;
        let abi_version_symbol =
            unsafe { lib.get::<unsafe extern "C" fn() -> i64>(b"enkai_abi_version") };
        let symbol_table_symbol =
            unsafe { lib.get::<unsafe extern "C" fn() -> FfiSlice>(b"enkai_symbol_table") };
        let policy = match (abi_version_symbol, symbol_table_symbol) {
            (Ok(version_fn), Ok(symbols_fn)) => {
                let version = unsafe { version_fn() };
                if version != FFI_ABI_VERSION {
                    return Err(RuntimeError::with_code(
                        E_FFI_ABI_VERSION,
                        &format!(
                            "FFI ABI version mismatch for '{}': expected {}, got {}",
                            lib_name, FFI_ABI_VERSION, version
                        ),
                    ));
                }
                let manifest = self.load_symbol_manifest(lib_name, unsafe { symbols_fn() })?;
                if manifest.abi_version != version {
                    return Err(RuntimeError::with_code(
                        E_FFI_SYMBOL_TABLE,
                        &format!(
                            "FFI symbol table version mismatch for '{}': expected {}, got {}",
                            lib_name, version, manifest.abi_version
                        ),
                    ));
                }
                let exported_symbols = if manifest.exports.iter().any(|item| item == "*") {
                    None
                } else {
                    Some(manifest.exports.into_iter().collect())
                };
                LibraryAbiPolicy {
                    abi_version: version,
                    exported_symbols,
                }
            }
            (Err(_), Err(_)) => LibraryAbiPolicy {
                abi_version: 0,
                exported_symbols: None,
            },
            _ => {
                return Err(RuntimeError::with_code(
                    E_FFI_SYMBOL_TABLE,
                    &format!(
                        "FFI library '{}' must expose both enkai_abi_version and enkai_symbol_table or neither",
                        lib_name
                    ),
                ))
            }
        };
        self.abi_policies.insert(lib_name.to_string(), policy);
        Ok(())
    }

    fn ensure_declared_symbol(&self, lib_name: &str, symbol: &str) -> Result<(), RuntimeError> {
        let Some(policy) = self.abi_policies.get(lib_name) else {
            return Ok(());
        };
        let Some(exports) = &policy.exported_symbols else {
            return Ok(());
        };
        if exports.contains(symbol) {
            return Ok(());
        }
        Err(RuntimeError::with_code(
            E_FFI_SYMBOL_TABLE,
            &format!(
                "FFI symbol '{}' is not declared by '{}' ABI manifest v{}",
                symbol, lib_name, policy.abi_version
            ),
        ))
    }

    fn load_symbol_manifest(
        &mut self,
        lib_name: &str,
        slice: FfiSlice,
    ) -> Result<SymbolTableManifest, RuntimeError> {
        if slice.ptr.is_null() {
            return Err(RuntimeError::with_code(
                E_FFI_SYMBOL_TABLE,
                &format!("FFI symbol table returned null for '{}'", lib_name),
            ));
        }
        let bytes = unsafe { std::slice::from_raw_parts(slice.ptr, slice.len) }.to_vec();
        let library = self.load_library(lib_name)?;
        let free = unsafe { library.get::<unsafe extern "C" fn(*mut u8, usize)>(b"enkai_free") }
            .map_err(|err| {
                RuntimeError::with_code(
                    E_FFI_FREE_MISSING,
                    &format!(
                        "enkai_free symbol not found while reading ABI manifest: {}",
                        err
                    ),
                )
            })?;
        unsafe { free(slice.ptr, slice.len) };
        let text = String::from_utf8(bytes).map_err(|_| {
            RuntimeError::with_code(
                E_FFI_SYMBOL_TABLE,
                &format!("FFI symbol table for '{}' returned invalid UTF-8", lib_name),
            )
        })?;
        let manifest = serde_json::from_str::<SymbolTableManifest>(&text).map_err(|err| {
            RuntimeError::with_code(
                E_FFI_SYMBOL_TABLE,
                &format!(
                    "Failed to parse FFI symbol table for '{}': {}",
                    lib_name, err
                ),
            )
        })?;
        Ok(manifest)
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
    if let Ok(cwd) = std::env::current_dir() {
        if dir_set.insert(cwd.clone()) {
            dirs.push(cwd);
        }
    }
    let mut out = Vec::new();
    let mut out_set = HashSet::new();
    for dir in dirs {
        for name in &names {
            let candidate = dir.join(name);
            if out_set.insert(candidate.clone()) {
                out.push(candidate);
            }
        }
    }
    for fallback in names {
        let candidate = PathBuf::from(fallback);
        if out_set.insert(candidate.clone()) {
            out.push(candidate);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_policy_allows_declared_symbol() {
        let mut loader = FfiLoader::new();
        loader.abi_policies.insert(
            "demo".to_string(),
            LibraryAbiPolicy {
                abi_version: FFI_ABI_VERSION,
                exported_symbols: Some(["ok_symbol".to_string()].into_iter().collect()),
            },
        );
        loader
            .ensure_declared_symbol("demo", "ok_symbol")
            .expect("declared");
    }

    #[test]
    fn abi_policy_rejects_undeclared_symbol() {
        let mut loader = FfiLoader::new();
        loader.abi_policies.insert(
            "demo".to_string(),
            LibraryAbiPolicy {
                abi_version: FFI_ABI_VERSION,
                exported_symbols: Some(["ok_symbol".to_string()].into_iter().collect()),
            },
        );
        let err = loader
            .ensure_declared_symbol("demo", "missing_symbol")
            .expect_err("missing");
        assert_eq!(err.code(), Some(E_FFI_SYMBOL_TABLE));
        assert!(err.message.contains("missing_symbol"));
    }

    #[test]
    fn abi_policy_wildcard_allows_all_symbols() {
        let mut loader = FfiLoader::new();
        loader.abi_policies.insert(
            "demo".to_string(),
            LibraryAbiPolicy {
                abi_version: FFI_ABI_VERSION,
                exported_symbols: None,
            },
        );
        loader
            .ensure_declared_symbol("demo", "anything")
            .expect("wildcard");
    }
}
