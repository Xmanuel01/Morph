use std::collections::BTreeMap;
use std::env;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use enkai_compiler::bytecode::Program;
use enkai_runtime::{Value, VM};

use crate::{bootstrap, emit_command_backend_report, load_cached_program, resolve_entry};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RuntimeBackendMode {
    Auto,
    Selfhost,
    Rust,
}

#[derive(Debug, Clone)]
pub(crate) struct RuntimeExecOptions {
    pub(crate) trace_vm: bool,
    pub(crate) disasm: bool,
    pub(crate) trace_task: bool,
    pub(crate) trace_net: bool,
    pub(crate) runtime_backend: RuntimeBackendMode,
    pub(crate) report_command: Option<&'static str>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ScopedEnv {
    pub(crate) vars: BTreeMap<String, String>,
    pub(crate) std_override: Option<PathBuf>,
}

pub(crate) fn execute_runtime_target(
    target: &Path,
    options: RuntimeExecOptions,
) -> Result<i32, String> {
    let (root, entry) = resolve_entry(target)?;
    execute_runtime_entry(&root, &entry, options)
}

pub(crate) fn execute_runtime_target_with_env(
    target: &Path,
    options: RuntimeExecOptions,
    env_scope: &ScopedEnv,
) -> Result<i32, String> {
    with_scoped_env(env_scope, || execute_runtime_target(target, options))
}

pub(crate) fn execute_runtime_entry(
    root: &Path,
    entry: &Path,
    options: RuntimeExecOptions,
) -> Result<i32, String> {
    if options.runtime_backend != RuntimeBackendMode::Rust {
        if options.trace_vm || options.disasm || options.trace_task || options.trace_net {
            if options.runtime_backend == RuntimeBackendMode::Selfhost {
                return Err(
                    "enkai run --runtime-backend selfhost does not support --trace-vm/--disasm/--trace-task/--trace-net"
                        .to_string(),
                );
            }
        } else {
            match bootstrap::try_run_selfhost_entry(entry) {
                Ok(Some(outcome)) => {
                    if let Some(command) = options.report_command {
                        emit_command_backend_report(command, entry, root, outcome.backend);
                    }
                    return Ok(outcome.exit_code);
                }
                Ok(None) => {
                    if options.runtime_backend == RuntimeBackendMode::Selfhost {
                        return Err(format!(
                            "enkai run --runtime-backend selfhost could not execute {} on the self-host runtime path",
                            entry.display()
                        ));
                    }
                }
                Err(err) => {
                    if options.runtime_backend == RuntimeBackendMode::Selfhost {
                        return Err(format!("enkai run selfhost runtime failed: {}", err));
                    }
                }
            }
        }
    }

    let program = match load_cached_program(root, entry) {
        Ok(Some(program)) => program,
        Ok(None) => compile_program_prefer_selfhost(entry)?.0,
        Err(err) => {
            eprintln!("cache disabled: {}", err);
            compile_program_prefer_selfhost(entry)?.0
        }
    };
    let mut vm = VM::new(
        options.trace_vm,
        options.disasm,
        options.trace_task,
        options.trace_net,
    );
    if let Some(command) = options.report_command {
        emit_command_backend_report(command, entry, root, bootstrap::SelfhostRunBackend::Rust);
    }
    match vm.run(&program) {
        Ok(Value::Int(code)) => Ok(code as i32),
        Ok(_) => Ok(0),
        Err(err) => Err(format!("Runtime error: {}", err)),
    }
}

pub(crate) fn compile_program_prefer_selfhost(
    entry: &Path,
) -> Result<(Program, bootstrap::SelfhostRunBackend), String> {
    match bootstrap::try_compile_selfhost_program(entry) {
        Ok(Some(compiled)) => Ok((compiled.program, compiled.backend)),
        Ok(None) | Err(_) => compile_program_with_rust_backend(entry),
    }
}

fn compile_program_with_rust_backend(
    entry: &Path,
) -> Result<(Program, bootstrap::SelfhostRunBackend), String> {
    let package = enkai_compiler::modules::load_package(entry).map_err(|err| err.to_string())?;
    enkai_compiler::TypeChecker::check_package(&package).map_err(crate::type_error_message)?;
    let program = enkai_compiler::compiler::compile_package(&package)
        .map_err(crate::compile_error_message)?;
    Ok((program, bootstrap::SelfhostRunBackend::Rust))
}

pub(crate) fn default_options() -> RuntimeExecOptions {
    RuntimeExecOptions {
        trace_vm: false,
        disasm: false,
        trace_task: false,
        trace_net: false,
        runtime_backend: RuntimeBackendMode::Auto,
        report_command: None,
    }
}

pub(crate) fn bundled_std_override() -> Option<PathBuf> {
    if env::var_os("ENKAI_STD").is_some() {
        return None;
    }
    let bundled_std = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")))
        .join("std");
    bundled_std.is_dir().then_some(bundled_std)
}

pub(crate) fn with_scoped_env<T, F>(env_scope: &ScopedEnv, run: F) -> Result<T, String>
where
    F: FnOnce() -> Result<T, String>,
{
    let _guard = scoped_env_lock()
        .lock()
        .map_err(|_| "runtime env lock poisoned".to_string())?;
    let previous_values = env_scope
        .vars
        .keys()
        .map(|key| (key.clone(), env::var_os(key)))
        .collect::<Vec<_>>();
    let previous_std = if env_scope.std_override.is_some() {
        Some(env::var_os("ENKAI_STD"))
    } else {
        None
    };
    for (key, value) in &env_scope.vars {
        unsafe {
            env::set_var(key, value);
        }
    }
    if let Some(std_path) = &env_scope.std_override {
        unsafe {
            env::set_var("ENKAI_STD", std_path);
        }
    }
    let result = run();
    for (key, previous) in previous_values {
        restore_env_var(&key, previous);
    }
    if let Some(previous_std) = previous_std {
        restore_env_var("ENKAI_STD", previous_std);
    }
    result
}

fn scoped_env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn restore_env_var(key: &str, value: Option<std::ffi::OsString>) {
    unsafe {
        if let Some(value) = value {
            env::set_var(key, value);
        } else {
            env::remove_var(key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scoped_env_restores_previous_values() {
        let key = "ENKAI_RUNTIME_EXEC_TEST";
        unsafe {
            std::env::set_var(key, "before");
        }
        let mut vars = BTreeMap::new();
        vars.insert(key.to_string(), "during".to_string());
        let scope = ScopedEnv {
            vars,
            std_override: None,
        };
        let observed = with_scoped_env(&scope, || Ok(std::env::var(key).expect("scoped value")))
            .expect("scope");
        assert_eq!(observed, "during");
        assert_eq!(std::env::var(key).as_deref(), Ok("before"));
        unsafe {
            std::env::remove_var(key);
        }
    }
}
