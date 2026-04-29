use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

pub(crate) enum SimExecution {
    Run(crate::sim::SimRunOptions),
    Replay(crate::sim::SimReplayOptions),
}

pub(crate) fn run_sim_execution_with_logs(
    execution: &SimExecution,
    child_env: &BTreeMap<String, String>,
    stdout_path: &Path,
    stderr_path: &Path,
) -> Result<i32, String> {
    let _guard = cluster_env_lock()
        .lock()
        .map_err(|_| "cluster env lock poisoned".to_string())?;
    let previous_values = child_env
        .keys()
        .map(|key| (key.clone(), env::var_os(key)))
        .collect::<Vec<_>>();
    for (key, value) in child_env {
        unsafe {
            env::set_var(key, value);
        }
    }
    let (result, note) = match execution {
        SimExecution::Run(options) => (
            crate::sim::execute_sim_run(options, None, None)?,
            format!(
                "in-process sim execution: run target={} output={}\nexit_code={{result}}\n",
                options.target.display(),
                options
                    .output
                    .as_deref()
                    .map(|path| path.display().to_string())
                    .unwrap_or_default()
            ),
        ),
        SimExecution::Replay(options) => (
            crate::sim::execute_sim_replay(options)?,
            format!(
                "in-process sim execution: replay snapshot={} steps={} output={}\nexit_code={{result}}\n",
                options.snapshot.display(),
                options.steps,
                options
                    .output
                    .as_deref()
                    .map(|path| path.display().to_string())
                    .unwrap_or_default()
            ),
        ),
    };
    if let Some(parent) = stdout_path.parent() {
        fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }
    let stdout_note = note.replace("{result}", &result.to_string());
    fs::write(stdout_path, stdout_note.as_bytes()).map_err(|err| err.to_string())?;
    fs::write(stderr_path, b"").map_err(|err| err.to_string())?;
    for (key, previous) in previous_values {
        unsafe {
            if let Some(value) = previous {
                env::set_var(&key, value);
            } else {
                env::remove_var(&key);
            }
        }
    }
    Ok(result)
}

pub(crate) fn should_inject_retry_failure(rank: usize, completed_windows: usize) -> bool {
    let target_rank = env::var("ENKAI_CLUSTER_INJECT_FAIL_ONCE_RANK")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok());
    let Some(target_rank) = target_rank else {
        return false;
    };
    if target_rank != rank {
        return false;
    }
    let after_windows = env::var("ENKAI_CLUSTER_INJECT_FAIL_ONCE_AFTER_WINDOWS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(0);
    completed_windows >= after_windows
}

fn cluster_env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}
