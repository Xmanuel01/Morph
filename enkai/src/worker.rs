use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use enkai_compiler::compiler::compile_package;
use enkai_compiler::modules::load_package;
use enkai_compiler::TypeChecker;
use enkai_runtime::{Value, VM};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QueueMessage {
    schema_version: u32,
    id: String,
    queue: String,
    payload: serde_json::Value,
    tenant: Option<String>,
    attempts: u32,
    max_attempts: u32,
    enqueued_ms: u64,
    last_error: Option<String>,
}

#[derive(Debug, Clone)]
enum WorkerCommand {
    Enqueue(EnqueueArgs),
    Run(RunArgs),
}

#[derive(Debug, Clone)]
struct EnqueueArgs {
    dir: PathBuf,
    queue: String,
    payload: serde_json::Value,
    tenant: Option<String>,
    id: Option<String>,
    max_attempts: u32,
    json: bool,
    output: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct RunArgs {
    dir: PathBuf,
    queue: String,
    handler: PathBuf,
    once: bool,
    json: bool,
    output: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize)]
struct WorkerRunReport {
    schema_version: u32,
    queue: String,
    status: String,
    processed: usize,
    acked: usize,
    requeued: usize,
    dead_lettered: usize,
    last_message_id: Option<String>,
    last_attempt: Option<u32>,
    dead_letter_path: String,
    pending_path: String,
}

#[derive(Debug, Clone)]
enum WorkerOutcome {
    Acked(QueueMessage),
    Requeued(QueueMessage),
    DeadLettered(QueueMessage),
}

pub fn print_worker_usage() {
    eprintln!("  enkai worker enqueue --queue <name> --dir <state_dir> --payload <json> [--tenant <tenant>] [--id <id>] [--max-attempts <n>] [--json] [--output <file>]");
    eprintln!("  enkai worker run --queue <name> --dir <state_dir> --handler <file.enk> [--once] [--json] [--output <file>]");
}

pub fn worker_command(args: &[String]) -> i32 {
    let parsed = match parse_worker_args(args) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("enkai worker: {}", err);
            print_worker_usage();
            return 1;
        }
    };
    match parsed {
        WorkerCommand::Enqueue(options) => enqueue_command(&options),
        WorkerCommand::Run(options) => run_command(&options),
    }
}

fn parse_worker_args(args: &[String]) -> Result<WorkerCommand, String> {
    let Some(subcommand) = args.first().map(|value| value.as_str()) else {
        return Err("missing subcommand (enqueue|run)".to_string());
    };
    match subcommand {
        "enqueue" => parse_enqueue_args(&args[1..]).map(WorkerCommand::Enqueue),
        "run" => parse_run_args(&args[1..]).map(WorkerCommand::Run),
        _ => Err(format!(
            "unknown worker subcommand '{}'; expected enqueue|run",
            subcommand
        )),
    }
}

fn parse_enqueue_args(args: &[String]) -> Result<EnqueueArgs, String> {
    let mut dir: Option<PathBuf> = None;
    let mut queue: Option<String> = None;
    let mut payload: Option<serde_json::Value> = None;
    let mut tenant: Option<String> = None;
    let mut id: Option<String> = None;
    let mut max_attempts = 3u32;
    let mut json = false;
    let mut output: Option<PathBuf> = None;
    let mut idx = 0usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--dir" => {
                idx += 1;
                dir = Some(PathBuf::from(
                    args.get(idx)
                        .ok_or_else(|| "--dir requires a value".to_string())?,
                ));
            }
            "--queue" => {
                idx += 1;
                queue = Some(
                    args.get(idx)
                        .ok_or_else(|| "--queue requires a value".to_string())?
                        .trim()
                        .to_string(),
                );
            }
            "--payload" => {
                idx += 1;
                let raw = args
                    .get(idx)
                    .ok_or_else(|| "--payload requires a value".to_string())?;
                payload = Some(
                    serde_json::from_str(raw)
                        .map_err(|err| format!("--payload must be valid JSON: {}", err))?,
                );
            }
            "--tenant" => {
                idx += 1;
                tenant = Some(
                    args.get(idx)
                        .ok_or_else(|| "--tenant requires a value".to_string())?
                        .to_string(),
                );
            }
            "--id" => {
                idx += 1;
                id = Some(
                    args.get(idx)
                        .ok_or_else(|| "--id requires a value".to_string())?
                        .to_string(),
                );
            }
            "--max-attempts" => {
                idx += 1;
                max_attempts = args
                    .get(idx)
                    .ok_or_else(|| "--max-attempts requires a value".to_string())?
                    .parse::<u32>()
                    .map_err(|_| "--max-attempts must be an integer".to_string())?;
                if max_attempts == 0 {
                    return Err("--max-attempts must be >= 1".to_string());
                }
            }
            "--json" => json = true,
            "--output" => {
                idx += 1;
                output = Some(PathBuf::from(
                    args.get(idx)
                        .ok_or_else(|| "--output requires a value".to_string())?,
                ));
            }
            other => return Err(format!("unknown option '{}'", other)),
        }
        idx += 1;
    }
    let dir = dir.ok_or_else(|| "--dir is required".to_string())?;
    let queue = queue.ok_or_else(|| "--queue is required".to_string())?;
    let payload = payload.ok_or_else(|| "--payload is required".to_string())?;
    Ok(EnqueueArgs {
        dir,
        queue,
        payload,
        tenant,
        id,
        max_attempts,
        json,
        output,
    })
}

fn parse_run_args(args: &[String]) -> Result<RunArgs, String> {
    let mut dir: Option<PathBuf> = None;
    let mut queue: Option<String> = None;
    let mut handler: Option<PathBuf> = None;
    let mut once = false;
    let mut json = false;
    let mut output: Option<PathBuf> = None;
    let mut idx = 0usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--dir" => {
                idx += 1;
                dir = Some(PathBuf::from(
                    args.get(idx)
                        .ok_or_else(|| "--dir requires a value".to_string())?,
                ));
            }
            "--queue" => {
                idx += 1;
                queue = Some(
                    args.get(idx)
                        .ok_or_else(|| "--queue requires a value".to_string())?
                        .trim()
                        .to_string(),
                );
            }
            "--handler" => {
                idx += 1;
                handler = Some(PathBuf::from(
                    args.get(idx)
                        .ok_or_else(|| "--handler requires a value".to_string())?,
                ));
            }
            "--once" => once = true,
            "--json" => json = true,
            "--output" => {
                idx += 1;
                output = Some(PathBuf::from(
                    args.get(idx)
                        .ok_or_else(|| "--output requires a value".to_string())?,
                ));
            }
            other => return Err(format!("unknown option '{}'", other)),
        }
        idx += 1;
    }
    Ok(RunArgs {
        dir: dir.ok_or_else(|| "--dir is required".to_string())?,
        queue: queue.ok_or_else(|| "--queue is required".to_string())?,
        handler: handler.ok_or_else(|| "--handler is required".to_string())?,
        once,
        json,
        output,
    })
}

fn enqueue_command(args: &EnqueueArgs) -> i32 {
    if let Err(err) = fs::create_dir_all(queue_root(&args.dir, &args.queue)) {
        eprintln!("enkai worker enqueue: failed to create queue dir: {}", err);
        return 1;
    }
    let message = QueueMessage {
        schema_version: 1,
        id: args.id.clone().unwrap_or_else(generate_message_id),
        queue: args.queue.clone(),
        payload: args.payload.clone(),
        tenant: args.tenant.clone(),
        attempts: 0,
        max_attempts: args.max_attempts,
        enqueued_ms: now_ms(),
        last_error: None,
    };
    if let Err(err) = append_jsonl(&pending_path(&args.dir, &args.queue), &message) {
        eprintln!("enkai worker enqueue: {}", err);
        return 1;
    }
    let payload = serde_json::json!({
        "schema_version": 1,
        "status": "queued",
        "queue": args.queue,
        "id": message.id,
        "max_attempts": message.max_attempts,
        "pending_path": pending_path(&args.dir, &args.queue),
    });
    if let Err(err) = emit_json(&payload, args.json, args.output.as_deref()) {
        eprintln!("enkai worker enqueue: {}", err);
        return 1;
    }
    if !args.json {
        println!("queued message {} on {}", message.id, args.queue);
    }
    0
}

fn run_command(args: &RunArgs) -> i32 {
    let report = match run_worker_once(args) {
        Ok(report) => report,
        Err(err) => {
            eprintln!("enkai worker run: {}", err);
            return 1;
        }
    };
    if let Err(err) = emit_json(&report, args.json, args.output.as_deref()) {
        eprintln!("enkai worker run: {}", err);
        return 1;
    }
    if !args.json {
        println!(
            "worker queue={} status={} acked={} requeued={} dead_lettered={}",
            report.queue, report.status, report.acked, report.requeued, report.dead_lettered
        );
    }
    if report.status == "failed" {
        1
    } else {
        0
    }
}

fn run_worker_once(args: &RunArgs) -> Result<WorkerRunReport, String> {
    let pending_file = pending_path(&args.dir, &args.queue);
    let dead_letter_file = dead_letter_path(&args.dir, &args.queue);
    let mut pending = load_jsonl::<QueueMessage>(&pending_file)?;
    if pending.is_empty() {
        return Ok(WorkerRunReport {
            schema_version: 1,
            queue: args.queue.clone(),
            status: "idle".to_string(),
            processed: 0,
            acked: 0,
            requeued: 0,
            dead_lettered: 0,
            last_message_id: None,
            last_attempt: None,
            dead_letter_path: dead_letter_file.display().to_string(),
            pending_path: pending_file.display().to_string(),
        });
    }

    let mut message = pending.remove(0);
    message.attempts = message.attempts.saturating_add(1);
    let exit_code = run_handler(&args.handler, &message)?;
    let mut status = "acked".to_string();
    let mut acked = 1usize;
    let mut requeued = 0usize;
    let mut dead_lettered = 0usize;
    match apply_worker_exit(message.clone(), exit_code) {
        WorkerOutcome::Acked(next) => {
            message = next;
        }
        WorkerOutcome::Requeued(next) => {
            message = next.clone();
            pending.push(next);
            status = "requeued".to_string();
            acked = 0;
            requeued = 1;
        }
        WorkerOutcome::DeadLettered(next) => {
            message = next.clone();
            append_jsonl(&dead_letter_file, &next)?;
            status = "dead_lettered".to_string();
            acked = 0;
            dead_lettered = 1;
        }
    }
    write_jsonl(&pending_file, &pending)?;

    if !args.once {
        while !load_jsonl::<QueueMessage>(&pending_file)?.is_empty() {
            let _ = run_worker_once(&RunArgs {
                once: true,
                ..args.clone()
            })?;
        }
    }

    Ok(WorkerRunReport {
        schema_version: 1,
        queue: args.queue.clone(),
        status,
        processed: 1,
        acked,
        requeued,
        dead_lettered,
        last_message_id: Some(message.id),
        last_attempt: Some(message.attempts),
        dead_letter_path: dead_letter_file.display().to_string(),
        pending_path: pending_file.display().to_string(),
    })
}

fn apply_worker_exit(mut message: QueueMessage, exit_code: i32) -> WorkerOutcome {
    if exit_code == 0 {
        return WorkerOutcome::Acked(message);
    }
    message.last_error = Some(format!("handler exited with {}", exit_code));
    if message.attempts < message.max_attempts {
        WorkerOutcome::Requeued(message)
    } else {
        WorkerOutcome::DeadLettered(message)
    }
}

fn run_handler(handler: &Path, message: &QueueMessage) -> Result<i32, String> {
    if !handler.is_file() {
        return Err(format!("handler not found: {}", handler.display()));
    }
    if let Some(exe) = resolve_worker_exe()? {
        let mut command = Command::new(exe);
        command.arg("run").arg(handler);
        populate_worker_env(&mut command, message)?;
        let status = command
            .status()
            .map_err(|err| format!("failed to spawn handler {}: {}", handler.display(), err))?;
        return Ok(status.code().unwrap_or(1));
    }
    run_handler_in_process(handler, message)
}

fn populate_worker_env(command: &mut Command, message: &QueueMessage) -> Result<(), String> {
    command.env(
        "ENKAI_WORKER_PAYLOAD",
        serde_json::to_string(&message.payload).map_err(|err| err.to_string())?,
    );
    command.env("ENKAI_WORKER_ID", &message.id);
    command.env("ENKAI_WORKER_QUEUE", &message.queue);
    command.env("ENKAI_WORKER_ATTEMPT", message.attempts.to_string());
    command.env(
        "ENKAI_WORKER_MAX_ATTEMPTS",
        message.max_attempts.to_string(),
    );
    if env::var_os("ENKAI_STD").is_none() {
        let bundled_std = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")))
            .join("std");
        if bundled_std.is_dir() {
            command.env("ENKAI_STD", bundled_std);
        }
    }
    if let Some(tenant) = &message.tenant {
        command.env("ENKAI_WORKER_TENANT", tenant);
    }
    Ok(())
}

fn resolve_worker_exe() -> Result<Option<PathBuf>, String> {
    if let Some(value) = env::var_os("ENKAI_WORKER_EXE") {
        return Ok(Some(PathBuf::from(value)));
    }
    let exe = env::current_exe().map_err(|err| format!("current exe: {}", err))?;
    let Some(parent) = exe.parent() else {
        return Ok(None);
    };
    if parent.file_name().and_then(|value| value.to_str()) == Some("deps") {
        let base_dir = parent
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| parent.to_path_buf());
        let mut candidates = Vec::new();
        if cfg!(windows) {
            candidates.push(base_dir.join("enkai.exe"));
        }
        candidates.push(base_dir.join("enkai"));
        for candidate in candidates {
            if candidate.is_file() {
                return Ok(Some(candidate));
            }
        }
        return Ok(None);
    }
    Ok(Some(exe))
}

fn worker_env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn run_handler_in_process(handler: &Path, message: &QueueMessage) -> Result<i32, String> {
    let _guard = worker_env_lock()
        .lock()
        .map_err(|_| "worker env lock poisoned".to_string())?;
    let payload = serde_json::to_string(&message.payload).map_err(|err| err.to_string())?;
    let std_override = if env::var_os("ENKAI_STD").is_none() {
        let bundled_std = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")))
            .join("std");
        bundled_std.is_dir().then_some(bundled_std)
    } else {
        None
    };
    let previous_payload = env::var_os("ENKAI_WORKER_PAYLOAD");
    let previous_id = env::var_os("ENKAI_WORKER_ID");
    let previous_queue = env::var_os("ENKAI_WORKER_QUEUE");
    let previous_attempt = env::var_os("ENKAI_WORKER_ATTEMPT");
    let previous_max_attempts = env::var_os("ENKAI_WORKER_MAX_ATTEMPTS");
    let previous_tenant = env::var_os("ENKAI_WORKER_TENANT");
    let previous_std = if std_override.is_some() {
        env::var_os("ENKAI_STD")
    } else {
        None
    };
    let result = {
        unsafe {
            env::set_var("ENKAI_WORKER_PAYLOAD", payload);
            env::set_var("ENKAI_WORKER_ID", &message.id);
            env::set_var("ENKAI_WORKER_QUEUE", &message.queue);
            env::set_var("ENKAI_WORKER_ATTEMPT", message.attempts.to_string());
            env::set_var(
                "ENKAI_WORKER_MAX_ATTEMPTS",
                message.max_attempts.to_string(),
            );
            if let Some(tenant) = &message.tenant {
                env::set_var("ENKAI_WORKER_TENANT", tenant);
            } else {
                env::remove_var("ENKAI_WORKER_TENANT");
            }
            if let Some(std_path) = &std_override {
                env::set_var("ENKAI_STD", std_path);
            }
        }
        let package = load_package(handler).map_err(|err| err.to_string())?;
        TypeChecker::check_package(&package).map_err(|err| format!("{:?}", err))?;
        let program = compile_package(&package).map_err(|err| format!("{:?}", err))?;
        let mut vm = VM::new(false, false, false, false);
        match vm.run(&program) {
            Ok(Value::Int(code)) => Ok(code as i32),
            Ok(_) => Ok(0),
            Err(err) => Err(err.to_string()),
        }
    };
    unsafe {
        restore_env_var("ENKAI_WORKER_PAYLOAD", previous_payload);
        restore_env_var("ENKAI_WORKER_ID", previous_id);
        restore_env_var("ENKAI_WORKER_QUEUE", previous_queue);
        restore_env_var("ENKAI_WORKER_ATTEMPT", previous_attempt);
        restore_env_var("ENKAI_WORKER_MAX_ATTEMPTS", previous_max_attempts);
        restore_env_var("ENKAI_WORKER_TENANT", previous_tenant);
        if std_override.is_some() {
            restore_env_var("ENKAI_STD", previous_std);
        }
    }
    result
}

unsafe fn restore_env_var(key: &str, value: Option<std::ffi::OsString>) {
    if let Some(value) = value {
        env::set_var(key, value);
    } else {
        env::remove_var(key);
    }
}

fn queue_root(dir: &Path, queue: &str) -> PathBuf {
    dir.join("queues").join(queue)
}

fn pending_path(dir: &Path, queue: &str) -> PathBuf {
    queue_root(dir, queue).join("pending.jsonl")
}

fn dead_letter_path(dir: &Path, queue: &str) -> PathBuf {
    queue_root(dir, queue).join("dead_letter.jsonl")
}

fn load_jsonl<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Vec<T>, String> {
    if !path.is_file() {
        return Ok(Vec::new());
    }
    let text = fs::read_to_string(path)
        .map_err(|err| format!("failed to read {}: {}", path.display(), err))?;
    let mut out = Vec::new();
    for (line_no, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value = serde_json::from_str(trimmed).map_err(|err| {
            format!(
                "failed to parse {} line {}: {}",
                path.display(),
                line_no + 1,
                err
            )
        })?;
        out.push(value);
    }
    Ok(out)
}

fn write_jsonl<T: Serialize>(path: &Path, items: &[T]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create {}: {}", parent.display(), err))?;
    }
    let mut lines = Vec::with_capacity(items.len());
    for item in items {
        lines.push(serde_json::to_string(item).map_err(|err| err.to_string())?);
    }
    fs::write(path, lines.join("\n"))
        .map_err(|err| format!("failed to write {}: {}", path.display(), err))
}

fn append_jsonl<T: Serialize>(path: &Path, item: &T) -> Result<(), String> {
    let mut items = load_jsonl::<serde_json::Value>(path)?;
    items.push(serde_json::to_value(item).map_err(|err| err.to_string())?);
    write_jsonl(path, &items)
}

fn emit_json<T: Serialize>(
    payload: &T,
    emit_stdout_json: bool,
    output: Option<&Path>,
) -> Result<(), String> {
    let text = serde_json::to_string_pretty(payload).map_err(|err| err.to_string())?;
    if emit_stdout_json {
        println!("{}", text);
    }
    if let Some(path) = output {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)
                    .map_err(|err| format!("failed to create {}: {}", parent.display(), err))?;
            }
        }
        fs::write(path, text)
            .map_err(|err| format!("failed to write {}: {}", path.display(), err))?;
    }
    Ok(())
}

fn generate_message_id() -> String {
    format!("msg-{}", now_ms())
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn parse_enqueue_args_accepts_required_fields() {
        let parsed = parse_worker_args(&[
            "enqueue".to_string(),
            "--queue".to_string(),
            "jobs".to_string(),
            "--dir".to_string(),
            "state".to_string(),
            "--payload".to_string(),
            "{\"job\":1}".to_string(),
        ])
        .expect("parse");
        match parsed {
            WorkerCommand::Enqueue(args) => {
                assert_eq!(args.queue, "jobs");
                assert_eq!(args.max_attempts, 3);
            }
            _ => panic!("expected enqueue"),
        }
    }

    #[test]
    fn parse_run_args_accepts_required_fields() {
        let parsed = parse_worker_args(&[
            "run".to_string(),
            "--queue".to_string(),
            "jobs".to_string(),
            "--dir".to_string(),
            "state".to_string(),
            "--handler".to_string(),
            "handler.enk".to_string(),
            "--once".to_string(),
        ])
        .expect("parse");
        match parsed {
            WorkerCommand::Run(args) => {
                assert!(args.once);
                assert_eq!(args.queue, "jobs");
            }
            _ => panic!("expected run"),
        }
    }

    #[test]
    fn worker_queue_requeues_then_dead_letters() {
        let dir = tempdir().expect("tempdir");
        let state_dir = dir.path().join("state");
        fs::create_dir_all(&state_dir).expect("state dir");
        let handler = dir.path().join("handler.enk");
        fs::write(
            &handler,
            "import std::env\n\
policy default ::\n\
    allow env\n\
::\n\
fn main() ::\n\
    let id := env.get(\"ENKAI_WORKER_ID\")?\n\
    if id == \"dead\" ::\n\
        return 2\n\
    ::\n\
    return 0\n\
::\n\
main()\n",
        )
        .expect("handler");

        assert_eq!(
            worker_command(&[
                "enqueue".to_string(),
                "--queue".to_string(),
                "jobs".to_string(),
                "--dir".to_string(),
                state_dir.to_string_lossy().to_string(),
                "--payload".to_string(),
                "{\"ok\":true}".to_string(),
                "--id".to_string(),
                "ok".to_string(),
                "--max-attempts".to_string(),
                "2".to_string(),
            ]),
            0
        );
        assert_eq!(
            worker_command(&[
                "enqueue".to_string(),
                "--queue".to_string(),
                "jobs".to_string(),
                "--dir".to_string(),
                state_dir.to_string_lossy().to_string(),
                "--payload".to_string(),
                "{\"dead\":true}".to_string(),
                "--id".to_string(),
                "dead".to_string(),
                "--max-attempts".to_string(),
                "2".to_string(),
            ]),
            0
        );

        assert_eq!(
            worker_command(&[
                "run".to_string(),
                "--queue".to_string(),
                "jobs".to_string(),
                "--dir".to_string(),
                state_dir.to_string_lossy().to_string(),
                "--handler".to_string(),
                handler.to_string_lossy().to_string(),
                "--once".to_string(),
            ]),
            0
        );
        let pending =
            load_jsonl::<QueueMessage>(&pending_path(&state_dir, "jobs")).expect("pending");
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].id, "dead");

        assert_eq!(
            worker_command(&[
                "run".to_string(),
                "--queue".to_string(),
                "jobs".to_string(),
                "--dir".to_string(),
                state_dir.to_string_lossy().to_string(),
                "--handler".to_string(),
                handler.to_string_lossy().to_string(),
                "--once".to_string(),
            ]),
            0
        );
        let pending =
            load_jsonl::<QueueMessage>(&pending_path(&state_dir, "jobs")).expect("pending");
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].attempts, 1);

        assert_eq!(
            worker_command(&[
                "run".to_string(),
                "--queue".to_string(),
                "jobs".to_string(),
                "--dir".to_string(),
                state_dir.to_string_lossy().to_string(),
                "--handler".to_string(),
                handler.to_string_lossy().to_string(),
                "--once".to_string(),
            ]),
            0
        );
        let pending =
            load_jsonl::<QueueMessage>(&pending_path(&state_dir, "jobs")).expect("pending");
        assert!(pending.is_empty());
        let dead = load_jsonl::<QueueMessage>(&dead_letter_path(&state_dir, "jobs")).expect("dead");
        assert_eq!(dead.len(), 1);
        assert_eq!(dead[0].attempts, 2);
    }

    #[test]
    fn apply_worker_exit_requeues_and_dead_letters() {
        let base = QueueMessage {
            schema_version: 1,
            id: "job".to_string(),
            queue: "jobs".to_string(),
            payload: serde_json::json!({"kind":"test"}),
            tenant: None,
            attempts: 1,
            max_attempts: 2,
            enqueued_ms: 1,
            last_error: None,
        };

        match apply_worker_exit(base.clone(), 1) {
            WorkerOutcome::Requeued(msg) => {
                assert_eq!(msg.attempts, 1);
                assert!(msg.last_error.is_some());
            }
            _ => panic!("expected requeue"),
        }

        let mut exhausted = base;
        exhausted.attempts = 2;
        match apply_worker_exit(exhausted, 1) {
            WorkerOutcome::DeadLettered(msg) => {
                assert_eq!(msg.attempts, 2);
                assert!(msg.last_error.is_some());
            }
            _ => panic!("expected dead letter"),
        }
    }
}
