use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::queue_backend::WorkerQueueStore;
use crate::systems::{WorkerQueueManifest, WorkerRetryPolicyManifest, WorkerRunPolicyManifest};

pub(crate) use crate::queue_backend::QueueMessage;

#[derive(Debug, Clone)]
pub(crate) struct EnqueueRequest {
    queue: String,
    store: WorkerQueueStore,
    payload: serde_json::Value,
    tenant: Option<String>,
    id: Option<String>,
    max_attempts: u32,
    retry_policy: WorkerRetryPolicy,
}

#[derive(Debug, Clone)]
pub(crate) struct RunRequest {
    queue: String,
    store: WorkerQueueStore,
    handler: PathBuf,
    run_policy: WorkerRunPolicy,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct WorkerRunReport {
    pub(crate) schema_version: u32,
    pub(crate) queue: String,
    pub(crate) status: String,
    pub(crate) processed: usize,
    pub(crate) acked: usize,
    pub(crate) requeued: usize,
    pub(crate) dead_lettered: usize,
    pub(crate) last_message_id: Option<String>,
    pub(crate) last_attempt: Option<u32>,
    pub(crate) pending_path: String,
    pub(crate) inflight_path: String,
    pub(crate) schedule_path: String,
    pub(crate) dead_letter_path: String,
    pub(crate) state_path: String,
}

#[derive(Debug, Clone)]
struct WorkerRetryPolicy {
    max_attempts: u32,
    delay_ms: u64,
}

#[derive(Debug, Clone)]
pub(crate) struct WorkerRunPolicy {
    drain_mode: WorkerDrainMode,
    max_messages: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorkerDrainMode {
    Once,
    UntilIdle,
}

#[derive(Debug, Clone)]
struct ManifestIoOptions {
    json: bool,
    output: Option<PathBuf>,
}

#[derive(Debug, Clone)]
enum WorkerManifestCommand {
    Enqueue {
        request: EnqueueRequest,
        io: ManifestIoOptions,
    },
    Run {
        request: RunRequest,
        io: ManifestIoOptions,
    },
}

#[derive(Debug, Clone)]
struct WorkerQueueBackend {
    queue: String,
    store: WorkerQueueStore,
}

pub(crate) fn enqueue(request: &EnqueueRequest) -> Result<serde_json::Value, String> {
    request.store.ensure_layout()?;
    let message = QueueMessage {
        schema_version: 1,
        id: request.id.clone().unwrap_or_else(generate_message_id),
        queue: request.queue.clone(),
        payload: request.payload.clone(),
        tenant: request.tenant.clone(),
        attempts: 0,
        max_attempts: request.max_attempts,
        enqueued_ms: now_ms(),
        last_error: None,
    };
    request.store.enqueue(&message)?;
    Ok(serde_json::json!({
        "schema_version": 1,
        "status": "queued",
        "queue": request.queue,
        "backend_kind": "selfhost_jsonl_queue_v2",
        "id": message.id,
        "max_attempts": message.max_attempts,
        "retry_policy": {
            "max_attempts": request.retry_policy.max_attempts,
            "delay_ms": request.retry_policy.delay_ms,
        },
        "pending_path": request.store.pending_path,
        "inflight_path": request.store.inflight_path,
        "schedule_path": request.store.schedule_path,
        "dead_letter_path": request.store.dead_letter_path,
        "state_path": request.store.state_path,
    }))
}

pub(crate) fn run(request: &RunRequest) -> Result<WorkerRunReport, String> {
    let backend = WorkerQueueBackend::from_run_request(request);
    backend.run(request)
}

pub(crate) fn execute_manifest_cli(manifest: &WorkerQueueManifest) -> i32 {
    let command = match parse_manifest(manifest) {
        Ok(command) => command,
        Err(err) => {
            eprintln!("enkai worker: {}", err);
            return 1;
        }
    };
    match command {
        WorkerManifestCommand::Enqueue { request, io } => {
            let payload = match enqueue(&request) {
                Ok(payload) => payload,
                Err(err) => {
                    eprintln!("enkai worker enqueue: {}", err);
                    return 1;
                }
            };
            if let Err(err) = emit_json(&payload, io.json, io.output.as_deref()) {
                eprintln!("enkai worker enqueue: {}", err);
                return 1;
            }
            if !io.json {
                println!(
                    "queued message {} on {}",
                    payload["id"].as_str().unwrap_or("<unknown>"),
                    request.queue
                );
            }
            0
        }
        WorkerManifestCommand::Run { request, io } => {
            let report = match run(&request) {
                Ok(report) => report,
                Err(err) => {
                    eprintln!("enkai worker run: {}", err);
                    return 1;
                }
            };
            if let Err(err) = emit_json(&report, io.json, io.output.as_deref()) {
                eprintln!("enkai worker run: {}", err);
                return 1;
            }
            if !io.json {
                println!(
                    "worker queue={} status={} acked={} requeued={} dead_lettered={}",
                    report.queue,
                    report.status,
                    report.acked,
                    report.requeued,
                    report.dead_lettered
                );
            }
            if report.status == "failed" {
                1
            } else {
                0
            }
        }
    }
}

impl WorkerQueueBackend {
    fn from_run_request(request: &RunRequest) -> Self {
        Self {
            queue: request.queue.clone(),
            store: request.store.clone(),
        }
    }

    fn run(&self, request: &RunRequest) -> Result<WorkerRunReport, String> {
        self.store.ensure_layout()?;
        if matches!(request.run_policy.drain_mode, WorkerDrainMode::Once) {
            return self.execute_once(&request.handler);
        }
        self.run_until_idle(&request.handler, request.run_policy.max_messages)
    }

    fn execute_once(&self, handler: &Path) -> Result<WorkerRunReport, String> {
        let Some(mut message) = self.store.lease_next()? else {
            return Ok(self.idle_report());
        };
        message.attempts = message.attempts.saturating_add(1);
        let execution_result = crate::worker_handler_runtime::execute_handler(handler, &message);
        let mut report = self.idle_report();
        report.processed = 1;
        report.last_message_id = Some(message.id.clone());
        report.last_attempt = Some(message.attempts);
        let exit_code = match execution_result {
            Ok(code) => code,
            Err(err) => {
                message.last_error = Some(err);
                return self.complete_failed_message(&message, &mut report);
            }
        };
        if exit_code == 0 {
            self.store.ack(&message.id)?;
            report.status = "acked".to_string();
            report.acked = 1;
            return Ok(report);
        }
        message.last_error = Some(format!("handler exited with {}", exit_code));
        self.complete_failed_message(&message, &mut report)
    }

    fn complete_failed_message(
        &self,
        message: &QueueMessage,
        report: &mut WorkerRunReport,
    ) -> Result<WorkerRunReport, String> {
        if message.attempts < message.max_attempts {
            self.store.requeue(message, 0)?;
            report.status = "requeued".to_string();
            report.requeued = 1;
            return Ok(report.clone());
        }
        self.store.dead_letter(message)?;
        report.status = "dead_lettered".to_string();
        report.dead_lettered = 1;
        Ok(report.clone())
    }

    fn run_until_idle(
        &self,
        handler: &Path,
        max_messages: Option<usize>,
    ) -> Result<WorkerRunReport, String> {
        let mut aggregate = self.idle_report();
        loop {
            let report = self.execute_once(handler)?;
            if report.processed == 0 {
                if aggregate.processed == 0 {
                    return Ok(report);
                }
                let state = self.store.load_state()?;
                aggregate.status = if self.store.load_pending()?.is_empty()
                    && self.store.load_inflight()?.is_empty()
                    && self.store.load_scheduled()?.is_empty()
                {
                    "drained".to_string()
                } else {
                    aggregate.status
                };
                aggregate.last_message_id = state.last_message_id;
                return Ok(aggregate);
            }
            aggregate.processed += report.processed;
            aggregate.acked += report.acked;
            aggregate.requeued += report.requeued;
            aggregate.dead_lettered += report.dead_lettered;
            aggregate.last_message_id = report.last_message_id;
            aggregate.last_attempt = report.last_attempt;
            aggregate.status = match (aggregate.status.as_str(), report.status.as_str()) {
                ("dead_lettered", _) => "dead_lettered".to_string(),
                (_, "dead_lettered") => "dead_lettered".to_string(),
                ("requeued", _) => "requeued".to_string(),
                (_, "requeued") => "requeued".to_string(),
                ("idle", status) => status.to_string(),
                (status, _) => status.to_string(),
            };
            if let Some(max_messages) = max_messages {
                if aggregate.processed >= max_messages {
                    return Ok(aggregate);
                }
            }
        }
    }

    fn idle_report(&self) -> WorkerRunReport {
        WorkerRunReport {
            schema_version: 1,
            queue: self.queue.clone(),
            status: "idle".to_string(),
            processed: 0,
            acked: 0,
            requeued: 0,
            dead_lettered: 0,
            last_message_id: None,
            last_attempt: None,
            pending_path: self.store.pending_path.display().to_string(),
            inflight_path: self.store.inflight_path.display().to_string(),
            schedule_path: self.store.schedule_path.display().to_string(),
            dead_letter_path: self.store.dead_letter_path.display().to_string(),
            state_path: self.store.state_path.display().to_string(),
        }
    }
}

fn parse_manifest(manifest: &WorkerQueueManifest) -> Result<WorkerManifestCommand, String> {
    match manifest {
        WorkerQueueManifest::Enqueue {
            queue,
            queue_root,
            pending_path,
            inflight_path,
            schedule_path,
            dead_letter_path,
            state_path,
            payload,
            tenant,
            id,
            max_attempts,
            retry_policy,
            json,
            output,
            ..
        } => {
            let retry_policy = resolve_retry_policy(*max_attempts, retry_policy)?;
            let request = EnqueueRequest {
                queue: queue.clone(),
                store: WorkerQueueStore::new(
                    queue.clone(),
                    PathBuf::from(queue_root),
                    PathBuf::from(pending_path),
                    PathBuf::from(inflight_path),
                    PathBuf::from(schedule_path),
                    PathBuf::from(dead_letter_path),
                    PathBuf::from(state_path),
                ),
                payload: payload.clone(),
                tenant: tenant.clone(),
                id: id.clone(),
                max_attempts: retry_policy.max_attempts,
                retry_policy,
            };
            Ok(WorkerManifestCommand::Enqueue {
                request,
                io: ManifestIoOptions {
                    json: *json,
                    output: output.as_ref().map(PathBuf::from),
                },
            })
        }
        WorkerQueueManifest::Run {
            queue,
            queue_root,
            pending_path,
            inflight_path,
            schedule_path,
            dead_letter_path,
            state_path,
            handler,
            once,
            run_policy,
            json,
            output,
            ..
        } => {
            let request = RunRequest {
                queue: queue.clone(),
                store: WorkerQueueStore::new(
                    queue.clone(),
                    PathBuf::from(queue_root),
                    PathBuf::from(pending_path),
                    PathBuf::from(inflight_path),
                    PathBuf::from(schedule_path),
                    PathBuf::from(dead_letter_path),
                    PathBuf::from(state_path),
                ),
                handler: PathBuf::from(handler),
                run_policy: resolve_run_policy(*once, run_policy)?,
            };
            Ok(WorkerManifestCommand::Run {
                request,
                io: ManifestIoOptions {
                    json: *json,
                    output: output.as_ref().map(PathBuf::from),
                },
            })
        }
    }
}

fn resolve_retry_policy(
    max_attempts: u32,
    retry_policy: &WorkerRetryPolicyManifest,
) -> Result<WorkerRetryPolicy, String> {
    if retry_policy.max_attempts == 0 {
        return Err("worker retry policy max_attempts must be >= 1".to_string());
    }
    if retry_policy.max_attempts != max_attempts {
        return Err(format!(
            "worker retry policy max_attempts {} does not match manifest max_attempts {}",
            retry_policy.max_attempts, max_attempts
        ));
    }
    Ok(WorkerRetryPolicy {
        max_attempts: retry_policy.max_attempts,
        delay_ms: retry_policy.delay_ms,
    })
}

fn resolve_run_policy(
    once: bool,
    run_policy: &WorkerRunPolicyManifest,
) -> Result<WorkerRunPolicy, String> {
    let drain_mode = match run_policy.drain_mode.trim() {
        "once" => WorkerDrainMode::Once,
        "until_idle" => WorkerDrainMode::UntilIdle,
        other => {
            return Err(format!(
                "invalid worker run_policy.drain_mode '{}'; expected once|until_idle",
                other
            ))
        }
    };
    if matches!(drain_mode, WorkerDrainMode::Once) != once {
        return Err(format!(
            "worker run policy mismatch: once={} but drain_mode={}",
            once, run_policy.drain_mode
        ));
    }
    let max_messages = run_policy
        .max_messages
        .map(|value| {
            usize::try_from(value)
                .map_err(|_| "worker run_policy.max_messages overflow".to_string())
        })
        .transpose()?;
    if matches!(drain_mode, WorkerDrainMode::Once) && max_messages != Some(1) {
        return Err("worker once mode requires run_policy.max_messages == 1".to_string());
    }
    Ok(WorkerRunPolicy {
        drain_mode,
        max_messages,
    })
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
pub(crate) fn load_jsonl<T: for<'de> serde::Deserialize<'de>>(
    path: &Path,
) -> Result<Vec<T>, String> {
    crate::queue_backend::load_jsonl(path)
}
