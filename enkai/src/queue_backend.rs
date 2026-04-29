use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct QueueMessage {
    pub(crate) schema_version: u32,
    pub(crate) id: String,
    pub(crate) queue: String,
    pub(crate) payload: serde_json::Value,
    pub(crate) tenant: Option<String>,
    pub(crate) attempts: u32,
    pub(crate) max_attempts: u32,
    pub(crate) enqueued_ms: u64,
    pub(crate) last_error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ScheduledMessage {
    pub(crate) schema_version: u32,
    pub(crate) message: QueueMessage,
    pub(crate) available_at_ms: u64,
    pub(crate) retry_delay_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct QueueState {
    pub(crate) schema_version: u32,
    pub(crate) queue: String,
    pub(crate) enqueue_count: u64,
    pub(crate) lease_count: u64,
    pub(crate) acked_count: u64,
    pub(crate) retry_count: u64,
    pub(crate) dead_letter_count: u64,
    pub(crate) last_message_id: Option<String>,
    pub(crate) updated_ms: u64,
}

#[derive(Debug, Clone)]
pub(crate) struct WorkerQueueStore {
    pub(crate) queue: String,
    pub(crate) queue_root: PathBuf,
    pub(crate) pending_path: PathBuf,
    pub(crate) inflight_path: PathBuf,
    pub(crate) schedule_path: PathBuf,
    pub(crate) dead_letter_path: PathBuf,
    pub(crate) state_path: PathBuf,
}

impl WorkerQueueStore {
    pub(crate) fn new(
        queue: String,
        queue_root: PathBuf,
        pending_path: PathBuf,
        inflight_path: PathBuf,
        schedule_path: PathBuf,
        dead_letter_path: PathBuf,
        state_path: PathBuf,
    ) -> Self {
        Self {
            queue,
            queue_root,
            pending_path,
            inflight_path,
            schedule_path,
            dead_letter_path,
            state_path,
        }
    }

    pub(crate) fn ensure_layout(&self) -> Result<(), String> {
        fs::create_dir_all(&self.queue_root).map_err(|err| {
            format!(
                "failed to create queue dir {}: {}",
                self.queue_root.display(),
                err
            )
        })?;
        for path in [
            &self.pending_path,
            &self.inflight_path,
            &self.schedule_path,
            &self.dead_letter_path,
        ] {
            if !path.is_file() {
                write_jsonl::<serde_json::Value>(path, &[])?;
            }
        }
        if !self.state_path.is_file() {
            self.write_state(&QueueState {
                schema_version: 1,
                queue: self.queue.clone(),
                enqueue_count: 0,
                lease_count: 0,
                acked_count: 0,
                retry_count: 0,
                dead_letter_count: 0,
                last_message_id: None,
                updated_ms: now_ms(),
            })?;
        }
        Ok(())
    }

    pub(crate) fn enqueue(&self, message: &QueueMessage) -> Result<(), String> {
        self.ensure_layout()?;
        append_jsonl(&self.pending_path, message)?;
        self.update_state(|state| {
            state.enqueue_count = state.enqueue_count.saturating_add(1);
            state.last_message_id = Some(message.id.clone());
        })
    }

    pub(crate) fn lease_next(&self) -> Result<Option<QueueMessage>, String> {
        self.ensure_layout()?;
        self.promote_due_scheduled()?;
        let mut pending = self.load_pending()?;
        let Some(message) = pending.first().cloned() else {
            return Ok(None);
        };
        pending.remove(0);
        self.write_pending(&pending)?;
        let mut inflight = self.load_inflight()?;
        inflight.push(message.clone());
        self.write_inflight(&inflight)?;
        self.update_state(|state| {
            state.lease_count = state.lease_count.saturating_add(1);
            state.last_message_id = Some(message.id.clone());
        })?;
        Ok(Some(message))
    }

    pub(crate) fn ack(&self, message_id: &str) -> Result<(), String> {
        self.ensure_layout()?;
        self.remove_inflight(message_id)?;
        self.update_state(|state| {
            state.acked_count = state.acked_count.saturating_add(1);
            state.last_message_id = Some(message_id.to_string());
        })
    }

    pub(crate) fn requeue(
        &self,
        message: &QueueMessage,
        retry_delay_ms: u64,
    ) -> Result<(), String> {
        self.ensure_layout()?;
        self.remove_inflight(&message.id)?;
        let available_at_ms = now_ms().saturating_add(retry_delay_ms);
        append_jsonl(
            &self.schedule_path,
            &ScheduledMessage {
                schema_version: 1,
                message: message.clone(),
                available_at_ms,
                retry_delay_ms,
            },
        )?;
        self.update_state(|state| {
            state.retry_count = state.retry_count.saturating_add(1);
            state.last_message_id = Some(message.id.clone());
        })
    }

    pub(crate) fn dead_letter(&self, message: &QueueMessage) -> Result<(), String> {
        self.ensure_layout()?;
        self.remove_inflight(&message.id)?;
        append_jsonl(&self.dead_letter_path, message)?;
        self.update_state(|state| {
            state.dead_letter_count = state.dead_letter_count.saturating_add(1);
            state.last_message_id = Some(message.id.clone());
        })
    }

    pub(crate) fn load_pending(&self) -> Result<Vec<QueueMessage>, String> {
        load_jsonl(&self.pending_path)
    }

    pub(crate) fn load_inflight(&self) -> Result<Vec<QueueMessage>, String> {
        load_jsonl(&self.inflight_path)
    }

    pub(crate) fn load_scheduled(&self) -> Result<Vec<ScheduledMessage>, String> {
        load_jsonl(&self.schedule_path)
    }

    pub(crate) fn load_state(&self) -> Result<QueueState, String> {
        if !self.state_path.is_file() {
            return Ok(QueueState {
                schema_version: 1,
                queue: self.queue.clone(),
                enqueue_count: 0,
                lease_count: 0,
                acked_count: 0,
                retry_count: 0,
                dead_letter_count: 0,
                last_message_id: None,
                updated_ms: now_ms(),
            });
        }
        let text = fs::read_to_string(&self.state_path)
            .map_err(|err| format!("failed to read {}: {}", self.state_path.display(), err))?;
        serde_json::from_str(&text)
            .map_err(|err| format!("failed to parse {}: {}", self.state_path.display(), err))
    }

    pub(crate) fn write_pending(&self, items: &[QueueMessage]) -> Result<(), String> {
        write_jsonl(&self.pending_path, items)
    }

    pub(crate) fn write_inflight(&self, items: &[QueueMessage]) -> Result<(), String> {
        write_jsonl(&self.inflight_path, items)
    }

    pub(crate) fn write_scheduled(&self, items: &[ScheduledMessage]) -> Result<(), String> {
        write_jsonl(&self.schedule_path, items)
    }

    pub(crate) fn write_state(&self, state: &QueueState) -> Result<(), String> {
        if let Some(parent) = self.state_path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)
                    .map_err(|err| format!("failed to create {}: {}", parent.display(), err))?;
            }
        }
        let text = serde_json::to_string_pretty(state).map_err(|err| err.to_string())?;
        fs::write(&self.state_path, text)
            .map_err(|err| format!("failed to write {}: {}", self.state_path.display(), err))
    }

    fn update_state(&self, mut update: impl FnMut(&mut QueueState)) -> Result<(), String> {
        let mut state = self.load_state()?;
        update(&mut state);
        state.updated_ms = now_ms();
        self.write_state(&state)
    }

    fn promote_due_scheduled(&self) -> Result<(), String> {
        let scheduled = self.load_scheduled()?;
        if scheduled.is_empty() {
            return Ok(());
        }
        let now = now_ms();
        let mut keep = Vec::new();
        let mut due = Vec::new();
        for item in scheduled {
            if item.available_at_ms <= now {
                due.push(item.message);
            } else {
                keep.push(item);
            }
        }
        if due.is_empty() {
            return Ok(());
        }
        let mut pending = self.load_pending()?;
        pending.extend(due);
        self.write_pending(&pending)?;
        self.write_scheduled(&keep)
    }

    fn remove_inflight(&self, message_id: &str) -> Result<(), String> {
        let mut inflight = self.load_inflight()?;
        inflight.retain(|item| item.id != message_id);
        self.write_inflight(&inflight)
    }
}

pub(crate) fn load_jsonl<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Vec<T>, String> {
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
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|err| format!("failed to create {}: {}", parent.display(), err))?;
        }
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

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
