use crate::queue_runtime;
use crate::systems::WorkerQueueManifest;

pub fn print_worker_usage() {
    eprintln!("  enkai worker enqueue --queue <name> --dir <state_dir> --payload <json> [--tenant <tenant>] [--id <id>] [--max-attempts <n>] [--json] [--output <file>]");
    eprintln!("  enkai worker run --queue <name> --dir <state_dir> --handler <file.enk> [--once] [--json] [--output <file>]");
}

pub fn worker_command(args: &[String]) -> i32 {
    let parsed = match crate::systems::build_worker_queue_manifest(args) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("enkai worker: {}", err);
            print_worker_usage();
            return 1;
        }
    };
    execute_worker_manifest(&parsed)
}

pub(crate) fn execute_worker_manifest(manifest: &WorkerQueueManifest) -> i32 {
    queue_runtime::execute_manifest_cli(manifest)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::queue_backend::{QueueState, ScheduledMessage};
    use crate::queue_runtime::QueueMessage;
    use tempfile::tempdir;

    #[test]
    fn worker_queue_processes_messages_end_to_end() {
        let dir = tempdir().expect("tempdir");
        let state_dir = dir.path().join("state");
        fs::create_dir_all(&state_dir).expect("state dir");
        let handler = dir.path().join("handler.enk");
        let failing_handler = dir.path().join("handler_fail.enk");
        fs::write(
            &handler,
            "fn main() ::\n\
    return 0\n::\n\
main()\n",
        )
        .expect("handler");
        fs::write(
            &failing_handler,
            "fn main() ::\n\
    missing_symbol()\n\
::\n\
main()\n",
        )
        .expect("failing handler");

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
        let pending = queue_runtime::load_jsonl::<QueueMessage>(
            &state_dir.join("queues").join("jobs").join("pending.jsonl"),
        )
        .expect("pending");
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
                failing_handler.to_string_lossy().to_string(),
                "--once".to_string(),
            ]),
            0
        );
        let scheduled = queue_runtime::load_jsonl::<ScheduledMessage>(
            &state_dir
                .join("queues")
                .join("jobs")
                .join("scheduled.jsonl"),
        )
        .expect("scheduled");
        assert_eq!(scheduled.len(), 1);
        assert_eq!(scheduled[0].message.id, "dead");
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
        let dead = queue_runtime::load_jsonl::<QueueMessage>(
            &state_dir
                .join("queues")
                .join("jobs")
                .join("dead_letter.jsonl"),
        )
        .expect("dead");
        assert!(dead.is_empty());
        let pending = queue_runtime::load_jsonl::<QueueMessage>(
            &state_dir.join("queues").join("jobs").join("pending.jsonl"),
        )
        .expect("pending");
        assert!(pending.is_empty());
        let scheduled = queue_runtime::load_jsonl::<ScheduledMessage>(
            &state_dir
                .join("queues")
                .join("jobs")
                .join("scheduled.jsonl"),
        )
        .expect("scheduled");
        assert!(scheduled.is_empty());
        let state_text = fs::read_to_string(
            state_dir
                .join("queues")
                .join("jobs")
                .join("queue_state.json"),
        )
        .expect("state");
        let state: QueueState = serde_json::from_str(&state_text).expect("state json");
        assert_eq!(state.enqueue_count, 2);
        assert_eq!(state.acked_count, 2);
        assert_eq!(state.dead_letter_count, 0);
        assert_eq!(state.retry_count, 1);
    }
}
