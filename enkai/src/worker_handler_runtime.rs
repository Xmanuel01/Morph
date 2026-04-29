use std::collections::BTreeMap;
use std::path::Path;

use crate::queue_runtime::QueueMessage;

pub(crate) fn execute_handler(handler: &Path, message: &QueueMessage) -> Result<i32, String> {
    if !handler.is_file() {
        return Err(format!("handler not found: {}", handler.display()));
    }
    let mut vars = BTreeMap::new();
    vars.insert(
        "ENKAI_WORKER_PAYLOAD".to_string(),
        serde_json::to_string(&message.payload).map_err(|err| err.to_string())?,
    );
    vars.insert("ENKAI_WORKER_ID".to_string(), message.id.clone());
    vars.insert("ENKAI_WORKER_QUEUE".to_string(), message.queue.clone());
    vars.insert(
        "ENKAI_WORKER_ATTEMPT".to_string(),
        message.attempts.to_string(),
    );
    vars.insert(
        "ENKAI_WORKER_MAX_ATTEMPTS".to_string(),
        message.max_attempts.to_string(),
    );
    if let Some(tenant) = &message.tenant {
        vars.insert("ENKAI_WORKER_TENANT".to_string(), tenant.clone());
    }
    let env_scope = crate::runtime_exec::ScopedEnv {
        vars,
        std_override: crate::runtime_exec::bundled_std_override(),
    };
    crate::runtime_exec::execute_runtime_target_with_env(
        handler,
        crate::runtime_exec::RuntimeExecOptions {
            report_command: None,
            ..crate::runtime_exec::default_options()
        },
        &env_scope,
    )
}
