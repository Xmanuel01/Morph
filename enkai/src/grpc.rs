use std::env;
use std::fs;
use std::io::Write;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::json;
use tokio::sync::oneshot;
use tonic::{Request, Response, Status};

pub mod proto {
    tonic::include_proto!("enkai.chat.v1");
}

use proto::chat_service_server::ChatService;
use proto::{
    ChatReply, ChatRequest, HealthReply, HealthRequest, ReadyReply, ReadyRequest, StreamEvent,
};

type BoxStatus = Box<Status>;
pub(crate) type StartedChannel = (SyncSender<Result<(), String>>, Receiver<Result<(), String>>);

#[derive(Debug, Clone)]
pub(crate) struct GrpcRuntimeConfig {
    pub(crate) api_version: String,
    pub(crate) conversation_dir: PathBuf,
    pub(crate) log_path: Option<PathBuf>,
    pub(crate) startup_issue: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GrpcLaunchConfig {
    pub(crate) host: String,
    pub(crate) port: u16,
    pub(crate) api_version: String,
    pub(crate) conversation_dir: PathBuf,
    pub(crate) log_path: Option<PathBuf>,
}

#[derive(Debug)]
pub(crate) struct GrpcRuntimeState {
    config: GrpcRuntimeConfig,
    sequence: AtomicU64,
    log_lock: Mutex<()>,
}

pub struct GrpcServerHandle {
    pub(crate) shutdown: Option<oneshot::Sender<()>>,
    pub(crate) join: Option<thread::JoinHandle<()>>,
    #[allow(dead_code)]
    pub(crate) addr: SocketAddr,
}

impl GrpcServerHandle {
    #[cfg(test)]
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    pub fn shutdown(mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

#[derive(Clone)]
pub(crate) struct EnkaiGrpcService {
    pub(crate) state: Arc<GrpcRuntimeState>,
}

impl GrpcRuntimeConfig {
    #[allow(dead_code)]
    pub(crate) fn from_env() -> Self {
        let api_version = env::var("ENKAI_API_VERSION")
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "v1".to_string());
        let conversation_dir = env::var("ENKAI_CONVERSATION_DIR")
            .ok()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));
        let log_path = env::var("ENKAI_LOG_PATH")
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .map(PathBuf::from);
        let db_engine = env::var("ENKAI_DB_ENGINE").unwrap_or_else(|_| "sqlite".to_string());
        let startup_issue = if api_version.trim().is_empty() {
            Some("ENKAI_API_VERSION is required".to_string())
        } else if !matches!(db_engine.trim(), "sqlite" | "postgres" | "mysql") {
            Some("ENKAI_DB_ENGINE must be sqlite|postgres|mysql".to_string())
        } else if let Err(err) = fs::create_dir_all(&conversation_dir) {
            Some(format!(
                "failed to create conversation dir {}: {}",
                conversation_dir.display(),
                err
            ))
        } else {
            None
        };
        Self {
            api_version,
            conversation_dir,
            log_path,
            startup_issue,
        }
    }

    pub(crate) fn from_launch_config(config: &GrpcLaunchConfig) -> Self {
        let startup_issue = if config.api_version.trim().is_empty() {
            Some("ENKAI_API_VERSION is required".to_string())
        } else if let Err(err) = fs::create_dir_all(&config.conversation_dir) {
            Some(format!(
                "failed to create conversation dir {}: {}",
                config.conversation_dir.display(),
                err
            ))
        } else {
            None
        };
        Self {
            api_version: config.api_version.clone(),
            conversation_dir: config.conversation_dir.clone(),
            log_path: config.log_path.clone(),
            startup_issue,
        }
    }
}

impl GrpcRuntimeState {
    pub(crate) fn new(config: GrpcRuntimeConfig) -> Self {
        Self {
            config,
            sequence: AtomicU64::new(0),
            log_lock: Mutex::new(()),
        }
    }

    fn next_conversation_id(&self, requested: &str) -> String {
        if !requested.trim().is_empty() {
            return requested.trim().to_string();
        }
        let seq = self.sequence.fetch_add(1, Ordering::Relaxed) + 1;
        format!("grpc-{}-{}", unix_ms(), seq)
    }

    fn ensure_ready(&self) -> Result<(), BoxStatus> {
        if let Some(issue) = &self.config.startup_issue {
            return Err(Box::new(Status::failed_precondition(format!(
                "service_not_ready:{issue}"
            ))));
        }
        Ok(())
    }

    fn validate_api_version(&self, api_version: &str) -> Result<(), BoxStatus> {
        if api_version.trim().is_empty() {
            return Err(Box::new(Status::invalid_argument("missing_api_version")));
        }
        if api_version.trim() != self.config.api_version {
            return Err(Box::new(Status::invalid_argument("api_version_mismatch")));
        }
        Ok(())
    }

    fn validate_prompt(&self, prompt: &str) -> Result<(), BoxStatus> {
        if prompt.trim().is_empty() {
            return Err(Box::new(Status::invalid_argument("missing_prompt")));
        }
        Ok(())
    }

    fn reply_text(&self, stream: bool) -> &'static str {
        if stream {
            "hello from enkai"
        } else {
            "hello from enkai backend"
        }
    }

    fn save_conversation(
        &self,
        conversation_id: &str,
        prompt: &str,
        reply: &str,
        source: &str,
    ) -> Result<(), BoxStatus> {
        let state = json!({
            "schema_version": 1,
            "id": conversation_id,
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": reply}
            ],
            "user_text": prompt,
            "reply": reply,
            "source": source,
            "updated_ms": unix_ms()
        });
        let payload = serde_json::to_string(&state).map_err(|err| {
            Box::new(Status::internal(format!(
                "conversation_state_serialize_failed:{err}"
            )))
        })?;
        let primary = self.config.conversation_dir.join("conversation_state.json");
        let backup = self
            .config
            .conversation_dir
            .join("conversation_state.backup.json");
        fs::write(&primary, &payload).map_err(|err| {
            Box::new(Status::internal(format!(
                "conversation_state_write_failed:{}:{err}",
                primary.display()
            )))
        })?;
        fs::write(&backup, &payload).map_err(|err| {
            Box::new(Status::internal(format!(
                "conversation_state_write_failed:{}:{err}",
                backup.display()
            )))
        })?;
        Ok(())
    }

    fn append_log(&self, rpc: &str, conversation_id: &str, prompt: &str) {
        let Some(path) = &self.config.log_path else {
            return;
        };
        let _guard = self.log_lock.lock().unwrap_or_else(|err| err.into_inner());
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        if let Ok(mut file) = fs::OpenOptions::new().create(true).append(true).open(path) {
            let line = json!({
                "ts_ms": unix_ms(),
                "protocol": "grpc",
                "rpc": rpc,
                "conversation_id": conversation_id,
                "prompt_bytes": prompt.len(),
                "api_version": self.config.api_version,
            });
            let _ = writeln!(file, "{}", line);
        }
    }
}

#[tonic::async_trait]
impl ChatService for EnkaiGrpcService {
    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthReply>, Status> {
        Ok(Response::new(HealthReply {
            status: "ok".to_string(),
            api_version: self.state.config.api_version.clone(),
        }))
    }

    async fn ready(&self, _request: Request<ReadyRequest>) -> Result<Response<ReadyReply>, Status> {
        let status = if self.state.config.startup_issue.is_none() {
            "ready"
        } else {
            "not_ready"
        };
        Ok(Response::new(ReadyReply {
            status: status.to_string(),
            api_version: self.state.config.api_version.clone(),
        }))
    }

    async fn chat(&self, request: Request<ChatRequest>) -> Result<Response<ChatReply>, Status> {
        self.state.ensure_ready().map_err(|err| *err)?;
        let inner = request.into_inner();
        self.state
            .validate_api_version(&inner.api_version)
            .map_err(|err| *err)?;
        self.state
            .validate_prompt(&inner.prompt)
            .map_err(|err| *err)?;
        let conversation_id = self.state.next_conversation_id(&inner.conversation_id);
        let reply = self.state.reply_text(false).to_string();
        self.state
            .save_conversation(&conversation_id, &inner.prompt, &reply, "grpc-chat")
            .map_err(|err| *err)?;
        self.state
            .append_log("Chat", &conversation_id, &inner.prompt);
        Ok(Response::new(ChatReply {
            id: conversation_id,
            reply,
            api_version: self.state.config.api_version.clone(),
        }))
    }

    type StreamChatStream = std::pin::Pin<
        Box<dyn tokio_stream::Stream<Item = Result<StreamEvent, Status>> + Send + 'static>,
    >;

    async fn stream_chat(
        &self,
        request: Request<ChatRequest>,
    ) -> Result<Response<Self::StreamChatStream>, Status> {
        self.state.ensure_ready().map_err(|err| *err)?;
        let inner = request.into_inner();
        self.state
            .validate_api_version(&inner.api_version)
            .map_err(|err| *err)?;
        self.state
            .validate_prompt(&inner.prompt)
            .map_err(|err| *err)?;
        let conversation_id = self.state.next_conversation_id(&inner.conversation_id);
        let api_version = self.state.config.api_version.clone();
        let reply = self.state.reply_text(true).to_string();
        self.state
            .save_conversation(&conversation_id, &inner.prompt, &reply, "grpc-stream")
            .map_err(|err| *err)?;
        self.state
            .append_log("StreamChat", &conversation_id, &inner.prompt);
        let stream = tokio_stream::iter(vec![
            Ok(StreamEvent {
                event: "token".to_string(),
                value: "hello".to_string(),
                conversation_id: conversation_id.clone(),
                api_version: api_version.clone(),
            }),
            Ok(StreamEvent {
                event: "token".to_string(),
                value: " from".to_string(),
                conversation_id: conversation_id.clone(),
                api_version: api_version.clone(),
            }),
            Ok(StreamEvent {
                event: "token".to_string(),
                value: " enkai".to_string(),
                conversation_id: conversation_id.clone(),
                api_version: api_version.clone(),
            }),
            Ok(StreamEvent {
                event: "done".to_string(),
                value: String::new(),
                conversation_id,
                api_version,
            }),
        ]);
        Ok(Response::new(Box::pin(stream) as Self::StreamChatStream))
    }
}

pub fn grpc_command(args: &[String]) -> i32 {
    crate::grpc_runtime::grpc_command(args)
}

pub fn print_grpc_usage() {
    crate::grpc_runtime::print_grpc_usage();
}

#[allow(dead_code)]
pub fn maybe_start_grpc_server(
    host: Option<&str>,
    port: Option<&str>,
) -> Result<Option<GrpcServerHandle>, String> {
    crate::grpc_runtime::maybe_start_grpc_server(host, port)
}

fn unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_millis()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn parse_probe_args_defaults() {
        let parsed = crate::grpc_runtime::parse_grpc_probe_args(&[]).expect("defaults");
        assert_eq!(parsed.address, "http://127.0.0.1:9090");
        assert_eq!(parsed.api_version, "v1");
        assert_eq!(parsed.prompt, "hello from grpc");
    }

    #[test]
    fn grpc_server_probe_roundtrip() {
        let dir = tempdir().expect("tempdir");
        let state = GrpcRuntimeConfig {
            api_version: "v1".to_string(),
            conversation_dir: dir.path().to_path_buf(),
            log_path: Some(dir.path().join("server.jsonl")),
            startup_issue: None,
        };
        let handle = crate::grpc_runtime::start_grpc_server(state, "127.0.0.1", 0).expect("server");
        let payload =
            crate::grpc_runtime::execute_grpc_probe(&crate::grpc_runtime::GrpcProbeArgs {
                address: format!("http://{}", handle.addr()),
                api_version: "v1".to_string(),
                prompt: "hi".to_string(),
                conversation_id: None,
                json: false,
                output: None,
            })
            .expect("probe");
        assert_eq!(payload["health"]["status"], "ok");
        assert_eq!(payload["ready"]["status"], "ready");
        assert_eq!(payload["chat"]["reply"], "hello from enkai backend");
        assert_eq!(
            payload["stream"].as_array().map(|items| items.len()),
            Some(4)
        );
        handle.shutdown();
    }
}
