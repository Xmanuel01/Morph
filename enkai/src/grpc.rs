use std::env;
use std::fs;
use std::io::Write;
use std::net::{SocketAddr, TcpListener, ToSocketAddrs};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::json;
use tokio::runtime::Builder;
use tokio::sync::oneshot;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::Server;
use tonic::{Request, Response, Status};

pub mod proto {
    tonic::include_proto!("enkai.chat.v1");
}

use proto::chat_service_client::ChatServiceClient;
use proto::chat_service_server::{ChatService, ChatServiceServer};
use proto::{
    ChatReply, ChatRequest, HealthReply, HealthRequest, ReadyReply, ReadyRequest, StreamEvent,
};

type BoxStatus = Box<Status>;
type StartedChannel = (SyncSender<Result<(), String>>, Receiver<Result<(), String>>);

#[derive(Debug, Clone, PartialEq, Eq)]
struct GrpcProbeArgs {
    address: String,
    api_version: String,
    prompt: String,
    conversation_id: Option<String>,
    json: bool,
    output: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct GrpcRuntimeConfig {
    api_version: String,
    conversation_dir: PathBuf,
    log_path: Option<PathBuf>,
    startup_issue: Option<String>,
}

#[derive(Debug)]
struct GrpcRuntimeState {
    config: GrpcRuntimeConfig,
    sequence: AtomicU64,
    log_lock: Mutex<()>,
}

pub struct GrpcServerHandle {
    shutdown: Option<oneshot::Sender<()>>,
    join: Option<thread::JoinHandle<()>>,
    #[allow(dead_code)]
    addr: SocketAddr,
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
struct EnkaiGrpcService {
    state: Arc<GrpcRuntimeState>,
}

impl GrpcRuntimeConfig {
    fn from_env() -> Self {
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
}

impl GrpcRuntimeState {
    fn new(config: GrpcRuntimeConfig) -> Self {
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

pub fn print_grpc_usage() {
    eprintln!(
        "  enkai grpc probe [--address <http://host:port>] [--api-version <v>] [--prompt <text>] [--conversation-id <id>] [--json] [--output <file>]"
    );
}

pub fn grpc_command(args: &[String]) -> i32 {
    if args.is_empty() {
        print_grpc_usage();
        return 1;
    }
    match args[0].as_str() {
        "probe" => grpc_probe_command(&args[1..]),
        _ => {
            eprintln!("enkai grpc: unknown subcommand '{}'", args[0]);
            print_grpc_usage();
            1
        }
    }
}

fn grpc_probe_command(args: &[String]) -> i32 {
    let parsed = match parse_grpc_probe_args(args) {
        Ok(parsed) => parsed,
        Err(err) => {
            eprintln!("enkai grpc probe: {err}");
            return 1;
        }
    };
    let payload = match execute_grpc_probe(&parsed) {
        Ok(payload) => payload,
        Err(err) => {
            eprintln!("enkai grpc probe: {err}");
            return 1;
        }
    };
    let text = serde_json::to_string_pretty(&payload)
        .unwrap_or_else(|err| json!({"error": err.to_string()}).to_string());
    if let Some(output) = parsed.output.as_ref() {
        if let Some(parent) = output.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = fs::create_dir_all(parent);
            }
        }
        if let Err(err) = fs::write(output, text.as_bytes()) {
            eprintln!(
                "enkai grpc probe: failed to write {}: {}",
                output.display(),
                err
            );
            return 1;
        }
    }
    if parsed.json || parsed.output.is_none() {
        println!("{}", text);
    }
    0
}

fn parse_grpc_probe_args(args: &[String]) -> Result<GrpcProbeArgs, String> {
    let mut parsed = GrpcProbeArgs {
        address: "http://127.0.0.1:9090".to_string(),
        api_version: "v1".to_string(),
        prompt: "hello from grpc".to_string(),
        conversation_id: None,
        json: false,
        output: None,
    };
    let mut idx = 0usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--address" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--address requires a value".to_string());
                }
                parsed.address = args[idx].clone();
            }
            "--api-version" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--api-version requires a value".to_string());
                }
                parsed.api_version = args[idx].clone();
            }
            "--prompt" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--prompt requires a value".to_string());
                }
                parsed.prompt = args[idx].clone();
            }
            "--conversation-id" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--conversation-id requires a value".to_string());
                }
                parsed.conversation_id = Some(args[idx].clone());
            }
            "--json" => parsed.json = true,
            "--output" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--output requires a value".to_string());
                }
                parsed.output = Some(PathBuf::from(&args[idx]));
            }
            other => return Err(format!("unknown option {other}")),
        }
        idx += 1;
    }
    Ok(parsed)
}

fn execute_grpc_probe(args: &GrpcProbeArgs) -> Result<serde_json::Value, String> {
    let runtime = Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|err| format!("failed to build tokio runtime: {err}"))?;
    runtime.block_on(async move {
        let mut client = ChatServiceClient::connect(args.address.clone())
            .await
            .map_err(|err| format!("connect failed: {err}"))?;
        let health = client
            .health(Request::new(HealthRequest {}))
            .await
            .map_err(|err| format!("health failed: {err}"))?
            .into_inner();
        let ready = client
            .ready(Request::new(ReadyRequest {}))
            .await
            .map_err(|err| format!("ready failed: {err}"))?
            .into_inner();
        let chat = client
            .chat(Request::new(ChatRequest {
                prompt: args.prompt.clone(),
                conversation_id: args.conversation_id.clone().unwrap_or_default(),
                api_version: args.api_version.clone(),
            }))
            .await
            .map_err(|err| format!("chat failed: {err}"))?
            .into_inner();
        let mut stream = client
            .stream_chat(Request::new(ChatRequest {
                prompt: args.prompt.clone(),
                conversation_id: chat.id.clone(),
                api_version: args.api_version.clone(),
            }))
            .await
            .map_err(|err| format!("stream_chat failed: {err}"))?
            .into_inner();
        let mut events = Vec::new();
        while let Some(item) = stream
            .message()
            .await
            .map_err(|err| format!("stream read failed: {err}"))?
        {
            events.push(json!({
                "event": item.event,
                "value": item.value,
                "conversation_id": item.conversation_id,
                "api_version": item.api_version,
            }));
        }
        Ok(json!({
            "schema_version": 1,
            "address": args.address,
            "health": {
                "status": health.status,
                "api_version": health.api_version,
            },
            "ready": {
                "status": ready.status,
                "api_version": ready.api_version,
            },
            "chat": {
                "id": chat.id,
                "reply": chat.reply,
                "api_version": chat.api_version,
            },
            "stream": events,
        }))
    })
}

pub fn maybe_start_grpc_server(
    host: Option<&str>,
    port: Option<&str>,
) -> Result<Option<GrpcServerHandle>, String> {
    let port_value = match port {
        Some(value) if !value.trim().is_empty() => value.trim().to_string(),
        _ => match env::var("ENKAI_GRPC_PORT") {
            Ok(value) if !value.trim().is_empty() => value.trim().to_string(),
            _ => return Ok(None),
        },
    };
    let host_value = host
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .or_else(|| {
            env::var("ENKAI_GRPC_HOST")
                .ok()
                .filter(|value| !value.trim().is_empty())
        })
        .or_else(|| {
            env::var("ENKAI_SERVE_HOST")
                .ok()
                .filter(|value| !value.trim().is_empty())
        })
        .unwrap_or_else(|| "0.0.0.0".to_string());
    let port_num = port_value
        .parse::<u16>()
        .map_err(|_| format!("invalid gRPC port '{port_value}'"))?;
    if port_num == 0 {
        return Err("gRPC port must be in range 1..65535".to_string());
    }
    start_grpc_server(GrpcRuntimeConfig::from_env(), &host_value, port_num).map(Some)
}

fn start_grpc_server(
    config: GrpcRuntimeConfig,
    host: &str,
    port: u16,
) -> Result<GrpcServerHandle, String> {
    let addr = resolve_socket_addr(host, port)?;
    let listener =
        TcpListener::bind(addr).map_err(|err| format!("gRPC bind failed for {}: {}", addr, err))?;
    listener
        .set_nonblocking(true)
        .map_err(|err| format!("gRPC nonblocking setup failed for {}: {}", addr, err))?;
    let local_addr = listener
        .local_addr()
        .map_err(|err| format!("gRPC local address lookup failed: {err}"))?;
    let state = Arc::new(GrpcRuntimeState::new(config));
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let (started_tx, started_rx): StartedChannel = sync_channel(1);
    let join = thread::spawn(move || {
        let runtime = match Builder::new_current_thread().enable_all().build() {
            Ok(runtime) => runtime,
            Err(err) => {
                let _ = started_tx.send(Err(format!("failed to build gRPC runtime: {err}")));
                return;
            }
        };
        runtime.block_on(async move {
            let listener = match tokio::net::TcpListener::from_std(listener) {
                Ok(listener) => listener,
                Err(err) => {
                    let _ = started_tx.send(Err(format!("failed to adopt gRPC listener: {err}")));
                    return;
                }
            };
            let incoming = TcpListenerStream::new(listener);
            let service = EnkaiGrpcService { state };
            let _ = started_tx.send(Ok(()));
            let result = Server::builder()
                .add_service(ChatServiceServer::new(service))
                .serve_with_incoming_shutdown(incoming, async move {
                    let _ = shutdown_rx.await;
                })
                .await;
            if let Err(err) = result {
                eprintln!("[grpc] server error: {}", err);
            }
        });
    });
    match started_rx.recv_timeout(Duration::from_secs(3)) {
        Ok(Ok(())) => Ok(GrpcServerHandle {
            shutdown: Some(shutdown_tx),
            join: Some(join),
            addr: local_addr,
        }),
        Ok(Err(err)) => {
            let _ = shutdown_tx.send(());
            let _ = join.join();
            Err(err)
        }
        Err(_) => {
            let _ = shutdown_tx.send(());
            let _ = join.join();
            Err("gRPC server startup timed out".to_string())
        }
    }
}

fn resolve_socket_addr(host: &str, port: u16) -> Result<SocketAddr, String> {
    let mut addrs = (host, port)
        .to_socket_addrs()
        .map_err(|err| format!("failed to resolve gRPC address {host}:{port}: {err}"))?;
    addrs
        .next()
        .ok_or_else(|| format!("failed to resolve gRPC address {host}:{port}"))
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
        let parsed = parse_grpc_probe_args(&[]).expect("defaults");
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
        let handle = start_grpc_server(state, "127.0.0.1", 0).expect("server");
        let payload = execute_grpc_probe(&GrpcProbeArgs {
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
