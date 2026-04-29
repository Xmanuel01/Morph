use std::env;
use std::fs;
use std::net::{SocketAddr, TcpListener, ToSocketAddrs};
use std::path::PathBuf;
use std::sync::mpsc::sync_channel;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use tokio::runtime::Builder;
use tokio::sync::oneshot;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::Server;
use tonic::Request;

use crate::grpc::{
    proto::chat_service_client::ChatServiceClient,
    proto::chat_service_server::ChatServiceServer,
    proto::{ChatRequest, HealthRequest, ReadyRequest},
    EnkaiGrpcService, GrpcLaunchConfig, GrpcRuntimeConfig, GrpcRuntimeState, GrpcServerHandle,
    StartedChannel,
};
use crate::systems::ServeRuntimeManifest;

pub(crate) fn start_from_serve_manifest(
    manifest: &ServeRuntimeManifest,
) -> Result<Option<GrpcServerHandle>, String> {
    let Some(port) = manifest.grpc.port.as_deref() else {
        return Ok(None);
    };
    let port_num = port
        .trim()
        .parse::<u16>()
        .map_err(|_| format!("invalid gRPC port '{}'", port.trim()))?;
    if port_num == 0 {
        return Err("gRPC port must be in range 1..65535".to_string());
    }
    let host = manifest
        .grpc
        .host
        .as_deref()
        .or(manifest.http.host.as_deref())
        .unwrap_or("0.0.0.0")
        .trim()
        .to_string();
    let runtime = manifest
        .grpc_runtime
        .clone()
        .unwrap_or_else(|| crate::systems::default_grpc_runtime_manifest(&manifest.target));
    let launch = GrpcLaunchConfig {
        host,
        port: port_num,
        api_version: runtime.api_version,
        conversation_dir: PathBuf::from(runtime.conversation_dir),
        log_path: runtime.log_path.map(PathBuf::from),
    };
    start_grpc_server_from_launch(&launch)
}

pub(crate) fn maybe_start_grpc_server(
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GrpcProbeArgs {
    pub(crate) address: String,
    pub(crate) api_version: String,
    pub(crate) prompt: String,
    pub(crate) conversation_id: Option<String>,
    pub(crate) json: bool,
    pub(crate) output: Option<PathBuf>,
}

pub(crate) fn print_grpc_usage() {
    eprintln!(
        "  enkai grpc probe [--address <http://host:port>] [--api-version <v>] [--prompt <text>] [--conversation-id <id>] [--json] [--output <file>]"
    );
}

pub(crate) fn grpc_command(args: &[String]) -> i32 {
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
        .unwrap_or_else(|err| serde_json::json!({"error": err.to_string()}).to_string());
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

pub(crate) fn parse_grpc_probe_args(args: &[String]) -> Result<GrpcProbeArgs, String> {
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

pub(crate) fn execute_grpc_probe(args: &GrpcProbeArgs) -> Result<serde_json::Value, String> {
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
            events.push(serde_json::json!({
                "event": item.event,
                "value": item.value,
                "conversation_id": item.conversation_id,
                "api_version": item.api_version,
            }));
        }
        Ok(serde_json::json!({
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

pub(crate) fn start_grpc_server_from_launch(
    config: &GrpcLaunchConfig,
) -> Result<Option<GrpcServerHandle>, String> {
    if config.port == 0 {
        return Err("gRPC port must be in range 1..65535".to_string());
    }
    let runtime = GrpcRuntimeConfig::from_launch_config(config);
    start_grpc_server(runtime, &config.host, config.port).map(Some)
}

pub(crate) fn start_grpc_server(
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
