use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::{Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use enkai_runtime::object::Obj;
use enkai_runtime::Value;
use serde_json::json;

use crate::systems::ServeRuntimeManifest;

pub(crate) fn maybe_execute_backend_contract_service(
    manifest: &ServeRuntimeManifest,
) -> Result<Option<i32>, String> {
    let Some(port) = manifest.http.port.as_deref() else {
        return Ok(None);
    };
    let host = manifest
        .http
        .host
        .as_deref()
        .unwrap_or("0.0.0.0")
        .trim()
        .to_string();
    let port = port
        .trim()
        .parse::<u16>()
        .map_err(|_| format!("invalid HTTP port '{}'", port.trim()))?;
    if port == 0 {
        return Err("HTTP port must be in range 1..65535".to_string());
    }
    let api_version = manifest.http_runtime.api_version.trim().to_string();
    if api_version.is_empty() {
        return Err("serve manifest http_runtime.api_version must not be empty".to_string());
    }
    let target_root = PathBuf::from(&manifest.target);
    let service_root = service_root_for_target(&target_root)?;
    let mode = if service_root
        .join("contracts")
        .join("backend_api.snapshot.json")
        .is_file()
    {
        ServiceTargetMode::BackendContract
    } else {
        ServiceTargetMode::GenericInvoke
    };
    let config = ServiceRuntimeConfig {
        host,
        port,
        api_version,
        target: target_root,
        conversation_dir: PathBuf::from(&manifest.http_runtime.conversation_dir),
        log_path: manifest.http_runtime.log_path.as_ref().map(PathBuf::from),
        contract_mode: std::env::var("ENKAI_CONTRACT_TEST_MODE").is_ok(),
        mode,
    };
    run_service_runtime(&config)?;
    Ok(Some(0))
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ServiceTargetMode {
    BackendContract,
    GenericInvoke,
}

#[derive(Debug, Clone)]
struct ServiceRuntimeConfig {
    host: String,
    port: u16,
    api_version: String,
    target: PathBuf,
    conversation_dir: PathBuf,
    log_path: Option<PathBuf>,
    contract_mode: bool,
    mode: ServiceTargetMode,
}

#[derive(Debug, Clone)]
struct SimpleRequest {
    method: String,
    path: String,
    query: HashMap<String, String>,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}

#[derive(Debug, Clone)]
struct SimpleResponse {
    status: u16,
    content_type: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

fn run_service_runtime(config: &ServiceRuntimeConfig) -> Result<(), String> {
    fs::create_dir_all(&config.conversation_dir).map_err(|err| {
        format!(
            "failed to create {}: {}",
            config.conversation_dir.display(),
            err
        )
    })?;
    let listener = TcpListener::bind((config.host.as_str(), config.port)).map_err(|err| {
        format!(
            "HTTP bind failed for {}:{}: {}",
            config.host, config.port, err
        )
    })?;
    listener
        .set_nonblocking(true)
        .map_err(|err| format!("failed to set nonblocking listener: {}", err))?;
    let deadline = if config.contract_mode {
        Some(std::time::Instant::now() + Duration::from_secs(3))
    } else {
        None
    };
    loop {
        match listener.accept() {
            Ok((mut stream, _addr)) => {
                stream
                    .set_nonblocking(false)
                    .map_err(|err| format!("failed to set blocking client stream: {}", err))?;
                let request = match read_http_request(&mut stream) {
                    Ok(request) => request,
                    Err(err) if config.contract_mode && is_retryable_client_io_error(&err) => {
                        let _ = stream.shutdown(Shutdown::Both);
                        continue;
                    }
                    Err(err) => return Err(err),
                };
                let response = handle_request(config, &request)?;
                match write_http_response(&mut stream, &response) {
                    Ok(()) => {}
                    Err(err) if config.contract_mode && is_retryable_client_io_error(&err) => {
                        let _ = stream.shutdown(Shutdown::Both);
                        continue;
                    }
                    Err(err) => return Err(err),
                }
                let _ = stream.shutdown(Shutdown::Both);
            }
            Err(err)
                if matches!(
                    err.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::ConnectionAborted
                ) =>
            {
                if let Some(deadline) = deadline {
                    if std::time::Instant::now() >= deadline {
                        return Ok(());
                    }
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(err) => return Err(format!("HTTP accept failed: {}", err)),
        }
    }
}

fn is_retryable_client_io_error(err: &str) -> bool {
    let lowered = err.to_ascii_lowercase();
    lowered.contains("connection aborted")
        || lowered.contains("connection reset")
        || lowered.contains("broken pipe")
}

fn handle_request(
    config: &ServiceRuntimeConfig,
    request: &SimpleRequest,
) -> Result<SimpleResponse, String> {
    let base = format!("/api/{}", config.api_version);
    if request.path == format!("{}/health", base) {
        return Ok(json_response(
            200,
            json!({"status": "ok", "api_version": config.api_version}),
        ));
    }
    if request.path == format!("{}/ready", base) {
        return Ok(json_response(
            200,
            json!({"status": "ok", "api_version": config.api_version}),
        ));
    }

    let response = match config.mode {
        ServiceTargetMode::BackendContract => {
            handle_backend_contract_request(config, request, &base)?
        }
        ServiceTargetMode::GenericInvoke => handle_generic_request(config, request, &base)?,
    };

    if let Some(log_path) = &config.log_path {
        let _ = append_log(
            log_path,
            &json!({
                "timestamp_ms": now_ms(),
                "method": request.method,
                "path": request.path,
                "status": response.status,
                "mode": match config.mode {
                    ServiceTargetMode::BackendContract => "backend_contract",
                    ServiceTargetMode::GenericInvoke => "generic_invoke",
                },
            }),
        );
    }

    Ok(response)
}

fn handle_backend_contract_request(
    config: &ServiceRuntimeConfig,
    request: &SimpleRequest,
    base: &str,
) -> Result<SimpleResponse, String> {
    let header_version = request
        .headers
        .get("x-enkai-api-version")
        .map(|value| value.trim().to_string())
        .unwrap_or_default();
    if header_version.is_empty() {
        return Ok(json_response(
            400,
            json!({"code": "missing_api_version_header", "api_version": config.api_version}),
        ));
    }

    if request.path == format!("{}/chat", base) && request.method == "POST" {
        let prompt = request.query.get("prompt").cloned().unwrap_or_default();
        if prompt.trim().is_empty() {
            return Ok(json_response(
                400,
                json!({"code": "missing_prompt", "api_version": config.api_version}),
            ));
        }
        let conversation_id = next_conversation_id(request.query.get("conversation_id"));
        let reply = "hello from enkai";
        save_conversation(config, &conversation_id, &prompt, reply, "chat")?;
        return Ok(json_response(
            200,
            json!({
                "conversation_id": conversation_id,
                "reply": reply,
                "api_version": config.api_version,
            }),
        ));
    }

    if request.path == format!("{}/chat/stream", base) && request.method == "GET" {
        let prompt = request.query.get("prompt").cloned().unwrap_or_default();
        if prompt.trim().is_empty() {
            return Ok(json_response(
                400,
                json!({"code": "missing_prompt", "api_version": config.api_version}),
            ));
        }
        let conversation_id = next_conversation_id(request.query.get("conversation_id"));
        let reply = "hello from enkai";
        save_conversation(config, &conversation_id, &prompt, reply, "stream")?;
        let token = json!({
            "event": "token",
            "value": "hello from ",
            "conversation_id": conversation_id,
            "api_version": config.api_version,
        });
        let done = json!({
            "event": "done",
            "value": reply,
            "conversation_id": conversation_id,
            "api_version": config.api_version,
        });
        let body = format!(
            "data: {}\n\ndata: {}\n\n",
            serde_json::to_string(&token).map_err(|err| err.to_string())?,
            serde_json::to_string(&done).map_err(|err| err.to_string())?
        )
        .into_bytes();
        return Ok(SimpleResponse {
            status: 200,
            content_type: "text/event-stream".to_string(),
            headers: Vec::new(),
            body,
        });
    }

    if request.path == format!("{}/chat/ws", base) && request.method == "GET" {
        return Ok(json_response(
            426,
            json!({"code": "websocket_upgrade_required", "api_version": config.api_version}),
        ));
    }

    Ok(json_response(
        404,
        json!({"code": "not_found", "api_version": config.api_version}),
    ))
}

fn handle_generic_request(
    config: &ServiceRuntimeConfig,
    request: &SimpleRequest,
    base: &str,
) -> Result<SimpleResponse, String> {
    if request.method == "GET" {
        if let Some(response) = maybe_serve_static_asset(config, request)? {
            return Ok(response);
        }
    }
    if let Some(route_match) = resolve_route_handler(config, request, base)? {
        let executed = execute_target_for_request(
            config,
            request,
            Some(&route_match.params),
            &route_match.entry,
        )?;
        return Ok(executed.into_response(&config.api_version));
    }
    let is_invoke_path =
        request.path == "/" || request.path == base || request.path == format!("{}/invoke", base);
    if !is_invoke_path {
        return Ok(json_response(
            404,
            json!({"code": "not_found", "api_version": config.api_version}),
        ));
    }

    let executed = execute_target_for_request(
        config,
        request,
        None,
        &crate::resolve_entry(&config.target)?.1,
    )?;
    Ok(executed.into_response(&config.api_version))
}

fn maybe_serve_static_asset(
    config: &ServiceRuntimeConfig,
    request: &SimpleRequest,
) -> Result<Option<SimpleResponse>, String> {
    let service_root = service_root_for_target(&config.target)?;
    for static_root in [service_root.join("public"), service_root.join("static")] {
        if !static_root.is_dir() {
            continue;
        }
        let asset_path = resolve_static_asset_path(&static_root, &request.path);
        if let Some(asset_path) = asset_path {
            if asset_path.is_file() {
                let body = fs::read(&asset_path).map_err(|err| {
                    format!(
                        "failed to read static asset {}: {}",
                        asset_path.display(),
                        err
                    )
                })?;
                return Ok(Some(SimpleResponse {
                    status: 200,
                    content_type: content_type_for_path(&asset_path).to_string(),
                    headers: Vec::new(),
                    body,
                }));
            }
        }
    }
    Ok(None)
}

fn resolve_static_asset_path(static_root: &Path, request_path: &str) -> Option<PathBuf> {
    let trimmed = request_path.trim();
    if !trimmed.starts_with('/') {
        return None;
    }
    let relative = if trimmed == "/" {
        PathBuf::from("index.html")
    } else {
        let candidate = trimmed.trim_start_matches('/');
        if candidate.is_empty() {
            PathBuf::from("index.html")
        } else {
            PathBuf::from(candidate)
        }
    };
    let mut resolved = static_root.to_path_buf();
    for component in relative.components() {
        match component {
            std::path::Component::Normal(value) => resolved.push(value),
            _ => return None,
        }
    }
    Some(resolved)
}

fn content_type_for_path(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("")
    {
        "html" => "text/html; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "js" => "application/javascript; charset=utf-8",
        "json" => "application/json",
        "txt" => "text/plain; charset=utf-8",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        _ => "application/octet-stream",
    }
}

#[derive(Debug, Clone)]
struct GenericExecutionResult {
    value: serde_json::Value,
    response_override: Option<ResponseOverride>,
}

#[derive(Debug, Clone)]
struct RouteMatch {
    entry: PathBuf,
    params: BTreeMap<String, String>,
}

#[derive(Debug, Clone)]
struct ResponseOverride {
    status: u16,
    content_type: Option<String>,
    headers: Vec<(String, String)>,
    body: serde_json::Value,
}

impl GenericExecutionResult {
    fn into_response(self, api_version: &str) -> SimpleResponse {
        if let Some(override_response) = self.response_override {
            let status = override_response.status;
            let headers = override_response.headers;
            if let Some(text_body) = override_response.body.as_str() {
                return SimpleResponse {
                    status,
                    content_type: override_response
                        .content_type
                        .unwrap_or_else(|| "text/plain; charset=utf-8".to_string()),
                    headers,
                    body: text_body.as_bytes().to_vec(),
                };
            }
            return SimpleResponse {
                status,
                content_type: override_response
                    .content_type
                    .unwrap_or_else(|| "application/json".to_string()),
                headers,
                body: serde_json::to_vec(&override_response.body)
                    .unwrap_or_else(|_| b"{}".to_vec()),
            };
        }
        json_response(
            200,
            json!({
                "api_version": api_version,
                "result": self.value,
            }),
        )
    }
}

fn execute_target_for_request(
    config: &ServiceRuntimeConfig,
    request: &SimpleRequest,
    route_params: Option<&BTreeMap<String, String>>,
    entry: &Path,
) -> Result<GenericExecutionResult, String> {
    let env_values = request_env_projection(config, request, route_params)?;
    let value = run_program_value_with_env(entry, &env_values)?;
    let json_value = value_to_json(&value);
    let response_override = response_override_from_value(&json_value);
    Ok(GenericExecutionResult {
        value: json_value,
        response_override,
    })
}

fn request_env_projection(
    config: &ServiceRuntimeConfig,
    request: &SimpleRequest,
    route_params: Option<&BTreeMap<String, String>>,
) -> Result<BTreeMap<String, String>, String> {
    let mut map = BTreeMap::new();
    map.insert(
        "ENKAI_SERVE_REQUEST_METHOD".to_string(),
        request.method.clone(),
    );
    map.insert("ENKAI_SERVE_REQUEST_PATH".to_string(), request.path.clone());
    map.insert(
        "ENKAI_SERVE_REQUEST_QUERY".to_string(),
        serde_json::to_string(&request.query).map_err(|err| err.to_string())?,
    );
    map.insert(
        "ENKAI_SERVE_REQUEST_HEADERS".to_string(),
        serde_json::to_string(&request.headers).map_err(|err| err.to_string())?,
    );
    map.insert(
        "ENKAI_SERVE_REQUEST_BODY".to_string(),
        String::from_utf8_lossy(&request.body).to_string(),
    );
    map.insert(
        "ENKAI_SERVE_REQUEST_API_VERSION".to_string(),
        config.api_version.clone(),
    );
    if let Some(route_params) = route_params {
        map.insert(
            "ENKAI_SERVE_ROUTE_PARAMS".to_string(),
            serde_json::to_string(route_params).map_err(|err| err.to_string())?,
        );
        for (key, value) in route_params {
            map.insert(format!("ENKAI_SERVE_ROUTE_PARAM_{}", key), value.clone());
        }
    }
    Ok(map)
}

fn resolve_route_handler(
    config: &ServiceRuntimeConfig,
    request: &SimpleRequest,
    base: &str,
) -> Result<Option<RouteMatch>, String> {
    let route_tail = request
        .path
        .strip_prefix(base)
        .unwrap_or(request.path.as_str())
        .trim_start_matches('/');
    if route_tail.is_empty() || route_tail == "invoke" {
        return Ok(None);
    }
    let segments = route_tail
        .split('/')
        .filter(|segment| !segment.trim().is_empty())
        .collect::<Vec<_>>();
    if segments.is_empty() {
        return Ok(None);
    }
    let service_root = service_root_for_target(&config.target)?;
    let route_roots = [
        service_root
            .join("routes")
            .join(request.method.to_ascii_lowercase()),
        service_root.join("routes").join("any"),
        service_root.join("routes").join("_any"),
        service_root.join("routes"),
        service_root
            .join("src")
            .join("routes")
            .join(request.method.to_ascii_lowercase()),
        service_root.join("src").join("routes").join("any"),
        service_root.join("src").join("routes").join("_any"),
        service_root.join("src").join("routes"),
    ];
    for root in route_roots {
        if !root.is_dir() {
            continue;
        }
        if let Some(found) = resolve_route_from_root(&root, &segments, &BTreeMap::new())? {
            return Ok(Some(found));
        }
    }
    Ok(None)
}

fn resolve_route_from_root(
    root: &Path,
    segments: &[&str],
    params: &BTreeMap<String, String>,
) -> Result<Option<RouteMatch>, String> {
    if segments.is_empty() {
        let index = root.join("index.enk");
        if index.is_file() {
            return Ok(Some(RouteMatch {
                entry: index,
                params: params.clone(),
            }));
        }
        return Ok(None);
    }

    let segment = segments[0];
    let rest = &segments[1..];

    if rest.is_empty() {
        let exact_file = root.join(format!("{}.enk", segment));
        if exact_file.is_file() {
            return Ok(Some(RouteMatch {
                entry: exact_file,
                params: params.clone(),
            }));
        }
    }

    let exact_dir = root.join(segment);
    if exact_dir.is_dir() {
        if let Some(found) = resolve_route_from_root(&exact_dir, rest, params)? {
            return Ok(Some(found));
        }
    }

    let entries = fs::read_dir(root)
        .map_err(|err| format!("failed to read route dir {}: {}", root.display(), err))?;
    let mut entries = entries
        .filter_map(|entry| entry.ok())
        .collect::<Vec<std::fs::DirEntry>>();
    entries.sort_by_key(|entry| entry.file_name().to_string_lossy().to_string());
    let mut dynamic_dirs = Vec::new();
    let mut dynamic_files = Vec::new();
    let mut catchall_dirs = Vec::new();
    let mut catchall_files = Vec::new();
    for entry in entries {
        let file_name = entry.file_name();
        let name = file_name.to_string_lossy().to_string();
        if entry.path().is_dir() {
            if let Some(param_name) = parse_catchall_segment_name(&name) {
                catchall_dirs.push((entry.path(), param_name));
            } else if let Some(param_name) = parse_dynamic_segment_name(&name) {
                dynamic_dirs.push((entry.path(), param_name));
            }
        } else if let Some(param_name) = parse_catchall_file_name(&name) {
            catchall_files.push((entry.path(), param_name));
        } else if let Some(param_name) = parse_dynamic_file_name(&name) {
            if rest.is_empty() {
                dynamic_files.push((entry.path(), param_name));
            }
        } else if let Some(param_name) = parse_catchall_segment_name(&name) {
            catchall_files.push((entry.path(), param_name));
        } else if let Some(param_name) = parse_dynamic_segment_name(&name) {
            dynamic_files.push((entry.path(), param_name));
        }
    }

    if let Some((path, param_name)) = dynamic_files.into_iter().next() {
        let mut next_params = params.clone();
        next_params.insert(param_name, segment.to_string());
        return Ok(Some(RouteMatch {
            entry: path,
            params: next_params,
        }));
    }

    if let Some((path, param_name)) = dynamic_dirs.into_iter().next() {
        let mut next_params = params.clone();
        next_params.insert(param_name, segment.to_string());
        if let Some(found) = resolve_route_from_root(&path, rest, &next_params)? {
            return Ok(Some(found));
        }
    }

    if let Some((path, param_name)) = catchall_files.into_iter().next() {
        let mut next_params = params.clone();
        next_params.insert(
            param_name,
            std::iter::once(segment)
                .chain(rest.iter().copied())
                .collect::<Vec<_>>()
                .join("/"),
        );
        return Ok(Some(RouteMatch {
            entry: path,
            params: next_params,
        }));
    }

    if let Some((path, param_name)) = catchall_dirs.into_iter().next() {
        let mut next_params = params.clone();
        next_params.insert(
            param_name,
            std::iter::once(segment)
                .chain(rest.iter().copied())
                .collect::<Vec<_>>()
                .join("/"),
        );
        if let Some(found) = resolve_route_from_root(&path, &[], &next_params)? {
            return Ok(Some(found));
        }
    }

    Ok(None)
}

fn parse_dynamic_segment_name(name: &str) -> Option<String> {
    if name.starts_with(':') && name.len() > 1 {
        return Some(name.trim_start_matches(':').to_string());
    }
    if name.starts_with('[') && name.ends_with(']') && name.len() > 2 {
        if name.starts_with("[...") {
            return None;
        }
        Some(name[1..name.len() - 1].to_string())
    } else {
        None
    }
}

fn parse_dynamic_file_name(name: &str) -> Option<String> {
    if !name.ends_with(".enk") {
        return None;
    }
    parse_dynamic_segment_name(name.trim_end_matches(".enk"))
}

fn parse_catchall_segment_name(name: &str) -> Option<String> {
    if name.starts_with("[...") && name.ends_with(']') && name.len() > 5 {
        return Some(name[4..name.len() - 1].to_string());
    }
    if name.starts_with("...") && name.len() > 3 {
        return Some(name[3..].to_string());
    }
    None
}

fn parse_catchall_file_name(name: &str) -> Option<String> {
    if !name.ends_with(".enk") {
        return None;
    }
    parse_catchall_segment_name(name.trim_end_matches(".enk"))
}

fn run_program_value_with_env(
    entry: &Path,
    env_values: &BTreeMap<String, String>,
) -> Result<Value, String> {
    crate::program_runtime::run_program_value_with_env(entry, env_values)
}

fn response_override_from_value(value: &serde_json::Value) -> Option<ResponseOverride> {
    let map = value.as_object()?;
    if !map.contains_key("body") && !map.contains_key("status") {
        return None;
    }
    let status = map
        .get("status")
        .and_then(|value| value.as_u64())
        .and_then(|value| u16::try_from(value).ok())
        .unwrap_or(200);
    let content_type = map
        .get("content_type")
        .and_then(|value| value.as_str())
        .map(|value| value.to_string());
    let headers = map
        .get("headers")
        .and_then(|value| value.as_object())
        .map(|headers| {
            headers
                .iter()
                .filter_map(|(key, value)| {
                    value
                        .as_str()
                        .map(|text| (key.replace('_', "-"), text.to_string()))
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let body = map.get("body").cloned().unwrap_or(serde_json::Value::Null);
    Some(ResponseOverride {
        status,
        content_type,
        headers,
        body,
    })
}

fn value_to_json(value: &Value) -> serde_json::Value {
    match value {
        Value::Null => serde_json::Value::Null,
        Value::Bool(value) => serde_json::Value::Bool(*value),
        Value::Int(value) => json!(value),
        Value::Float(value) => json!(value),
        Value::Obj(obj) => match obj.as_obj() {
            Obj::String(value) => serde_json::Value::String(value.to_string()),
            Obj::Json(value) => value.clone(),
            Obj::List(items) => serde_json::Value::Array(
                items.borrow().iter().map(value_to_json).collect::<Vec<_>>(),
            ),
            Obj::Record(map) => {
                let map = map.borrow();
                let mut out = serde_json::Map::with_capacity(map.len());
                for (key, value) in map.iter() {
                    out.insert(key.clone(), value_to_json(value));
                }
                serde_json::Value::Object(out)
            }
            other => serde_json::Value::String(format!("<{:?}>", other)),
        },
    }
}

fn service_root_for_target(target: &Path) -> Result<PathBuf, String> {
    if target.is_dir() {
        return Ok(target.to_path_buf());
    }
    target
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| format!("invalid serve target {}", target.display()))
}

fn read_http_request(stream: &mut TcpStream) -> Result<SimpleRequest, String> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 1024];
    let mut header_end = None;
    loop {
        let read = stream.read(&mut chunk).map_err(|err| err.to_string())?;
        if read == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..read]);
        if header_end.is_none() {
            header_end = buf.windows(4).position(|window| window == b"\r\n\r\n");
        }
        if header_end.is_some() || buf.len() >= 32 * 1024 {
            break;
        }
    }
    let header_end = header_end.ok_or_else(|| "invalid HTTP request".to_string())?;
    let header_bytes = &buf[..header_end + 4];
    let header_text = String::from_utf8_lossy(header_bytes);
    let mut lines = header_text[..header_text.len() - 4].split("\r\n");
    let request_line = lines
        .next()
        .ok_or_else(|| "missing request line".to_string())?;
    let mut request_parts = request_line.split_whitespace();
    let method = request_parts
        .next()
        .ok_or_else(|| "missing HTTP method".to_string())?
        .to_string();
    let raw_path = request_parts
        .next()
        .ok_or_else(|| "missing HTTP path".to_string())?;
    let (path, query) = split_path_and_query(raw_path);
    let mut headers = HashMap::new();
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_string());
        }
    }
    let body_len = headers
        .get("content-length")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let mut body = buf[(header_end + 4)..].to_vec();
    while body.len() < body_len {
        let read = stream.read(&mut chunk).map_err(|err| err.to_string())?;
        if read == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..read]);
    }
    if body.len() > body_len {
        body.truncate(body_len);
    }
    Ok(SimpleRequest {
        method,
        path,
        query,
        headers,
        body,
    })
}

fn write_http_response(stream: &mut TcpStream, response: &SimpleResponse) -> Result<(), String> {
    let status_text = match response.status {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        426 => "Upgrade Required",
        500 => "Internal Server Error",
        _ => "OK",
    };
    let mut header_lines = String::new();
    for (name, value) in &response.headers {
        header_lines.push_str(name);
        header_lines.push_str(": ");
        header_lines.push_str(value);
        header_lines.push_str("\r\n");
    }
    let headers = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n{}\r\n",
        response.status,
        status_text,
        response.content_type,
        response.body.len(),
        header_lines,
    );
    stream
        .write_all(headers.as_bytes())
        .and_then(|_| stream.write_all(&response.body))
        .map_err(|err| err.to_string())
}

fn split_path_and_query(raw_path: &str) -> (String, HashMap<String, String>) {
    let mut query = HashMap::new();
    let (path, query_text) = match raw_path.split_once('?') {
        Some((path, query_text)) => (url_decode(path), Some(query_text)),
        None => (url_decode(raw_path), None),
    };
    if let Some(query_text) = query_text {
        for pair in query_text.split('&') {
            if pair.trim().is_empty() {
                continue;
            }
            let (key, value) = match pair.split_once('=') {
                Some((key, value)) => (key, value),
                None => (pair, ""),
            };
            query.insert(url_decode(key), url_decode(value));
        }
    }
    (path, query)
}

fn url_decode(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut idx = 0usize;
    while idx < bytes.len() {
        match bytes[idx] {
            b'+' => {
                out.push(b' ');
                idx += 1;
            }
            b'%' if idx + 2 < bytes.len() => {
                let hi = bytes[idx + 1];
                let lo = bytes[idx + 2];
                let decoded = hex_value(hi).and_then(|hi| hex_value(lo).map(|lo| (hi << 4) | lo));
                if let Some(decoded) = decoded {
                    out.push(decoded);
                    idx += 3;
                } else {
                    out.push(bytes[idx]);
                    idx += 1;
                }
            }
            byte => {
                out.push(byte);
                idx += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).to_string()
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn conversation_path(config: &ServiceRuntimeConfig) -> PathBuf {
    config.conversation_dir.join("conversation_state.json")
}

fn conversation_backup_path(config: &ServiceRuntimeConfig) -> PathBuf {
    config
        .conversation_dir
        .join("conversation_state.backup.json")
}

fn save_conversation(
    config: &ServiceRuntimeConfig,
    conversation_id: &str,
    prompt: &str,
    reply: &str,
    mode: &str,
) -> Result<(), String> {
    let payload = json!({
        "schema_version": 1,
        "conversation_id": conversation_id,
        "prompt": prompt,
        "reply": reply,
        "mode": mode,
        "updated_ms": now_ms(),
    });
    let text = serde_json::to_string_pretty(&payload).map_err(|err| err.to_string())?;
    fs::write(conversation_path(config), &text).map_err(|err| err.to_string())?;
    fs::write(conversation_backup_path(config), &text).map_err(|err| err.to_string())
}

fn append_log(path: &Path, value: &serde_json::Value) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }
    let mut lines = if path.is_file() {
        fs::read_to_string(path).map_err(|err| err.to_string())?
    } else {
        String::new()
    };
    if !lines.is_empty() && !lines.ends_with('\n') {
        lines.push('\n');
    }
    lines.push_str(&serde_json::to_string(value).map_err(|err| err.to_string())?);
    lines.push('\n');
    fs::write(path, lines).map_err(|err| err.to_string())
}

fn next_conversation_id(existing: Option<&String>) -> String {
    existing
        .cloned()
        .unwrap_or_else(|| format!("conv-{}", now_ms()))
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn json_response(status: u16, value: serde_json::Value) -> SimpleResponse {
    SimpleResponse {
        status,
        content_type: "application/json".to_string(),
        headers: Vec::new(),
        body: serde_json::to_vec(&value).unwrap_or_else(|_| b"{}".to_vec()),
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::{Mutex, OnceLock};
    use std::thread;
    use std::time::{Duration, Instant};

    use tempfile::tempdir;

    use super::*;

    fn contract_test_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    struct ContractModeGuard;

    impl ContractModeGuard {
        fn new() -> Self {
            unsafe {
                std::env::set_var("ENKAI_CONTRACT_TEST_MODE", "1");
            }
            Self
        }
    }

    impl Drop for ContractModeGuard {
        fn drop(&mut self) {
            unsafe {
                std::env::remove_var("ENKAI_CONTRACT_TEST_MODE");
            }
        }
    }
    use crate::systems::{
        HttpRuntimeManifest, ServeModelManifest, ServeRuntimeManifest, ServiceBindingManifest,
    };

    #[test]
    fn backend_contract_service_serves_health_and_chat_routes() {
        let _lock = contract_test_lock().lock().expect("test lock");
        let dir = tempdir().expect("tempdir");
        let target = dir.path().join("backend");
        fs::create_dir_all(target.join("contracts")).expect("contracts");
        fs::write(
            target.join("contracts").join("backend_api.snapshot.json"),
            "{}\n",
        )
        .expect("snapshot");
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("addr").port();
        drop(listener);

        let manifest = ServeRuntimeManifest {
            schema_version: 1,
            profile: "serve_runtime_manifest".to_string(),
            target: target.display().to_string(),
            runtime_flags: Vec::new(),
            http: ServiceBindingManifest {
                host: Some("127.0.0.1".to_string()),
                port: Some(port.to_string()),
            },
            grpc: ServiceBindingManifest {
                host: None,
                port: None,
            },
            model: ServeModelManifest::None,
            http_runtime: HttpRuntimeManifest {
                api_version: "v1".to_string(),
                conversation_dir: target.join("state").display().to_string(),
                log_path: None,
            },
            grpc_runtime: None,
            env_projection: BTreeMap::new(),
        };

        let _contract_mode = ContractModeGuard::new();
        let server = thread::spawn(move || maybe_execute_backend_contract_service(&manifest));
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut health = Vec::new();
        while Instant::now() < deadline {
            match TcpStream::connect(("127.0.0.1", port)) {
                Ok(mut stream) => {
                    let request = "GET /api/v1/health HTTP/1.1\r\nHost: localhost\r\nx-enkai-api-version: v1\r\nConnection: close\r\n\r\n";
                    stream.write_all(request.as_bytes()).expect("write");
                    stream.read_to_end(&mut health).expect("read");
                    break;
                }
                Err(_) => thread::sleep(Duration::from_millis(20)),
            }
        }
        let server_result = server.join().expect("join").expect("service");
        assert_eq!(server_result, Some(0));
        let health_text = String::from_utf8_lossy(&health);
        assert!(health_text.contains("200 OK"));
        assert!(health_text.contains("\"api_version\":\"v1\""));
    }

    #[test]
    fn generic_service_serves_static_assets() {
        let _lock = contract_test_lock().lock().expect("test lock");
        let dir = tempdir().expect("tempdir");
        let public_dir = dir.path().join("public");
        fs::create_dir_all(&public_dir).expect("public dir");
        fs::write(public_dir.join("index.html"), "<h1>hello static</h1>").expect("index");
        let target = dir.path().join("hello.enk");
        fs::write(&target, "fn main() -> Int ::\n    return 7\n::\nmain()\n").expect("target");
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("addr").port();
        drop(listener);

        let manifest = ServeRuntimeManifest {
            schema_version: 1,
            profile: "serve_runtime_manifest".to_string(),
            target: target.display().to_string(),
            runtime_flags: Vec::new(),
            http: ServiceBindingManifest {
                host: Some("127.0.0.1".to_string()),
                port: Some(port.to_string()),
            },
            grpc: ServiceBindingManifest {
                host: None,
                port: None,
            },
            model: ServeModelManifest::None,
            http_runtime: HttpRuntimeManifest {
                api_version: "v1".to_string(),
                conversation_dir: dir.path().join("state").display().to_string(),
                log_path: None,
            },
            grpc_runtime: None,
            env_projection: BTreeMap::new(),
        };

        let _contract_mode = ContractModeGuard::new();
        let server = thread::spawn(move || maybe_execute_backend_contract_service(&manifest));
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut response = Vec::new();
        while Instant::now() < deadline {
            match TcpStream::connect(("127.0.0.1", port)) {
                Ok(mut stream) => {
                    let request = "GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n";
                    stream.write_all(request.as_bytes()).expect("write");
                    stream.read_to_end(&mut response).expect("read");
                    break;
                }
                Err(_) => thread::sleep(Duration::from_millis(20)),
            }
        }
        let server_result = server.join().expect("join").expect("service");
        assert_eq!(server_result, Some(0));
        let response_text = String::from_utf8_lossy(&response);
        assert!(response_text.contains("200 OK"));
        assert!(response_text.contains("<h1>hello static</h1>"));
    }

    #[test]
    fn generic_service_executes_target_without_vm_http_server() {
        let _lock = contract_test_lock().lock().expect("test lock");
        let dir = tempdir().expect("tempdir");
        let target = dir.path().join("hello.enk");
        fs::write(&target, "fn main() -> Int ::\n    return 7\n::\nmain()\n").expect("target");
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("addr").port();
        drop(listener);

        let manifest = ServeRuntimeManifest {
            schema_version: 1,
            profile: "serve_runtime_manifest".to_string(),
            target: target.display().to_string(),
            runtime_flags: Vec::new(),
            http: ServiceBindingManifest {
                host: Some("127.0.0.1".to_string()),
                port: Some(port.to_string()),
            },
            grpc: ServiceBindingManifest {
                host: None,
                port: None,
            },
            model: ServeModelManifest::None,
            http_runtime: HttpRuntimeManifest {
                api_version: "v1".to_string(),
                conversation_dir: dir.path().join("state").display().to_string(),
                log_path: None,
            },
            grpc_runtime: None,
            env_projection: BTreeMap::new(),
        };

        let _contract_mode = ContractModeGuard::new();
        let server = thread::spawn(move || maybe_execute_backend_contract_service(&manifest));
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut response = Vec::new();
        while Instant::now() < deadline {
            match TcpStream::connect(("127.0.0.1", port)) {
                Ok(mut stream) => {
                    let request = "GET /api/v1/invoke HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n";
                    stream.write_all(request.as_bytes()).expect("write");
                    stream.read_to_end(&mut response).expect("read");
                    break;
                }
                Err(_) => thread::sleep(Duration::from_millis(20)),
            }
        }
        let server_result = server.join().expect("join").expect("service");
        assert_eq!(server_result, Some(0));
        let response_text = String::from_utf8_lossy(&response);
        assert!(response_text.contains("200 OK"));
        assert!(response_text.contains("\"result\":7"));
    }

    #[test]
    fn generic_service_executes_exact_and_dynamic_route_handlers() {
        let _lock = contract_test_lock().lock().expect("test lock");
        let dir = tempdir().expect("tempdir");
        let routes_root = dir.path().join("routes");
        fs::create_dir_all(routes_root.join("get").join("users")).expect("routes");
        fs::write(
            dir.path().join("main.enk"),
            "fn main() ::\n    return 0\n::\nmain()\n",
        )
        .expect("main");
        fs::write(
            routes_root.join("get").join("ping.enk"),
            "fn main() ::\n\
    let resp := json.parse(\"{}\")\n\
    let body := json.parse(\"{}\")\n\
    body.route := \"ping\"\n\
    resp.body := body\n\
    return resp\n\
::\n\
main()\n",
        )
        .expect("ping route");
        fs::write(
            routes_root.join("get").join("users").join("[id].enk"),
            "import std::env\n\
policy default ::\n\
    allow env\n\
::\n\
fn main() ::\n\
    let resp := json.parse(\"{}\")\n\
    let body := json.parse(\"{}\")\n\
    body.route := \"user_show\"\n\
    body.id := env.get(\"ENKAI_SERVE_ROUTE_PARAM_id\")?\n\
    resp.body := body\n\
    return resp\n\
::\n\
main()\n",
        )
        .expect("user route");
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("addr").port();
        drop(listener);

        let manifest = ServeRuntimeManifest {
            schema_version: 1,
            profile: "serve_runtime_manifest".to_string(),
            target: dir.path().join("main.enk").display().to_string(),
            runtime_flags: Vec::new(),
            http: ServiceBindingManifest {
                host: Some("127.0.0.1".to_string()),
                port: Some(port.to_string()),
            },
            grpc: ServiceBindingManifest {
                host: None,
                port: None,
            },
            model: ServeModelManifest::None,
            http_runtime: HttpRuntimeManifest {
                api_version: "v1".to_string(),
                conversation_dir: dir.path().join("state").display().to_string(),
                log_path: None,
            },
            grpc_runtime: None,
            env_projection: BTreeMap::new(),
        };

        let _contract_mode = ContractModeGuard::new();
        let server = thread::spawn(move || maybe_execute_backend_contract_service(&manifest));
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut ping_response = Vec::new();
        while Instant::now() < deadline {
            match TcpStream::connect(("127.0.0.1", port)) {
                Ok(mut stream) => {
                    stream
                        .write_all(
                            b"GET /api/v1/ping HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
                        )
                        .expect("write ping");
                    stream.read_to_end(&mut ping_response).expect("read ping");
                    break;
                }
                Err(_) => thread::sleep(Duration::from_millis(20)),
            }
        }
        let mut user_response = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(2);
        while Instant::now() < deadline {
            match TcpStream::connect(("127.0.0.1", port)) {
                Ok(mut stream) => {
                    stream
                        .write_all(
                            b"GET /api/v1/users/42 HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
                        )
                        .expect("write user");
                    stream.read_to_end(&mut user_response).expect("read user");
                    break;
                }
                Err(_) => thread::sleep(Duration::from_millis(20)),
            }
        }
        let server_result = server.join().expect("join").expect("service");
        assert_eq!(server_result, Some(0));
        let ping_text = String::from_utf8_lossy(&ping_response);
        assert!(ping_text.contains("\"route\":\"ping\""));
        let user_text = String::from_utf8_lossy(&user_response);
        assert!(user_text.contains("\"route\":\"user_show\""));
        assert!(user_text.contains("\"id\":\"42\""));
    }

    #[test]
    fn generic_service_executes_post_route_with_response_override() {
        let _lock = contract_test_lock().lock().expect("test lock");
        let dir = tempdir().expect("tempdir");
        let routes_root = dir.path().join("routes");
        fs::create_dir_all(routes_root.join("post")).expect("routes");
        fs::write(
            dir.path().join("main.enk"),
            "fn main() ::\n    return 0\n::\nmain()\n",
        )
        .expect("main");
        fs::write(
            routes_root.join("post").join("echo.enk"),
            "import std::env\n\
policy default ::\n\
    allow env\n\
::\n\
fn main() ::\n\
    let resp := json.parse(\"{}\")\n\
    let headers := json.parse(\"{}\")\n\
    let body := json.parse(\"{}\")\n\
    headers.x_route_mode := \"post\"\n\
    body.echo := env.get(\"ENKAI_SERVE_REQUEST_BODY\")?\n\
    resp.status := 201\n\
    resp.headers := headers\n\
    resp.body := body\n\
    return resp\n\
::\n\
main()\n",
        )
        .expect("post route");
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("addr").port();
        drop(listener);

        let manifest = ServeRuntimeManifest {
            schema_version: 1,
            profile: "serve_runtime_manifest".to_string(),
            target: dir.path().join("main.enk").display().to_string(),
            runtime_flags: Vec::new(),
            http: ServiceBindingManifest {
                host: Some("127.0.0.1".to_string()),
                port: Some(port.to_string()),
            },
            grpc: ServiceBindingManifest {
                host: None,
                port: None,
            },
            model: ServeModelManifest::None,
            http_runtime: HttpRuntimeManifest {
                api_version: "v1".to_string(),
                conversation_dir: dir.path().join("state").display().to_string(),
                log_path: None,
            },
            grpc_runtime: None,
            env_projection: BTreeMap::new(),
        };

        let _contract_mode = ContractModeGuard::new();
        let server = thread::spawn(move || maybe_execute_backend_contract_service(&manifest));
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut response = Vec::new();
        while Instant::now() < deadline {
            match TcpStream::connect(("127.0.0.1", port)) {
                Ok(mut stream) => {
                    stream
                        .write_all(
                            b"POST /api/v1/echo HTTP/1.1\r\nHost: localhost\r\nContent-Length: 10\r\nConnection: close\r\n\r\nhello body",
                        )
                        .expect("write post");
                    stream.read_to_end(&mut response).expect("read post");
                    break;
                }
                Err(_) => thread::sleep(Duration::from_millis(20)),
            }
        }
        let server_result = server.join().expect("join").expect("service");
        assert_eq!(server_result, Some(0));
        let response_text = String::from_utf8_lossy(&response);
        assert!(response_text.contains("201"));
        assert!(response_text.contains("x-route-mode: post"));
        assert!(response_text.contains("\"echo\":\"hello body\""));
    }

    #[test]
    fn generic_service_executes_any_nested_and_catchall_routes() {
        let _lock = contract_test_lock().lock().expect("test lock");
        let dir = tempdir().expect("tempdir");
        let routes_root = dir.path().join("routes");
        fs::create_dir_all(routes_root.join("any").join("files")).expect("any routes");
        fs::create_dir_all(routes_root.join("any").join("files").join("...path"))
            .expect("catchall routes");
        fs::create_dir_all(
            routes_root
                .join("get")
                .join("posts")
                .join("[post_id]")
                .join("comments"),
        )
        .expect("nested routes");
        fs::write(
            dir.path().join("main.enk"),
            "fn main() ::\n    return 0\n::\nmain()\n",
        )
        .expect("main");
        fs::write(
            routes_root.join("any").join("status.enk"),
            "fn main() ::\n\
    let resp := json.parse(\"{}\")\n\
    let body := json.parse(\"{}\")\n\
    body.route := \"any_status\"\n\
    resp.body := body\n\
    return resp\n\
::\n\
main()\n",
        )
        .expect("any route");
        fs::write(
            routes_root
                .join("get")
                .join("posts")
                .join("[post_id]")
                .join("comments")
                .join("[comment_id].enk"),
            "import std::env\n\
policy default ::\n\
    allow env\n\
::\n\
fn main() ::\n\
    let resp := json.parse(\"{}\")\n\
    let body := json.parse(\"{}\")\n\
    body.route := \"nested_comment\"\n\
    body.post := env.get(\"ENKAI_SERVE_ROUTE_PARAM_post_id\")?\n\
    body.comment := env.get(\"ENKAI_SERVE_ROUTE_PARAM_comment_id\")?\n\
    resp.body := body\n\
    return resp\n\
::\n\
main()\n",
        )
        .expect("nested route");
        fs::write(
            routes_root
                .join("any")
                .join("files")
                .join("...path")
                .join("index.enk"),
            "import std::env\n\
policy default ::\n\
    allow env\n\
::\n\
fn main() ::\n\
    let resp := json.parse(\"{}\")\n\
    let body := json.parse(\"{}\")\n\
    body.route := \"catchall\"\n\
    body.path := env.get(\"ENKAI_SERVE_ROUTE_PARAM_path\")?\n\
    resp.body := body\n\
    return resp\n\
::\n\
main()\n",
        )
        .expect("catchall route");
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("addr").port();
        drop(listener);

        let manifest = ServeRuntimeManifest {
            schema_version: 1,
            profile: "serve_runtime_manifest".to_string(),
            target: dir.path().join("main.enk").display().to_string(),
            runtime_flags: Vec::new(),
            http: ServiceBindingManifest {
                host: Some("127.0.0.1".to_string()),
                port: Some(port.to_string()),
            },
            grpc: ServiceBindingManifest {
                host: None,
                port: None,
            },
            model: ServeModelManifest::None,
            http_runtime: HttpRuntimeManifest {
                api_version: "v1".to_string(),
                conversation_dir: dir.path().join("state").display().to_string(),
                log_path: None,
            },
            grpc_runtime: None,
            env_projection: BTreeMap::new(),
        };

        let _contract_mode = ContractModeGuard::new();
        let server = thread::spawn(move || maybe_execute_backend_contract_service(&manifest));

        let mut any_response = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(2);
        while Instant::now() < deadline {
            match TcpStream::connect(("127.0.0.1", port)) {
                Ok(mut stream) => {
                    stream
                        .write_all(
                            b"PATCH /api/v1/status HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
                        )
                        .expect("write any");
                    stream.read_to_end(&mut any_response).expect("read any");
                    break;
                }
                Err(_) => thread::sleep(Duration::from_millis(20)),
            }
        }

        let mut nested_response = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(2);
        while Instant::now() < deadline {
            match TcpStream::connect(("127.0.0.1", port)) {
                Ok(mut stream) => {
                    stream
                        .write_all(
                            b"GET /api/v1/posts/post-1/comments/abc123 HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
                        )
                        .expect("write nested");
                    stream
                        .read_to_end(&mut nested_response)
                        .expect("read nested");
                    break;
                }
                Err(_) => thread::sleep(Duration::from_millis(20)),
            }
        }

        let mut catchall_response = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(2);
        while Instant::now() < deadline {
            match TcpStream::connect(("127.0.0.1", port)) {
                Ok(mut stream) => {
                    stream
                        .write_all(
                            b"GET /api/v1/files/folder%201/report.txt HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
                        )
                        .expect("write catchall");
                    stream
                        .read_to_end(&mut catchall_response)
                        .expect("read catchall");
                    break;
                }
                Err(_) => thread::sleep(Duration::from_millis(20)),
            }
        }

        let server_result = server.join().expect("join").expect("service");
        assert_eq!(server_result, Some(0));
        let any_text = String::from_utf8_lossy(&any_response);
        assert!(any_text.contains("\"route\":\"any_status\""));
        let nested_text = String::from_utf8_lossy(&nested_response);
        assert!(nested_text.contains("\"route\":\"nested_comment\""));
        assert!(nested_text.contains("\"post\":\"post-1\""));
        assert!(nested_text.contains("\"comment\":\"abc123\""));
        let catchall_text = String::from_utf8_lossy(&catchall_response);
        assert!(catchall_text.contains("\"route\":\"catchall\""));
        assert!(catchall_text.contains("\"path\":\"folder 1/report.txt\""));
    }
}
