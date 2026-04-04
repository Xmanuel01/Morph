use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::rc::Weak;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use base64::Engine;
use enkaic::ast::{Arg, Block, Expr, Item, LValue, Module, Stmt};
use enkaic::bytecode::{Constant, FfiSignature, FfiType, Instruction, NativeFunctionDecl, Program};
use enkaic::compiler::compile_package;
use enkaic::formatter::{check_format, format_source};
use enkaic::modules::load_package;
use enkaic::parser::parse_module_named;
use enkaic::TypeChecker;
use sha1::{Digest, Sha1};

use crate::checkpoint::{
    latest_checkpoint, load_checkpoint, rotate_checkpoints, save_checkpoint, CheckpointMeta,
    CheckpointState,
};
use crate::dataset::{resolve_dataset_paths, Batch, DatasetConfig, DatasetStream};
use crate::error::{RuntimeError, RuntimeFrame};
use crate::ffi::FfiFunction;
use crate::ffi::{ffi_stats_snapshot, FfiLoader, FfiStats};
use crate::object::{
    agent_env_value, buffer_value, channel_value, event_queue_value_with_native, function_value,
    pool_value_with_native, record_value, rng_stream_value, sim_coroutine_value, sim_world_value,
    snn_network_value, sparse_matrix_value_with_native, sparse_vector_value_with_native,
    spatial_index_value_with_native, string_value, task_handle_value, BoundFunctionObj, HttpStream,
    NativeFunction, NativeImpl, Obj, StreamCommand, WebSocketHandle, WsCommand, WsIncoming,
};
use crate::tokenizer::{bytes_to_ids, ids_to_bytes, Tokenizer, TrainConfig};
use crate::value::{object_allocation_count, ObjRef, Value};

#[derive(Debug)]
struct CallFrame {
    func_index: u16,
    ip: usize,
    base: usize,      // start of locals/args
    caller_sp: usize, // stack height where callee was placed
    prev_policy: Option<String>,
}

#[derive(Clone)]
enum TaskState {
    Ready,
    Sleeping(Instant),
    BlockedJoin,
    BlockedChannel,
    BlockedIo,
    Finished,
}

struct Task {
    id: usize,
    stack: Vec<Value>,
    frames: Vec<CallFrame>,
    state: TaskState,
    result: Option<Result<Value, RuntimeError>>,
    join_waiters: Vec<usize>,
    pending_error: Option<RuntimeError>,
    http_conn: Option<TcpStream>,
    http_meta: Option<HttpRequestMeta>,
    policy: Option<String>,
}

enum TaskRunOutcome {
    Completed(Value),
    Yielded,
    Errored(RuntimeError),
}

enum IoResult {
    Accept(Result<TcpStream, String>),
    Read(Result<Vec<u8>, String>),
    ReadAll(Result<Vec<u8>, String>),
    Write(Result<usize, String>),
    HttpResponse(Result<HttpResponseData, String>),
}

enum SimAccelState {
    Uninitialized,
    Disabled,
    Ready(Box<SimAccelBindings>),
}

struct SimAccelBindings {
    sparse_vector_new: FfiFunction,
    sparse_vector_set: FfiFunction,
    sparse_vector_dot: FfiFunction,
    sparse_matrix_new: FfiFunction,
    sparse_matrix_set: FfiFunction,
    sparse_matrix_matvec: FfiFunction,
    event_queue_new: FfiFunction,
    event_queue_push: FfiFunction,
    event_queue_pop: FfiFunction,
    event_queue_peek: FfiFunction,
    event_queue_len: FfiFunction,
    pool_new: FfiFunction,
    pool_release: FfiFunction,
    pool_acquire: FfiFunction,
    pool_reset: FfiFunction,
    pool_available: FfiFunction,
    pool_capacity: FfiFunction,
    pool_stats: FfiFunction,
    spatial_index_new: FfiFunction,
    spatial_upsert: FfiFunction,
    spatial_remove: FfiFunction,
    spatial_radius: FfiFunction,
    spatial_nearest: FfiFunction,
    spatial_occupancy: FfiFunction,
    rng_stream_new: FfiFunction,
    rng_stream_next_float: FfiFunction,
    rng_stream_next_int: FfiFunction,
    snn_network_new: FfiFunction,
    snn_set_potential: FfiFunction,
    snn_get_potential: FfiFunction,
    snn_set_threshold: FfiFunction,
    snn_get_threshold: FfiFunction,
    snn_set_decay: FfiFunction,
    snn_get_decay: FfiFunction,
    snn_connect: FfiFunction,
    snn_step: FfiFunction,
}

impl SimAccelBindings {
    fn load(loader: &mut FfiLoader) -> Result<Self, RuntimeError> {
        fn bind(
            loader: &mut FfiLoader,
            name: &str,
            params: Vec<FfiType>,
            ret: FfiType,
        ) -> Result<FfiFunction, RuntimeError> {
            loader.bind(&NativeFunctionDecl {
                library: "enkai_native".to_string(),
                name: name.to_string(),
                signature: FfiSignature { params, ret },
            })
        }

        Ok(Self {
            sparse_vector_new: bind(loader, "sim_sparse_vector_new", vec![], FfiType::Handle)?,
            sparse_vector_set: bind(
                loader,
                "sim_sparse_vector_set",
                vec![FfiType::Handle, FfiType::Int, FfiType::Float],
                FfiType::Bool,
            )?,
            sparse_vector_dot: bind(
                loader,
                "sim_sparse_vector_dot",
                vec![FfiType::Handle, FfiType::Buffer],
                FfiType::Float,
            )?,
            sparse_matrix_new: bind(loader, "sim_sparse_matrix_new", vec![], FfiType::Handle)?,
            sparse_matrix_set: bind(
                loader,
                "sim_sparse_matrix_set",
                vec![FfiType::Handle, FfiType::Int, FfiType::Int, FfiType::Float],
                FfiType::Bool,
            )?,
            sparse_matrix_matvec: bind(
                loader,
                "sim_sparse_matrix_matvec",
                vec![FfiType::Handle, FfiType::Buffer],
                FfiType::Buffer,
            )?,
            event_queue_new: bind(loader, "sim_event_queue_new", vec![], FfiType::Handle)?,
            event_queue_push: bind(
                loader,
                "sim_event_queue_push",
                vec![FfiType::Handle, FfiType::Float, FfiType::Int],
                FfiType::Bool,
            )?,
            event_queue_pop: bind(
                loader,
                "sim_event_queue_pop",
                vec![FfiType::Handle],
                FfiType::Buffer,
            )?,
            event_queue_peek: bind(
                loader,
                "sim_event_queue_peek",
                vec![FfiType::Handle],
                FfiType::Buffer,
            )?,
            event_queue_len: bind(
                loader,
                "sim_event_queue_len",
                vec![FfiType::Handle],
                FfiType::Int,
            )?,
            pool_new: bind(
                loader,
                "sim_pool_new",
                vec![FfiType::Int, FfiType::Bool],
                FfiType::Handle,
            )?,
            pool_release: bind(
                loader,
                "sim_pool_release",
                vec![FfiType::Handle],
                FfiType::Bool,
            )?,
            pool_acquire: bind(
                loader,
                "sim_pool_acquire",
                vec![FfiType::Handle],
                FfiType::Bool,
            )?,
            pool_reset: bind(
                loader,
                "sim_pool_reset",
                vec![FfiType::Handle],
                FfiType::Void,
            )?,
            pool_available: bind(
                loader,
                "sim_pool_available",
                vec![FfiType::Handle],
                FfiType::Int,
            )?,
            pool_capacity: bind(
                loader,
                "sim_pool_capacity",
                vec![FfiType::Handle],
                FfiType::Int,
            )?,
            pool_stats: bind(
                loader,
                "sim_pool_stats",
                vec![FfiType::Handle],
                FfiType::Buffer,
            )?,
            spatial_index_new: bind(loader, "sim_spatial_index_new", vec![], FfiType::Handle)?,
            spatial_upsert: bind(
                loader,
                "sim_spatial_upsert",
                vec![
                    FfiType::Handle,
                    FfiType::Int,
                    FfiType::Float,
                    FfiType::Float,
                ],
                FfiType::Bool,
            )?,
            spatial_remove: bind(
                loader,
                "sim_spatial_remove",
                vec![FfiType::Handle, FfiType::Int],
                FfiType::Bool,
            )?,
            spatial_radius: bind(
                loader,
                "sim_spatial_radius",
                vec![
                    FfiType::Handle,
                    FfiType::Float,
                    FfiType::Float,
                    FfiType::Float,
                ],
                FfiType::Buffer,
            )?,
            spatial_nearest: bind(
                loader,
                "sim_spatial_nearest",
                vec![FfiType::Handle, FfiType::Float, FfiType::Float],
                FfiType::Int,
            )?,
            spatial_occupancy: bind(
                loader,
                "sim_spatial_occupancy",
                vec![
                    FfiType::Handle,
                    FfiType::Float,
                    FfiType::Float,
                    FfiType::Float,
                    FfiType::Float,
                ],
                FfiType::Int,
            )?,
            rng_stream_new: bind(
                loader,
                "sim_rng_stream_new",
                vec![FfiType::Int, FfiType::Int, FfiType::Int],
                FfiType::Handle,
            )?,
            rng_stream_next_float: bind(
                loader,
                "sim_rng_stream_next_float",
                vec![FfiType::Handle],
                FfiType::Float,
            )?,
            rng_stream_next_int: bind(
                loader,
                "sim_rng_stream_next_int",
                vec![FfiType::Handle, FfiType::Int],
                FfiType::Int,
            )?,
            snn_network_new: bind(
                loader,
                "sim_snn_network_new",
                vec![FfiType::Int],
                FfiType::Handle,
            )?,
            snn_set_potential: bind(
                loader,
                "sim_snn_set_potential",
                vec![FfiType::Handle, FfiType::Int, FfiType::Float],
                FfiType::Bool,
            )?,
            snn_get_potential: bind(
                loader,
                "sim_snn_get_potential",
                vec![FfiType::Handle, FfiType::Int],
                FfiType::Float,
            )?,
            snn_set_threshold: bind(
                loader,
                "sim_snn_set_threshold",
                vec![FfiType::Handle, FfiType::Int, FfiType::Float],
                FfiType::Bool,
            )?,
            snn_get_threshold: bind(
                loader,
                "sim_snn_get_threshold",
                vec![FfiType::Handle, FfiType::Int],
                FfiType::Float,
            )?,
            snn_set_decay: bind(
                loader,
                "sim_snn_set_decay",
                vec![FfiType::Handle, FfiType::Float],
                FfiType::Bool,
            )?,
            snn_get_decay: bind(
                loader,
                "sim_snn_get_decay",
                vec![FfiType::Handle],
                FfiType::Float,
            )?,
            snn_connect: bind(
                loader,
                "sim_snn_connect",
                vec![FfiType::Handle, FfiType::Int, FfiType::Int, FfiType::Float],
                FfiType::Bool,
            )?,
            snn_step: bind(
                loader,
                "sim_snn_step",
                vec![FfiType::Handle, FfiType::Buffer],
                FfiType::Buffer,
            )?,
        })
    }
}

struct IoEvent {
    task_id: usize,
    result: IoResult,
}

static TOOL_IO_FILE_COUNTER: AtomicU64 = AtomicU64::new(1);

type ToolProcessOutput = (Vec<u8>, Vec<u8>, Option<i32>);

struct ServerEvent {
    server_id: usize,
    request: HttpRequestData,
    stream: TcpStream,
    accepted_at: Instant,
}

#[derive(Clone)]
struct HttpRoute {
    method: String,
    path: String,
    handler: Value,
}

struct ServerAuthConfig {
    header: String,
    tokens: HashMap<String, String>,
    allow_anonymous: bool,
}

struct RateLimitConfig {
    capacity: f64,
    refill_per_sec: f64,
    key: RateLimitKey,
}

#[derive(Clone, Copy)]
enum RateLimitKey {
    Ip,
    Token,
    Tenant,
    Model,
    TenantModel,
}

struct RateLimitBucket {
    tokens: f64,
    last: Instant,
}

struct ServerLogger {
    path: std::path::PathBuf,
}

struct HttpServer {
    handler: Value,
    routes: Vec<HttpRoute>,
    default_handler: Option<Value>,
    auth: Option<ServerAuthConfig>,
    rate_limit: Option<RateLimitConfig>,
    rate_state: HashMap<String, RateLimitBucket>,
    logger: Option<ServerLogger>,
    policy: Option<String>,
    inflight: Arc<AtomicUsize>,
    max_inflight: usize,
    require_model_version_header: bool,
    model_name: Option<String>,
    model_version: Option<String>,
    model_registry: Option<String>,
    multi_model_registry: Option<String>,
    stop: mpsc::Sender<()>,
}

#[derive(Clone)]
struct HttpRequestData {
    method: String,
    path: String,
    query: String,
    headers: HashMap<String, String>,
    body: Vec<u8>,
    remote_addr: String,
}

struct HttpRequestMeta {
    id: u64,
    server_id: usize,
    start: Instant,
    queue_ms: u64,
    method: String,
    path: String,
    remote_addr: String,
    correlation_id: String,
    tenant: Option<String>,
    model_name: Option<String>,
    model_version: Option<String>,
    model_registry: Option<String>,
    inflight_at_start: usize,
    error_code: Option<String>,
}

#[derive(Clone)]
struct HttpResponseData {
    status: u16,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}

#[derive(Default, Clone, Copy)]
struct VmBenchCounters {
    opcode_dispatch: u64,
    arithmetic_ops: u64,
    compare_ops: u64,
    native_calls: u64,
    sim_coroutines_spawned: u64,
    sim_coroutine_emits: u64,
    sim_coroutine_next_waits: u64,
}

#[derive(Clone)]
struct VmBenchProfile {
    out_path: PathBuf,
    case: Option<String>,
    started: Instant,
    start_object_allocs: u64,
    start_ffi: FfiStats,
    counters: VmBenchCounters,
}

#[derive(Clone)]
struct HttpRequestOptions {
    headers: HashMap<String, String>,
    timeout_ms: Option<u64>,
    retries: usize,
    retry_backoff_ms: u64,
}

impl Default for HttpRequestOptions {
    fn default() -> Self {
        Self {
            headers: HashMap::new(),
            timeout_ms: None,
            retries: 0,
            retry_backoff_ms: 100,
        }
    }
}

type CapabilityRequirement = (Vec<String>, Option<CapabilityContext>);
type HttpServerConfigParsed = (
    Option<Value>,
    Option<ServerAuthConfig>,
    Option<RateLimitConfig>,
    Option<ServerLogger>,
    usize,
    bool,
);
type PreparedHttpRequest = (
    Option<Value>,
    HashMap<String, String>,
    Option<String>,
    String,
    Option<String>,
    Option<String>,
    Option<HttpResponseData>,
    Option<String>,
);

#[derive(Debug, Clone)]
struct Policy {
    rules: Vec<PolicyRuleRuntime>,
}

#[derive(Debug, Clone)]
struct PolicyRuleRuntime {
    allow: bool,
    capability: Vec<String>,
    filters: Vec<PolicyFilterRuntime>,
}

#[derive(Debug, Clone)]
struct PolicyFilterRuntime {
    name: String,
    values: Vec<String>,
}

#[derive(Debug, Clone)]
enum CapabilityContext {
    Path(String),
    Domain(String),
}

impl CapabilityContext {
    fn for_path(path: &str) -> Self {
        CapabilityContext::Path(path.to_string())
    }

    fn for_domain(domain: &str) -> Self {
        CapabilityContext::Domain(domain.to_string())
    }
}

impl Policy {
    fn is_allowed(&self, capability: &[String], context: Option<&CapabilityContext>) -> bool {
        let mut allowed = false;
        for rule in &self.rules {
            if capability_matches(&rule.capability, capability)
                && filters_match(&rule.filters, context)
            {
                if !rule.allow {
                    return false;
                }
                allowed = true;
            }
        }
        allowed
    }
}

pub struct VM {
    stack: Vec<Value>,
    frames: Vec<CallFrame>,
    globals: Vec<Value>,
    globals_map: HashMap<String, u16>,
    trace: bool,
    disasm: bool,
    trace_task: bool,
    trace_net: bool,
    ffi_loader: FfiLoader,
    tasks: Vec<Option<Task>>,
    sim_coroutines: Vec<Option<Weak<Obj>>>,
    ready: VecDeque<usize>,
    current_task: Option<usize>,
    next_task_id: usize,
    yield_now: bool,
    pending_state: Option<TaskState>,
    io_sender: mpsc::Sender<IoEvent>,
    io_receiver: mpsc::Receiver<IoEvent>,
    servers: Vec<HttpServer>,
    server_sender: mpsc::Sender<ServerEvent>,
    server_receiver: mpsc::Receiver<ServerEvent>,
    next_request_id: u64,
    active_http_conn: Option<TcpStream>,
    policies: HashMap<String, Policy>,
    active_policy: Option<String>,
    bench_profile: Option<VmBenchProfile>,
    sim_accel: SimAccelState,
}

impl VM {
    pub fn new(trace: bool, disasm: bool, trace_task: bool, trace_net: bool) -> Self {
        let (io_sender, io_receiver) = mpsc::channel();
        let (server_sender, server_receiver) = mpsc::channel();
        let bench_profile = std::env::var("ENKAI_BENCH_PROFILE_OUT")
            .ok()
            .map(|raw| raw.trim().to_string())
            .filter(|raw| !raw.is_empty())
            .map(|raw| VmBenchProfile {
                out_path: PathBuf::from(raw),
                case: std::env::var("ENKAI_BENCH_PROFILE_CASE")
                    .ok()
                    .map(|v| v.trim().to_string())
                    .filter(|v| !v.is_empty()),
                started: Instant::now(),
                start_object_allocs: 0,
                start_ffi: FfiStats::default(),
                counters: VmBenchCounters::default(),
            });
        Self {
            stack: Vec::new(),
            frames: Vec::new(),
            globals: Vec::new(),
            globals_map: HashMap::new(),
            trace,
            disasm,
            trace_task,
            trace_net,
            ffi_loader: FfiLoader::new(),
            tasks: Vec::new(),
            sim_coroutines: Vec::new(),
            ready: VecDeque::new(),
            current_task: None,
            next_task_id: 0,
            yield_now: false,
            pending_state: None,
            io_sender,
            io_receiver,
            servers: Vec::new(),
            server_sender,
            server_receiver,
            next_request_id: 0,
            active_http_conn: None,
            policies: HashMap::new(),
            active_policy: None,
            bench_profile,
            sim_accel: SimAccelState::Uninitialized,
        }
    }

    pub fn set_sim_accel_enabled(&mut self, enabled: bool) {
        self.sim_accel = if enabled {
            SimAccelState::Uninitialized
        } else {
            SimAccelState::Disabled
        };
    }

    pub fn run(&mut self, program: &Program) -> Result<Value, RuntimeError> {
        self.begin_bench_profile();
        self.install_globals(program)?;
        if self.disasm {
            println!("{}", program.disassemble());
        }
        let main_func = function_value(program.main, program);
        let main_id = self.spawn_task_internal(program, main_func)?;
        let result = self.scheduler_loop(program, main_id);
        self.finish_bench_profile(result.as_ref().err());
        result
    }

    fn begin_bench_profile(&mut self) {
        let Some(profile) = self.bench_profile.as_mut() else {
            return;
        };
        profile.started = Instant::now();
        profile.start_object_allocs = object_allocation_count();
        profile.start_ffi = ffi_stats_snapshot();
        profile.counters = VmBenchCounters::default();
    }

    fn finish_bench_profile(&mut self, error: Option<&RuntimeError>) {
        let Some(profile) = self.bench_profile.as_ref() else {
            return;
        };
        let finished = Instant::now();
        let ffi_now = ffi_stats_snapshot();
        let obj_now = object_allocation_count();
        let ffi_call_count = ffi_now
            .call_count
            .saturating_sub(profile.start_ffi.call_count);
        let marshal_in_bytes = ffi_now
            .marshal_in_bytes
            .saturating_sub(profile.start_ffi.marshal_in_bytes);
        let marshal_out_bytes = ffi_now
            .marshal_out_bytes
            .saturating_sub(profile.start_ffi.marshal_out_bytes);
        let ffi_copy_count = ffi_now
            .copy_count
            .saturating_sub(profile.start_ffi.copy_count);
        let ffi_handle_count = ffi_now
            .handle_count
            .saturating_sub(profile.start_ffi.handle_count);
        let ffi_time_ns = ffi_now
            .native_time_ns
            .saturating_sub(profile.start_ffi.native_time_ns);
        let total_ms = finished.duration_since(profile.started).as_secs_f64() * 1000.0;
        let ffi_ms = (ffi_time_ns as f64) / 1_000_000.0;
        let vm_exec_ms = (total_ms - ffi_ms).max(0.0);
        let status = if error.is_none() { "ok" } else { "error" };
        let report = serde_json::json!({
            "schema_version": 1,
            "case": profile.case,
            "status": status,
            "error": error.map(|e| {
                serde_json::json!({
                    "code": e.code(),
                    "message": e.message,
                })
            }),
            "timing_ms": {
                "total": total_ms,
                "vm_exec": vm_exec_ms,
                "native_calls": ffi_ms,
                "gc": 0.0,
                "io": 0.0,
            },
            "counters": {
                "opcode_dispatch": profile.counters.opcode_dispatch,
                "arithmetic_ops": profile.counters.arithmetic_ops,
                "compare_ops": profile.counters.compare_ops,
                "native_function_calls": profile.counters.native_calls,
                "sim_coroutines_spawned": profile.counters.sim_coroutines_spawned,
                "sim_coroutine_emits": profile.counters.sim_coroutine_emits,
                "sim_coroutine_next_waits": profile.counters.sim_coroutine_next_waits,
                "ffi_calls": ffi_call_count,
                "object_allocations": obj_now.saturating_sub(profile.start_object_allocs),
                "marshal_in_bytes": marshal_in_bytes,
                "marshal_out_bytes": marshal_out_bytes,
                "marshal_copy_ops": ffi_copy_count,
                "ffi_handle_objects": ffi_handle_count,
            }
        });
        if let Some(parent) = profile.out_path.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = std::fs::create_dir_all(parent);
            }
        }
        if let Ok(text) = serde_json::to_string_pretty(&report) {
            let _ = std::fs::write(&profile.out_path, text);
        }
    }

    fn try_fast_numeric_binary(
        op: &Instruction,
        left: &Value,
        right: &Value,
    ) -> Result<Option<Value>, RuntimeError> {
        let out = match op {
            Instruction::Add => match (left, right) {
                (Value::Int(x), Value::Int(y)) => Some(Value::Int(x.wrapping_add(*y))),
                (Value::Int(x), Value::Float(y)) => Some(Value::Float((*x as f64) + *y)),
                (Value::Float(x), Value::Int(y)) => Some(Value::Float(*x + (*y as f64))),
                (Value::Float(x), Value::Float(y)) => Some(Value::Float(*x + *y)),
                _ => None,
            },
            Instruction::Sub => match (left, right) {
                (Value::Int(x), Value::Int(y)) => Some(Value::Int(x.wrapping_sub(*y))),
                (Value::Int(x), Value::Float(y)) => Some(Value::Float((*x as f64) - *y)),
                (Value::Float(x), Value::Int(y)) => Some(Value::Float(*x - (*y as f64))),
                (Value::Float(x), Value::Float(y)) => Some(Value::Float(*x - *y)),
                _ => None,
            },
            Instruction::Mul => match (left, right) {
                (Value::Int(x), Value::Int(y)) => Some(Value::Int(x.wrapping_mul(*y))),
                (Value::Int(x), Value::Float(y)) => Some(Value::Float((*x as f64) * *y)),
                (Value::Float(x), Value::Int(y)) => Some(Value::Float(*x * (*y as f64))),
                (Value::Float(x), Value::Float(y)) => Some(Value::Float(*x * *y)),
                _ => None,
            },
            Instruction::Div => match (left, right) {
                (Value::Int(_), Value::Int(0)) => {
                    return Err(RuntimeError::new("Division by zero"));
                }
                (Value::Int(x), Value::Int(y)) => {
                    if *x == i64::MIN && *y == -1 {
                        Some(Value::Int(i64::MIN))
                    } else {
                        Some(Value::Int(*x / *y))
                    }
                }
                (Value::Int(_), Value::Float(v)) if *v == 0.0 => {
                    return Err(RuntimeError::new("Division by zero"));
                }
                (Value::Float(_), Value::Int(0)) => {
                    return Err(RuntimeError::new("Division by zero"));
                }
                (Value::Float(_), Value::Float(v)) if *v == 0.0 => {
                    return Err(RuntimeError::new("Division by zero"));
                }
                (Value::Int(x), Value::Float(y)) => Some(Value::Float((*x as f64) / *y)),
                (Value::Float(x), Value::Int(y)) => Some(Value::Float(*x / (*y as f64))),
                (Value::Float(x), Value::Float(y)) => Some(Value::Float(*x / *y)),
                _ => None,
            },
            Instruction::Lt => match (left, right) {
                (Value::Int(x), Value::Int(y)) => Some(Value::Bool(x < y)),
                (Value::Int(x), Value::Float(y)) => Some(Value::Bool((*x as f64) < *y)),
                (Value::Float(x), Value::Int(y)) => Some(Value::Bool(*x < (*y as f64))),
                (Value::Float(x), Value::Float(y)) => Some(Value::Bool(*x < *y)),
                _ => None,
            },
            Instruction::Gt => match (left, right) {
                (Value::Int(x), Value::Int(y)) => Some(Value::Bool(x > y)),
                (Value::Int(x), Value::Float(y)) => Some(Value::Bool((*x as f64) > *y)),
                (Value::Float(x), Value::Int(y)) => Some(Value::Bool(*x > (*y as f64))),
                (Value::Float(x), Value::Float(y)) => Some(Value::Bool(*x > *y)),
                _ => None,
            },
            Instruction::Le => match (left, right) {
                (Value::Int(x), Value::Int(y)) => Some(Value::Bool(x <= y)),
                (Value::Int(x), Value::Float(y)) => Some(Value::Bool((*x as f64) <= *y)),
                (Value::Float(x), Value::Int(y)) => Some(Value::Bool(*x <= (*y as f64))),
                (Value::Float(x), Value::Float(y)) => Some(Value::Bool(*x <= *y)),
                _ => None,
            },
            Instruction::Ge => match (left, right) {
                (Value::Int(x), Value::Int(y)) => Some(Value::Bool(x >= y)),
                (Value::Int(x), Value::Float(y)) => Some(Value::Bool((*x as f64) >= *y)),
                (Value::Float(x), Value::Int(y)) => Some(Value::Bool(*x >= (*y as f64))),
                (Value::Float(x), Value::Float(y)) => Some(Value::Bool(*x >= *y)),
                _ => None,
            },
            _ => None,
        };
        Ok(out)
    }

    fn install_globals(&mut self, program: &Program) -> Result<(), RuntimeError> {
        self.globals = Vec::with_capacity(program.globals.len());
        for name in &program.globals {
            self.globals_map
                .insert(name.clone(), self.globals.len() as u16);
            self.globals.push(Value::Null);
        }
        // populate initial values (functions mostly)
        for (idx, init) in program.global_inits.iter().enumerate() {
            if let Some(c) = init {
                let v = self.constant_to_value(c, program)?;
                self.globals[idx] = v;
            }
        }
        // install native print for convenience
        let print = Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
            name: "print".to_string(),
            arity: 1,
            kind: NativeImpl::Rust(std::rc::Rc::new(|_, args| {
                println!("{}", display_value(&args[0]));
                Ok(Value::Null)
            })),
            bound: None,
        })));
        if let Some(idx) = self.globals_map.get("print").copied() {
            self.globals[idx as usize] = print;
        } else {
            self.globals_map
                .insert("print".to_string(), self.globals.len() as u16);
            self.globals.push(print);
        }
        let mut task_record = std::collections::HashMap::new();
        task_record.insert(
            "spawn".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "task.spawn".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        task_record.insert(
            "join".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "task.join".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        task_record.insert(
            "sleep".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "task.sleep".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        let task_value = record_value(task_record);
        if let Some(idx) = self.globals_map.get("task").copied() {
            self.globals[idx as usize] = task_value;
        } else {
            self.globals_map
                .insert("task".to_string(), self.globals.len() as u16);
            self.globals.push(task_value);
        }
        let mut chan_record = std::collections::HashMap::new();
        chan_record.insert(
            "make".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "chan.make".to_string(),
                arity: 0,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        chan_record.insert(
            "send".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "chan.send".to_string(),
                arity: 2,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        chan_record.insert(
            "recv".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "chan.recv".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        let chan_value = record_value(chan_record);
        if let Some(idx) = self.globals_map.get("chan").copied() {
            self.globals[idx as usize] = chan_value;
        } else {
            self.globals_map
                .insert("chan".to_string(), self.globals.len() as u16);
            self.globals.push(chan_value);
        }
        let mut net_record = std::collections::HashMap::new();
        net_record.insert(
            "bind".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "net.bind".to_string(),
                arity: 2,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        let net_value = record_value(net_record);
        if let Some(idx) = self.globals_map.get("net").copied() {
            self.globals[idx as usize] = net_value;
        } else {
            self.globals_map
                .insert("net".to_string(), self.globals.len() as u16);
            self.globals.push(net_value);
        }
        let mut http_record = std::collections::HashMap::new();
        http_record.insert(
            "serve".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "http.serve".to_string(),
                arity: 3,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        http_record.insert(
            "serve_with".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "http.serve_with".to_string(),
                arity: 4,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        http_record.insert(
            "route".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "http.route".to_string(),
                arity: 3,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        http_record.insert(
            "middleware".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "http.middleware".to_string(),
                arity: 2,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        http_record.insert(
            "get".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "http.get".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        http_record.insert(
            "post".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "http.post".to_string(),
                arity: 2,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        http_record.insert(
            "request".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "http.request".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        http_record.insert(
            "header".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "http.header".to_string(),
                arity: 2,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        http_record.insert(
            "query".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "http.query".to_string(),
                arity: 2,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        http_record.insert(
            "response".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "http.response".to_string(),
                arity: 2,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        http_record.insert(
            "stream_open".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "http.stream_open".to_string(),
                arity: 2,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        http_record.insert(
            "stream_send".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "http.stream_send".to_string(),
                arity: 2,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        http_record.insert(
            "stream_close".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "http.stream_close".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        http_record.insert(
            "ws_open".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "http.ws_open".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        http_record.insert(
            "ws_send".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "http.ws_send".to_string(),
                arity: 2,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        http_record.insert(
            "ws_recv".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "http.ws_recv".to_string(),
                arity: 2,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        http_record.insert(
            "ws_close".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "http.ws_close".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        http_record.insert(
            "ok".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "http.ok".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        http_record.insert(
            "bad_request".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "http.bad_request".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        http_record.insert(
            "not_found".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "http.not_found".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        let http_value = record_value(http_record);
        if let Some(idx) = self.globals_map.get("http").copied() {
            self.globals[idx as usize] = http_value;
        } else {
            self.globals_map
                .insert("http".to_string(), self.globals.len() as u16);
            self.globals.push(http_value);
        }
        let mut tool_record = std::collections::HashMap::new();
        tool_record.insert(
            "invoke".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "tool.invoke".to_string(),
                arity: 2,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        let tool_value = record_value(tool_record);
        if let Some(idx) = self.globals_map.get("tool").copied() {
            self.globals[idx as usize] = tool_value;
        } else {
            self.globals_map
                .insert("tool".to_string(), self.globals.len() as u16);
            self.globals.push(tool_value);
        }
        let mut policy_record = std::collections::HashMap::new();
        policy_record.insert(
            "register".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "policy.register".to_string(),
                arity: 3,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        let policy_value = record_value(policy_record);
        if let Some(idx) = self.globals_map.get("policy").copied() {
            self.globals[idx as usize] = policy_value;
        } else {
            self.globals_map
                .insert("policy".to_string(), self.globals.len() as u16);
            self.globals.push(policy_value);
        }
        let mut json_record = std::collections::HashMap::new();
        json_record.insert(
            "parse".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "json.parse".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        json_record.insert(
            "stringify".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "json.stringify".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        json_record.insert(
            "parse_many".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "json.parse_many".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        json_record.insert(
            "stringify_many".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "json.stringify_many".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        let json_value = record_value(json_record);
        if let Some(idx) = self.globals_map.get("json").copied() {
            self.globals[idx as usize] = json_value;
        } else {
            self.globals_map
                .insert("json".to_string(), self.globals.len() as u16);
            self.globals.push(json_value);
        }
        let mut bootstrap_record = std::collections::HashMap::new();
        bootstrap_record.insert(
            "format".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "bootstrap.format".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        bootstrap_record.insert(
            "check".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "bootstrap.check".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        bootstrap_record.insert(
            "lint".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "bootstrap.lint".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        bootstrap_record.insert(
            "lint_count".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "bootstrap.lint_count".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        bootstrap_record.insert(
            "lint_json".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "bootstrap.lint_json".to_string(),
                arity: 2,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        let bootstrap_value = record_value(bootstrap_record);
        if let Some(idx) = self.globals_map.get("bootstrap").copied() {
            self.globals[idx as usize] = bootstrap_value;
        } else {
            self.globals_map
                .insert("bootstrap".to_string(), self.globals.len() as u16);
            self.globals.push(bootstrap_value);
        }
        let mut compiler_record = std::collections::HashMap::new();
        compiler_record.insert(
            "parse_subset".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "compiler.parse_subset".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        compiler_record.insert(
            "check_subset".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "compiler.check_subset".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        compiler_record.insert(
            "emit_subset".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "compiler.emit_subset".to_string(),
                arity: 2,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        let compiler_value = record_value(compiler_record);
        if let Some(idx) = self.globals_map.get("compiler").copied() {
            self.globals[idx as usize] = compiler_value;
        } else {
            self.globals_map
                .insert("compiler".to_string(), self.globals.len() as u16);
            self.globals.push(compiler_value);
        }
        let mut tokenizer_record = std::collections::HashMap::new();
        tokenizer_record.insert(
            "train".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "tokenizer.train".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        tokenizer_record.insert(
            "load".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "tokenizer.load".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        tokenizer_record.insert(
            "save".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "tokenizer.save".to_string(),
                arity: 2,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        tokenizer_record.insert(
            "encode".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "tokenizer.encode".to_string(),
                arity: 2,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        tokenizer_record.insert(
            "decode".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "tokenizer.decode".to_string(),
                arity: 2,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        let tokenizer_value = record_value(tokenizer_record);
        if let Some(idx) = self.globals_map.get("tokenizer").copied() {
            self.globals[idx as usize] = tokenizer_value;
        } else {
            self.globals_map
                .insert("tokenizer".to_string(), self.globals.len() as u16);
            self.globals.push(tokenizer_value);
        }
        let mut dataset_record = std::collections::HashMap::new();
        dataset_record.insert(
            "open".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "dataset.open".to_string(),
                arity: 3,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        dataset_record.insert(
            "next_batch".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "dataset.next_batch".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        let dataset_value = record_value(dataset_record);
        if let Some(idx) = self.globals_map.get("dataset").copied() {
            self.globals[idx as usize] = dataset_value;
        } else {
            self.globals_map
                .insert("dataset".to_string(), self.globals.len() as u16);
            self.globals.push(dataset_value);
        }
        let mut checkpoint_record = std::collections::HashMap::new();
        checkpoint_record.insert(
            "save".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "checkpoint.save".to_string(),
                arity: 2,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        checkpoint_record.insert(
            "load".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "checkpoint.load".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        checkpoint_record.insert(
            "latest".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "checkpoint.latest".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        checkpoint_record.insert(
            "rotate".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "checkpoint.rotate".to_string(),
                arity: 2,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        let checkpoint_value = record_value(checkpoint_record);
        if let Some(idx) = self.globals_map.get("checkpoint").copied() {
            self.globals[idx as usize] = checkpoint_value;
        } else {
            self.globals_map
                .insert("checkpoint".to_string(), self.globals.len() as u16);
            self.globals.push(checkpoint_value);
        }
        let mut sparse_record = std::collections::HashMap::new();
        sparse_record.insert(
            "vector".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "sparse.vector".to_string(),
                arity: 0,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        sparse_record.insert(
            "matrix".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "sparse.matrix".to_string(),
                arity: 0,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        sparse_record.insert(
            "get".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "sparse.get".to_string(),
                arity: 3,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        sparse_record.insert(
            "set".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "sparse.set".to_string(),
                arity: 4,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        sparse_record.insert(
            "get_vector".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "sparse.get_vector".to_string(),
                arity: 2,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        sparse_record.insert(
            "set_vector".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "sparse.set_vector".to_string(),
                arity: 3,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        sparse_record.insert(
            "nonzero".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "sparse.nonzero".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        sparse_record.insert(
            "nonzero_vector".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "sparse.nonzero_vector".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        sparse_record.insert(
            "dot".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "sparse.dot".to_string(),
                arity: 2,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        sparse_record.insert(
            "matvec".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "sparse.matvec".to_string(),
                arity: 2,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        sparse_record.insert(
            "nnz".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "sparse.nnz".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        let sparse_value = record_value(sparse_record);
        if let Some(idx) = self.globals_map.get("sparse").copied() {
            self.globals[idx as usize] = sparse_value;
        } else {
            self.globals_map
                .insert("sparse".to_string(), self.globals.len() as u16);
            self.globals.push(sparse_value);
        }
        let mut event_record = std::collections::HashMap::new();
        event_record.insert(
            "make".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "event.make".to_string(),
                arity: 0,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        event_record.insert(
            "push".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "event.push".to_string(),
                arity: 3,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        event_record.insert(
            "pop".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "event.pop".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        event_record.insert(
            "peek".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "event.peek".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        event_record.insert(
            "len".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "event.len".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        event_record.insert(
            "is_empty".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "event.is_empty".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        let event_value = record_value(event_record);
        if let Some(idx) = self.globals_map.get("event").copied() {
            self.globals[idx as usize] = event_value;
        } else {
            self.globals_map
                .insert("event".to_string(), self.globals.len() as u16);
            self.globals.push(event_value);
        }
        let mut pool_record = std::collections::HashMap::new();
        pool_record.insert(
            "make".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "pool.make".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        pool_record.insert(
            "make_growable".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "pool.make_growable".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        pool_record.insert(
            "acquire".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "pool.acquire".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        pool_record.insert(
            "release".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "pool.release".to_string(),
                arity: 2,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        pool_record.insert(
            "reset".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "pool.reset".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        pool_record.insert(
            "available".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "pool.available".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        pool_record.insert(
            "capacity".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "pool.capacity".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        pool_record.insert(
            "stats".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "pool.stats".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        let pool_value = record_value(pool_record);
        if let Some(idx) = self.globals_map.get("pool").copied() {
            self.globals[idx as usize] = pool_value;
        } else {
            self.globals_map
                .insert("pool".to_string(), self.globals.len() as u16);
            self.globals.push(pool_value);
        }
        let mut sim_record = std::collections::HashMap::new();
        sim_record.insert(
            "make".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "sim.make".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        sim_record.insert(
            "make_seeded".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "sim.make_seeded".to_string(),
                arity: 2,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        sim_record.insert(
            "time".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "sim.time".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        sim_record.insert(
            "seed".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "sim.seed".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        sim_record.insert(
            "pending".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "sim.pending".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        sim_record.insert(
            "schedule".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "sim.schedule".to_string(),
                arity: 3,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        sim_record.insert(
            "step".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "sim.step".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        sim_record.insert(
            "run".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "sim.run".to_string(),
                arity: 2,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        sim_record.insert(
            "snapshot".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "sim.snapshot".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        sim_record.insert(
            "restore".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "sim.restore".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        sim_record.insert(
            "replay".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "sim.replay".to_string(),
                arity: 3,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        sim_record.insert(
            "log".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "sim.log".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        sim_record.insert(
            "entity_set".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "sim.entity_set".to_string(),
                arity: 3,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        sim_record.insert(
            "entity_get".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "sim.entity_get".to_string(),
                arity: 2,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        sim_record.insert(
            "entity_remove".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "sim.entity_remove".to_string(),
                arity: 2,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        sim_record.insert(
            "entity_ids".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "sim.entity_ids".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        sim_record.insert(
            "coroutine".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "sim.coroutine".to_string(),
                arity: 2,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        sim_record.insert(
            "coroutine_with".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "sim.coroutine_with".to_string(),
                arity: 3,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        sim_record.insert(
            "coroutine_args".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "sim.coroutine_args".to_string(),
                arity: 3,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        sim_record.insert(
            "world".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "sim.world".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        sim_record.insert(
            "state".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "sim.state".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        sim_record.insert(
            "emit".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "sim.emit".to_string(),
                arity: 2,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        sim_record.insert(
            "next".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "sim.next".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        sim_record.insert(
            "join".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "sim.join".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        sim_record.insert(
            "done".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "sim.done".to_string(),
                arity: 1,
                kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                bound: None,
            }))),
        );
        let sim_value = record_value(sim_record);
        if let Some(idx) = self.globals_map.get("sim").copied() {
            self.globals[idx as usize] = sim_value;
        } else {
            self.globals_map
                .insert("sim".to_string(), self.globals.len() as u16);
            self.globals.push(sim_value);
        }
        let mut spatial_record = std::collections::HashMap::new();
        for (name, arity) in [
            ("make", 0),
            ("upsert", 4),
            ("remove", 2),
            ("radius", 4),
            ("nearest", 3),
            ("occupancy", 5),
        ] {
            spatial_record.insert(
                name.to_string(),
                Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                    name: format!("spatial.{}", name),
                    arity,
                    kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                    bound: None,
                }))),
            );
        }
        let spatial_value = record_value(spatial_record);
        if let Some(idx) = self.globals_map.get("spatial").copied() {
            self.globals[idx as usize] = spatial_value;
        } else {
            self.globals_map
                .insert("spatial".to_string(), self.globals.len() as u16);
            self.globals.push(spatial_value);
        }
        let mut snn_record = std::collections::HashMap::new();
        for (name, arity) in [
            ("make", 1),
            ("connect", 4),
            ("set_potential", 3),
            ("get_potential", 2),
            ("set_threshold", 3),
            ("get_threshold", 2),
            ("set_decay", 2),
            ("get_decay", 1),
            ("step", 2),
            ("spikes", 1),
            ("potentials", 1),
            ("synapses", 1),
        ] {
            snn_record.insert(
                name.to_string(),
                Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                    name: format!("snn.{}", name),
                    arity,
                    kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                    bound: None,
                }))),
            );
        }
        let snn_value = record_value(snn_record);
        if let Some(idx) = self.globals_map.get("snn").copied() {
            self.globals[idx as usize] = snn_value;
        } else {
            self.globals_map
                .insert("snn".to_string(), self.globals.len() as u16);
            self.globals.push(snn_value);
        }
        let mut agent_record = std::collections::HashMap::new();
        for (name, arity) in [
            ("make", 2),
            ("register", 6),
            ("state", 2),
            ("body", 2),
            ("memory", 2),
            ("set_body", 3),
            ("set_memory", 3),
            ("position", 2),
            ("set_position", 4),
            ("neighbors", 3),
            ("reward_add", 3),
            ("reward_get", 2),
            ("reward_take", 2),
            ("sense_push", 3),
            ("sense_take", 2),
            ("action_push", 3),
            ("action_take", 2),
            ("stream", 3),
            ("next_float", 1),
            ("next_int", 2),
        ] {
            agent_record.insert(
                name.to_string(),
                Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                    name: format!("agent.{}", name),
                    arity,
                    kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
                    bound: None,
                }))),
            );
        }
        let agent_value = record_value(agent_record);
        if let Some(idx) = self.globals_map.get("agent").copied() {
            self.globals[idx as usize] = agent_value;
        } else {
            self.globals_map
                .insert("agent".to_string(), self.globals.len() as u16);
            self.globals.push(agent_value);
        }
        Ok(())
    }

    fn scheduler_loop(&mut self, program: &Program, main_id: usize) -> Result<Value, RuntimeError> {
        loop {
            self.drain_io_events();
            self.drain_server_events(program);
            self.wake_sleepers();
            if self.ready.is_empty() {
                if self.all_tasks_finished() {
                    self.stop_servers();
                    if let Some(task) = self.tasks.get(main_id).and_then(|t| t.as_ref()) {
                        if let Some(result) = &task.result {
                            return result.clone();
                        }
                    }
                    return Ok(Value::Null);
                }
                if let Some(next_wake) = self.next_wake_time() {
                    let now = Instant::now();
                    if next_wake > now {
                        std::thread::sleep(next_wake - now);
                    }
                } else {
                    std::thread::sleep(Duration::from_millis(1));
                }
                continue;
            }
            let task_id = self.ready.pop_front().unwrap();
            if let Some(task) = self.tasks.get(task_id).and_then(|t| t.as_ref()) {
                if matches!(task.state, TaskState::Finished) {
                    continue;
                }
            }
            self.current_task = Some(task_id);
            let outcome = self.run_task(program, task_id, 10_000);
            self.current_task = None;
            match outcome {
                TaskRunOutcome::Completed(value) => {
                    self.finish_task(task_id, Ok(value));
                }
                TaskRunOutcome::Errored(err) => {
                    if task_id == main_id {
                        self.stop_servers();
                        return Err(err);
                    }
                    self.finish_task(task_id, Err(err));
                }
                TaskRunOutcome::Yielded => {
                    if let Some(task) = self.tasks.get(task_id).and_then(|t| t.as_ref()) {
                        if matches!(task.state, TaskState::Ready) {
                            self.ready.push_back(task_id);
                        }
                    }
                }
            }
        }
    }

    fn run_task(&mut self, program: &Program, task_id: usize, budget: usize) -> TaskRunOutcome {
        let mut task = match self.tasks.get_mut(task_id).and_then(|entry| entry.take()) {
            Some(task) => task,
            None => return TaskRunOutcome::Errored(RuntimeError::new("Unknown task")),
        };
        if let Some(err) = task.pending_error.take() {
            self.tasks[task_id] = Some(task);
            return TaskRunOutcome::Errored(err);
        }
        self.pending_state = None;
        std::mem::swap(&mut self.active_policy, &mut task.policy);
        std::mem::swap(&mut self.active_http_conn, &mut task.http_conn);
        std::mem::swap(&mut self.stack, &mut task.stack);
        std::mem::swap(&mut self.frames, &mut task.frames);
        let outcome = self.execute(program, budget);
        std::mem::swap(&mut self.stack, &mut task.stack);
        std::mem::swap(&mut self.frames, &mut task.frames);
        std::mem::swap(&mut self.active_http_conn, &mut task.http_conn);
        std::mem::swap(&mut self.active_policy, &mut task.policy);
        if let Some(state) = self.pending_state.take() {
            task.state = state;
        }
        self.tasks[task_id] = Some(task);
        outcome
    }

    fn spawn_task_internal(
        &mut self,
        program: &Program,
        func: Value,
    ) -> Result<usize, RuntimeError> {
        let id = self.next_task_id;
        self.next_task_id += 1;
        if self.trace_task {
            println!("[task] spawn {}", id);
        }
        let mut task = Task {
            id,
            stack: Vec::new(),
            frames: Vec::new(),
            state: TaskState::Ready,
            result: None,
            join_waiters: Vec::new(),
            pending_error: None,
            http_conn: None,
            http_meta: None,
            policy: self.active_policy.clone(),
        };
        task.stack.push(func);
        std::mem::swap(&mut self.stack, &mut task.stack);
        std::mem::swap(&mut self.frames, &mut task.frames);
        let previous = self.current_task;
        self.current_task = Some(id);
        let result = self.call_value(program, 0);
        self.current_task = previous;
        std::mem::swap(&mut self.stack, &mut task.stack);
        std::mem::swap(&mut self.frames, &mut task.frames);
        if let Err(err) = result {
            task.state = TaskState::Finished;
            task.result = Some(Err(err));
        }
        self.insert_task(task);
        if let Some(task) = self.tasks.get(id).and_then(|t| t.as_ref()) {
            if matches!(task.state, TaskState::Ready) {
                self.ready.push_back(id);
            }
        }
        Ok(id)
    }

    fn spawn_task_with_args(
        &mut self,
        program: &Program,
        func: Value,
        args: Vec<Value>,
        http_conn: Option<TcpStream>,
    ) -> Result<usize, RuntimeError> {
        let argc = args.len();
        let id = self.next_task_id;
        self.next_task_id += 1;
        if self.trace_task {
            println!("[task] spawn {}", id);
        }
        let mut task = Task {
            id,
            stack: Vec::new(),
            frames: Vec::new(),
            state: TaskState::Ready,
            result: None,
            join_waiters: Vec::new(),
            pending_error: None,
            http_conn,
            http_meta: None,
            policy: self.active_policy.clone(),
        };
        task.stack.push(func);
        task.stack.extend(args);
        std::mem::swap(&mut self.stack, &mut task.stack);
        std::mem::swap(&mut self.frames, &mut task.frames);
        let previous = self.current_task;
        self.current_task = Some(id);
        let result = self.call_value(program, argc);
        self.current_task = previous;
        std::mem::swap(&mut self.stack, &mut task.stack);
        std::mem::swap(&mut self.frames, &mut task.frames);
        if let Err(err) = result {
            task.state = TaskState::Finished;
            task.result = Some(Err(err));
        }
        self.insert_task(task);
        if let Some(task) = self.tasks.get(id).and_then(|t| t.as_ref()) {
            if matches!(task.state, TaskState::Ready) {
                self.ready.push_back(id);
            }
        }
        Ok(id)
    }

    fn insert_task(&mut self, task: Task) {
        let id = task.id;
        if self.tasks.len() <= id {
            self.tasks.resize_with(id + 1, || None);
        }
        self.tasks[id] = Some(task);
    }

    fn finish_task(&mut self, task_id: usize, result: Result<Value, RuntimeError>) {
        let mut joiners = Vec::new();
        let mut http_conn = None;
        let mut http_meta = None;
        if let Some(task) = self.tasks.get_mut(task_id).and_then(|t| t.as_mut()) {
            task.state = TaskState::Finished;
            task.result = Some(result.clone());
            joiners.append(&mut task.join_waiters);
            http_conn = task.http_conn.take();
            http_meta = task.http_meta.take();
        }
        let _ = self.complete_sim_coroutine(task_id, Some(result.clone()));
        if self.trace_task {
            println!("[task] finish {}", task_id);
        }
        let mut ready_ids = Vec::new();
        for joiner_id in joiners {
            if let Some(joiner) = self.tasks.get_mut(joiner_id).and_then(|t| t.as_mut()) {
                match &result {
                    Ok(value) => {
                        joiner.stack.push(value.clone());
                    }
                    Err(err) => {
                        joiner.pending_error = Some(err.clone());
                    }
                }
                joiner.state = TaskState::Ready;
                ready_ids.push(joiner_id);
            }
        }
        for id in ready_ids {
            self.ready.push_back(id);
        }
        if let Some(stream) = http_conn {
            let mut response = self.response_from_result(&result);
            if let Some(meta) = &http_meta {
                self.attach_response_meta_headers(&mut response, meta);
                self.log_http_meta(meta, response.status, false);
            }
            std::thread::spawn(move || {
                let _ = write_http_response(stream, response);
            });
        } else if let Some(meta) = &http_meta {
            self.log_http_meta(meta, 0, true);
        }
        if let Some(meta) = &http_meta {
            if let Some(server) = self.servers.get(meta.server_id) {
                let _ =
                    server
                        .inflight
                        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                            Some(value.saturating_sub(1))
                        });
            }
        }
    }

    fn all_tasks_finished(&self) -> bool {
        self.tasks
            .iter()
            .filter_map(|t| t.as_ref())
            .all(|task| matches!(task.state, TaskState::Finished))
    }

    fn stop_servers(&self) {
        for server in &self.servers {
            let _ = server.stop.send(());
        }
    }

    fn wake_sleepers(&mut self) {
        let now = Instant::now();
        let mut ready_ids = Vec::new();
        for task in self.tasks.iter_mut().filter_map(|t| t.as_mut()) {
            if let TaskState::Sleeping(deadline) = task.state {
                if deadline <= now {
                    task.state = TaskState::Ready;
                    ready_ids.push(task.id);
                }
            }
        }
        for id in ready_ids {
            self.ready.push_back(id);
        }
    }

    fn next_wake_time(&self) -> Option<Instant> {
        self.tasks
            .iter()
            .filter_map(|t| t.as_ref())
            .filter_map(|task| match task.state {
                TaskState::Sleeping(deadline) => Some(deadline),
                _ => None,
            })
            .min()
    }

    fn drain_io_events(&mut self) {
        let mut ready_ids = Vec::new();
        while let Ok(event) = self.io_receiver.try_recv() {
            let mut value = None;
            let mut error = None;
            match event.result {
                IoResult::Accept(result) => match result {
                    Ok(stream) => {
                        value = Some(Value::Obj(ObjRef::new(Obj::TcpConnection(
                            std::cell::RefCell::new(stream),
                        ))));
                    }
                    Err(err) => {
                        error = Some(RuntimeError::new(&format!("accept failed: {}", err)));
                    }
                },
                IoResult::Read(result) | IoResult::ReadAll(result) => match result {
                    Ok(bytes) => {
                        value = Some(Value::Obj(ObjRef::new(Obj::Buffer(bytes))));
                    }
                    Err(err) => {
                        error = Some(RuntimeError::new(&format!("read failed: {}", err)));
                    }
                },
                IoResult::Write(result) => match result {
                    Ok(count) => value = Some(Value::Int(count as i64)),
                    Err(err) => {
                        error = Some(RuntimeError::new(&format!("write failed: {}", err)));
                    }
                },
                IoResult::HttpResponse(result) => match result {
                    Ok(resp) => {
                        value = Some(self.http_response_value(resp));
                    }
                    Err(err) => {
                        error = Some(RuntimeError::new(&format!("http failed: {}", err)));
                    }
                },
            }
            if let Some(task) = self.tasks.get_mut(event.task_id).and_then(|t| t.as_mut()) {
                if let Some(value) = value {
                    task.stack.push(value);
                }
                if let Some(err) = error {
                    task.pending_error = Some(err);
                }
                task.state = TaskState::Ready;
                ready_ids.push(task.id);
            }
        }
        for id in ready_ids {
            self.ready.push_back(id);
        }
    }

    fn drain_server_events(&mut self, program: &Program) {
        while let Ok(event) = self.server_receiver.try_recv() {
            let stream = event.stream;
            let request_id = self.next_request_id;
            self.next_request_id = self.next_request_id.saturating_add(1);
            let queue_ms = event.accepted_at.elapsed().as_millis() as u64;
            let (
                handler,
                params,
                tenant,
                correlation_id,
                request_model_name,
                request_model_version,
                error_resp,
                error_code,
            ) = match self.prepare_http_request(event.server_id, &event.request, request_id) {
                Ok(value) => value,
                Err(err) => {
                    self.write_http_error(stream, &err.to_string());
                    continue;
                }
            };
            let (model_registry, inflight_counter, inflight_now, max_inflight, server_policy) =
                match self.servers.get(event.server_id) {
                    Some(server) => (
                        server.model_registry.clone(),
                        Arc::clone(&server.inflight),
                        server.inflight.load(Ordering::Relaxed),
                        server.max_inflight,
                        server.policy.clone(),
                    ),
                    None => (None, Arc::new(AtomicUsize::new(0)), 0, 0, None),
                };
            if let Some(mut resp) = error_resp {
                let meta = HttpRequestMeta {
                    id: request_id,
                    server_id: event.server_id,
                    start: Instant::now(),
                    queue_ms,
                    method: event.request.method.clone(),
                    path: event.request.path.clone(),
                    remote_addr: event.request.remote_addr.clone(),
                    correlation_id: correlation_id.clone(),
                    tenant: tenant.clone(),
                    model_name: request_model_name.clone(),
                    model_version: request_model_version.clone(),
                    model_registry: model_registry.clone(),
                    inflight_at_start: inflight_now,
                    error_code: error_code.clone(),
                };
                self.attach_response_meta_headers(&mut resp, &meta);
                self.log_http_meta(&meta, resp.status, false);
                let _ = write_http_response(stream, resp);
                continue;
            }
            let handler = match handler {
                Some(h) => h,
                None => {
                    let mut resp = error_response(404, "not_found", "Not Found");
                    let meta = HttpRequestMeta {
                        id: request_id,
                        server_id: event.server_id,
                        start: Instant::now(),
                        queue_ms,
                        method: event.request.method.clone(),
                        path: event.request.path.clone(),
                        remote_addr: event.request.remote_addr.clone(),
                        correlation_id: correlation_id.clone(),
                        tenant: tenant.clone(),
                        model_name: request_model_name.clone(),
                        model_version: request_model_version.clone(),
                        model_registry: model_registry.clone(),
                        inflight_at_start: inflight_now,
                        error_code: Some("not_found".to_string()),
                    };
                    self.attach_response_meta_headers(&mut resp, &meta);
                    self.log_http_meta(&meta, resp.status, false);
                    let _ = write_http_response(stream, resp);
                    continue;
                }
            };
            if max_inflight > 0 && inflight_now >= max_inflight {
                let mut resp = error_response(
                    503,
                    "backpressure_overloaded",
                    "Server is overloaded; try again later",
                );
                let meta = HttpRequestMeta {
                    id: request_id,
                    server_id: event.server_id,
                    start: Instant::now(),
                    queue_ms,
                    method: event.request.method.clone(),
                    path: event.request.path.clone(),
                    remote_addr: event.request.remote_addr.clone(),
                    correlation_id: correlation_id.clone(),
                    tenant,
                    model_name: request_model_name.clone(),
                    model_version: request_model_version.clone(),
                    model_registry: model_registry.clone(),
                    inflight_at_start: inflight_now,
                    error_code: Some("backpressure_overloaded".to_string()),
                };
                self.attach_response_meta_headers(&mut resp, &meta);
                self.log_http_meta(&meta, resp.status, false);
                let _ = write_http_response(stream, resp);
                continue;
            }
            let mut meta = HttpRequestMeta {
                id: request_id,
                server_id: event.server_id,
                start: Instant::now(),
                queue_ms,
                method: event.request.method.clone(),
                path: event.request.path.clone(),
                remote_addr: event.request.remote_addr.clone(),
                correlation_id,
                tenant,
                model_name: request_model_name,
                model_version: request_model_version,
                model_registry: model_registry.clone(),
                inflight_at_start: inflight_now.saturating_add(1),
                error_code: None,
            };
            if let Some(code) = error_code {
                meta.error_code = Some(code);
            }
            let mut request_data = event.request;
            request_data
                .headers
                .entry("x-enkai-correlation-id".to_string())
                .or_insert_with(|| meta.correlation_id.clone());
            if let Some(tenant_id) = meta.tenant.as_ref() {
                request_data
                    .headers
                    .entry("x-enkai-tenant".to_string())
                    .or_insert_with(|| tenant_id.clone());
            }
            if let Some(model_name) = meta.model_name.as_ref() {
                request_data
                    .headers
                    .entry("x-enkai-model-name".to_string())
                    .or_insert_with(|| model_name.clone());
            }
            if let Some(model_version) = meta.model_version.as_ref() {
                request_data
                    .headers
                    .entry("x-enkai-model-version".to_string())
                    .or_insert_with(|| model_version.clone());
            }
            let request = self.http_request_value_with_params(request_data, params);
            let stream_for_task = match stream.try_clone() {
                Ok(clone) => clone,
                Err(_) => {
                    let mut resp = error_response(
                        500,
                        "stream_clone_failed",
                        "failed to prepare request stream",
                    );
                    let mut fallback = meta;
                    fallback.error_code = Some("stream_clone_failed".to_string());
                    self.attach_response_meta_headers(&mut resp, &fallback);
                    self.log_http_meta(&fallback, resp.status, false);
                    let _ = write_http_response(stream, resp);
                    continue;
                }
            };
            if let Ok(task_id) =
                self.spawn_task_with_args(program, handler, vec![request], Some(stream_for_task))
            {
                let inflight_after = inflight_counter.fetch_add(1, Ordering::Relaxed) + 1;
                meta.inflight_at_start = inflight_after;
                if let Some(task) = self.tasks.get_mut(task_id).and_then(|t| t.as_mut()) {
                    task.http_meta = Some(meta);
                    if let Some(policy) = server_policy {
                        task.policy = Some(policy);
                    }
                }
            } else {
                let mut resp =
                    error_response(500, "task_spawn_failed", "failed to spawn request handler");
                let mut fallback = meta;
                fallback.error_code = Some("task_spawn_failed".to_string());
                fallback.inflight_at_start = inflight_counter.load(Ordering::Relaxed);
                self.attach_response_meta_headers(&mut resp, &fallback);
                self.log_http_meta(&fallback, resp.status, false);
                let _ = write_http_response(stream, resp);
            }
        }
    }

    fn execute(&mut self, program: &Program, mut budget: usize) -> TaskRunOutcome {
        while let Some(frame_view) = self.frames.last() {
            if budget == 0 {
                return TaskRunOutcome::Yielded;
            }
            let func_index = frame_view.func_index;
            let ip = frame_view.ip;
            let base = frame_view.base;
            let caller_sp = frame_view.caller_sp;
            let func = &program.functions[func_index as usize];
            let trace = |vm: &VM, err: RuntimeError| err.with_frames(vm.stack_trace(program, ip));
            if ip >= func.chunk.code.len() {
                return TaskRunOutcome::Errored(trace(
                    self,
                    RuntimeError::new("Instruction pointer out of bounds"),
                ));
            }
            let instr = func.chunk.code[ip].clone();
            if let Some(profile) = self.bench_profile.as_mut() {
                profile.counters.opcode_dispatch =
                    profile.counters.opcode_dispatch.saturating_add(1);
            }
            if self.trace {
                println!(
                    "[frame {} ip {}] {:?} | stack {:?}",
                    func_index, ip, instr, self.stack
                );
            }
            // advance ip
            let mut next_ip = ip + 1;
            let mut update_ip = true;
            match instr {
                Instruction::Const(idx) => {
                    // Fold "Const(Int) + next arithmetic/compare op" into a single step when
                    // lhs is already on top of the stack.
                    let mut fast_path_taken = false;
                    if let Some(Constant::Int(rhs)) = func.chunk.constants.get(idx as usize) {
                        if let Some(Value::Int(lhs)) = self.stack.last().cloned() {
                            if let Some(next_instr) = func.chunk.code.get(ip + 1) {
                                let maybe_value = match next_instr {
                                    Instruction::Add => Some(Value::Int(lhs.wrapping_add(*rhs))),
                                    Instruction::Sub => Some(Value::Int(lhs.wrapping_sub(*rhs))),
                                    Instruction::Mul => Some(Value::Int(lhs.wrapping_mul(*rhs))),
                                    Instruction::Div => {
                                        if *rhs == 0 {
                                            return TaskRunOutcome::Errored(trace(
                                                self,
                                                RuntimeError::new("Division by zero"),
                                            ));
                                        }
                                        let value = if lhs == i64::MIN && *rhs == -1 {
                                            i64::MIN
                                        } else {
                                            lhs / *rhs
                                        };
                                        Some(Value::Int(value))
                                    }
                                    Instruction::Mod => {
                                        if *rhs == 0 {
                                            return TaskRunOutcome::Errored(trace(
                                                self,
                                                RuntimeError::new("Modulo by zero"),
                                            ));
                                        }
                                        let value = if lhs == i64::MIN && *rhs == -1 {
                                            0
                                        } else {
                                            lhs % *rhs
                                        };
                                        Some(Value::Int(value))
                                    }
                                    Instruction::Eq => Some(Value::Bool(lhs == *rhs)),
                                    Instruction::Neq => Some(Value::Bool(lhs != *rhs)),
                                    Instruction::Lt => Some(Value::Bool(lhs < *rhs)),
                                    Instruction::Gt => Some(Value::Bool(lhs > *rhs)),
                                    Instruction::Le => Some(Value::Bool(lhs <= *rhs)),
                                    Instruction::Ge => Some(Value::Bool(lhs >= *rhs)),
                                    _ => None,
                                };
                                if let Some(value) = maybe_value {
                                    self.stack.pop();
                                    self.stack.push(value);
                                    next_ip = ip + 2;
                                    if let Some(profile) = self.bench_profile.as_mut() {
                                        match next_instr {
                                            Instruction::Add
                                            | Instruction::Sub
                                            | Instruction::Mul
                                            | Instruction::Div
                                            | Instruction::Mod => {
                                                profile.counters.arithmetic_ops = profile
                                                    .counters
                                                    .arithmetic_ops
                                                    .saturating_add(1);
                                            }
                                            _ => {
                                                profile.counters.compare_ops =
                                                    profile.counters.compare_ops.saturating_add(1);
                                            }
                                        }
                                    }
                                    fast_path_taken = true;
                                }
                            }
                        }
                    }

                    if !fast_path_taken {
                        if let Some(Constant::Float(rhs)) = func.chunk.constants.get(idx as usize) {
                            if let Some(Value::Float(lhs)) = self.stack.last().cloned() {
                                if let Some(next_instr) = func.chunk.code.get(ip + 1) {
                                    let maybe_value = match next_instr {
                                        Instruction::Add => Some(Value::Float(lhs + *rhs)),
                                        Instruction::Sub => Some(Value::Float(lhs - *rhs)),
                                        Instruction::Mul => Some(Value::Float(lhs * *rhs)),
                                        Instruction::Div => {
                                            if *rhs == 0.0 {
                                                return TaskRunOutcome::Errored(trace(
                                                    self,
                                                    RuntimeError::new("Division by zero"),
                                                ));
                                            }
                                            Some(Value::Float(lhs / *rhs))
                                        }
                                        Instruction::Eq => Some(Value::Bool(lhs == *rhs)),
                                        Instruction::Neq => Some(Value::Bool(lhs != *rhs)),
                                        Instruction::Lt => Some(Value::Bool(lhs < *rhs)),
                                        Instruction::Gt => Some(Value::Bool(lhs > *rhs)),
                                        Instruction::Le => Some(Value::Bool(lhs <= *rhs)),
                                        Instruction::Ge => Some(Value::Bool(lhs >= *rhs)),
                                        _ => None,
                                    };
                                    if let Some(value) = maybe_value {
                                        self.stack.pop();
                                        self.stack.push(value);
                                        next_ip = ip + 2;
                                        if let Some(profile) = self.bench_profile.as_mut() {
                                            match next_instr {
                                                Instruction::Add
                                                | Instruction::Sub
                                                | Instruction::Mul
                                                | Instruction::Div => {
                                                    profile.counters.arithmetic_ops = profile
                                                        .counters
                                                        .arithmetic_ops
                                                        .saturating_add(1);
                                                }
                                                _ => {
                                                    profile.counters.compare_ops = profile
                                                        .counters
                                                        .compare_ops
                                                        .saturating_add(1);
                                                }
                                            }
                                        }
                                        fast_path_taken = true;
                                    }
                                }
                            }
                        }
                    }

                    if !fast_path_taken {
                        let v = match &func.chunk.constants[idx as usize] {
                            Constant::Int(i) => Value::Int(*i),
                            Constant::Float(f) => Value::Float(*f),
                            Constant::Bool(b) => Value::Bool(*b),
                            Constant::Null => Value::Null,
                            other => match self.constant_to_value(other, program) {
                                Ok(v) => v,
                                Err(err) => return TaskRunOutcome::Errored(trace(self, err)),
                            },
                        };
                        self.stack.push(v);
                    }
                }
                Instruction::Pop => {
                    self.stack.pop();
                }
                Instruction::DefineGlobal(idx) => {
                    let val = match self.stack.pop() {
                        Some(val) => val,
                        None => {
                            return TaskRunOutcome::Errored(trace(
                                self,
                                RuntimeError::new("Stack underflow"),
                            ))
                        }
                    };
                    let slot = idx as usize;
                    if slot >= self.globals.len() {
                        return TaskRunOutcome::Errored(trace(
                            self,
                            RuntimeError::new("Global not found"),
                        ));
                    }
                    self.globals[slot] = val;
                }
                Instruction::LoadLocal(idx) => {
                    let slot = base + idx as usize;
                    if slot >= self.stack.len() {
                        return TaskRunOutcome::Errored(trace(
                            self,
                            RuntimeError::new("LoadLocal out of range"),
                        ));
                    }

                    // Numeric super-instructions for loop-heavy workloads.
                    let mut fast_path_taken = false;
                    if let Value::Int(local_int) = self.stack[slot] {
                        let const_int = |const_idx: u16| -> Option<i64> {
                            match func.chunk.constants.get(const_idx as usize) {
                                Some(Constant::Int(i)) => Some(*i),
                                _ => None,
                            }
                        };

                        if let (
                            Some(Instruction::Const(const_idx)),
                            Some(Instruction::Add),
                            Some(Instruction::StoreLocal(target)),
                        ) = (
                            func.chunk.code.get(ip + 1),
                            func.chunk.code.get(ip + 2),
                            func.chunk.code.get(ip + 3),
                        ) {
                            if *target == idx {
                                if let Some(rhs) = const_int(*const_idx) {
                                    self.stack[slot] = Value::Int(local_int.wrapping_add(rhs));
                                    next_ip = ip + 4;
                                    if let Some(profile) = self.bench_profile.as_mut() {
                                        profile.counters.arithmetic_ops =
                                            profile.counters.arithmetic_ops.saturating_add(1);
                                    }
                                    fast_path_taken = true;
                                }
                            }
                        }

                        if !fast_path_taken {
                            if let (
                                Some(Instruction::Const(const_idx)),
                                Some(Instruction::Sub),
                                Some(Instruction::StoreLocal(target)),
                            ) = (
                                func.chunk.code.get(ip + 1),
                                func.chunk.code.get(ip + 2),
                                func.chunk.code.get(ip + 3),
                            ) {
                                if *target == idx {
                                    if let Some(rhs) = const_int(*const_idx) {
                                        self.stack[slot] = Value::Int(local_int.wrapping_sub(rhs));
                                        next_ip = ip + 4;
                                        if let Some(profile) = self.bench_profile.as_mut() {
                                            profile.counters.arithmetic_ops =
                                                profile.counters.arithmetic_ops.saturating_add(1);
                                        }
                                        fast_path_taken = true;
                                    }
                                }
                            }
                        }

                        if !fast_path_taken {
                            if let (Some(Instruction::Const(const_idx)), Some(Instruction::Mul)) =
                                (func.chunk.code.get(ip + 1), func.chunk.code.get(ip + 2))
                            {
                                if let Some(rhs) = const_int(*const_idx) {
                                    self.stack.push(Value::Int(local_int.wrapping_mul(rhs)));
                                    next_ip = ip + 3;
                                    if let Some(profile) = self.bench_profile.as_mut() {
                                        profile.counters.arithmetic_ops =
                                            profile.counters.arithmetic_ops.saturating_add(1);
                                    }
                                    fast_path_taken = true;
                                }
                            }
                        }

                        if !fast_path_taken {
                            if let (Some(Instruction::Const(const_idx)), Some(Instruction::Div)) =
                                (func.chunk.code.get(ip + 1), func.chunk.code.get(ip + 2))
                            {
                                if let Some(rhs) = const_int(*const_idx) {
                                    if rhs == 0 {
                                        return TaskRunOutcome::Errored(trace(
                                            self,
                                            RuntimeError::new("Division by zero"),
                                        ));
                                    }
                                    let result = if local_int == i64::MIN && rhs == -1 {
                                        i64::MIN
                                    } else {
                                        local_int / rhs
                                    };
                                    self.stack.push(Value::Int(result));
                                    next_ip = ip + 3;
                                    if let Some(profile) = self.bench_profile.as_mut() {
                                        profile.counters.arithmetic_ops =
                                            profile.counters.arithmetic_ops.saturating_add(1);
                                    }
                                    fast_path_taken = true;
                                }
                            }
                        }

                        if !fast_path_taken {
                            if let (Some(Instruction::Const(const_idx)), Some(Instruction::Add)) =
                                (func.chunk.code.get(ip + 1), func.chunk.code.get(ip + 2))
                            {
                                if let Some(rhs) = const_int(*const_idx) {
                                    self.stack.push(Value::Int(local_int.wrapping_add(rhs)));
                                    next_ip = ip + 3;
                                    if let Some(profile) = self.bench_profile.as_mut() {
                                        profile.counters.arithmetic_ops =
                                            profile.counters.arithmetic_ops.saturating_add(1);
                                    }
                                    fast_path_taken = true;
                                }
                            }
                        }

                        if !fast_path_taken {
                            if let (Some(Instruction::Const(const_idx)), Some(Instruction::Sub)) =
                                (func.chunk.code.get(ip + 1), func.chunk.code.get(ip + 2))
                            {
                                if let Some(rhs) = const_int(*const_idx) {
                                    self.stack.push(Value::Int(local_int.wrapping_sub(rhs)));
                                    next_ip = ip + 3;
                                    if let Some(profile) = self.bench_profile.as_mut() {
                                        profile.counters.arithmetic_ops =
                                            profile.counters.arithmetic_ops.saturating_add(1);
                                    }
                                    fast_path_taken = true;
                                }
                            }
                        }

                        if !fast_path_taken {
                            if let (Some(Instruction::Const(const_idx)), Some(cmp_instr)) =
                                (func.chunk.code.get(ip + 1), func.chunk.code.get(ip + 2))
                            {
                                if let Some(rhs) = const_int(*const_idx) {
                                    let cmp_result = match cmp_instr {
                                        Instruction::Lt => Some(local_int < rhs),
                                        Instruction::Gt => Some(local_int > rhs),
                                        Instruction::Le => Some(local_int <= rhs),
                                        Instruction::Ge => Some(local_int >= rhs),
                                        _ => None,
                                    };
                                    if let Some(value) = cmp_result {
                                        self.stack.push(Value::Bool(value));
                                        next_ip = ip + 3;
                                        if let Some(profile) = self.bench_profile.as_mut() {
                                            profile.counters.compare_ops =
                                                profile.counters.compare_ops.saturating_add(1);
                                        }
                                        fast_path_taken = true;
                                    }
                                }
                            }
                        }
                    }

                    if !fast_path_taken {
                        if let Value::Float(local_float) = self.stack[slot] {
                            let const_float = |const_idx: u16| -> Option<f64> {
                                match func.chunk.constants.get(const_idx as usize) {
                                    Some(Constant::Float(v)) => Some(*v),
                                    _ => None,
                                }
                            };

                            if let (
                                Some(Instruction::Const(const_idx)),
                                Some(Instruction::Add),
                                Some(Instruction::StoreLocal(target)),
                            ) = (
                                func.chunk.code.get(ip + 1),
                                func.chunk.code.get(ip + 2),
                                func.chunk.code.get(ip + 3),
                            ) {
                                if *target == idx {
                                    if let Some(rhs) = const_float(*const_idx) {
                                        self.stack[slot] = Value::Float(local_float + rhs);
                                        next_ip = ip + 4;
                                        if let Some(profile) = self.bench_profile.as_mut() {
                                            profile.counters.arithmetic_ops =
                                                profile.counters.arithmetic_ops.saturating_add(1);
                                        }
                                        fast_path_taken = true;
                                    }
                                }
                            }

                            if !fast_path_taken {
                                if let (
                                    Some(Instruction::Const(const_idx)),
                                    Some(Instruction::Sub),
                                    Some(Instruction::StoreLocal(target)),
                                ) = (
                                    func.chunk.code.get(ip + 1),
                                    func.chunk.code.get(ip + 2),
                                    func.chunk.code.get(ip + 3),
                                ) {
                                    if *target == idx {
                                        if let Some(rhs) = const_float(*const_idx) {
                                            self.stack[slot] = Value::Float(local_float - rhs);
                                            next_ip = ip + 4;
                                            if let Some(profile) = self.bench_profile.as_mut() {
                                                profile.counters.arithmetic_ops = profile
                                                    .counters
                                                    .arithmetic_ops
                                                    .saturating_add(1);
                                            }
                                            fast_path_taken = true;
                                        }
                                    }
                                }
                            }

                            if !fast_path_taken {
                                if let (
                                    Some(Instruction::Const(const_idx)),
                                    Some(Instruction::Mul),
                                    Some(Instruction::StoreLocal(target)),
                                ) = (
                                    func.chunk.code.get(ip + 1),
                                    func.chunk.code.get(ip + 2),
                                    func.chunk.code.get(ip + 3),
                                ) {
                                    if *target == idx {
                                        if let Some(rhs) = const_float(*const_idx) {
                                            self.stack[slot] = Value::Float(local_float * rhs);
                                            next_ip = ip + 4;
                                            if let Some(profile) = self.bench_profile.as_mut() {
                                                profile.counters.arithmetic_ops = profile
                                                    .counters
                                                    .arithmetic_ops
                                                    .saturating_add(1);
                                            }
                                            fast_path_taken = true;
                                        }
                                    }
                                }
                            }

                            if !fast_path_taken {
                                if let (
                                    Some(Instruction::Const(const_idx)),
                                    Some(Instruction::Div),
                                    Some(Instruction::StoreLocal(target)),
                                ) = (
                                    func.chunk.code.get(ip + 1),
                                    func.chunk.code.get(ip + 2),
                                    func.chunk.code.get(ip + 3),
                                ) {
                                    if *target == idx {
                                        if let Some(rhs) = const_float(*const_idx) {
                                            if rhs == 0.0 {
                                                return TaskRunOutcome::Errored(trace(
                                                    self,
                                                    RuntimeError::new("Division by zero"),
                                                ));
                                            }
                                            self.stack[slot] = Value::Float(local_float / rhs);
                                            next_ip = ip + 4;
                                            if let Some(profile) = self.bench_profile.as_mut() {
                                                profile.counters.arithmetic_ops = profile
                                                    .counters
                                                    .arithmetic_ops
                                                    .saturating_add(1);
                                            }
                                            fast_path_taken = true;
                                        }
                                    }
                                }
                            }

                            if !fast_path_taken {
                                if let (
                                    Some(Instruction::Const(const_idx)),
                                    Some(Instruction::Mul),
                                ) = (func.chunk.code.get(ip + 1), func.chunk.code.get(ip + 2))
                                {
                                    if let Some(rhs) = const_float(*const_idx) {
                                        self.stack.push(Value::Float(local_float * rhs));
                                        next_ip = ip + 3;
                                        if let Some(profile) = self.bench_profile.as_mut() {
                                            profile.counters.arithmetic_ops =
                                                profile.counters.arithmetic_ops.saturating_add(1);
                                        }
                                        fast_path_taken = true;
                                    }
                                }
                            }

                            if !fast_path_taken {
                                if let (
                                    Some(Instruction::Const(const_idx)),
                                    Some(Instruction::Div),
                                ) = (func.chunk.code.get(ip + 1), func.chunk.code.get(ip + 2))
                                {
                                    if let Some(rhs) = const_float(*const_idx) {
                                        if rhs == 0.0 {
                                            return TaskRunOutcome::Errored(trace(
                                                self,
                                                RuntimeError::new("Division by zero"),
                                            ));
                                        }
                                        self.stack.push(Value::Float(local_float / rhs));
                                        next_ip = ip + 3;
                                        if let Some(profile) = self.bench_profile.as_mut() {
                                            profile.counters.arithmetic_ops =
                                                profile.counters.arithmetic_ops.saturating_add(1);
                                        }
                                        fast_path_taken = true;
                                    }
                                }
                            }

                            if !fast_path_taken {
                                if let (
                                    Some(Instruction::Const(const_idx)),
                                    Some(Instruction::Add),
                                ) = (func.chunk.code.get(ip + 1), func.chunk.code.get(ip + 2))
                                {
                                    if let Some(rhs) = const_float(*const_idx) {
                                        self.stack.push(Value::Float(local_float + rhs));
                                        next_ip = ip + 3;
                                        if let Some(profile) = self.bench_profile.as_mut() {
                                            profile.counters.arithmetic_ops =
                                                profile.counters.arithmetic_ops.saturating_add(1);
                                        }
                                        fast_path_taken = true;
                                    }
                                }
                            }

                            if !fast_path_taken {
                                if let (
                                    Some(Instruction::Const(const_idx)),
                                    Some(Instruction::Sub),
                                ) = (func.chunk.code.get(ip + 1), func.chunk.code.get(ip + 2))
                                {
                                    if let Some(rhs) = const_float(*const_idx) {
                                        self.stack.push(Value::Float(local_float - rhs));
                                        next_ip = ip + 3;
                                        if let Some(profile) = self.bench_profile.as_mut() {
                                            profile.counters.arithmetic_ops =
                                                profile.counters.arithmetic_ops.saturating_add(1);
                                        }
                                        fast_path_taken = true;
                                    }
                                }
                            }

                            if !fast_path_taken {
                                if let (Some(Instruction::Const(const_idx)), Some(cmp_instr)) =
                                    (func.chunk.code.get(ip + 1), func.chunk.code.get(ip + 2))
                                {
                                    if let Some(rhs) = const_float(*const_idx) {
                                        let cmp_result = match cmp_instr {
                                            Instruction::Lt => Some(local_float < rhs),
                                            Instruction::Gt => Some(local_float > rhs),
                                            Instruction::Le => Some(local_float <= rhs),
                                            Instruction::Ge => Some(local_float >= rhs),
                                            _ => None,
                                        };
                                        if let Some(value) = cmp_result {
                                            self.stack.push(Value::Bool(value));
                                            next_ip = ip + 3;
                                            if let Some(profile) = self.bench_profile.as_mut() {
                                                profile.counters.compare_ops =
                                                    profile.counters.compare_ops.saturating_add(1);
                                            }
                                            fast_path_taken = true;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if !fast_path_taken {
                        let val = self.stack[slot].clone();
                        self.stack.push(val);
                    }
                }
                Instruction::StoreLocal(idx) => {
                    let val = match self.stack.pop() {
                        Some(val) => val,
                        None => {
                            return TaskRunOutcome::Errored(trace(
                                self,
                                RuntimeError::new("Stack underflow"),
                            ))
                        }
                    };
                    let slot = base + idx as usize;
                    if slot >= self.stack.len() {
                        self.stack.resize(slot + 1, Value::Null);
                    }
                    self.stack[slot] = val;
                }
                Instruction::LoadGlobal(idx) => {
                    let slot = idx as usize;
                    if slot >= self.globals.len() {
                        return TaskRunOutcome::Errored(trace(
                            self,
                            RuntimeError::new("Global not found"),
                        ));
                    }

                    let mut fast_path_taken = false;
                    if let (Some(Instruction::LoadGlobal(rhs_idx)), Some(op_instr)) =
                        (func.chunk.code.get(ip + 1), func.chunk.code.get(ip + 2))
                    {
                        if matches!(
                            op_instr,
                            Instruction::Add
                                | Instruction::Sub
                                | Instruction::Mul
                                | Instruction::Div
                                | Instruction::Lt
                                | Instruction::Gt
                                | Instruction::Le
                                | Instruction::Ge
                        ) {
                            let rhs_slot = *rhs_idx as usize;
                            if rhs_slot < self.globals.len() {
                                match Self::try_fast_numeric_binary(
                                    op_instr,
                                    &self.globals[slot],
                                    &self.globals[rhs_slot],
                                ) {
                                    Ok(Some(value)) => {
                                        self.stack.push(value);
                                        next_ip = ip + 3;
                                        if let Some(profile) = self.bench_profile.as_mut() {
                                            match op_instr {
                                                Instruction::Add
                                                | Instruction::Sub
                                                | Instruction::Mul
                                                | Instruction::Div => {
                                                    profile.counters.arithmetic_ops = profile
                                                        .counters
                                                        .arithmetic_ops
                                                        .saturating_add(1);
                                                }
                                                _ => {
                                                    profile.counters.compare_ops = profile
                                                        .counters
                                                        .compare_ops
                                                        .saturating_add(1);
                                                }
                                            }
                                        }
                                        fast_path_taken = true;
                                    }
                                    Ok(None) => {}
                                    Err(err) => {
                                        return TaskRunOutcome::Errored(trace(self, err));
                                    }
                                }
                            }
                        }
                    }

                    if let Value::Int(global_int_ref) = &self.globals[slot] {
                        let global_int = *global_int_ref;
                        let const_int = |const_idx: u16| -> Option<i64> {
                            match func.chunk.constants.get(const_idx as usize) {
                                Some(Constant::Int(i)) => Some(*i),
                                _ => None,
                            }
                        };

                        if let (
                            Some(Instruction::Const(const_idx)),
                            Some(Instruction::Add),
                            Some(Instruction::StoreGlobal(target)),
                        ) = (
                            func.chunk.code.get(ip + 1),
                            func.chunk.code.get(ip + 2),
                            func.chunk.code.get(ip + 3),
                        ) {
                            if *target == idx {
                                if let Some(rhs) = const_int(*const_idx) {
                                    self.globals[slot] = Value::Int(global_int.wrapping_add(rhs));
                                    next_ip = ip + 4;
                                    if let Some(profile) = self.bench_profile.as_mut() {
                                        profile.counters.arithmetic_ops =
                                            profile.counters.arithmetic_ops.saturating_add(1);
                                    }
                                    fast_path_taken = true;
                                }
                            }
                        }

                        if !fast_path_taken {
                            if let (
                                Some(Instruction::Const(const_idx)),
                                Some(Instruction::Sub),
                                Some(Instruction::StoreGlobal(target)),
                            ) = (
                                func.chunk.code.get(ip + 1),
                                func.chunk.code.get(ip + 2),
                                func.chunk.code.get(ip + 3),
                            ) {
                                if *target == idx {
                                    if let Some(rhs) = const_int(*const_idx) {
                                        self.globals[slot] =
                                            Value::Int(global_int.wrapping_sub(rhs));
                                        next_ip = ip + 4;
                                        if let Some(profile) = self.bench_profile.as_mut() {
                                            profile.counters.arithmetic_ops =
                                                profile.counters.arithmetic_ops.saturating_add(1);
                                        }
                                        fast_path_taken = true;
                                    }
                                }
                            }
                        }

                        if !fast_path_taken {
                            if let (Some(Instruction::Const(const_idx)), Some(Instruction::Mul)) =
                                (func.chunk.code.get(ip + 1), func.chunk.code.get(ip + 2))
                            {
                                if let Some(rhs) = const_int(*const_idx) {
                                    self.stack.push(Value::Int(global_int.wrapping_mul(rhs)));
                                    next_ip = ip + 3;
                                    if let Some(profile) = self.bench_profile.as_mut() {
                                        profile.counters.arithmetic_ops =
                                            profile.counters.arithmetic_ops.saturating_add(1);
                                    }
                                    fast_path_taken = true;
                                }
                            }
                        }

                        if !fast_path_taken {
                            if let (Some(Instruction::Const(const_idx)), Some(Instruction::Div)) =
                                (func.chunk.code.get(ip + 1), func.chunk.code.get(ip + 2))
                            {
                                if let Some(rhs) = const_int(*const_idx) {
                                    if rhs == 0 {
                                        return TaskRunOutcome::Errored(trace(
                                            self,
                                            RuntimeError::new("Division by zero"),
                                        ));
                                    }
                                    let result = if global_int == i64::MIN && rhs == -1 {
                                        i64::MIN
                                    } else {
                                        global_int / rhs
                                    };
                                    self.stack.push(Value::Int(result));
                                    next_ip = ip + 3;
                                    if let Some(profile) = self.bench_profile.as_mut() {
                                        profile.counters.arithmetic_ops =
                                            profile.counters.arithmetic_ops.saturating_add(1);
                                    }
                                    fast_path_taken = true;
                                }
                            }
                        }

                        if !fast_path_taken {
                            if let (Some(Instruction::Const(const_idx)), Some(Instruction::Add)) =
                                (func.chunk.code.get(ip + 1), func.chunk.code.get(ip + 2))
                            {
                                if let Some(rhs) = const_int(*const_idx) {
                                    self.stack.push(Value::Int(global_int.wrapping_add(rhs)));
                                    next_ip = ip + 3;
                                    if let Some(profile) = self.bench_profile.as_mut() {
                                        profile.counters.arithmetic_ops =
                                            profile.counters.arithmetic_ops.saturating_add(1);
                                    }
                                    fast_path_taken = true;
                                }
                            }
                        }

                        if !fast_path_taken {
                            if let (Some(Instruction::Const(const_idx)), Some(Instruction::Sub)) =
                                (func.chunk.code.get(ip + 1), func.chunk.code.get(ip + 2))
                            {
                                if let Some(rhs) = const_int(*const_idx) {
                                    self.stack.push(Value::Int(global_int.wrapping_sub(rhs)));
                                    next_ip = ip + 3;
                                    if let Some(profile) = self.bench_profile.as_mut() {
                                        profile.counters.arithmetic_ops =
                                            profile.counters.arithmetic_ops.saturating_add(1);
                                    }
                                    fast_path_taken = true;
                                }
                            }
                        }

                        if !fast_path_taken {
                            if let (Some(Instruction::Const(const_idx)), Some(cmp_instr)) =
                                (func.chunk.code.get(ip + 1), func.chunk.code.get(ip + 2))
                            {
                                if let Some(rhs) = const_int(*const_idx) {
                                    let cmp_result = match cmp_instr {
                                        Instruction::Lt => Some(global_int < rhs),
                                        Instruction::Gt => Some(global_int > rhs),
                                        Instruction::Le => Some(global_int <= rhs),
                                        Instruction::Ge => Some(global_int >= rhs),
                                        _ => None,
                                    };
                                    if let Some(value) = cmp_result {
                                        self.stack.push(Value::Bool(value));
                                        next_ip = ip + 3;
                                        if let Some(profile) = self.bench_profile.as_mut() {
                                            profile.counters.compare_ops =
                                                profile.counters.compare_ops.saturating_add(1);
                                        }
                                        fast_path_taken = true;
                                    }
                                }
                            }
                        }
                    }

                    if !fast_path_taken {
                        if let Value::Float(global_float_ref) = &self.globals[slot] {
                            let global_float = *global_float_ref;
                            let const_float = |const_idx: u16| -> Option<f64> {
                                match func.chunk.constants.get(const_idx as usize) {
                                    Some(Constant::Float(v)) => Some(*v),
                                    _ => None,
                                }
                            };

                            if let (
                                Some(Instruction::Const(const_idx)),
                                Some(Instruction::Add),
                                Some(Instruction::StoreGlobal(target)),
                            ) = (
                                func.chunk.code.get(ip + 1),
                                func.chunk.code.get(ip + 2),
                                func.chunk.code.get(ip + 3),
                            ) {
                                if *target == idx {
                                    if let Some(rhs) = const_float(*const_idx) {
                                        self.globals[slot] = Value::Float(global_float + rhs);
                                        next_ip = ip + 4;
                                        if let Some(profile) = self.bench_profile.as_mut() {
                                            profile.counters.arithmetic_ops =
                                                profile.counters.arithmetic_ops.saturating_add(1);
                                        }
                                        fast_path_taken = true;
                                    }
                                }
                            }

                            if !fast_path_taken {
                                if let (
                                    Some(Instruction::Const(const_idx)),
                                    Some(Instruction::Sub),
                                    Some(Instruction::StoreGlobal(target)),
                                ) = (
                                    func.chunk.code.get(ip + 1),
                                    func.chunk.code.get(ip + 2),
                                    func.chunk.code.get(ip + 3),
                                ) {
                                    if *target == idx {
                                        if let Some(rhs) = const_float(*const_idx) {
                                            self.globals[slot] = Value::Float(global_float - rhs);
                                            next_ip = ip + 4;
                                            if let Some(profile) = self.bench_profile.as_mut() {
                                                profile.counters.arithmetic_ops = profile
                                                    .counters
                                                    .arithmetic_ops
                                                    .saturating_add(1);
                                            }
                                            fast_path_taken = true;
                                        }
                                    }
                                }
                            }

                            if !fast_path_taken {
                                if let (
                                    Some(Instruction::Const(const_idx)),
                                    Some(Instruction::Mul),
                                    Some(Instruction::StoreGlobal(target)),
                                ) = (
                                    func.chunk.code.get(ip + 1),
                                    func.chunk.code.get(ip + 2),
                                    func.chunk.code.get(ip + 3),
                                ) {
                                    if *target == idx {
                                        if let Some(rhs) = const_float(*const_idx) {
                                            self.globals[slot] = Value::Float(global_float * rhs);
                                            next_ip = ip + 4;
                                            if let Some(profile) = self.bench_profile.as_mut() {
                                                profile.counters.arithmetic_ops = profile
                                                    .counters
                                                    .arithmetic_ops
                                                    .saturating_add(1);
                                            }
                                            fast_path_taken = true;
                                        }
                                    }
                                }
                            }

                            if !fast_path_taken {
                                if let (
                                    Some(Instruction::Const(const_idx)),
                                    Some(Instruction::Div),
                                    Some(Instruction::StoreGlobal(target)),
                                ) = (
                                    func.chunk.code.get(ip + 1),
                                    func.chunk.code.get(ip + 2),
                                    func.chunk.code.get(ip + 3),
                                ) {
                                    if *target == idx {
                                        if let Some(rhs) = const_float(*const_idx) {
                                            if rhs == 0.0 {
                                                return TaskRunOutcome::Errored(trace(
                                                    self,
                                                    RuntimeError::new("Division by zero"),
                                                ));
                                            }
                                            self.globals[slot] = Value::Float(global_float / rhs);
                                            next_ip = ip + 4;
                                            if let Some(profile) = self.bench_profile.as_mut() {
                                                profile.counters.arithmetic_ops = profile
                                                    .counters
                                                    .arithmetic_ops
                                                    .saturating_add(1);
                                            }
                                            fast_path_taken = true;
                                        }
                                    }
                                }
                            }

                            if !fast_path_taken {
                                if let (
                                    Some(Instruction::Const(const_idx)),
                                    Some(Instruction::Mul),
                                ) = (func.chunk.code.get(ip + 1), func.chunk.code.get(ip + 2))
                                {
                                    if let Some(rhs) = const_float(*const_idx) {
                                        self.stack.push(Value::Float(global_float * rhs));
                                        next_ip = ip + 3;
                                        if let Some(profile) = self.bench_profile.as_mut() {
                                            profile.counters.arithmetic_ops =
                                                profile.counters.arithmetic_ops.saturating_add(1);
                                        }
                                        fast_path_taken = true;
                                    }
                                }
                            }

                            if !fast_path_taken {
                                if let (
                                    Some(Instruction::Const(const_idx)),
                                    Some(Instruction::Div),
                                ) = (func.chunk.code.get(ip + 1), func.chunk.code.get(ip + 2))
                                {
                                    if let Some(rhs) = const_float(*const_idx) {
                                        if rhs == 0.0 {
                                            return TaskRunOutcome::Errored(trace(
                                                self,
                                                RuntimeError::new("Division by zero"),
                                            ));
                                        }
                                        self.stack.push(Value::Float(global_float / rhs));
                                        next_ip = ip + 3;
                                        if let Some(profile) = self.bench_profile.as_mut() {
                                            profile.counters.arithmetic_ops =
                                                profile.counters.arithmetic_ops.saturating_add(1);
                                        }
                                        fast_path_taken = true;
                                    }
                                }
                            }

                            if !fast_path_taken {
                                if let (
                                    Some(Instruction::Const(const_idx)),
                                    Some(Instruction::Add),
                                ) = (func.chunk.code.get(ip + 1), func.chunk.code.get(ip + 2))
                                {
                                    if let Some(rhs) = const_float(*const_idx) {
                                        self.stack.push(Value::Float(global_float + rhs));
                                        next_ip = ip + 3;
                                        if let Some(profile) = self.bench_profile.as_mut() {
                                            profile.counters.arithmetic_ops =
                                                profile.counters.arithmetic_ops.saturating_add(1);
                                        }
                                        fast_path_taken = true;
                                    }
                                }
                            }

                            if !fast_path_taken {
                                if let (
                                    Some(Instruction::Const(const_idx)),
                                    Some(Instruction::Sub),
                                ) = (func.chunk.code.get(ip + 1), func.chunk.code.get(ip + 2))
                                {
                                    if let Some(rhs) = const_float(*const_idx) {
                                        self.stack.push(Value::Float(global_float - rhs));
                                        next_ip = ip + 3;
                                        if let Some(profile) = self.bench_profile.as_mut() {
                                            profile.counters.arithmetic_ops =
                                                profile.counters.arithmetic_ops.saturating_add(1);
                                        }
                                        fast_path_taken = true;
                                    }
                                }
                            }

                            if !fast_path_taken {
                                if let (Some(Instruction::Const(const_idx)), Some(cmp_instr)) =
                                    (func.chunk.code.get(ip + 1), func.chunk.code.get(ip + 2))
                                {
                                    if let Some(rhs) = const_float(*const_idx) {
                                        let cmp_result = match cmp_instr {
                                            Instruction::Lt => Some(global_float < rhs),
                                            Instruction::Gt => Some(global_float > rhs),
                                            Instruction::Le => Some(global_float <= rhs),
                                            Instruction::Ge => Some(global_float >= rhs),
                                            _ => None,
                                        };
                                        if let Some(value) = cmp_result {
                                            self.stack.push(Value::Bool(value));
                                            next_ip = ip + 3;
                                            if let Some(profile) = self.bench_profile.as_mut() {
                                                profile.counters.compare_ops =
                                                    profile.counters.compare_ops.saturating_add(1);
                                            }
                                            fast_path_taken = true;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if !fast_path_taken {
                        let v = self.globals[slot].clone();
                        self.stack.push(v);
                    }
                }
                Instruction::StoreGlobal(idx) => {
                    let val = match self.stack.pop() {
                        Some(val) => val,
                        None => {
                            return TaskRunOutcome::Errored(trace(
                                self,
                                RuntimeError::new("Stack underflow"),
                            ))
                        }
                    };
                    let slot = idx as usize;
                    if slot >= self.globals.len() {
                        return TaskRunOutcome::Errored(trace(
                            self,
                            RuntimeError::new("Global not found"),
                        ));
                    }
                    self.globals[slot] = val;
                }
                Instruction::Add
                | Instruction::Sub
                | Instruction::Mul
                | Instruction::Div
                | Instruction::Mod
                | Instruction::Eq
                | Instruction::Neq
                | Instruction::Lt
                | Instruction::Gt
                | Instruction::Le
                | Instruction::Ge => {
                    let b = match self.stack.pop() {
                        Some(val) => val,
                        None => {
                            return TaskRunOutcome::Errored(trace(
                                self,
                                RuntimeError::new("Stack underflow"),
                            ))
                        }
                    };
                    let a = match self.stack.pop() {
                        Some(val) => val,
                        None => {
                            return TaskRunOutcome::Errored(trace(
                                self,
                                RuntimeError::new("Stack underflow"),
                            ))
                        }
                    };
                    if let Some(profile) = self.bench_profile.as_mut() {
                        match instr {
                            Instruction::Add
                            | Instruction::Sub
                            | Instruction::Mul
                            | Instruction::Div
                            | Instruction::Mod => {
                                profile.counters.arithmetic_ops =
                                    profile.counters.arithmetic_ops.saturating_add(1);
                            }
                            _ => {
                                profile.counters.compare_ops =
                                    profile.counters.compare_ops.saturating_add(1);
                            }
                        }
                    }
                    let result = match instr {
                        Instruction::Add => match (a, b) {
                            (Value::Int(x), Value::Int(y)) => Value::Int(x.wrapping_add(y)),
                            (Value::Int(x), Value::Float(y)) => Value::Float((x as f64) + y),
                            (Value::Float(x), Value::Int(y)) => Value::Float(x + (y as f64)),
                            (Value::Float(x), Value::Float(y)) => Value::Float(x + y),
                            _ => {
                                return TaskRunOutcome::Errored(trace(
                                    self,
                                    RuntimeError::new("Add expects numbers"),
                                ))
                            }
                        },
                        Instruction::Sub => match (a, b) {
                            (Value::Int(x), Value::Int(y)) => Value::Int(x.wrapping_sub(y)),
                            (Value::Int(x), Value::Float(y)) => Value::Float((x as f64) - y),
                            (Value::Float(x), Value::Int(y)) => Value::Float(x - (y as f64)),
                            (Value::Float(x), Value::Float(y)) => Value::Float(x - y),
                            _ => {
                                return TaskRunOutcome::Errored(trace(
                                    self,
                                    RuntimeError::new("Sub expects numbers"),
                                ))
                            }
                        },
                        Instruction::Mul => match (a, b) {
                            (Value::Int(x), Value::Int(y)) => Value::Int(x.wrapping_mul(y)),
                            (Value::Int(x), Value::Float(y)) => Value::Float((x as f64) * y),
                            (Value::Float(x), Value::Int(y)) => Value::Float(x * (y as f64)),
                            (Value::Float(x), Value::Float(y)) => Value::Float(x * y),
                            _ => {
                                return TaskRunOutcome::Errored(trace(
                                    self,
                                    RuntimeError::new("Mul expects numbers"),
                                ))
                            }
                        },
                        Instruction::Div => match (a, b) {
                            (Value::Int(_), Value::Int(0)) => {
                                return TaskRunOutcome::Errored(trace(
                                    self,
                                    RuntimeError::new("Division by zero"),
                                ))
                            }
                            (Value::Int(x), Value::Int(y)) => {
                                if x == i64::MIN && y == -1 {
                                    Value::Int(i64::MIN)
                                } else {
                                    Value::Int(x / y)
                                }
                            }
                            (Value::Int(_), Value::Float(0.0)) => {
                                return TaskRunOutcome::Errored(trace(
                                    self,
                                    RuntimeError::new("Division by zero"),
                                ))
                            }
                            (Value::Float(_), Value::Int(0)) => {
                                return TaskRunOutcome::Errored(trace(
                                    self,
                                    RuntimeError::new("Division by zero"),
                                ))
                            }
                            (Value::Float(_), Value::Float(0.0)) => {
                                return TaskRunOutcome::Errored(trace(
                                    self,
                                    RuntimeError::new("Division by zero"),
                                ))
                            }
                            (Value::Int(x), Value::Float(y)) => Value::Float((x as f64) / y),
                            (Value::Float(x), Value::Int(y)) => Value::Float(x / (y as f64)),
                            (Value::Float(x), Value::Float(y)) => Value::Float(x / y),
                            _ => {
                                return TaskRunOutcome::Errored(trace(
                                    self,
                                    RuntimeError::new("Div expects numbers"),
                                ))
                            }
                        },
                        Instruction::Mod => match (a, b) {
                            (Value::Int(_), Value::Int(0)) => {
                                return TaskRunOutcome::Errored(trace(
                                    self,
                                    RuntimeError::new("Modulo by zero"),
                                ))
                            }
                            (Value::Int(x), Value::Int(y)) => {
                                if x == i64::MIN && y == -1 {
                                    Value::Int(0)
                                } else {
                                    Value::Int(x % y)
                                }
                            }
                            _ => {
                                return TaskRunOutcome::Errored(trace(
                                    self,
                                    RuntimeError::new("Mod expects Int"),
                                ))
                            }
                        },
                        Instruction::Eq => Value::Bool(a == b),
                        Instruction::Neq => Value::Bool(a != b),
                        Instruction::Lt => match compare_lt(a, b) {
                            Ok(v) => v,
                            Err(err) => return TaskRunOutcome::Errored(trace(self, err)),
                        },
                        Instruction::Gt => match compare_gt(a, b) {
                            Ok(v) => v,
                            Err(err) => return TaskRunOutcome::Errored(trace(self, err)),
                        },
                        Instruction::Le => match compare_le(a, b) {
                            Ok(v) => v,
                            Err(err) => return TaskRunOutcome::Errored(trace(self, err)),
                        },
                        Instruction::Ge => match compare_ge(a, b) {
                            Ok(v) => v,
                            Err(err) => return TaskRunOutcome::Errored(trace(self, err)),
                        },
                        _ => unreachable!(),
                    };
                    self.stack.push(result);
                }
                Instruction::Neg => {
                    let v = match self.stack.pop() {
                        Some(val) => val,
                        None => {
                            return TaskRunOutcome::Errored(trace(
                                self,
                                RuntimeError::new("Stack underflow"),
                            ))
                        }
                    };
                    let result = match v {
                        Value::Int(i) => Value::Int(-i),
                        Value::Float(f) => Value::Float(-f),
                        _ => {
                            return TaskRunOutcome::Errored(trace(
                                self,
                                RuntimeError::new("Neg expects number"),
                            ))
                        }
                    };
                    self.stack.push(result);
                }
                Instruction::Not => {
                    let v = match self.stack.pop() {
                        Some(val) => val,
                        None => {
                            return TaskRunOutcome::Errored(trace(
                                self,
                                RuntimeError::new("Stack underflow"),
                            ))
                        }
                    };
                    let b = v.is_truthy();
                    self.stack.push(Value::Bool(!b));
                }
                Instruction::Jump(target) => {
                    next_ip = target;
                }
                Instruction::JumpIfFalse(target) => {
                    let cond = match self.stack.pop() {
                        Some(val) => val,
                        None => {
                            return TaskRunOutcome::Errored(trace(
                                self,
                                RuntimeError::new("Stack underflow"),
                            ))
                        }
                    };
                    if !cond.is_truthy() {
                        next_ip = target;
                    }
                }
                Instruction::MakeRecord(count) => {
                    let mut map = std::collections::HashMap::new();
                    for _ in 0..count {
                        let value = match self.stack.pop() {
                            Some(val) => val,
                            None => {
                                return TaskRunOutcome::Errored(trace(
                                    self,
                                    RuntimeError::new("Stack underflow"),
                                ))
                            }
                        };
                        let key = match self.stack.pop() {
                            Some(val) => val,
                            None => {
                                return TaskRunOutcome::Errored(trace(
                                    self,
                                    RuntimeError::new("Stack underflow"),
                                ))
                            }
                        };
                        let key = match key {
                            Value::Obj(obj) => match obj.as_obj() {
                                Obj::String(s) => s.clone(),
                                _ => {
                                    return TaskRunOutcome::Errored(trace(
                                        self,
                                        RuntimeError::new("Record key must be string"),
                                    ))
                                }
                            },
                            _ => {
                                return TaskRunOutcome::Errored(trace(
                                    self,
                                    RuntimeError::new("Record key must be string"),
                                ))
                            }
                        };
                        map.insert(key, value);
                    }
                    self.stack.push(record_value(map));
                }
                Instruction::MakeList(count) => {
                    let mut values = Vec::with_capacity(count as usize);
                    for _ in 0..count {
                        let value = match self.stack.pop() {
                            Some(val) => val,
                            None => {
                                return TaskRunOutcome::Errored(trace(
                                    self,
                                    RuntimeError::new("Stack underflow"),
                                ))
                            }
                        };
                        values.push(value);
                    }
                    values.reverse();
                    self.stack
                        .push(Value::Obj(ObjRef::new(Obj::List(RefCell::new(values)))));
                }
                Instruction::GetField(name_idx) => {
                    let target = match self.stack.pop() {
                        Some(val) => val,
                        None => {
                            return TaskRunOutcome::Errored(trace(
                                self,
                                RuntimeError::new("Stack underflow"),
                            ))
                        }
                    };
                    let name = match &func.chunk.constants[name_idx as usize] {
                        Constant::String(s) => s.clone(),
                        _ => {
                            return TaskRunOutcome::Errored(trace(
                                self,
                                RuntimeError::new("Field name must be string constant"),
                            ))
                        }
                    };
                    let value = match target.clone() {
                        Value::Obj(obj) => match obj.as_obj() {
                            Obj::Record(map) => {
                                let (field_value, type_name) = {
                                    let map = map.borrow();
                                    let value = map.get(&name).cloned();
                                    let ty = map.get("__type").cloned();
                                    (value, ty)
                                };
                                if let Some(val) = field_value {
                                    val
                                } else if let Some(Value::Obj(obj)) = type_name {
                                    if let Obj::String(type_name) = obj.as_obj() {
                                        if let Some(val) =
                                            self.lookup_method(type_name, &name, target.clone())
                                        {
                                            val
                                        } else {
                                            return TaskRunOutcome::Errored(trace(
                                                self,
                                                RuntimeError::new("Unknown field"),
                                            ));
                                        }
                                    } else {
                                        return TaskRunOutcome::Errored(trace(
                                            self,
                                            RuntimeError::new("Unknown field"),
                                        ));
                                    }
                                } else {
                                    return TaskRunOutcome::Errored(trace(
                                        self,
                                        RuntimeError::new("Unknown field"),
                                    ));
                                }
                            }
                            Obj::TcpListener(_) => match name.as_str() {
                                "accept" => self.bound_native("net.accept", 0, target),
                                "port" => self.bound_native("net.listener.port", 0, target),
                                "close" => self.bound_native("net.listener.close", 0, target),
                                _ => {
                                    return TaskRunOutcome::Errored(trace(
                                        self,
                                        RuntimeError::new("Unknown field"),
                                    ))
                                }
                            },
                            Obj::TcpConnection(_) => match name.as_str() {
                                "read" => self.bound_native("net.read", 1, target),
                                "read_all" => self.bound_native("net.read_all", 0, target),
                                "write" => self.bound_native("net.write", 1, target),
                                "close" => self.bound_native("net.close", 0, target),
                                _ => {
                                    return TaskRunOutcome::Errored(trace(
                                        self,
                                        RuntimeError::new("Unknown field"),
                                    ))
                                }
                            },
                            Obj::Tokenizer(_) => match name.as_str() {
                                "encode" => self.bound_native("tokenizer.encode", 1, target),
                                "decode" => self.bound_native("tokenizer.decode", 1, target),
                                "save" => self.bound_native("tokenizer.save", 1, target),
                                _ => {
                                    return TaskRunOutcome::Errored(trace(
                                        self,
                                        RuntimeError::new("Unknown field"),
                                    ))
                                }
                            },
                            Obj::DatasetStream(_) => match name.as_str() {
                                "next_batch" => self.bound_native("dataset.next_batch", 0, target),
                                _ => {
                                    return TaskRunOutcome::Errored(trace(
                                        self,
                                        RuntimeError::new("Unknown field"),
                                    ))
                                }
                            },
                            _ => {
                                return TaskRunOutcome::Errored(trace(
                                    self,
                                    RuntimeError::new("Field access expects record"),
                                ))
                            }
                        },
                        _ => {
                            return TaskRunOutcome::Errored(trace(
                                self,
                                RuntimeError::new("Field access expects record"),
                            ))
                        }
                    };
                    self.stack.push(value);
                }
                Instruction::GetIndex => {
                    let index = match self.stack.pop() {
                        Some(val) => val,
                        None => {
                            return TaskRunOutcome::Errored(trace(
                                self,
                                RuntimeError::new("Stack underflow"),
                            ))
                        }
                    };
                    let target = match self.stack.pop() {
                        Some(val) => val,
                        None => {
                            return TaskRunOutcome::Errored(trace(
                                self,
                                RuntimeError::new("Stack underflow"),
                            ))
                        }
                    };
                    let idx = match index {
                        Value::Int(i) => i,
                        _ => {
                            return TaskRunOutcome::Errored(trace(
                                self,
                                RuntimeError::new("Index expects Int"),
                            ))
                        }
                    };
                    let value = match target {
                        Value::Obj(obj) => match obj.as_obj() {
                            Obj::List(items) => {
                                if idx < 0 {
                                    return TaskRunOutcome::Errored(trace(
                                        self,
                                        RuntimeError::new("Index out of range"),
                                    ));
                                }
                                let items = items.borrow();
                                let idx = idx as usize;
                                match items.get(idx).cloned() {
                                    Some(val) => val,
                                    None => {
                                        return TaskRunOutcome::Errored(trace(
                                            self,
                                            RuntimeError::new("Index out of range"),
                                        ))
                                    }
                                }
                            }
                            _ => {
                                return TaskRunOutcome::Errored(trace(
                                    self,
                                    RuntimeError::new("Indexing expects a list"),
                                ))
                            }
                        },
                        _ => {
                            return TaskRunOutcome::Errored(trace(
                                self,
                                RuntimeError::new("Indexing expects a list"),
                            ))
                        }
                    };
                    self.stack.push(value);
                }
                Instruction::SetField(name_idx) => {
                    let value = match self.stack.pop() {
                        Some(val) => val,
                        None => {
                            return TaskRunOutcome::Errored(trace(
                                self,
                                RuntimeError::new("Stack underflow"),
                            ))
                        }
                    };
                    let target = match self.stack.pop() {
                        Some(val) => val,
                        None => {
                            return TaskRunOutcome::Errored(trace(
                                self,
                                RuntimeError::new("Stack underflow"),
                            ))
                        }
                    };
                    let name = match &func.chunk.constants[name_idx as usize] {
                        Constant::String(s) => s.clone(),
                        _ => {
                            return TaskRunOutcome::Errored(trace(
                                self,
                                RuntimeError::new("Field name must be string constant"),
                            ))
                        }
                    };
                    match target {
                        Value::Obj(obj) => match obj.as_obj() {
                            Obj::Record(map) => {
                                map.borrow_mut().insert(name, value);
                            }
                            _ => {
                                return TaskRunOutcome::Errored(trace(
                                    self,
                                    RuntimeError::new("Field assignment expects record"),
                                ))
                            }
                        },
                        _ => {
                            return TaskRunOutcome::Errored(trace(
                                self,
                                RuntimeError::new("Field assignment expects record"),
                            ))
                        }
                    }
                }
                Instruction::SetIndex => {
                    let value = match self.stack.pop() {
                        Some(val) => val,
                        None => {
                            return TaskRunOutcome::Errored(trace(
                                self,
                                RuntimeError::new("Stack underflow"),
                            ))
                        }
                    };
                    let index = match self.stack.pop() {
                        Some(val) => val,
                        None => {
                            return TaskRunOutcome::Errored(trace(
                                self,
                                RuntimeError::new("Stack underflow"),
                            ))
                        }
                    };
                    let target = match self.stack.pop() {
                        Some(val) => val,
                        None => {
                            return TaskRunOutcome::Errored(trace(
                                self,
                                RuntimeError::new("Stack underflow"),
                            ))
                        }
                    };
                    let idx = match index {
                        Value::Int(i) => i,
                        _ => {
                            return TaskRunOutcome::Errored(trace(
                                self,
                                RuntimeError::new("Index expects Int"),
                            ))
                        }
                    };
                    match target {
                        Value::Obj(obj) => match obj.as_obj() {
                            Obj::List(items) => {
                                if idx < 0 {
                                    return TaskRunOutcome::Errored(trace(
                                        self,
                                        RuntimeError::new("Index out of range"),
                                    ));
                                }
                                let mut items = items.borrow_mut();
                                let idx = idx as usize;
                                if idx >= items.len() {
                                    return TaskRunOutcome::Errored(trace(
                                        self,
                                        RuntimeError::new("Index out of range"),
                                    ));
                                }
                                items[idx] = value;
                            }
                            _ => {
                                return TaskRunOutcome::Errored(trace(
                                    self,
                                    RuntimeError::new("Index assignment expects list"),
                                ))
                            }
                        },
                        _ => {
                            return TaskRunOutcome::Errored(trace(
                                self,
                                RuntimeError::new("Index assignment expects list"),
                            ))
                        }
                    }
                }
                Instruction::Call(argc) => {
                    if let Some(caller) = self.frames.last_mut() {
                        caller.ip = next_ip;
                    }
                    if let Err(err) = self.call_value(program, argc as usize) {
                        return TaskRunOutcome::Errored(trace(self, err));
                    }
                    if self.yield_now {
                        self.yield_now = false;
                        return TaskRunOutcome::Yielded;
                    }
                    update_ip = false;
                }
                Instruction::Return => {
                    let ret = self.stack.pop().unwrap_or(Value::Null);
                    let frame = self.frames.pop().unwrap();
                    self.active_policy = frame.prev_policy;
                    self.stack.truncate(caller_sp);
                    self.stack.push(ret);
                    continue;
                }
                Instruction::TryUnwrap => {
                    let value = match self.stack.pop() {
                        Some(val) => val,
                        None => {
                            return TaskRunOutcome::Errored(trace(
                                self,
                                RuntimeError::new("Stack underflow"),
                            ))
                        }
                    };
                    match value {
                        Value::Null => {
                            return TaskRunOutcome::Errored(trace(
                                self,
                                RuntimeError::new("Tried to unwrap none"),
                            ))
                        }
                        _ => self.stack.push(value),
                    }
                }
            }
            budget = budget.saturating_sub(1);
            if update_ip {
                if let Some(frame_mut) = self.frames.last_mut() {
                    frame_mut.ip = next_ip;
                }
            }
            if self.yield_now {
                self.yield_now = false;
                return TaskRunOutcome::Yielded;
            }
        }
        TaskRunOutcome::Completed(self.stack.pop().unwrap_or(Value::Null))
    }

    fn call_value(&mut self, program: &Program, argc: usize) -> Result<(), RuntimeError> {
        let callee_index = self
            .stack
            .len()
            .checked_sub(argc + 1)
            .ok_or_else(|| RuntimeError::new("Call stack underflow"))?;
        let callee = self
            .stack
            .get(callee_index)
            .cloned()
            .ok_or_else(|| RuntimeError::new("Missing callee"))?;
        match callee {
            Value::Obj(obj) => {
                match obj.as_obj() {
                    Obj::Function(f) => {
                        if argc as u16 != f.arity {
                            return Err(RuntimeError::new("Arity mismatch"));
                        }
                        self.frames.push(CallFrame {
                            func_index: f.func_index,
                            ip: 0,
                            base: callee_index + 1, // locals start at first arg
                            caller_sp: callee_index,
                            prev_policy: self.active_policy.clone(),
                        });
                    }
                    Obj::BoundFunction(bf) => {
                        if argc as u16 != bf.arity {
                            return Err(RuntimeError::new("Arity mismatch"));
                        }
                        let func = program
                            .functions
                            .get(bf.func_index as usize)
                            .ok_or_else(|| RuntimeError::new("Function not found"))?;
                        if func.arity != bf.arity + 1 {
                            return Err(RuntimeError::new("Bound function arity mismatch"));
                        }
                        self.stack.insert(callee_index + 1, bf.bound.clone());
                        let prev_policy = self.active_policy.clone();
                        if let Some(policy_name) = bound_policy(&bf.bound) {
                            self.active_policy = Some(policy_name);
                        }
                        self.frames.push(CallFrame {
                            func_index: bf.func_index,
                            ip: 0,
                            base: callee_index + 1,
                            caller_sp: callee_index,
                            prev_policy,
                        });
                    }
                    Obj::NativeFunction(nf) => {
                        if argc as u16 != nf.arity {
                            return Err(RuntimeError::new("Arity mismatch"));
                        }
                        let args_start = self.stack.len() - argc;
                        let mut args: Vec<Value> = self.stack[args_start..].to_vec();
                        if let Some(bound) = nf.bound.clone() {
                            args.insert(0, bound);
                        }
                        if let Some(profile) = self.bench_profile.as_mut() {
                            profile.counters.native_calls =
                                profile.counters.native_calls.saturating_add(1);
                        }
                        if nf.name == "policy.register" {
                            self.stack.truncate(callee_index);
                            self.policy_register(args)?;
                            self.stack.push(Value::Null);
                            return Ok(());
                        }
                        if let Some((capability, context)) =
                            self.native_capability_context(&nf.name, &args)?
                        {
                            self.check_capability(&capability, context.as_ref())?;
                        }
                        if nf.name == "task.join" {
                            self.stack.truncate(callee_index);
                            if let Some(value) =
                                self.task_join(args.first().cloned().unwrap_or(Value::Null))?
                            {
                                self.stack.push(value);
                            }
                            return Ok(());
                        }
                        if nf.name == "task.sleep" {
                            let ms = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("sleep expects Int"))?;
                            self.stack.truncate(callee_index);
                            self.task_sleep(ms)?;
                            self.stack.push(Value::Null);
                            return Ok(());
                        }
                        if nf.name == "task.spawn" {
                            let func = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("spawn expects function"))?;
                            let handle = self.task_spawn(program, func)?;
                            self.stack.truncate(callee_index);
                            self.stack.push(handle);
                            return Ok(());
                        }
                        if nf.name == "chan.make" {
                            if !args.is_empty() {
                                return Err(RuntimeError::new("chan.make expects no args"));
                            }
                            let channel = self.channel_make();
                            self.stack.truncate(callee_index);
                            self.stack.push(channel);
                            return Ok(());
                        }
                        if nf.name == "chan.send" {
                            let channel = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("chan.send expects channel"))?;
                            let value = args
                                .get(1)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("chan.send expects value"))?;
                            self.stack.truncate(callee_index);
                            self.channel_send(channel, value)?;
                            self.stack.push(Value::Null);
                            return Ok(());
                        }
                        if nf.name == "chan.recv" {
                            let channel = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("chan.recv expects channel"))?;
                            self.stack.truncate(callee_index);
                            if let Some(value) = self.channel_recv(channel)? {
                                self.stack.push(value);
                            }
                            return Ok(());
                        }
                        if nf.name == "net.bind" {
                            let host = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("net.bind expects host"))?;
                            let port = args
                                .get(1)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("net.bind expects port"))?;
                            self.stack.truncate(callee_index);
                            let listener = self.net_bind(host, port)?;
                            self.stack.push(listener);
                            return Ok(());
                        }
                        if nf.name == "net.accept" {
                            let listener = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("accept expects listener"))?;
                            self.stack.truncate(callee_index);
                            if let Some(value) = self.net_accept(listener)? {
                                self.stack.push(value);
                            }
                            return Ok(());
                        }
                        if nf.name == "net.read" {
                            let conn = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("read expects connection"))?;
                            let count = args
                                .get(1)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("read expects count"))?;
                            self.stack.truncate(callee_index);
                            if let Some(value) = self.net_read(conn, count)? {
                                self.stack.push(value);
                            }
                            return Ok(());
                        }
                        if nf.name == "net.read_all" {
                            let conn = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("read_all expects connection"))?;
                            self.stack.truncate(callee_index);
                            if let Some(value) = self.net_read_all(conn)? {
                                self.stack.push(value);
                            }
                            return Ok(());
                        }
                        if nf.name == "net.write" {
                            let conn = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("write expects connection"))?;
                            let buf = args
                                .get(1)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("write expects buffer"))?;
                            self.stack.truncate(callee_index);
                            if let Some(value) = self.net_write(conn, buf)? {
                                self.stack.push(value);
                            }
                            return Ok(());
                        }
                        if nf.name == "net.close" {
                            let conn = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("close expects connection"))?;
                            self.stack.truncate(callee_index);
                            self.net_close(conn)?;
                            self.stack.push(Value::Null);
                            return Ok(());
                        }
                        if nf.name == "net.listener.close" {
                            let listener = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("close expects listener"))?;
                            self.stack.truncate(callee_index);
                            self.net_listener_close(listener)?;
                            self.stack.push(Value::Null);
                            return Ok(());
                        }
                        if nf.name == "net.listener.port" {
                            let listener = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("port expects listener"))?;
                            self.stack.truncate(callee_index);
                            let port = self.net_listener_port(listener)?;
                            self.stack.push(Value::Int(port as i64));
                            return Ok(());
                        }
                        if nf.name == "http.serve" {
                            let host = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("serve expects host"))?;
                            let port = args
                                .get(1)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("serve expects port"))?;
                            let handler = args
                                .get(2)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("serve expects handler"))?;
                            self.stack.truncate(callee_index);
                            self.http_serve(host, port, handler)?;
                            self.stack.push(Value::Null);
                            return Ok(());
                        }
                        if nf.name == "http.serve_with" {
                            let host = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("serve_with expects host"))?;
                            let port = args
                                .get(1)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("serve_with expects port"))?;
                            let routes = args
                                .get(2)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("serve_with expects routes"))?;
                            let config = args
                                .get(3)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("serve_with expects config"))?;
                            self.stack.truncate(callee_index);
                            self.http_serve_with(host, port, routes, config)?;
                            self.stack.push(Value::Null);
                            return Ok(());
                        }
                        if nf.name == "http.route" {
                            let method = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("route expects method"))?;
                            let path = args
                                .get(1)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("route expects path"))?;
                            let handler = args
                                .get(2)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("route expects handler"))?;
                            self.stack.truncate(callee_index);
                            let route = self.http_route(method, path, handler)?;
                            self.stack.push(route);
                            return Ok(());
                        }
                        if nf.name == "http.middleware" {
                            let name = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("middleware expects name"))?;
                            let config = args
                                .get(1)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("middleware expects config"))?;
                            self.stack.truncate(callee_index);
                            let middleware = self.http_middleware(name, config)?;
                            self.stack.push(middleware);
                            return Ok(());
                        }
                        if nf.name == "http.get" {
                            let url = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("get expects url"))?;
                            self.stack.truncate(callee_index);
                            if let Some(value) = self.http_get(url)? {
                                self.stack.push(value);
                            }
                            return Ok(());
                        }
                        if nf.name == "http.post" {
                            let url = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("post expects url"))?;
                            let body = args
                                .get(1)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("post expects body"))?;
                            self.stack.truncate(callee_index);
                            if let Some(value) = self.http_post(url, body)? {
                                self.stack.push(value);
                            }
                            return Ok(());
                        }
                        if nf.name == "http.request" {
                            let config = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("request expects config"))?;
                            self.stack.truncate(callee_index);
                            if let Some(value) = self.http_request(config)? {
                                self.stack.push(value);
                            }
                            return Ok(());
                        }
                        if nf.name == "http.header" {
                            let req = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("header expects request"))?;
                            let name = args
                                .get(1)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("header expects name"))?;
                            self.stack.truncate(callee_index);
                            let value = self.http_header(req, name)?;
                            self.stack.push(value);
                            return Ok(());
                        }
                        if nf.name == "http.query" {
                            let req = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("query expects request"))?;
                            let name = args
                                .get(1)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("query expects name"))?;
                            self.stack.truncate(callee_index);
                            let value = self.http_query(req, name)?;
                            self.stack.push(value);
                            return Ok(());
                        }
                        if nf.name == "http.response" {
                            let status = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("response expects status"))?;
                            let body = args
                                .get(1)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("response expects body"))?;
                            self.stack.truncate(callee_index);
                            let resp = self.http_response(status, body)?;
                            self.stack.push(resp);
                            return Ok(());
                        }
                        if nf.name == "http.stream_open" {
                            let status = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("stream_open expects status"))?;
                            let headers = args
                                .get(1)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("stream_open expects headers"))?;
                            self.stack.truncate(callee_index);
                            let stream = self.http_stream_open(status, headers)?;
                            self.stack.push(stream);
                            return Ok(());
                        }
                        if nf.name == "http.stream_send" {
                            let stream = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("stream_send expects stream"))?;
                            let chunk = args
                                .get(1)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("stream_send expects chunk"))?;
                            self.stack.truncate(callee_index);
                            self.http_stream_send(stream, chunk)?;
                            self.stack.push(Value::Null);
                            return Ok(());
                        }
                        if nf.name == "http.stream_close" {
                            let stream = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("stream_close expects stream"))?;
                            self.stack.truncate(callee_index);
                            self.http_stream_close(stream)?;
                            self.stack.push(Value::Null);
                            return Ok(());
                        }
                        if nf.name == "http.ws_open" {
                            let req = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("ws_open expects request"))?;
                            self.stack.truncate(callee_index);
                            let ws = self.http_ws_open(req)?;
                            self.stack.push(ws);
                            return Ok(());
                        }
                        if nf.name == "http.ws_send" {
                            let ws = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("ws_send expects websocket"))?;
                            let message = args
                                .get(1)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("ws_send expects message"))?;
                            self.stack.truncate(callee_index);
                            self.http_ws_send(ws, message)?;
                            self.stack.push(Value::Null);
                            return Ok(());
                        }
                        if nf.name == "http.ws_recv" {
                            let ws = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("ws_recv expects websocket"))?;
                            let timeout_ms = args
                                .get(1)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("ws_recv expects timeout_ms"))?;
                            self.stack.truncate(callee_index);
                            let msg = self.http_ws_recv(ws, timeout_ms)?;
                            self.stack.push(msg);
                            return Ok(());
                        }
                        if nf.name == "http.ws_close" {
                            let ws = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("ws_close expects websocket"))?;
                            self.stack.truncate(callee_index);
                            self.http_ws_close(ws)?;
                            self.stack.push(Value::Null);
                            return Ok(());
                        }
                        if nf.name == "tool.invoke" {
                            let tool_name = args.first().cloned().ok_or_else(|| {
                                RuntimeError::new("tool.invoke expects tool name")
                            })?;
                            let tool_args = args.get(1).cloned().ok_or_else(|| {
                                RuntimeError::new("tool.invoke expects args record")
                            })?;
                            self.stack.truncate(callee_index);
                            let value = self.tool_invoke(tool_name, tool_args)?;
                            self.stack.push(value);
                            return Ok(());
                        }
                        if nf.name == "http.ok" {
                            let body = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("ok expects body"))?;
                            self.stack.truncate(callee_index);
                            let resp = self.http_response(Value::Int(200), body)?;
                            self.stack.push(resp);
                            return Ok(());
                        }
                        if nf.name == "http.bad_request" {
                            let body = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("bad_request expects body"))?;
                            self.stack.truncate(callee_index);
                            let resp = self.http_response(Value::Int(400), body)?;
                            self.stack.push(resp);
                            return Ok(());
                        }
                        if nf.name == "http.not_found" {
                            let body = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("not_found expects body"))?;
                            self.stack.truncate(callee_index);
                            let resp = self.http_response(Value::Int(404), body)?;
                            self.stack.push(resp);
                            return Ok(());
                        }
                        if nf.name == "json.parse" {
                            let text = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("parse expects string"))?;
                            self.stack.truncate(callee_index);
                            let value = self.json_parse(text)?;
                            self.stack.push(value);
                            return Ok(());
                        }
                        if nf.name == "json.stringify" {
                            let value = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("stringify expects value"))?;
                            self.stack.truncate(callee_index);
                            let text = self.json_stringify(value)?;
                            self.stack.push(text);
                            return Ok(());
                        }
                        if nf.name == "json.parse_many" {
                            let value = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("parse_many expects values"))?;
                            self.stack.truncate(callee_index);
                            let parsed = self.json_parse_many(value)?;
                            self.stack.push(parsed);
                            return Ok(());
                        }
                        if nf.name == "json.stringify_many" {
                            let value = args.first().cloned().ok_or_else(|| {
                                RuntimeError::new("stringify_many expects values")
                            })?;
                            self.stack.truncate(callee_index);
                            let text = self.json_stringify_many(value)?;
                            self.stack.push(text);
                            return Ok(());
                        }
                        if nf.name == "bootstrap.format" {
                            let value = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("format expects source"))?;
                            self.stack.truncate(callee_index);
                            let text = self.bootstrap_format(value)?;
                            self.stack.push(text);
                            return Ok(());
                        }
                        if nf.name == "bootstrap.check" {
                            let value = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("check expects source"))?;
                            self.stack.truncate(callee_index);
                            let ok = self.bootstrap_check(value)?;
                            self.stack.push(ok);
                            return Ok(());
                        }
                        if nf.name == "bootstrap.lint" {
                            let value = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("lint expects source"))?;
                            self.stack.truncate(callee_index);
                            let lint = self.bootstrap_lint(value)?;
                            self.stack.push(lint);
                            return Ok(());
                        }
                        if nf.name == "bootstrap.lint_count" {
                            let value = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("lint_count expects source"))?;
                            self.stack.truncate(callee_index);
                            let count = self.bootstrap_lint_count(value)?;
                            self.stack.push(count);
                            return Ok(());
                        }
                        if nf.name == "bootstrap.lint_json" {
                            let file = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("lint_json expects file"))?;
                            let source = args
                                .get(1)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("lint_json expects source"))?;
                            self.stack.truncate(callee_index);
                            let text = self.bootstrap_lint_json(file, source)?;
                            self.stack.push(text);
                            return Ok(());
                        }
                        if nf.name == "compiler.parse_subset" {
                            let source = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("parse_subset expects source"))?;
                            self.stack.truncate(callee_index);
                            let parsed = self.compiler_parse_subset(source)?;
                            self.stack.push(parsed);
                            return Ok(());
                        }
                        if nf.name == "compiler.check_subset" {
                            let source = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("check_subset expects source"))?;
                            self.stack.truncate(callee_index);
                            let ok = self.compiler_check_subset(source)?;
                            self.stack.push(ok);
                            return Ok(());
                        }
                        if nf.name == "compiler.emit_subset" {
                            let source = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("emit_subset expects source"))?;
                            let output = args
                                .get(1)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("emit_subset expects output"))?;
                            self.stack.truncate(callee_index);
                            let ok = self.compiler_emit_subset(source, output)?;
                            self.stack.push(ok);
                            return Ok(());
                        }
                        if nf.name == "tokenizer.train" {
                            let config = args.first().cloned().ok_or_else(|| {
                                RuntimeError::new("tokenizer.train expects config")
                            })?;
                            self.stack.truncate(callee_index);
                            let tokenizer = self.tokenizer_train(config)?;
                            self.stack.push(tokenizer);
                            return Ok(());
                        }
                        if nf.name == "tokenizer.load" {
                            let path = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("tokenizer.load expects path"))?;
                            self.stack.truncate(callee_index);
                            let tokenizer = self.tokenizer_load(path)?;
                            self.stack.push(tokenizer);
                            return Ok(());
                        }
                        if nf.name == "tokenizer.encode" {
                            let tokenizer = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("encode expects tokenizer"))?;
                            let text = args
                                .get(1)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("encode expects text"))?;
                            self.stack.truncate(callee_index);
                            let encoded = self.tokenizer_encode(tokenizer, text)?;
                            self.stack.push(encoded);
                            return Ok(());
                        }
                        if nf.name == "tokenizer.decode" {
                            let tokenizer = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("decode expects tokenizer"))?;
                            let tokens = args
                                .get(1)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("decode expects tokens"))?;
                            self.stack.truncate(callee_index);
                            let decoded = self.tokenizer_decode(tokenizer, tokens)?;
                            self.stack.push(decoded);
                            return Ok(());
                        }
                        if nf.name == "tokenizer.save" {
                            let tokenizer = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("save expects tokenizer"))?;
                            let path = args
                                .get(1)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("save expects path"))?;
                            self.stack.truncate(callee_index);
                            self.tokenizer_save(tokenizer, path)?;
                            self.stack.push(Value::Null);
                            return Ok(());
                        }
                        if nf.name == "dataset.open" {
                            let path = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("dataset.open expects path"))?;
                            let tokenizer = args.get(1).cloned().ok_or_else(|| {
                                RuntimeError::new("dataset.open expects tokenizer")
                            })?;
                            let config = args
                                .get(2)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("dataset.open expects config"))?;
                            self.stack.truncate(callee_index);
                            let stream = self.dataset_open(path, tokenizer, config)?;
                            self.stack.push(stream);
                            return Ok(());
                        }
                        if nf.name == "dataset.next_batch" {
                            let stream = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("next_batch expects stream"))?;
                            self.stack.truncate(callee_index);
                            if let Some(batch) = self.dataset_next_batch(stream)? {
                                self.stack.push(batch);
                            } else {
                                self.stack.push(Value::Null);
                            }
                            return Ok(());
                        }
                        if nf.name == "checkpoint.save" {
                            let dir = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("checkpoint.save expects dir"))?;
                            let state = args.get(1).cloned().ok_or_else(|| {
                                RuntimeError::new("checkpoint.save expects state")
                            })?;
                            self.stack.truncate(callee_index);
                            self.checkpoint_save(dir, state)?;
                            self.stack.push(Value::Null);
                            return Ok(());
                        }
                        if nf.name == "checkpoint.load" {
                            let path = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("checkpoint.load expects path"))?;
                            self.stack.truncate(callee_index);
                            let value = self.checkpoint_load(path)?;
                            self.stack.push(value);
                            return Ok(());
                        }
                        if nf.name == "checkpoint.latest" {
                            let dir = args.first().cloned().ok_or_else(|| {
                                RuntimeError::new("checkpoint.latest expects dir")
                            })?;
                            self.stack.truncate(callee_index);
                            let value = self.checkpoint_latest(dir)?;
                            self.stack.push(value);
                            return Ok(());
                        }
                        if nf.name == "checkpoint.rotate" {
                            let dir = args.first().cloned().ok_or_else(|| {
                                RuntimeError::new("checkpoint.rotate expects dir")
                            })?;
                            let keep = args.get(1).cloned().ok_or_else(|| {
                                RuntimeError::new("checkpoint.rotate expects keep")
                            })?;
                            self.stack.truncate(callee_index);
                            self.checkpoint_rotate(dir, keep)?;
                            self.stack.push(Value::Null);
                            return Ok(());
                        }
                        if nf.name == "sparse.vector" {
                            self.stack.truncate(callee_index);
                            let native = self
                                .sim_accel_bindings()
                                .and_then(|bindings| bindings.sparse_vector_new.call(&[]).ok());
                            self.stack.push(sparse_vector_value_with_native(native));
                            return Ok(());
                        }
                        if nf.name == "sparse.matrix" {
                            self.stack.truncate(callee_index);
                            let native = self
                                .sim_accel_bindings()
                                .and_then(|bindings| bindings.sparse_matrix_new.call(&[]).ok());
                            self.stack.push(sparse_matrix_value_with_native(native));
                            return Ok(());
                        }
                        if nf.name == "sparse.get" {
                            let matrix = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("sparse.get expects matrix"))?;
                            let row = args
                                .get(1)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("sparse.get expects row"))?;
                            let col = args
                                .get(2)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("sparse.get expects col"))?;
                            self.stack.truncate(callee_index);
                            let value = self.sparse_get(matrix, row, col)?;
                            self.stack.push(value);
                            return Ok(());
                        }
                        if nf.name == "sparse.set" {
                            let matrix = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("sparse.set expects matrix"))?;
                            let row = args
                                .get(1)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("sparse.set expects row"))?;
                            let col = args
                                .get(2)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("sparse.set expects col"))?;
                            let value = args
                                .get(3)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("sparse.set expects value"))?;
                            self.stack.truncate(callee_index);
                            self.sparse_set(matrix, row, col, value)?;
                            self.stack.push(Value::Null);
                            return Ok(());
                        }
                        if nf.name == "sparse.get_vector" {
                            let vector = args.first().cloned().ok_or_else(|| {
                                RuntimeError::new("sparse.get_vector expects vector")
                            })?;
                            let index = args.get(1).cloned().ok_or_else(|| {
                                RuntimeError::new("sparse.get_vector expects index")
                            })?;
                            self.stack.truncate(callee_index);
                            let value = self.sparse_vector_get(vector, index)?;
                            self.stack.push(value);
                            return Ok(());
                        }
                        if nf.name == "sparse.set_vector" {
                            let vector = args.first().cloned().ok_or_else(|| {
                                RuntimeError::new("sparse.set_vector expects vector")
                            })?;
                            let index = args.get(1).cloned().ok_or_else(|| {
                                RuntimeError::new("sparse.set_vector expects index")
                            })?;
                            let value = args.get(2).cloned().ok_or_else(|| {
                                RuntimeError::new("sparse.set_vector expects value")
                            })?;
                            self.stack.truncate(callee_index);
                            self.sparse_vector_set(vector, index, value)?;
                            self.stack.push(Value::Null);
                            return Ok(());
                        }
                        if nf.name == "sparse.nonzero" {
                            let matrix = args.first().cloned().ok_or_else(|| {
                                RuntimeError::new("sparse.nonzero expects matrix")
                            })?;
                            self.stack.truncate(callee_index);
                            let value = self.sparse_nonzero(matrix)?;
                            self.stack.push(value);
                            return Ok(());
                        }
                        if nf.name == "sparse.nonzero_vector" {
                            let vector = args.first().cloned().ok_or_else(|| {
                                RuntimeError::new("sparse.nonzero_vector expects vector")
                            })?;
                            self.stack.truncate(callee_index);
                            let value = self.sparse_vector_nonzero(vector)?;
                            self.stack.push(value);
                            return Ok(());
                        }
                        if nf.name == "sparse.dot" {
                            let vector = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("sparse.dot expects vector"))?;
                            let dense = args
                                .get(1)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("sparse.dot expects dense"))?;
                            self.stack.truncate(callee_index);
                            let value = self.sparse_dot(vector, dense)?;
                            self.stack.push(Value::Float(value));
                            return Ok(());
                        }
                        if nf.name == "sparse.matvec" {
                            let matrix = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("sparse.matvec expects matrix"))?;
                            let dense = args
                                .get(1)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("sparse.matvec expects dense"))?;
                            self.stack.truncate(callee_index);
                            let value = self.sparse_matvec(matrix, dense)?;
                            self.stack.push(value);
                            return Ok(());
                        }
                        if nf.name == "sparse.nnz" {
                            let value = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("sparse.nnz expects value"))?;
                            self.stack.truncate(callee_index);
                            let value = self.sparse_nnz(value)?;
                            self.stack.push(Value::Int(value as i64));
                            return Ok(());
                        }
                        if nf.name == "event.make" {
                            self.stack.truncate(callee_index);
                            let native = self
                                .sim_accel_bindings()
                                .and_then(|bindings| bindings.event_queue_new.call(&[]).ok());
                            self.stack.push(event_queue_value_with_native(native));
                            return Ok(());
                        }
                        if nf.name == "event.push" {
                            let queue = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("event.push expects queue"))?;
                            let time = args
                                .get(1)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("event.push expects time"))?;
                            let event = args
                                .get(2)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("event.push expects event"))?;
                            self.stack.truncate(callee_index);
                            self.event_push(queue, time, event)?;
                            self.stack.push(Value::Null);
                            return Ok(());
                        }
                        if nf.name == "event.pop" {
                            let queue = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("event.pop expects queue"))?;
                            self.stack.truncate(callee_index);
                            let value = self.event_pop(queue)?;
                            self.stack.push(value);
                            return Ok(());
                        }
                        if nf.name == "event.peek" {
                            let queue = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("event.peek expects queue"))?;
                            self.stack.truncate(callee_index);
                            let value = self.event_peek(queue)?;
                            self.stack.push(value);
                            return Ok(());
                        }
                        if nf.name == "event.len" {
                            let queue = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("event.len expects queue"))?;
                            self.stack.truncate(callee_index);
                            let value = self.event_len(queue)?;
                            self.stack.push(Value::Int(value as i64));
                            return Ok(());
                        }
                        if nf.name == "event.is_empty" {
                            let queue = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("event.is_empty expects queue"))?;
                            self.stack.truncate(callee_index);
                            let value = self.event_len(queue)?;
                            self.stack.push(Value::Bool(value == 0));
                            return Ok(());
                        }
                        if nf.name == "pool.make" {
                            let capacity = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("pool.make expects capacity"))?;
                            self.stack.truncate(callee_index);
                            let value = self.pool_make(capacity, false)?;
                            self.stack.push(value);
                            return Ok(());
                        }
                        if nf.name == "pool.make_growable" {
                            let capacity = args.first().cloned().ok_or_else(|| {
                                RuntimeError::new("pool.make_growable expects capacity")
                            })?;
                            self.stack.truncate(callee_index);
                            let value = self.pool_make(capacity, true)?;
                            self.stack.push(value);
                            return Ok(());
                        }
                        if nf.name == "pool.acquire" {
                            let pool = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("pool.acquire expects pool"))?;
                            self.stack.truncate(callee_index);
                            let value = self.pool_acquire(pool)?;
                            self.stack.push(value);
                            return Ok(());
                        }
                        if nf.name == "pool.release" {
                            let pool = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("pool.release expects pool"))?;
                            let value = args
                                .get(1)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("pool.release expects value"))?;
                            self.stack.truncate(callee_index);
                            let released = self.pool_release(pool, value)?;
                            self.stack.push(Value::Bool(released));
                            return Ok(());
                        }
                        if nf.name == "pool.reset" {
                            let pool = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("pool.reset expects pool"))?;
                            self.stack.truncate(callee_index);
                            self.pool_reset(pool)?;
                            self.stack.push(Value::Null);
                            return Ok(());
                        }
                        if nf.name == "pool.available" {
                            let pool = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("pool.available expects pool"))?;
                            self.stack.truncate(callee_index);
                            let value = self.pool_available(pool)?;
                            self.stack.push(Value::Int(value as i64));
                            return Ok(());
                        }
                        if nf.name == "pool.capacity" {
                            let pool = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("pool.capacity expects pool"))?;
                            self.stack.truncate(callee_index);
                            let value = self.pool_capacity(pool)?;
                            self.stack.push(Value::Int(value as i64));
                            return Ok(());
                        }
                        if nf.name == "pool.stats" {
                            let pool = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("pool.stats expects pool"))?;
                            self.stack.truncate(callee_index);
                            let value = self.pool_stats(pool)?;
                            self.stack.push(value);
                            return Ok(());
                        }
                        if nf.name == "sim.make" {
                            let max_events = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("sim.make expects max_events"))?;
                            self.stack.truncate(callee_index);
                            self.stack.push(self.sim_make(max_events, Value::Int(0))?);
                            return Ok(());
                        }
                        if nf.name == "sim.make_seeded" {
                            let max_events = args.first().cloned().ok_or_else(|| {
                                RuntimeError::new("sim.make_seeded expects max_events")
                            })?;
                            let seed = args
                                .get(1)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("sim.make_seeded expects seed"))?;
                            self.stack.truncate(callee_index);
                            self.stack.push(self.sim_make(max_events, seed)?);
                            return Ok(());
                        }
                        if nf.name == "sim.time" {
                            let world = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("sim.time expects world"))?;
                            self.stack.truncate(callee_index);
                            self.stack.push(Value::Float(self.sim_time(world)?));
                            return Ok(());
                        }
                        if nf.name == "sim.seed" {
                            let world = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("sim.seed expects world"))?;
                            self.stack.truncate(callee_index);
                            self.stack.push(Value::Int(self.sim_seed(world)?));
                            return Ok(());
                        }
                        if nf.name == "sim.pending" {
                            let world = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("sim.pending expects world"))?;
                            self.stack.truncate(callee_index);
                            self.stack.push(Value::Int(self.sim_pending(world)? as i64));
                            return Ok(());
                        }
                        if nf.name == "sim.schedule" {
                            let world = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("sim.schedule expects world"))?;
                            let time = args
                                .get(1)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("sim.schedule expects time"))?;
                            let event = args
                                .get(2)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("sim.schedule expects event"))?;
                            self.stack.truncate(callee_index);
                            self.sim_schedule(world, time, event)?;
                            self.stack.push(Value::Null);
                            return Ok(());
                        }
                        if nf.name == "sim.step" {
                            let world = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("sim.step expects world"))?;
                            self.stack.truncate(callee_index);
                            self.stack.push(self.sim_step(world)?);
                            return Ok(());
                        }
                        if nf.name == "sim.run" {
                            let world = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("sim.run expects world"))?;
                            let max_steps = args
                                .get(1)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("sim.run expects max_steps"))?;
                            self.stack.truncate(callee_index);
                            self.stack.push(self.sim_run(world, max_steps)?);
                            return Ok(());
                        }
                        if nf.name == "sim.snapshot" {
                            let world = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("sim.snapshot expects world"))?;
                            self.stack.truncate(callee_index);
                            self.stack.push(self.sim_snapshot(world)?);
                            return Ok(());
                        }
                        if nf.name == "sim.restore" {
                            let snapshot = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("sim.restore expects snapshot"))?;
                            self.stack.truncate(callee_index);
                            self.stack.push(self.sim_restore(snapshot)?);
                            return Ok(());
                        }
                        if nf.name == "sim.replay" {
                            let log = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("sim.replay expects log"))?;
                            let max_events = args.get(1).cloned().ok_or_else(|| {
                                RuntimeError::new("sim.replay expects max_events")
                            })?;
                            let seed = args
                                .get(2)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("sim.replay expects seed"))?;
                            self.stack.truncate(callee_index);
                            self.stack.push(self.sim_replay(log, max_events, seed)?);
                            return Ok(());
                        }
                        if nf.name == "sim.log" {
                            let world = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("sim.log expects world"))?;
                            self.stack.truncate(callee_index);
                            self.stack.push(self.sim_log(world)?);
                            return Ok(());
                        }
                        if nf.name == "sim.entity_set" {
                            let world = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("sim.entity_set expects world"))?;
                            let id = args
                                .get(1)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("sim.entity_set expects id"))?;
                            let value = args
                                .get(2)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("sim.entity_set expects value"))?;
                            self.stack.truncate(callee_index);
                            self.sim_entity_set(world, id, value)?;
                            self.stack.push(Value::Null);
                            return Ok(());
                        }
                        if nf.name == "sim.entity_get" {
                            let world = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("sim.entity_get expects world"))?;
                            let id = args
                                .get(1)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("sim.entity_get expects id"))?;
                            self.stack.truncate(callee_index);
                            self.stack.push(self.sim_entity_get(world, id)?);
                            return Ok(());
                        }
                        if nf.name == "sim.entity_remove" {
                            let world = args.first().cloned().ok_or_else(|| {
                                RuntimeError::new("sim.entity_remove expects world")
                            })?;
                            let id = args
                                .get(1)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("sim.entity_remove expects id"))?;
                            self.stack.truncate(callee_index);
                            self.stack
                                .push(Value::Bool(self.sim_entity_remove(world, id)?));
                            return Ok(());
                        }
                        if nf.name == "sim.entity_ids" {
                            let world = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("sim.entity_ids expects world"))?;
                            self.stack.truncate(callee_index);
                            self.stack.push(self.sim_entity_ids(world)?);
                            return Ok(());
                        }
                        if nf.name == "sim.coroutine" {
                            let world = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("sim.coroutine expects world"))?;
                            let func = args.get(1).cloned().ok_or_else(|| {
                                RuntimeError::new("sim.coroutine expects function")
                            })?;
                            self.stack.truncate(callee_index);
                            let coroutine = self.sim_coroutine_spawn(program, world, func, None)?;
                            self.stack.push(coroutine);
                            return Ok(());
                        }
                        if nf.name == "sim.coroutine_with" {
                            let world = args.first().cloned().ok_or_else(|| {
                                RuntimeError::new("sim.coroutine_with expects world")
                            })?;
                            let func = args.get(1).cloned().ok_or_else(|| {
                                RuntimeError::new("sim.coroutine_with expects function")
                            })?;
                            let state = args.get(2).cloned().ok_or_else(|| {
                                RuntimeError::new("sim.coroutine_with expects state")
                            })?;
                            self.stack.truncate(callee_index);
                            let coroutine =
                                self.sim_coroutine_spawn(program, world, func, Some(state))?;
                            self.stack.push(coroutine);
                            return Ok(());
                        }
                        if nf.name == "sim.coroutine_args" {
                            let world = args.first().cloned().ok_or_else(|| {
                                RuntimeError::new("sim.coroutine_args expects world")
                            })?;
                            let func = args.get(1).cloned().ok_or_else(|| {
                                RuntimeError::new("sim.coroutine_args expects function")
                            })?;
                            let arg_list = args.get(2).cloned().ok_or_else(|| {
                                RuntimeError::new("sim.coroutine_args expects args list")
                            })?;
                            self.stack.truncate(callee_index);
                            let coroutine =
                                self.sim_coroutine_spawn_args(program, world, func, arg_list)?;
                            self.stack.push(coroutine);
                            return Ok(());
                        }
                        if nf.name == "sim.world" {
                            let coroutine = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("sim.world expects coroutine"))?;
                            self.stack.truncate(callee_index);
                            self.stack.push(self.sim_coroutine_world(coroutine)?);
                            return Ok(());
                        }
                        if nf.name == "sim.state" {
                            let coroutine = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("sim.state expects coroutine"))?;
                            self.stack.truncate(callee_index);
                            self.stack.push(self.sim_coroutine_state(coroutine)?);
                            return Ok(());
                        }
                        if nf.name == "sim.emit" {
                            let coroutine = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("sim.emit expects coroutine"))?;
                            let value = args
                                .get(1)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("sim.emit expects value"))?;
                            self.stack.truncate(callee_index);
                            self.sim_coroutine_emit(coroutine, value)?;
                            self.stack.push(Value::Null);
                            return Ok(());
                        }
                        if nf.name == "sim.next" {
                            let coroutine = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("sim.next expects coroutine"))?;
                            self.stack.truncate(callee_index);
                            if let Some(value) = self.sim_coroutine_next(coroutine)? {
                                self.stack.push(value);
                            }
                            return Ok(());
                        }
                        if nf.name == "sim.join" {
                            let coroutine = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("sim.join expects coroutine"))?;
                            self.stack.truncate(callee_index);
                            if let Some(value) = self.sim_coroutine_join(coroutine)? {
                                self.stack.push(value);
                            }
                            return Ok(());
                        }
                        if nf.name == "sim.done" {
                            let coroutine = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("sim.done expects coroutine"))?;
                            self.stack.truncate(callee_index);
                            self.stack
                                .push(Value::Bool(self.sim_coroutine_done(coroutine)?));
                            return Ok(());
                        }
                        if nf.name == "spatial.make" {
                            self.stack.truncate(callee_index);
                            let value = self.spatial_make();
                            self.stack.push(value);
                            return Ok(());
                        }
                        if nf.name == "spatial.upsert" {
                            let spatial = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("spatial.upsert expects index"))?;
                            let entity_id = args.get(1).cloned().ok_or_else(|| {
                                RuntimeError::new("spatial.upsert expects entity id")
                            })?;
                            let x = args
                                .get(2)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("spatial.upsert expects x"))?;
                            let y = args
                                .get(3)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("spatial.upsert expects y"))?;
                            self.stack.truncate(callee_index);
                            self.spatial_upsert(spatial, entity_id, x, y)?;
                            self.stack.push(Value::Null);
                            return Ok(());
                        }
                        if nf.name == "spatial.remove" {
                            let spatial = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("spatial.remove expects index"))?;
                            let entity_id = args.get(1).cloned().ok_or_else(|| {
                                RuntimeError::new("spatial.remove expects entity id")
                            })?;
                            self.stack.truncate(callee_index);
                            let removed = self.spatial_remove(spatial, entity_id)?;
                            self.stack.push(Value::Bool(removed));
                            return Ok(());
                        }
                        if nf.name == "spatial.radius" {
                            let spatial = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("spatial.radius expects index"))?;
                            let x = args
                                .get(1)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("spatial.radius expects x"))?;
                            let y = args
                                .get(2)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("spatial.radius expects y"))?;
                            let radius = args.get(3).cloned().ok_or_else(|| {
                                RuntimeError::new("spatial.radius expects radius")
                            })?;
                            self.stack.truncate(callee_index);
                            let value = self.spatial_radius(spatial, x, y, radius)?;
                            self.stack.push(value);
                            return Ok(());
                        }
                        if nf.name == "spatial.nearest" {
                            let spatial = args.first().cloned().ok_or_else(|| {
                                RuntimeError::new("spatial.nearest expects index")
                            })?;
                            let x = args
                                .get(1)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("spatial.nearest expects x"))?;
                            let y = args
                                .get(2)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("spatial.nearest expects y"))?;
                            self.stack.truncate(callee_index);
                            let value = self.spatial_nearest(spatial, x, y)?;
                            self.stack.push(value);
                            return Ok(());
                        }
                        if nf.name == "spatial.occupancy" {
                            let spatial = args.first().cloned().ok_or_else(|| {
                                RuntimeError::new("spatial.occupancy expects index")
                            })?;
                            let min_x = args.get(1).cloned().ok_or_else(|| {
                                RuntimeError::new("spatial.occupancy expects min_x")
                            })?;
                            let min_y = args.get(2).cloned().ok_or_else(|| {
                                RuntimeError::new("spatial.occupancy expects min_y")
                            })?;
                            let max_x = args.get(3).cloned().ok_or_else(|| {
                                RuntimeError::new("spatial.occupancy expects max_x")
                            })?;
                            let max_y = args.get(4).cloned().ok_or_else(|| {
                                RuntimeError::new("spatial.occupancy expects max_y")
                            })?;
                            self.stack.truncate(callee_index);
                            let value =
                                self.spatial_occupancy(spatial, min_x, min_y, max_x, max_y)?;
                            self.stack.push(Value::Int(value));
                            return Ok(());
                        }
                        if nf.name == "snn.make" {
                            let neuron_count = args.first().cloned().ok_or_else(|| {
                                RuntimeError::new("snn.make expects neuron_count")
                            })?;
                            self.stack.truncate(callee_index);
                            let value = self.snn_make(neuron_count)?;
                            self.stack.push(value);
                            return Ok(());
                        }
                        if nf.name == "snn.connect" {
                            let network = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("snn.connect expects network"))?;
                            let from = args
                                .get(1)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("snn.connect expects from"))?;
                            let to = args
                                .get(2)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("snn.connect expects to"))?;
                            let weight = args
                                .get(3)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("snn.connect expects weight"))?;
                            self.stack.truncate(callee_index);
                            self.snn_connect(network, from, to, weight)?;
                            self.stack.push(Value::Null);
                            return Ok(());
                        }
                        if nf.name == "snn.set_potential" {
                            let network = args.first().cloned().ok_or_else(|| {
                                RuntimeError::new("snn.set_potential expects network")
                            })?;
                            let index = args.get(1).cloned().ok_or_else(|| {
                                RuntimeError::new("snn.set_potential expects index")
                            })?;
                            let value = args.get(2).cloned().ok_or_else(|| {
                                RuntimeError::new("snn.set_potential expects value")
                            })?;
                            self.stack.truncate(callee_index);
                            self.snn_set_potential(network, index, value)?;
                            self.stack.push(Value::Null);
                            return Ok(());
                        }
                        if nf.name == "snn.get_potential" {
                            let network = args.first().cloned().ok_or_else(|| {
                                RuntimeError::new("snn.get_potential expects network")
                            })?;
                            let index = args.get(1).cloned().ok_or_else(|| {
                                RuntimeError::new("snn.get_potential expects index")
                            })?;
                            self.stack.truncate(callee_index);
                            let value = self.snn_get_potential(network, index)?;
                            self.stack.push(Value::Float(value));
                            return Ok(());
                        }
                        if nf.name == "snn.set_threshold" {
                            let network = args.first().cloned().ok_or_else(|| {
                                RuntimeError::new("snn.set_threshold expects network")
                            })?;
                            let index = args.get(1).cloned().ok_or_else(|| {
                                RuntimeError::new("snn.set_threshold expects index")
                            })?;
                            let value = args.get(2).cloned().ok_or_else(|| {
                                RuntimeError::new("snn.set_threshold expects value")
                            })?;
                            self.stack.truncate(callee_index);
                            self.snn_set_threshold(network, index, value)?;
                            self.stack.push(Value::Null);
                            return Ok(());
                        }
                        if nf.name == "snn.get_threshold" {
                            let network = args.first().cloned().ok_or_else(|| {
                                RuntimeError::new("snn.get_threshold expects network")
                            })?;
                            let index = args.get(1).cloned().ok_or_else(|| {
                                RuntimeError::new("snn.get_threshold expects index")
                            })?;
                            self.stack.truncate(callee_index);
                            let value = self.snn_get_threshold(network, index)?;
                            self.stack.push(Value::Float(value));
                            return Ok(());
                        }
                        if nf.name == "snn.set_decay" {
                            let network = args.first().cloned().ok_or_else(|| {
                                RuntimeError::new("snn.set_decay expects network")
                            })?;
                            let value = args
                                .get(1)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("snn.set_decay expects value"))?;
                            self.stack.truncate(callee_index);
                            self.snn_set_decay(network, value)?;
                            self.stack.push(Value::Null);
                            return Ok(());
                        }
                        if nf.name == "snn.get_decay" {
                            let network = args.first().cloned().ok_or_else(|| {
                                RuntimeError::new("snn.get_decay expects network")
                            })?;
                            self.stack.truncate(callee_index);
                            let value = self.snn_get_decay(network)?;
                            self.stack.push(Value::Float(value));
                            return Ok(());
                        }
                        if nf.name == "snn.step" {
                            let network = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("snn.step expects network"))?;
                            let input = args
                                .get(1)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("snn.step expects input"))?;
                            self.stack.truncate(callee_index);
                            let value = self.snn_step(network, input)?;
                            self.stack.push(value);
                            return Ok(());
                        }
                        if nf.name == "snn.spikes" {
                            let network = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("snn.spikes expects network"))?;
                            self.stack.truncate(callee_index);
                            let value = self.snn_spikes(network)?;
                            self.stack.push(value);
                            return Ok(());
                        }
                        if nf.name == "snn.potentials" {
                            let network = args.first().cloned().ok_or_else(|| {
                                RuntimeError::new("snn.potentials expects network")
                            })?;
                            self.stack.truncate(callee_index);
                            let value = self.snn_potentials(network)?;
                            self.stack.push(value);
                            return Ok(());
                        }
                        if nf.name == "snn.synapses" {
                            let network = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("snn.synapses expects network"))?;
                            self.stack.truncate(callee_index);
                            let value = self.snn_synapses(network)?;
                            self.stack.push(value);
                            return Ok(());
                        }
                        if nf.name == "agent.make" {
                            let world = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("agent.make expects world"))?;
                            let spatial = args
                                .get(1)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("agent.make expects spatial"))?;
                            self.stack.truncate(callee_index);
                            let value = self.agent_make(world, spatial)?;
                            self.stack.push(value);
                            return Ok(());
                        }
                        if nf.name == "agent.register" {
                            let env = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("agent.register expects env"))?;
                            let agent_id = args.get(1).cloned().ok_or_else(|| {
                                RuntimeError::new("agent.register expects agent id")
                            })?;
                            let body = args
                                .get(2)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("agent.register expects body"))?;
                            let memory = args.get(3).cloned().ok_or_else(|| {
                                RuntimeError::new("agent.register expects memory")
                            })?;
                            let x = args
                                .get(4)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("agent.register expects x"))?;
                            let y = args
                                .get(5)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("agent.register expects y"))?;
                            self.stack.truncate(callee_index);
                            self.agent_register(env, agent_id, body, memory, x, y)?;
                            self.stack.push(Value::Null);
                            return Ok(());
                        }
                        if nf.name == "agent.state" {
                            let env = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("agent.state expects env"))?;
                            let agent_id = args
                                .get(1)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("agent.state expects agent id"))?;
                            self.stack.truncate(callee_index);
                            let value = self.agent_state(env, agent_id)?;
                            self.stack.push(value);
                            return Ok(());
                        }
                        if nf.name == "agent.body" {
                            let env = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("agent.body expects env"))?;
                            let agent_id = args
                                .get(1)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("agent.body expects agent id"))?;
                            self.stack.truncate(callee_index);
                            let value = self.agent_body(env, agent_id)?;
                            self.stack.push(value);
                            return Ok(());
                        }
                        if nf.name == "agent.memory" {
                            let env = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("agent.memory expects env"))?;
                            let agent_id = args.get(1).cloned().ok_or_else(|| {
                                RuntimeError::new("agent.memory expects agent id")
                            })?;
                            self.stack.truncate(callee_index);
                            let value = self.agent_memory(env, agent_id)?;
                            self.stack.push(value);
                            return Ok(());
                        }
                        if nf.name == "agent.set_body" {
                            let env = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("agent.set_body expects env"))?;
                            let agent_id = args.get(1).cloned().ok_or_else(|| {
                                RuntimeError::new("agent.set_body expects agent id")
                            })?;
                            let body = args
                                .get(2)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("agent.set_body expects body"))?;
                            self.stack.truncate(callee_index);
                            self.agent_set_body(env, agent_id, body)?;
                            self.stack.push(Value::Null);
                            return Ok(());
                        }
                        if nf.name == "agent.set_memory" {
                            let env = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("agent.set_memory expects env"))?;
                            let agent_id = args.get(1).cloned().ok_or_else(|| {
                                RuntimeError::new("agent.set_memory expects agent id")
                            })?;
                            let memory = args.get(2).cloned().ok_or_else(|| {
                                RuntimeError::new("agent.set_memory expects memory")
                            })?;
                            self.stack.truncate(callee_index);
                            self.agent_set_memory(env, agent_id, memory)?;
                            self.stack.push(Value::Null);
                            return Ok(());
                        }
                        if nf.name == "agent.position" {
                            let env = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("agent.position expects env"))?;
                            let agent_id = args.get(1).cloned().ok_or_else(|| {
                                RuntimeError::new("agent.position expects agent id")
                            })?;
                            self.stack.truncate(callee_index);
                            let value = self.agent_position(env, agent_id)?;
                            self.stack.push(value);
                            return Ok(());
                        }
                        if nf.name == "agent.set_position" {
                            let env = args.first().cloned().ok_or_else(|| {
                                RuntimeError::new("agent.set_position expects env")
                            })?;
                            let agent_id = args.get(1).cloned().ok_or_else(|| {
                                RuntimeError::new("agent.set_position expects agent id")
                            })?;
                            let x = args
                                .get(2)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("agent.set_position expects x"))?;
                            let y = args
                                .get(3)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("agent.set_position expects y"))?;
                            self.stack.truncate(callee_index);
                            self.agent_set_position(env, agent_id, x, y)?;
                            self.stack.push(Value::Null);
                            return Ok(());
                        }
                        if nf.name == "agent.neighbors" {
                            let env = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("agent.neighbors expects env"))?;
                            let agent_id = args.get(1).cloned().ok_or_else(|| {
                                RuntimeError::new("agent.neighbors expects agent id")
                            })?;
                            let radius = args.get(2).cloned().ok_or_else(|| {
                                RuntimeError::new("agent.neighbors expects radius")
                            })?;
                            self.stack.truncate(callee_index);
                            let value = self.agent_neighbors(env, agent_id, radius)?;
                            self.stack.push(value);
                            return Ok(());
                        }
                        if nf.name == "agent.reward_add" {
                            let env = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("agent.reward_add expects env"))?;
                            let agent_id = args.get(1).cloned().ok_or_else(|| {
                                RuntimeError::new("agent.reward_add expects agent id")
                            })?;
                            let delta = args.get(2).cloned().ok_or_else(|| {
                                RuntimeError::new("agent.reward_add expects delta")
                            })?;
                            self.stack.truncate(callee_index);
                            self.agent_reward_add(env, agent_id, delta)?;
                            self.stack.push(Value::Null);
                            return Ok(());
                        }
                        if nf.name == "agent.reward_get" {
                            let env = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("agent.reward_get expects env"))?;
                            let agent_id = args.get(1).cloned().ok_or_else(|| {
                                RuntimeError::new("agent.reward_get expects agent id")
                            })?;
                            self.stack.truncate(callee_index);
                            let value = self.agent_reward_get(env, agent_id)?;
                            self.stack.push(Value::Float(value));
                            return Ok(());
                        }
                        if nf.name == "agent.reward_take" {
                            let env = args.first().cloned().ok_or_else(|| {
                                RuntimeError::new("agent.reward_take expects env")
                            })?;
                            let agent_id = args.get(1).cloned().ok_or_else(|| {
                                RuntimeError::new("agent.reward_take expects agent id")
                            })?;
                            self.stack.truncate(callee_index);
                            let value = self.agent_reward_take(env, agent_id)?;
                            self.stack.push(Value::Float(value));
                            return Ok(());
                        }
                        if nf.name == "agent.sense_push" {
                            let env = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("agent.sense_push expects env"))?;
                            let agent_id = args.get(1).cloned().ok_or_else(|| {
                                RuntimeError::new("agent.sense_push expects agent id")
                            })?;
                            let value = args.get(2).cloned().ok_or_else(|| {
                                RuntimeError::new("agent.sense_push expects value")
                            })?;
                            self.stack.truncate(callee_index);
                            self.agent_sense_push(env, agent_id, value)?;
                            self.stack.push(Value::Null);
                            return Ok(());
                        }
                        if nf.name == "agent.sense_take" {
                            let env = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("agent.sense_take expects env"))?;
                            let agent_id = args.get(1).cloned().ok_or_else(|| {
                                RuntimeError::new("agent.sense_take expects agent id")
                            })?;
                            self.stack.truncate(callee_index);
                            let value = self.agent_sense_take(env, agent_id)?;
                            self.stack.push(value);
                            return Ok(());
                        }
                        if nf.name == "agent.action_push" {
                            let env = args.first().cloned().ok_or_else(|| {
                                RuntimeError::new("agent.action_push expects env")
                            })?;
                            let agent_id = args.get(1).cloned().ok_or_else(|| {
                                RuntimeError::new("agent.action_push expects agent id")
                            })?;
                            let value = args.get(2).cloned().ok_or_else(|| {
                                RuntimeError::new("agent.action_push expects value")
                            })?;
                            self.stack.truncate(callee_index);
                            self.agent_action_push(env, agent_id, value)?;
                            self.stack.push(Value::Null);
                            return Ok(());
                        }
                        if nf.name == "agent.action_take" {
                            let env = args.first().cloned().ok_or_else(|| {
                                RuntimeError::new("agent.action_take expects env")
                            })?;
                            let agent_id = args.get(1).cloned().ok_or_else(|| {
                                RuntimeError::new("agent.action_take expects agent id")
                            })?;
                            self.stack.truncate(callee_index);
                            let value = self.agent_action_take(env, agent_id)?;
                            self.stack.push(value);
                            return Ok(());
                        }
                        if nf.name == "agent.stream" {
                            let env = args
                                .first()
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("agent.stream expects env"))?;
                            let agent_id = args.get(1).cloned().ok_or_else(|| {
                                RuntimeError::new("agent.stream expects agent id")
                            })?;
                            let domain = args
                                .get(2)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("agent.stream expects domain"))?;
                            self.stack.truncate(callee_index);
                            let value = self.agent_stream(env, agent_id, domain)?;
                            self.stack.push(value);
                            return Ok(());
                        }
                        if nf.name == "agent.next_float" {
                            let stream = args.first().cloned().ok_or_else(|| {
                                RuntimeError::new("agent.next_float expects stream")
                            })?;
                            self.stack.truncate(callee_index);
                            let value = self.agent_next_float(stream)?;
                            self.stack.push(Value::Float(value));
                            return Ok(());
                        }
                        if nf.name == "agent.next_int" {
                            let stream = args.first().cloned().ok_or_else(|| {
                                RuntimeError::new("agent.next_int expects stream")
                            })?;
                            let upper = args
                                .get(1)
                                .cloned()
                                .ok_or_else(|| RuntimeError::new("agent.next_int expects upper"))?;
                            self.stack.truncate(callee_index);
                            let value = self.agent_next_int(stream, upper)?;
                            self.stack.push(Value::Int(value));
                            return Ok(());
                        }
                        let result = match &nf.kind {
                            NativeImpl::Rust(func) => (func)(self, &args)?,
                            NativeImpl::Ffi(func) => func.call(&args)?,
                        };
                        self.stack.truncate(callee_index);
                        self.stack.push(result);
                    }
                    _ => return Err(RuntimeError::new("Callee is not callable")),
                }
            }
            _ => return Err(RuntimeError::new("Callee is not callable")),
        }
        Ok(())
    }

    fn policy_register(&mut self, args: Vec<Value>) -> Result<(), RuntimeError> {
        if args.len() != 3 {
            return Err(RuntimeError::new("policy.register expects 3 arguments"));
        }
        let name = value_as_string(&args[0])?;
        let rules = value_as_list(&args[1])?;
        let is_default = value_as_bool(&args[2])?;
        if self.policies.contains_key(&name) {
            return Err(RuntimeError::new(&format!("Duplicate policy: {}", name)));
        }
        let mut parsed_rules = Vec::with_capacity(rules.len());
        for rule_val in rules {
            let rule_map = value_as_record(&rule_val)?;
            let allow = rule_map
                .get("allow")
                .ok_or_else(|| RuntimeError::new("policy rule missing allow"))?;
            let allow = value_as_bool(allow)?;
            let capability_val = rule_map
                .get("capability")
                .ok_or_else(|| RuntimeError::new("policy rule missing capability"))?;
            let capability_list = value_as_list(capability_val)?;
            let mut capability = Vec::with_capacity(capability_list.len());
            for seg in capability_list {
                capability.push(value_as_string(&seg)?);
            }
            let filters_val = rule_map
                .get("filters")
                .ok_or_else(|| RuntimeError::new("policy rule missing filters"))?;
            let filter_list = value_as_list(filters_val)?;
            let mut filters = Vec::with_capacity(filter_list.len());
            for filter_val in filter_list {
                let filter_map = value_as_record(&filter_val)?;
                let name_val = filter_map
                    .get("name")
                    .ok_or_else(|| RuntimeError::new("policy filter missing name"))?;
                let name = value_as_string(name_val)?;
                let values_val = filter_map
                    .get("values")
                    .ok_or_else(|| RuntimeError::new("policy filter missing values"))?;
                let values_list = value_as_list(values_val)?;
                let mut values = Vec::with_capacity(values_list.len());
                for value in values_list {
                    values.push(value_as_string(&value)?);
                }
                filters.push(PolicyFilterRuntime { name, values });
            }
            parsed_rules.push(PolicyRuleRuntime {
                allow,
                capability,
                filters,
            });
        }
        self.policies.insert(
            name.clone(),
            Policy {
                rules: parsed_rules,
            },
        );
        if is_default {
            if let Some(existing) = self.active_policy.clone() {
                if existing != name {
                    return Err(RuntimeError::new("Multiple default policies defined"));
                }
            } else {
                self.active_policy = Some(name);
            }
        }
        Ok(())
    }

    fn native_capability_context(
        &self,
        name: &str,
        args: &[Value],
    ) -> Result<Option<CapabilityRequirement>, RuntimeError> {
        let mut capability: Option<Vec<String>> = None;
        let mut context: Option<CapabilityContext> = None;
        match name {
            "print" => {
                capability = Some(vec!["io".to_string(), "print".to_string()]);
            }
            "io_read_stdin" => {
                capability = Some(vec!["io".to_string(), "read".to_string()]);
            }
            "io_write_stdout" | "io_write_stderr" => {
                capability = Some(vec!["io".to_string(), "write".to_string()]);
            }
            "log_emit" => {
                capability = Some(vec!["io".to_string(), "log".to_string()]);
            }
            "fsx_read_bytes" => {
                capability = Some(vec!["fs".to_string(), "read".to_string()]);
                if let Some(path) = args.first() {
                    context = Some(CapabilityContext::for_path(&value_as_string(path)?));
                }
            }
            "fsx_write_bytes" => {
                capability = Some(vec!["fs".to_string(), "write".to_string()]);
                if let Some(path) = args.first() {
                    context = Some(CapabilityContext::for_path(&value_as_string(path)?));
                }
            }
            "env_get" | "env_cwd" => {
                capability = Some(vec!["env".to_string(), "read".to_string()]);
            }
            "env_set" | "env_remove" | "env_set_cwd" => {
                capability = Some(vec!["env".to_string(), "write".to_string()]);
                if name == "env_set_cwd" {
                    if let Some(path) = args.first() {
                        context = Some(CapabilityContext::for_path(&value_as_string(path)?));
                    }
                }
            }
            "process_spawn" | "process_run" => {
                capability = Some(vec!["process".to_string(), "spawn".to_string()]);
                if let Some(cmd) = args.first() {
                    context = Some(CapabilityContext::for_path(&value_as_string(cmd)?));
                }
            }
            "process_wait" | "process_kill" => {
                capability = Some(vec!["process".to_string(), "control".to_string()]);
            }
            "process_exit" => {
                capability = Some(vec!["process".to_string(), "exit".to_string()]);
            }
            "db_sqlite_open"
            | "db_sqlite_exec"
            | "db_sqlite_exec_many"
            | "db_sqlite_transaction_begin"
            | "db_sqlite_transaction_commit"
            | "db_sqlite_close"
            | "db_postgres_open"
            | "db_postgres_exec"
            | "db_postgres_close"
            | "db_mysql_open"
            | "db_mysql_exec"
            | "db_mysql_close" => {
                capability = Some(vec!["db".to_string(), "write".to_string()]);
            }
            "db_sqlite_query" | "db_postgres_query" | "db_mysql_query" => {
                capability = Some(vec!["db".to_string(), "read".to_string()]);
            }
            "tls_fetch_server_info" => {
                capability = Some(vec!["net".to_string(), "tls".to_string()]);
                if let Some(host) = args.first() {
                    context = Some(CapabilityContext::for_domain(&value_as_string(host)?));
                }
            }
            "net.bind" => {
                capability = Some(vec!["net".to_string(), "listen".to_string()]);
                if let Some(host) = args.first() {
                    context = Some(CapabilityContext::for_domain(&value_as_string(host)?));
                }
            }
            "net.accept" => {
                capability = Some(vec!["net".to_string(), "accept".to_string()]);
            }
            "net.read" | "net.read_all" => {
                capability = Some(vec!["net".to_string(), "read".to_string()]);
            }
            "net.write" => {
                capability = Some(vec!["net".to_string(), "write".to_string()]);
            }
            "net.close" => {
                capability = Some(vec!["net".to_string(), "close".to_string()]);
            }
            "http.get" | "http.post" => {
                capability = Some(vec!["net".to_string(), "http".to_string()]);
                if let Some(url) = args.first() {
                    if let Some(domain) = domain_from_url(&value_as_string(url)?) {
                        context = Some(CapabilityContext::for_domain(&domain));
                    }
                }
            }
            "http.request" => {
                capability = Some(vec!["net".to_string(), "http".to_string()]);
                if let Some(Value::Obj(obj)) = args.first() {
                    if let Obj::Record(map) = obj.as_obj() {
                        if let Some(value) = map.borrow().get("url") {
                            if let Ok(url) = value_as_string(value) {
                                if let Some(domain) = domain_from_url(&url) {
                                    context = Some(CapabilityContext::for_domain(&domain));
                                }
                            }
                        }
                    }
                }
            }
            "http.serve" => {
                capability = Some(vec!["net".to_string(), "serve".to_string()]);
                if let Some(host) = args.first() {
                    context = Some(CapabilityContext::for_domain(&value_as_string(host)?));
                }
            }
            "http.serve_with" | "http.stream_open" | "http.stream_send" | "http.stream_close"
            | "http.ws_open" | "http.ws_send" | "http.ws_recv" | "http.ws_close" => {
                capability = Some(vec!["net".to_string(), "serve".to_string()]);
            }
            "tool.invoke" => {
                if let Some(path_value) = args.first() {
                    let tool_path = value_as_string(path_value)?;
                    capability = Some(tool_path_to_capability(&tool_path));
                    context = Some(CapabilityContext::for_path(&tool_path));
                } else {
                    capability = Some(vec!["tool".to_string(), "invoke".to_string()]);
                }
            }
            "compiler.emit_subset" => {
                capability = Some(vec!["fs".to_string(), "write".to_string()]);
                if let Some(path) = args.get(1) {
                    context = Some(CapabilityContext::for_path(&value_as_string(path)?));
                }
            }
            "tokenizer.train" => {
                capability = Some(vec!["fs".to_string(), "read".to_string()]);
                if let Some(config) = args.first() {
                    match config {
                        Value::Obj(obj) => match obj.as_obj() {
                            Obj::Record(map) => {
                                if let Some(path) = map.borrow().get("path") {
                                    context =
                                        Some(CapabilityContext::for_path(&value_as_string(path)?));
                                }
                            }
                            _ => {
                                context =
                                    Some(CapabilityContext::for_path(&value_as_string(config)?));
                            }
                        },
                        _ => {
                            context = Some(CapabilityContext::for_path(&value_as_string(config)?));
                        }
                    }
                }
            }
            "tokenizer.load" => {
                capability = Some(vec!["fs".to_string(), "read".to_string()]);
                if let Some(path) = args.first() {
                    context = Some(CapabilityContext::for_path(&value_as_string(path)?));
                }
            }
            "tokenizer.save" => {
                capability = Some(vec!["fs".to_string(), "write".to_string()]);
                if let Some(path) = args.get(1) {
                    context = Some(CapabilityContext::for_path(&value_as_string(path)?));
                }
            }
            "dataset.open" => {
                capability = Some(vec!["fs".to_string(), "read".to_string()]);
                if let Some(path) = args.first() {
                    context = Some(CapabilityContext::for_path(&value_as_string(path)?));
                }
            }
            "checkpoint.save" => {
                capability = Some(vec!["fs".to_string(), "write".to_string()]);
                if let Some(dir) = args.first() {
                    context = Some(CapabilityContext::for_path(&value_as_string(dir)?));
                }
            }
            "checkpoint.load" | "checkpoint.latest" => {
                capability = Some(vec!["fs".to_string(), "read".to_string()]);
                if let Some(dir) = args.first() {
                    context = Some(CapabilityContext::for_path(&value_as_string(dir)?));
                }
            }
            "checkpoint.rotate" => {
                capability = Some(vec!["fs".to_string(), "write".to_string()]);
                if let Some(dir) = args.first() {
                    context = Some(CapabilityContext::for_path(&value_as_string(dir)?));
                }
            }
            _ => {}
        }
        Ok(capability.map(|cap| (cap, context)))
    }

    fn check_capability(
        &self,
        capability: &[String],
        context: Option<&CapabilityContext>,
    ) -> Result<(), RuntimeError> {
        let policy_name = self.active_policy.clone().ok_or_else(|| {
            RuntimeError::with_code(
                "E_POLICY_DENIED",
                &format!("Policy denied: {}", capability.join(".")),
            )
        })?;
        let policy = self.policies.get(&policy_name).ok_or_else(|| {
            RuntimeError::with_code(
                "E_POLICY_UNKNOWN",
                &format!("Unknown policy: {}", policy_name),
            )
        })?;
        if policy.is_allowed(capability, context) {
            Ok(())
        } else {
            Err(RuntimeError::with_code(
                "E_POLICY_DENIED",
                &format!("Policy denied: {}", capability.join(".")),
            ))
        }
    }

    fn task_spawn(&mut self, program: &Program, func: Value) -> Result<Value, RuntimeError> {
        let (func_index, arity) = match func {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::Function(f) => (f.func_index, f.arity),
                _ => return Err(RuntimeError::new("task.spawn expects a function value")),
            },
            _ => return Err(RuntimeError::new("task.spawn expects a function value")),
        };
        if arity != 0 {
            return Err(RuntimeError::new("task.spawn expects arity 0"));
        }
        let func_value = function_value(func_index, program);
        let id = self.spawn_task_internal(program, func_value)?;
        Ok(task_handle_value(id))
    }

    fn task_join(&mut self, handle: Value) -> Result<Option<Value>, RuntimeError> {
        let target_id = match handle {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::TaskHandle(id) => *id,
                _ => return Err(RuntimeError::new("task.join expects TaskHandle")),
            },
            _ => return Err(RuntimeError::new("task.join expects TaskHandle")),
        };
        let result = match self.tasks.get(target_id).and_then(|t| t.as_ref()) {
            Some(task) => task.result.clone(),
            None => return Err(RuntimeError::new("Unknown task handle")),
        };
        if let Some(result) = result {
            return match result {
                Ok(value) => Ok(Some(value)),
                Err(err) => Err(err),
            };
        }
        let current_id = self
            .current_task
            .ok_or_else(|| RuntimeError::new("No current task"))?;
        if let Some(target) = self.tasks.get_mut(target_id).and_then(|t| t.as_mut()) {
            target.join_waiters.push(current_id);
        }
        self.pending_state = Some(TaskState::BlockedJoin);
        self.yield_now = true;
        Ok(None)
    }

    fn task_sleep(&mut self, value: Value) -> Result<(), RuntimeError> {
        let ms = match value {
            Value::Int(i) => i,
            _ => return Err(RuntimeError::new("sleep expects Int")),
        };
        let delay = if ms <= 0 { 0 } else { ms as u64 };
        let wake = Instant::now() + Duration::from_millis(delay);
        self.pending_state = Some(TaskState::Sleeping(wake));
        self.yield_now = true;
        Ok(())
    }

    fn channel_make(&self) -> Value {
        channel_value()
    }

    fn channel_send(&mut self, channel: Value, value: Value) -> Result<(), RuntimeError> {
        let channel_ref = match channel {
            Value::Obj(obj) => obj,
            _ => return Err(RuntimeError::new("chan.send expects Channel")),
        };
        if let Obj::Channel(state) = channel_ref.as_obj() {
            let mut state = state.borrow_mut();
            while let Some(waiter) = state.waiters.pop_front() {
                if let Some(task) = self.tasks.get_mut(waiter).and_then(|t| t.as_mut()) {
                    if matches!(task.state, TaskState::Finished) {
                        continue;
                    }
                    task.stack.push(value.clone());
                    task.state = TaskState::Ready;
                    self.ready.push_back(waiter);
                    return Ok(());
                }
            }
            state.queue.push_back(value);
        } else {
            return Err(RuntimeError::new("chan.send expects Channel"));
        }
        Ok(())
    }

    fn channel_recv(&mut self, channel: Value) -> Result<Option<Value>, RuntimeError> {
        let channel_ref = match channel {
            Value::Obj(obj) => obj,
            _ => return Err(RuntimeError::new("chan.recv expects Channel")),
        };
        if let Obj::Channel(state) = channel_ref.as_obj() {
            let mut state = state.borrow_mut();
            if let Some(value) = state.queue.pop_front() {
                return Ok(Some(value));
            }
            let current_id = self
                .current_task
                .ok_or_else(|| RuntimeError::new("No current task"))?;
            state.waiters.push_back(current_id);
            self.pending_state = Some(TaskState::BlockedChannel);
            self.yield_now = true;
            Ok(None)
        } else {
            Err(RuntimeError::new("chan.recv expects Channel"))
        }
    }

    fn bound_native(&self, name: &str, arity: u16, bound: Value) -> Value {
        Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
            name: name.to_string(),
            arity,
            kind: NativeImpl::Rust(std::rc::Rc::new(|_, _| Ok(Value::Null))),
            bound: Some(bound),
        })))
    }

    fn lookup_method(&self, type_name: &str, method: &str, receiver: Value) -> Option<Value> {
        let table_name = format!("__type_methods::{}", type_name);
        let idx = self.globals_map.get(&table_name).copied()?;
        let table = self.globals.get(idx as usize)?;
        let method_val = match table {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::Record(map) => {
                    let map = map.borrow();
                    map.get(method).cloned()?
                }
                _ => return None,
            },
            _ => return None,
        };
        Some(self.bind_method_value(method_val, receiver))
    }

    fn bind_method_value(&self, value: Value, receiver: Value) -> Value {
        match value {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::Function(f) => {
                    if f.arity == 0 {
                        return Value::Obj(obj);
                    }
                    Value::Obj(ObjRef::new(Obj::BoundFunction(BoundFunctionObj {
                        func_index: f.func_index,
                        arity: f.arity.saturating_sub(1),
                        bound: receiver,
                    })))
                }
                Obj::BoundFunction(_) => Value::Obj(obj),
                Obj::NativeFunction(nf) => {
                    Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                        name: nf.name.clone(),
                        arity: nf.arity.saturating_sub(1),
                        kind: nf.kind.clone(),
                        bound: Some(receiver),
                    })))
                }
                _ => Value::Obj(obj),
            },
            _ => value,
        }
    }

    fn net_bind(&mut self, host: Value, port: Value) -> Result<Value, RuntimeError> {
        let host = match host {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::String(s) => s.clone(),
                _ => return Err(RuntimeError::new("net.bind expects host string")),
            },
            _ => return Err(RuntimeError::new("net.bind expects host string")),
        };
        let port = match port {
            Value::Int(p) => p,
            _ => return Err(RuntimeError::new("net.bind expects port int")),
        };
        let addr = format!("{}:{}", host, port);
        let listener = TcpListener::bind(addr)
            .map_err(|err| RuntimeError::new(&format!("bind failed: {}", err)))?;
        if self.trace_net {
            if let Ok(addr) = listener.local_addr() {
                println!("[net] bind {}", addr);
            }
        }
        Ok(Value::Obj(ObjRef::new(Obj::TcpListener(
            std::cell::RefCell::new(listener),
        ))))
    }

    fn net_accept(&mut self, listener: Value) -> Result<Option<Value>, RuntimeError> {
        let listener = match listener {
            Value::Obj(obj) => obj,
            _ => return Err(RuntimeError::new("accept expects TcpListener")),
        };
        let listener = match listener.as_obj() {
            Obj::TcpListener(inner) => inner
                .borrow()
                .try_clone()
                .map_err(|err| RuntimeError::new(&format!("accept failed: {}", err)))?,
            _ => return Err(RuntimeError::new("accept expects TcpListener")),
        };
        let task_id = self
            .current_task
            .ok_or_else(|| RuntimeError::new("No current task"))?;
        let sender = self.io_sender.clone();
        if self.trace_net {
            println!("[net] accept (task {})", task_id);
        }
        std::thread::spawn(move || {
            let result = listener
                .accept()
                .map(|(stream, _)| stream)
                .map_err(|err| err.to_string());
            let _ = sender.send(IoEvent {
                task_id,
                result: IoResult::Accept(result),
            });
        });
        self.pending_state = Some(TaskState::BlockedIo);
        self.yield_now = true;
        Ok(None)
    }

    fn net_read(&mut self, conn: Value, count: Value) -> Result<Option<Value>, RuntimeError> {
        let stream = match conn {
            Value::Obj(obj) => obj,
            _ => return Err(RuntimeError::new("read expects TcpConnection")),
        };
        let count = match count {
            Value::Int(n) => n.max(0) as usize,
            _ => return Err(RuntimeError::new("read expects count int")),
        };
        let mut stream = match stream.as_obj() {
            Obj::TcpConnection(inner) => inner
                .borrow()
                .try_clone()
                .map_err(|err| RuntimeError::new(&format!("read failed: {}", err)))?,
            _ => return Err(RuntimeError::new("read expects TcpConnection")),
        };
        let task_id = self
            .current_task
            .ok_or_else(|| RuntimeError::new("No current task"))?;
        let sender = self.io_sender.clone();
        if self.trace_net {
            println!("[net] read {} bytes (task {})", count, task_id);
        }
        std::thread::spawn(move || {
            let mut buf = vec![0u8; count];
            let result = stream
                .read(&mut buf)
                .map(|n| {
                    buf.truncate(n);
                    buf
                })
                .map_err(|err| err.to_string());
            let _ = sender.send(IoEvent {
                task_id,
                result: IoResult::Read(result),
            });
        });
        self.pending_state = Some(TaskState::BlockedIo);
        self.yield_now = true;
        Ok(None)
    }

    fn net_read_all(&mut self, conn: Value) -> Result<Option<Value>, RuntimeError> {
        let stream = match conn {
            Value::Obj(obj) => obj,
            _ => return Err(RuntimeError::new("read_all expects TcpConnection")),
        };
        let mut stream = match stream.as_obj() {
            Obj::TcpConnection(inner) => inner
                .borrow()
                .try_clone()
                .map_err(|err| RuntimeError::new(&format!("read_all failed: {}", err)))?,
            _ => return Err(RuntimeError::new("read_all expects TcpConnection")),
        };
        let task_id = self
            .current_task
            .ok_or_else(|| RuntimeError::new("No current task"))?;
        let sender = self.io_sender.clone();
        if self.trace_net {
            println!("[net] read_all (task {})", task_id);
        }
        std::thread::spawn(move || {
            let mut buf = Vec::new();
            let result = stream
                .read_to_end(&mut buf)
                .map(|_| buf)
                .map_err(|err| err.to_string());
            let _ = sender.send(IoEvent {
                task_id,
                result: IoResult::ReadAll(result),
            });
        });
        self.pending_state = Some(TaskState::BlockedIo);
        self.yield_now = true;
        Ok(None)
    }

    fn net_write(&mut self, conn: Value, buf: Value) -> Result<Option<Value>, RuntimeError> {
        let stream = match conn {
            Value::Obj(obj) => obj,
            _ => return Err(RuntimeError::new("write expects TcpConnection")),
        };
        let bytes = match buf {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::Buffer(bytes) => bytes.clone(),
                Obj::String(s) => s.as_bytes().to_vec(),
                _ => return Err(RuntimeError::new("write expects Buffer")),
            },
            _ => return Err(RuntimeError::new("write expects Buffer")),
        };
        let mut stream = match stream.as_obj() {
            Obj::TcpConnection(inner) => inner
                .borrow()
                .try_clone()
                .map_err(|err| RuntimeError::new(&format!("write failed: {}", err)))?,
            _ => return Err(RuntimeError::new("write expects TcpConnection")),
        };
        let task_id = self
            .current_task
            .ok_or_else(|| RuntimeError::new("No current task"))?;
        let sender = self.io_sender.clone();
        if self.trace_net {
            println!("[net] write {} bytes (task {})", bytes.len(), task_id);
        }
        std::thread::spawn(move || {
            let result = stream.write(&bytes).map_err(|err| err.to_string());
            let _ = sender.send(IoEvent {
                task_id,
                result: IoResult::Write(result),
            });
        });
        self.pending_state = Some(TaskState::BlockedIo);
        self.yield_now = true;
        Ok(None)
    }

    fn net_close(&mut self, conn: Value) -> Result<(), RuntimeError> {
        let stream = match conn {
            Value::Obj(obj) => obj,
            _ => return Err(RuntimeError::new("close expects TcpConnection")),
        };
        if let Obj::TcpConnection(inner) = stream.as_obj() {
            if let Ok(s) = inner.borrow().try_clone() {
                let _ = s.shutdown(std::net::Shutdown::Both);
            }
            Ok(())
        } else {
            Err(RuntimeError::new("close expects TcpConnection"))
        }
    }

    fn net_listener_close(&mut self, listener: Value) -> Result<(), RuntimeError> {
        let listener = match listener {
            Value::Obj(obj) => obj,
            _ => return Err(RuntimeError::new("close expects TcpListener")),
        };
        if let Obj::TcpListener(_) = listener.as_obj() {
            Ok(())
        } else {
            Err(RuntimeError::new("close expects TcpListener"))
        }
    }

    fn net_listener_port(&mut self, listener: Value) -> Result<u16, RuntimeError> {
        let listener = match listener {
            Value::Obj(obj) => obj,
            _ => return Err(RuntimeError::new("port expects TcpListener")),
        };
        if let Obj::TcpListener(inner) = listener.as_obj() {
            let addr = inner
                .borrow()
                .local_addr()
                .map_err(|err| RuntimeError::new(&format!("port failed: {}", err)))?;
            Ok(addr.port())
        } else {
            Err(RuntimeError::new("port expects TcpListener"))
        }
    }

    fn http_serve(&mut self, host: Value, port: Value, handler: Value) -> Result<(), RuntimeError> {
        let host = match host {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::String(s) => s.clone(),
                _ => return Err(RuntimeError::new("http.serve expects host string")),
            },
            _ => return Err(RuntimeError::new("http.serve expects host string")),
        };
        let port = match port {
            Value::Int(p) => p,
            _ => return Err(RuntimeError::new("http.serve expects port int")),
        };
        if !(1..=65535).contains(&port) {
            return Err(RuntimeError::new(
                "http.serve port must be in range 1..65535",
            ));
        }
        self.validate_http_handler(&handler)?;
        let model_name = std::env::var("ENKAI_SERVE_MODEL_NAME")
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        let model_version = std::env::var("ENKAI_SERVE_MODEL_VERSION")
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        let model_registry = std::env::var("ENKAI_SERVE_MODEL_REGISTRY")
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        let multi_model_registry = if env_flag("ENKAI_SERVE_MULTI_MODEL") {
            model_registry.clone()
        } else {
            None
        };
        let require_model_version_header = if multi_model_registry.is_some() {
            true
        } else {
            env_flag("ENKAI_REQUIRE_MODEL_VERSION_HEADER")
        };
        let server = HttpServer {
            handler,
            routes: Vec::new(),
            default_handler: None,
            auth: None,
            rate_limit: None,
            rate_state: HashMap::new(),
            logger: None,
            policy: self.active_policy.clone(),
            inflight: Arc::new(AtomicUsize::new(0)),
            max_inflight: parse_env_usize("ENKAI_HTTP_MAX_INFLIGHT").unwrap_or(0),
            require_model_version_header,
            model_name: if multi_model_registry.is_some() {
                None
            } else {
                model_name
            },
            model_version: if multi_model_registry.is_some() {
                None
            } else {
                model_version
            },
            model_registry,
            multi_model_registry,
            stop: mpsc::channel().0,
        };
        self.start_http_server(host, port, server)
    }

    fn http_serve_with(
        &mut self,
        host: Value,
        port: Value,
        routes: Value,
        config: Value,
    ) -> Result<(), RuntimeError> {
        let host = match host {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::String(s) => s.clone(),
                _ => return Err(RuntimeError::new("http.serve_with expects host string")),
            },
            _ => return Err(RuntimeError::new("http.serve_with expects host string")),
        };
        let port = match port {
            Value::Int(p) => p,
            _ => return Err(RuntimeError::new("http.serve_with expects port int")),
        };
        if !(1..=65535).contains(&port) {
            return Err(RuntimeError::new(
                "http.serve_with port must be in range 1..65535",
            ));
        }
        let routes = self.parse_http_routes(routes)?;
        let (default_handler, auth, rate_limit, logger, max_inflight, require_model_version_header) =
            self.parse_http_server_config(config)?;
        let model_name = std::env::var("ENKAI_SERVE_MODEL_NAME")
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        let model_version = std::env::var("ENKAI_SERVE_MODEL_VERSION")
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        let model_registry = std::env::var("ENKAI_SERVE_MODEL_REGISTRY")
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        let multi_model_registry = if env_flag("ENKAI_SERVE_MULTI_MODEL") {
            model_registry.clone()
        } else {
            None
        };
        let server = HttpServer {
            handler: Value::Null,
            routes,
            default_handler,
            auth,
            rate_limit,
            rate_state: HashMap::new(),
            logger,
            policy: self.active_policy.clone(),
            inflight: Arc::new(AtomicUsize::new(0)),
            max_inflight,
            require_model_version_header: if multi_model_registry.is_some() {
                true
            } else {
                require_model_version_header
            },
            model_name: if multi_model_registry.is_some() {
                None
            } else {
                model_name
            },
            model_version: if multi_model_registry.is_some() {
                None
            } else {
                model_version
            },
            model_registry,
            multi_model_registry,
            stop: mpsc::channel().0,
        };
        self.start_http_server(host, port, server)
    }

    fn start_http_server(
        &mut self,
        host: String,
        port: i64,
        mut server: HttpServer,
    ) -> Result<(), RuntimeError> {
        let addr = format!("{}:{}", host, port);
        let listener = TcpListener::bind(addr)
            .map_err(|err| RuntimeError::new(&format!("http.serve bind failed: {}", err)))?;
        listener
            .set_nonblocking(true)
            .map_err(|err| RuntimeError::new(&format!("http.serve failed: {}", err)))?;
        let server_id = self.servers.len();
        let (stop_sender, stop_receiver) = mpsc::channel();
        server.stop = stop_sender;
        self.servers.push(server);
        let sender = self.server_sender.clone();
        std::thread::spawn(move || loop {
            if stop_receiver.try_recv().is_ok() {
                break;
            }
            match listener.accept() {
                Ok((mut stream, _)) => {
                    let remote_addr = stream
                        .peer_addr()
                        .map(|addr| addr.to_string())
                        .unwrap_or_else(|_| String::new());
                    if let Ok(mut req) = read_http_request(&mut stream) {
                        req.remote_addr = remote_addr;
                        let _ = sender.send(ServerEvent {
                            server_id,
                            request: req,
                            stream,
                            accepted_at: Instant::now(),
                        });
                    } else {
                        let _ = write_http_response(
                            stream,
                            error_response(400, "bad_request", "Bad Request"),
                        );
                    }
                }
                Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(_) => {
                    std::thread::sleep(Duration::from_millis(50));
                }
            }
        });
        Ok(())
    }

    fn http_route(
        &self,
        method: Value,
        path: Value,
        handler: Value,
    ) -> Result<Value, RuntimeError> {
        let method = value_as_string(&method)?.to_uppercase();
        let path = value_as_string(&path)?;
        if path.is_empty() || !path.starts_with('/') {
            return Err(RuntimeError::new("http.route path must start with '/'"));
        }
        self.validate_http_handler(&handler)?;
        let mut map = HashMap::new();
        map.insert("method".to_string(), string_value(&method));
        map.insert("path".to_string(), string_value(&path));
        map.insert("handler".to_string(), handler);
        Ok(record_value(map))
    }

    fn http_middleware(&self, name: Value, config: Value) -> Result<Value, RuntimeError> {
        let name = value_as_string(&name)?.to_lowercase();
        match name.as_str() {
            "auth" | "rate_limit" | "jsonl_log" | "backpressure" | "default" => {}
            _ => return Err(RuntimeError::new(
                "unsupported middleware; expected auth/rate_limit/jsonl_log/backpressure/default",
            )),
        }
        let mut map = HashMap::new();
        map.insert("name".to_string(), string_value(&name));
        map.insert("config".to_string(), config);
        Ok(record_value(map))
    }

    fn parse_http_routes(&self, routes: Value) -> Result<Vec<HttpRoute>, RuntimeError> {
        let list = match routes {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::List(items) => items.borrow().clone(),
                _ => return Err(RuntimeError::new("routes must be List")),
            },
            Value::Null => Vec::new(),
            _ => return Err(RuntimeError::new("routes must be List")),
        };
        let mut out = Vec::with_capacity(list.len());
        for route_val in list {
            let route_obj = match route_val {
                Value::Obj(obj) => obj,
                _ => return Err(RuntimeError::new("route must be record")),
            };
            let route_map = match route_obj.as_obj() {
                Obj::Record(map) => map.borrow().clone(),
                _ => return Err(RuntimeError::new("route must be record")),
            };
            let method = match route_map.get("method") {
                Some(value) => value_as_string(value)?.to_uppercase(),
                None => return Err(RuntimeError::new("route missing method")),
            };
            let path = match route_map.get("path") {
                Some(value) => value_as_string(value)?,
                None => return Err(RuntimeError::new("route missing path")),
            };
            if path.is_empty() || !path.starts_with('/') {
                return Err(RuntimeError::new("route path must start with '/'"));
            }
            let handler = match route_map.get("handler") {
                Some(value) => value.clone(),
                None => return Err(RuntimeError::new("route missing handler")),
            };
            self.validate_http_handler(&handler)?;
            out.push(HttpRoute {
                method,
                path,
                handler,
            });
        }
        Ok(out)
    }

    fn parse_http_server_config(
        &self,
        config: Value,
    ) -> Result<HttpServerConfigParsed, RuntimeError> {
        let default_max_inflight = parse_env_usize("ENKAI_HTTP_MAX_INFLIGHT").unwrap_or(0);
        let default_require_model_version_header = env_flag("ENKAI_REQUIRE_MODEL_VERSION_HEADER");
        if matches!(config, Value::Null) {
            return Ok((
                None,
                None,
                None,
                None,
                default_max_inflight,
                default_require_model_version_header,
            ));
        }
        let mut default_handler = None;
        let mut auth = None;
        let mut rate_limit = None;
        let mut logger = None;
        let mut max_inflight = default_max_inflight;
        let mut require_model_version_header = default_require_model_version_header;
        let obj = match config {
            Value::Obj(obj) => obj,
            _ => {
                return Err(RuntimeError::new(
                    "http.serve_with config must be record or list",
                ))
            }
        };
        match obj.as_obj() {
            Obj::Record(map) => {
                let map = map.borrow();
                default_handler = map.get("default").cloned();
                if let Some(handler) = &default_handler {
                    self.validate_http_handler(handler)?;
                }
                if let Some(value) = map.get("auth") {
                    auth = self.parse_auth_config(value)?;
                }
                if let Some(value) = map.get("rate_limit") {
                    rate_limit = self.parse_rate_limit_config(value)?;
                }
                if let Some(value) = map.get("log_path") {
                    logger = self.parse_logger_config(value)?;
                } else if let Some(value) = map.get("logger") {
                    logger = self.parse_logger_config(value)?;
                }
                if let Some(value) = map.get("max_inflight_requests") {
                    max_inflight =
                        parse_non_negative_usize(value, "max_inflight_requests must be Int >= 0")?;
                }
                if let Some(value) = map.get("require_model_version_header") {
                    require_model_version_header =
                        parse_bool_value(value, "require_model_version_header must be Bool")?;
                }
                if let Some(value) = map.get("middlewares") {
                    self.apply_http_middlewares(
                        value,
                        &mut default_handler,
                        &mut auth,
                        &mut rate_limit,
                        &mut logger,
                        &mut max_inflight,
                    )?;
                }
            }
            Obj::List(_) => {
                self.apply_http_middlewares(
                    &Value::Obj(obj.clone()),
                    &mut default_handler,
                    &mut auth,
                    &mut rate_limit,
                    &mut logger,
                    &mut max_inflight,
                )?;
            }
            _ => {
                return Err(RuntimeError::new(
                    "http.serve_with config must be record or list",
                ))
            }
        }
        Ok((
            default_handler,
            auth,
            rate_limit,
            logger,
            max_inflight,
            require_model_version_header,
        ))
    }

    fn parse_logger_config(&self, value: &Value) -> Result<Option<ServerLogger>, RuntimeError> {
        if matches!(value, Value::Null) {
            return Ok(None);
        }
        let path = match value {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::String(text) => text.clone(),
                Obj::Record(map) => {
                    let map = map.borrow();
                    let path_val = map
                        .get("path")
                        .ok_or_else(|| RuntimeError::new("logger.path missing"))?;
                    value_as_string(path_val)?
                }
                _ => return Err(RuntimeError::new("logger must be path string or record")),
            },
            _ => return Err(RuntimeError::new("logger must be path string or record")),
        };
        if path.trim().is_empty() {
            return Err(RuntimeError::new("logger path must not be empty"));
        }
        Ok(Some(ServerLogger {
            path: std::path::PathBuf::from(path),
        }))
    }

    fn apply_http_middlewares(
        &self,
        value: &Value,
        default_handler: &mut Option<Value>,
        auth: &mut Option<ServerAuthConfig>,
        rate_limit: &mut Option<RateLimitConfig>,
        logger: &mut Option<ServerLogger>,
        max_inflight: &mut usize,
    ) -> Result<(), RuntimeError> {
        let middlewares = value_as_list(value)?;
        for middleware in middlewares {
            let entry = value_as_record(&middleware)?;
            let name_value = entry
                .get("name")
                .or_else(|| entry.get("kind"))
                .ok_or_else(|| RuntimeError::new("middleware entry missing name"))?;
            let name = value_as_string(name_value)?.to_lowercase();
            let enabled = match entry.get("enabled") {
                Some(Value::Bool(value)) => *value,
                Some(_) => return Err(RuntimeError::new("middleware enabled must be Bool")),
                None => true,
            };
            if !enabled {
                continue;
            }
            let cfg = entry.get("config").cloned().unwrap_or(Value::Null);
            match name.as_str() {
                "auth" => *auth = self.parse_auth_config(&cfg)?,
                "rate_limit" => *rate_limit = self.parse_rate_limit_config(&cfg)?,
                "jsonl_log" => *logger = self.parse_logger_config(&cfg)?,
                "backpressure" => {
                    *max_inflight = self.parse_backpressure_config(&cfg)?;
                }
                "default" => {
                    if matches!(cfg, Value::Null) {
                        *default_handler = None;
                    } else {
                        self.validate_http_handler(&cfg)?;
                        *default_handler = Some(cfg);
                    }
                }
                _ => {
                    return Err(RuntimeError::new(
                        "unsupported middleware; expected auth/rate_limit/jsonl_log/backpressure/default",
                    ))
                }
            }
        }
        Ok(())
    }

    fn parse_backpressure_config(&self, value: &Value) -> Result<usize, RuntimeError> {
        match value {
            Value::Null => Ok(0),
            Value::Int(i) if *i >= 0 => Ok(*i as usize),
            Value::Obj(obj) => match obj.as_obj() {
                Obj::Record(map) => {
                    let map = map.borrow();
                    let Some(max) = map.get("max_inflight") else {
                        return Err(RuntimeError::new("backpressure.max_inflight missing"));
                    };
                    parse_non_negative_usize(max, "backpressure.max_inflight must be Int >= 0")
                }
                _ => Err(RuntimeError::new(
                    "backpressure config must be Int >= 0 or record",
                )),
            },
            _ => Err(RuntimeError::new(
                "backpressure config must be Int >= 0 or record",
            )),
        }
    }

    fn parse_auth_config(&self, value: &Value) -> Result<Option<ServerAuthConfig>, RuntimeError> {
        match value {
            Value::Null => Ok(None),
            Value::Obj(obj) => {
                let map = match obj.as_obj() {
                    Obj::Record(map) => map.borrow(),
                    _ => return Err(RuntimeError::new("auth must be record")),
                };
                let header = match map.get("header") {
                    Some(v) => value_as_string(v)?.to_lowercase(),
                    None => "authorization".to_string(),
                };
                let allow_anonymous = match map.get("allow_anonymous") {
                    Some(Value::Bool(b)) => *b,
                    Some(_) => return Err(RuntimeError::new("auth.allow_anonymous must be Bool")),
                    None => false,
                };
                let tokens_val = map
                    .get("tokens")
                    .ok_or_else(|| RuntimeError::new("auth.tokens missing"))?;
                let tokens_list = value_as_list(tokens_val)?;
                let mut tokens = HashMap::new();
                for entry in tokens_list {
                    let entry_obj = match entry {
                        Value::Obj(obj) => obj,
                        _ => return Err(RuntimeError::new("auth token entry must be record")),
                    };
                    let entry_map = match entry_obj.as_obj() {
                        Obj::Record(map) => map.borrow(),
                        _ => return Err(RuntimeError::new("auth token entry must be record")),
                    };
                    let token = entry_map
                        .get("token")
                        .ok_or_else(|| RuntimeError::new("auth token entry missing token"))
                        .and_then(value_as_string)?;
                    let tenant = match entry_map.get("tenant") {
                        Some(v) => value_as_string(v)?,
                        None => String::new(),
                    };
                    tokens.insert(token, tenant);
                }
                Ok(Some(ServerAuthConfig {
                    header,
                    tokens,
                    allow_anonymous,
                }))
            }
            _ => Err(RuntimeError::new("auth must be record")),
        }
    }

    fn parse_rate_limit_config(
        &self,
        value: &Value,
    ) -> Result<Option<RateLimitConfig>, RuntimeError> {
        match value {
            Value::Null => Ok(None),
            Value::Obj(obj) => {
                let map = match obj.as_obj() {
                    Obj::Record(map) => map.borrow(),
                    _ => return Err(RuntimeError::new("rate_limit must be record")),
                };
                let capacity = match map.get("capacity") {
                    Some(Value::Int(i)) if *i > 0 => *i as f64,
                    Some(Value::Float(f)) if *f > 0.0 => *f,
                    Some(_) => return Err(RuntimeError::new("rate_limit.capacity must be > 0")),
                    None => return Err(RuntimeError::new("rate_limit.capacity missing")),
                };
                let refill_per_sec = match map.get("refill_per_sec") {
                    Some(Value::Float(f)) if *f > 0.0 => *f,
                    Some(Value::Int(i)) if *i > 0 => *i as f64,
                    Some(_) => {
                        return Err(RuntimeError::new("rate_limit.refill_per_sec must be > 0"))
                    }
                    None => capacity, // default 1s burst
                };
                let key = match map.get("key") {
                    Some(v) => match value_as_string(v)?.to_lowercase().as_str() {
                        "token" => RateLimitKey::Token,
                        "tenant" => RateLimitKey::Tenant,
                        "model" => RateLimitKey::Model,
                        "tenant_model" => RateLimitKey::TenantModel,
                        _ => RateLimitKey::Ip,
                    },
                    None => RateLimitKey::Ip,
                };
                Ok(Some(RateLimitConfig {
                    capacity,
                    refill_per_sec,
                    key,
                }))
            }
            _ => Err(RuntimeError::new("rate_limit must be record")),
        }
    }

    fn http_get(&mut self, url: Value) -> Result<Option<Value>, RuntimeError> {
        self.http_request_simple("GET", url, None)
    }

    fn http_post(&mut self, url: Value, body: Value) -> Result<Option<Value>, RuntimeError> {
        self.http_request_simple("POST", url, Some(body))
    }

    fn http_request_simple(
        &mut self,
        method: &str,
        url: Value,
        body: Option<Value>,
    ) -> Result<Option<Value>, RuntimeError> {
        let url = match url {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::String(s) => s.clone(),
                _ => return Err(RuntimeError::new("http request expects url string")),
            },
            _ => return Err(RuntimeError::new("http request expects url string")),
        };
        let body_bytes = if let Some(body) = body {
            match body {
                Value::Obj(obj) => match obj.as_obj() {
                    Obj::Buffer(bytes) => bytes.clone(),
                    Obj::String(s) => s.as_bytes().to_vec(),
                    _ => return Err(RuntimeError::new("http body expects Buffer or String")),
                },
                Value::Null => Vec::new(),
                _ => return Err(RuntimeError::new("http body expects Buffer or String")),
            }
        } else {
            Vec::new()
        };
        let opts = HttpRequestOptions::default();
        self.http_request_threaded(method, url, body_bytes, opts)
    }

    fn http_request(&mut self, config: Value) -> Result<Option<Value>, RuntimeError> {
        let config_obj = match config {
            Value::Obj(obj) => obj,
            _ => return Err(RuntimeError::new("http.request expects config record")),
        };
        let map = match config_obj.as_obj() {
            Obj::Record(map) => map.borrow(),
            _ => return Err(RuntimeError::new("http.request expects config record")),
        };
        let method = match map.get("method") {
            Some(v) => value_as_string(v)?.to_uppercase(),
            None => "GET".to_string(),
        };
        let url = match map.get("url") {
            Some(v) => value_as_string(v)?,
            None => return Err(RuntimeError::new("http.request config missing url")),
        };
        let body_bytes = match map.get("body") {
            Some(Value::Obj(obj)) => match obj.as_obj() {
                Obj::Buffer(bytes) => bytes.clone(),
                Obj::String(s) => s.as_bytes().to_vec(),
                _ => return Err(RuntimeError::new("http.request body expects Buffer/String")),
            },
            Some(Value::Null) | None => Vec::new(),
            Some(_) => return Err(RuntimeError::new("http.request body expects Buffer/String")),
        };
        let headers = if let Some(Value::Obj(obj)) = map.get("headers") {
            if let Obj::Record(hmap) = obj.as_obj() {
                let mut out = HashMap::new();
                for (k, v) in hmap.borrow().iter() {
                    if let Value::Obj(vobj) = v {
                        if let Obj::String(s) = vobj.as_obj() {
                            out.insert(k.to_string(), s.clone());
                        }
                    }
                }
                out
            } else {
                HashMap::new()
            }
        } else {
            HashMap::new()
        };
        let timeout_ms = match map.get("timeout_ms") {
            Some(Value::Int(i)) if *i > 0 => Some(*i as u64),
            Some(Value::Float(f)) if *f > 0.0 => Some(*f as u64),
            _ => None,
        };
        let retries = match map.get("retries") {
            Some(Value::Int(i)) if *i >= 0 => *i as usize,
            Some(_) => return Err(RuntimeError::new("http.request retries must be >= 0")),
            None => 0,
        };
        let backoff_ms = match map.get("retry_backoff_ms") {
            Some(Value::Int(i)) if *i >= 0 => *i as u64,
            Some(Value::Float(f)) if *f >= 0.0 => *f as u64,
            Some(_) => {
                return Err(RuntimeError::new(
                    "http.request retry_backoff_ms must be >= 0",
                ))
            }
            None => 100,
        };
        let opts = HttpRequestOptions {
            headers,
            timeout_ms,
            retries,
            retry_backoff_ms: backoff_ms,
        };
        self.http_request_threaded(&method, url, body_bytes, opts)
    }

    fn http_request_threaded(
        &mut self,
        method: &str,
        url: String,
        body: Vec<u8>,
        opts: HttpRequestOptions,
    ) -> Result<Option<Value>, RuntimeError> {
        let task_id = self
            .current_task
            .ok_or_else(|| RuntimeError::new("No current task"))?;
        let sender = self.io_sender.clone();
        let method = method.to_string();
        std::thread::spawn(move || {
            let result = http_request_thread(&method, &url, &body, &opts);
            let _ = sender.send(IoEvent {
                task_id,
                result: IoResult::HttpResponse(result),
            });
        });
        self.pending_state = Some(TaskState::BlockedIo);
        self.yield_now = true;
        Ok(None)
    }

    fn http_header(&self, req: Value, name: Value) -> Result<Value, RuntimeError> {
        let name = value_as_string(&name)?.to_lowercase();
        let map = self.request_headers_map(req)?;
        Ok(match map.get(&name) {
            Some(value) => string_value(value),
            None => Value::Null,
        })
    }

    fn http_query(&self, req: Value, name: Value) -> Result<Value, RuntimeError> {
        let name = value_as_string(&name)?;
        let query = self.request_query(req)?;
        Ok(match query_param(&query, &name) {
            Some(v) => string_value(&v),
            None => Value::Null,
        })
    }

    fn http_stream_open(&mut self, status: Value, headers: Value) -> Result<Value, RuntimeError> {
        let status = match status {
            Value::Int(i) => i.clamp(100, 599) as u16,
            _ => return Err(RuntimeError::new("stream_open expects status int")),
        };
        let mut headers_map = HashMap::new();
        if let Value::Obj(obj) = headers {
            if let Obj::Record(hmap) = obj.as_obj() {
                for (k, v) in hmap.borrow().iter() {
                    if let Value::Obj(vobj) = v {
                        if let Obj::String(s) = vobj.as_obj() {
                            headers_map.insert(k.to_lowercase(), s.clone());
                        }
                    }
                }
            }
        }
        headers_map
            .entry("transfer-encoding".to_string())
            .or_insert_with(|| "chunked".to_string());
        headers_map
            .entry("connection".to_string())
            .or_insert_with(|| "close".to_string());
        let stream = self.active_http_conn.take();
        let stream =
            stream.ok_or_else(|| RuntimeError::new("stream_open requires http handler"))?;
        let (tx, rx) = mpsc::channel::<StreamCommand>();
        std::thread::spawn(move || {
            let _ = write_http_stream(stream, status, headers_map, rx);
        });
        Ok(Value::Obj(ObjRef::new(Obj::HttpStream(HttpStream {
            sender: tx,
        }))))
    }

    fn http_stream_send(&self, stream: Value, chunk: Value) -> Result<(), RuntimeError> {
        let sender = match stream {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::HttpStream(handle) => handle.sender.clone(),
                _ => return Err(RuntimeError::new("stream_send expects HttpStream")),
            },
            _ => return Err(RuntimeError::new("stream_send expects HttpStream")),
        };
        let bytes = match chunk {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::Buffer(bytes) => bytes.clone(),
                Obj::String(s) => s.as_bytes().to_vec(),
                _ => return Err(RuntimeError::new("stream_send expects Buffer/String")),
            },
            Value::Null => Vec::new(),
            _ => return Err(RuntimeError::new("stream_send expects Buffer/String")),
        };
        let _ = sender.send(StreamCommand::Data(bytes));
        Ok(())
    }

    fn http_stream_close(&self, stream: Value) -> Result<(), RuntimeError> {
        let sender = match stream {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::HttpStream(handle) => handle.sender.clone(),
                _ => return Err(RuntimeError::new("stream_close expects HttpStream")),
            },
            _ => return Err(RuntimeError::new("stream_close expects HttpStream")),
        };
        let _ = sender.send(StreamCommand::Close);
        Ok(())
    }

    fn http_ws_open(&mut self, req: Value) -> Result<Value, RuntimeError> {
        let headers = self.request_headers_map(req)?;
        let upgrade = headers
            .get("upgrade")
            .map(|v| v.to_ascii_lowercase())
            .unwrap_or_default();
        if upgrade != "websocket" {
            return Err(RuntimeError::new(
                "ws_open requires Upgrade: websocket request header",
            ));
        }
        let connection = headers
            .get("connection")
            .map(|v| v.to_ascii_lowercase())
            .unwrap_or_default();
        if !connection
            .split(',')
            .any(|part| part.trim().eq_ignore_ascii_case("upgrade"))
        {
            return Err(RuntimeError::new(
                "ws_open requires Connection: Upgrade request header",
            ));
        }
        let ws_key = headers
            .get("sec-websocket-key")
            .cloned()
            .ok_or_else(|| RuntimeError::new("ws_open missing Sec-WebSocket-Key header"))?;
        let accept = websocket_accept_key(&ws_key);
        let stream = self
            .active_http_conn
            .take()
            .ok_or_else(|| RuntimeError::new("ws_open requires http handler"))?;
        let (tx, rx) = mpsc::channel::<WsCommand>();
        let (incoming_tx, incoming_rx) = mpsc::channel::<WsIncoming>();
        let incoming = Arc::new(Mutex::new(incoming_rx));
        let incoming_for_thread = Arc::clone(&incoming);
        std::thread::spawn(move || {
            let _ = write_websocket_session(stream, &accept, rx, incoming_tx);
        });
        Ok(Value::Obj(ObjRef::new(Obj::WebSocket(WebSocketHandle {
            sender: tx,
            incoming: incoming_for_thread,
        }))))
    }

    fn http_ws_send(&self, ws: Value, message: Value) -> Result<(), RuntimeError> {
        let sender = match ws {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::WebSocket(handle) => handle.sender.clone(),
                _ => return Err(RuntimeError::new("ws_send expects WebSocket")),
            },
            _ => return Err(RuntimeError::new("ws_send expects WebSocket")),
        };
        let command = match message {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::String(text) => WsCommand::Text(text.clone()),
                Obj::Buffer(bytes) => WsCommand::Binary(bytes.clone()),
                _ => return Err(RuntimeError::new("ws_send expects String or Buffer")),
            },
            Value::Null => WsCommand::Text(String::new()),
            _ => return Err(RuntimeError::new("ws_send expects String or Buffer")),
        };
        sender
            .send(command)
            .map_err(|_| RuntimeError::new("websocket stream is closed"))?;
        Ok(())
    }

    fn http_ws_close(&self, ws: Value) -> Result<(), RuntimeError> {
        let sender = match ws {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::WebSocket(handle) => handle.sender.clone(),
                _ => return Err(RuntimeError::new("ws_close expects WebSocket")),
            },
            _ => return Err(RuntimeError::new("ws_close expects WebSocket")),
        };
        sender
            .send(WsCommand::Close)
            .map_err(|_| RuntimeError::new("websocket stream is closed"))?;
        Ok(())
    }

    fn http_ws_recv(&self, ws: Value, timeout_ms: Value) -> Result<Value, RuntimeError> {
        let incoming = match ws {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::WebSocket(handle) => Arc::clone(&handle.incoming),
                _ => return Err(RuntimeError::new("ws_recv expects WebSocket")),
            },
            _ => return Err(RuntimeError::new("ws_recv expects WebSocket")),
        };
        let timeout = match timeout_ms {
            Value::Int(ms) if ms >= 0 => ms as u64,
            _ => return Err(RuntimeError::new("ws_recv expects timeout_ms >= 0")),
        };
        let receiver = incoming
            .lock()
            .map_err(|_| RuntimeError::new("websocket receive lock poisoned"))?;
        let message = if timeout == 0 {
            match receiver.try_recv() {
                Ok(value) => Some(value),
                Err(mpsc::TryRecvError::Empty) => None,
                Err(mpsc::TryRecvError::Disconnected) => Some(WsIncoming::Closed),
            }
        } else {
            match receiver.recv_timeout(Duration::from_millis(timeout)) {
                Ok(value) => Some(value),
                Err(mpsc::RecvTimeoutError::Timeout) => None,
                Err(mpsc::RecvTimeoutError::Disconnected) => Some(WsIncoming::Closed),
            }
        };
        Ok(match message {
            Some(WsIncoming::Text(text)) => string_value(&text),
            Some(WsIncoming::Binary(bytes)) => Value::Obj(ObjRef::new(Obj::Buffer(bytes))),
            Some(WsIncoming::Closed) | None => Value::Null,
        })
    }

    fn prepare_http_request(
        &mut self,
        server_id: usize,
        req: &HttpRequestData,
        request_id: u64,
    ) -> Result<PreparedHttpRequest, RuntimeError> {
        let server = match self.servers.get_mut(server_id) {
            Some(server) => server,
            None => {
                return Ok((
                    None,
                    HashMap::new(),
                    None,
                    format!("req-{}", request_id),
                    None,
                    None,
                    Some(error_response(500, "server_missing", "Unknown server")),
                    Some("server_missing".to_string()),
                ))
            }
        };
        let correlation_id = resolve_correlation_id(req, request_id);
        let mut params = HashMap::new();
        let handler = if !server.routes.is_empty() {
            let mut selected = None;
            for route in &server.routes {
                if !route_method_match(&route.method, &req.method) {
                    continue;
                }
                if let Some(p) = match_route(&route.path, &req.path) {
                    params = p;
                    selected = Some(route.handler.clone());
                    break;
                }
            }
            selected.or_else(|| server.default_handler.clone())
        } else {
            Some(server.handler.clone())
        };
        let mut tenant: Option<String> = None;
        let mut token_value: Option<String> = None;
        let mut effective_model_name = server.model_name.clone();
        let mut effective_model_version = server.model_version.clone();
        if let Some(auth) = &server.auth {
            let token = extract_auth_token(req, auth);
            match token {
                Some(tok) => {
                    token_value = Some(tok.clone());
                    if let Some(t) = auth.tokens.get(&tok) {
                        if !t.is_empty() {
                            tenant = Some(t.clone());
                        }
                    } else {
                        return Ok((
                            None,
                            params,
                            None,
                            correlation_id,
                            effective_model_name,
                            effective_model_version,
                            Some(error_response(401, "unauthorized", "Invalid API token")),
                            Some("unauthorized".to_string()),
                        ));
                    }
                }
                None => {
                    if !auth.allow_anonymous {
                        return Ok((
                            None,
                            params,
                            None,
                            correlation_id,
                            effective_model_name,
                            effective_model_version,
                            Some(error_response(401, "unauthorized", "Missing API token")),
                            Some("unauthorized".to_string()),
                        ));
                    }
                }
            }
        }
        if let Some(registry_raw) = server.multi_model_registry.as_deref() {
            let model_name = match model_selector_header(req, "x-enkai-model-name") {
                Some(value) => value,
                None => {
                    return Ok((
                        None,
                        params,
                        tenant.clone(),
                        correlation_id,
                        effective_model_name,
                        effective_model_version,
                        Some(error_response(
                            400,
                            "missing_model_selector",
                            "Missing x-enkai-model-name or x-enkai-model-version header",
                        )),
                        Some("missing_model_selector".to_string()),
                    ))
                }
            };
            let model_version = match model_selector_header(req, "x-enkai-model-version") {
                Some(value) => value,
                None => {
                    return Ok((
                        None,
                        params,
                        tenant.clone(),
                        correlation_id,
                        effective_model_name,
                        effective_model_version,
                        Some(error_response(
                            400,
                            "missing_model_selector",
                            "Missing x-enkai-model-name or x-enkai-model-version header",
                        )),
                        Some("missing_model_selector".to_string()),
                    ))
                }
            };
            if !env_flag("ENKAI_SERVE_ALLOW_UNLOADED") {
                match is_model_loaded_in_registry(
                    Path::new(registry_raw),
                    &model_name,
                    &model_version,
                ) {
                    Ok(true) => {}
                    Ok(false) => {
                        return Ok((
                            None,
                            params,
                            tenant.clone(),
                            correlation_id,
                            Some(model_name),
                            Some(model_version),
                            Some(error_response(
                                409,
                                "model_not_loaded",
                                "Requested model version is not loaded",
                            )),
                            Some("model_not_loaded".to_string()),
                        ))
                    }
                    Err(err) => {
                        return Ok((
                            None,
                            params,
                            tenant.clone(),
                            correlation_id,
                            None,
                            None,
                            Some(error_response(500, "model_registry_error", &err)),
                            Some("model_registry_error".to_string()),
                        ))
                    }
                }
            }
            effective_model_name = Some(model_name);
            effective_model_version = Some(model_version);
        }
        if server.require_model_version_header && !req.headers.contains_key("x-enkai-model-version")
        {
            return Ok((
                None,
                params,
                tenant.clone(),
                correlation_id,
                effective_model_name,
                effective_model_version,
                Some(error_response(
                    400,
                    "missing_model_version",
                    "Missing x-enkai-model-version header",
                )),
                Some("missing_model_version".to_string()),
            ));
        }
        if let Some(expected) = effective_model_version.as_deref() {
            if let Some(actual) = req.headers.get("x-enkai-model-version") {
                if actual.trim() != expected {
                    return Ok((
                        None,
                        params,
                        tenant.clone(),
                        correlation_id,
                        effective_model_name,
                        effective_model_version,
                        Some(error_response(
                            409,
                            "model_version_mismatch",
                            "x-enkai-model-version mismatch",
                        )),
                        Some("model_version_mismatch".to_string()),
                    ));
                }
            }
        }
        if let Some(expected) = effective_model_name.as_deref() {
            if let Some(actual) = req.headers.get("x-enkai-model-name") {
                let trimmed = actual.trim();
                if !trimmed.is_empty() && trimmed != expected {
                    return Ok((
                        None,
                        params,
                        tenant.clone(),
                        correlation_id,
                        effective_model_name,
                        effective_model_version,
                        Some(error_response(
                            409,
                            "model_name_mismatch",
                            "x-enkai-model-name mismatch",
                        )),
                        Some("model_name_mismatch".to_string()),
                    ));
                }
            }
        }
        if let Some(rate) = &server.rate_limit {
            let key = rate_limit_key(
                rate.key,
                req,
                token_value.as_deref(),
                tenant.as_deref(),
                effective_model_name.as_deref(),
                effective_model_version.as_deref(),
            );
            if !rate_limit_allow(&mut server.rate_state, &key, rate) {
                return Ok((
                    None,
                    params,
                    tenant.clone(),
                    correlation_id,
                    effective_model_name,
                    effective_model_version,
                    Some(error_response(429, "rate_limited", "Rate limit exceeded")),
                    Some("rate_limited".to_string()),
                ));
            }
        }
        Ok((
            handler,
            params,
            tenant,
            correlation_id,
            effective_model_name,
            effective_model_version,
            None,
            None,
        ))
    }

    fn http_request_value_with_params(
        &self,
        req: HttpRequestData,
        params: HashMap<String, String>,
    ) -> Value {
        let value = self.http_request_value(req);
        if params.is_empty() {
            return value;
        }
        if let Value::Obj(obj) = &value {
            if let Obj::Record(map) = obj.as_obj() {
                let mut param_map = HashMap::new();
                for (k, v) in params {
                    param_map.insert(k, string_value(&v));
                }
                map.borrow_mut()
                    .insert("params".to_string(), record_value(param_map));
            }
        }
        value
    }

    fn request_headers_map(&self, req: Value) -> Result<HashMap<String, String>, RuntimeError> {
        let obj = match req {
            Value::Obj(obj) => obj,
            _ => return Err(RuntimeError::new("request expected record")),
        };
        let map = match obj.as_obj() {
            Obj::Record(map) => map.borrow(),
            _ => return Err(RuntimeError::new("request expected record")),
        };
        let headers = match map.get("headers") {
            Some(Value::Obj(obj)) => match obj.as_obj() {
                Obj::Record(hmap) => hmap.borrow(),
                _ => return Err(RuntimeError::new("request headers expected record")),
            },
            _ => return Err(RuntimeError::new("request headers missing")),
        };
        let mut out = HashMap::new();
        for (k, v) in headers.iter() {
            if let Value::Obj(vobj) = v {
                if let Obj::String(s) = vobj.as_obj() {
                    out.insert(k.to_lowercase(), s.clone());
                }
            }
        }
        Ok(out)
    }

    fn request_query(&self, req: Value) -> Result<String, RuntimeError> {
        let obj = match req {
            Value::Obj(obj) => obj,
            _ => return Err(RuntimeError::new("request expected record")),
        };
        let map = match obj.as_obj() {
            Obj::Record(map) => map.borrow(),
            _ => return Err(RuntimeError::new("request expected record")),
        };
        match map.get("query") {
            Some(v) => value_as_string(v),
            None => Ok(String::new()),
        }
    }

    fn http_response(&self, status: Value, body: Value) -> Result<Value, RuntimeError> {
        let status = match status {
            Value::Int(i) => {
                if !(100..=599).contains(&i) {
                    return Err(RuntimeError::new("http.response invalid status"));
                }
                i as u16
            }
            _ => return Err(RuntimeError::new("http.response expects status int")),
        };
        let body_bytes = match body {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::Buffer(bytes) => bytes.clone(),
                Obj::String(s) => s.as_bytes().to_vec(),
                _ => {
                    return Err(RuntimeError::new(
                        "http.response body expects Buffer/String",
                    ))
                }
            },
            Value::Null => Vec::new(),
            _ => {
                return Err(RuntimeError::new(
                    "http.response body expects Buffer/String",
                ))
            }
        };
        let mut map = HashMap::new();
        map.insert("status".to_string(), Value::Int(status as i64));
        map.insert("headers".to_string(), record_value(HashMap::new()));
        map.insert(
            "body".to_string(),
            Value::Obj(ObjRef::new(Obj::Buffer(body_bytes))),
        );
        Ok(record_value(map))
    }

    fn validate_http_handler(&self, handler: &Value) -> Result<(), RuntimeError> {
        match handler {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::Function(f) => {
                    if f.arity != 1 {
                        return Err(RuntimeError::new("http.serve expects handler arity 1"));
                    }
                }
                Obj::BoundFunction(bf) => {
                    if bf.arity != 1 {
                        return Err(RuntimeError::new("http.serve expects handler arity 1"));
                    }
                }
                Obj::NativeFunction(nf) => {
                    if nf.arity != 1 {
                        return Err(RuntimeError::new("http.serve expects handler arity 1"));
                    }
                }
                _ => return Err(RuntimeError::new("http.serve expects function handler")),
            },
            _ => return Err(RuntimeError::new("http.serve expects function handler")),
        }
        Ok(())
    }

    fn http_request_value(&self, req: HttpRequestData) -> Value {
        let mut headers = HashMap::new();
        for (k, v) in req.headers {
            headers.insert(k, string_value(&v));
        }
        let mut map = HashMap::new();
        map.insert("method".to_string(), string_value(&req.method));
        map.insert("path".to_string(), string_value(&req.path));
        map.insert("query".to_string(), string_value(&req.query));
        map.insert("headers".to_string(), record_value(headers));
        map.insert(
            "body".to_string(),
            Value::Obj(ObjRef::new(Obj::Buffer(req.body))),
        );
        map.insert("params".to_string(), record_value(HashMap::new()));
        map.insert("remote_addr".to_string(), string_value(&req.remote_addr));
        record_value(map)
    }

    fn http_response_value(&self, resp: HttpResponseData) -> Value {
        let mut headers = HashMap::new();
        for (k, v) in resp.headers {
            headers.insert(k, string_value(&v));
        }
        let mut map = HashMap::new();
        map.insert("status".to_string(), Value::Int(resp.status as i64));
        map.insert("headers".to_string(), record_value(headers));
        map.insert(
            "body".to_string(),
            Value::Obj(ObjRef::new(Obj::Buffer(resp.body))),
        );
        record_value(map)
    }

    fn response_from_result(&self, result: &Result<Value, RuntimeError>) -> HttpResponseData {
        match result {
            Ok(value) => match self.response_from_value(value.clone()) {
                Ok(resp) => resp,
                Err(err) => error_response(500, "invalid_response", &err.message),
            },
            Err(err) => {
                let code = err.code().unwrap_or("internal_error");
                error_response(500, code, &err.message)
            }
        }
    }

    fn log_http_meta(&self, meta: &HttpRequestMeta, status: u16, stream: bool) {
        let logger = match self
            .servers
            .get(meta.server_id)
            .and_then(|s| s.logger.as_ref())
        {
            Some(logger) => logger,
            None => return,
        };
        let latency_ms = meta.start.elapsed().as_millis() as u64;
        let entry = serde_json::json!({
            "ts_ms": unix_ms(),
            "request_id": meta.id,
            "correlation_id": meta.correlation_id.as_str(),
            "method": meta.method.as_str(),
            "path": meta.path.as_str(),
            "status": status,
            "queue_ms": meta.queue_ms,
            "latency_ms": latency_ms,
            "inflight": meta.inflight_at_start,
            "remote_addr": meta.remote_addr.as_str(),
            "tenant": meta.tenant.as_ref(),
            "model_name": meta.model_name.as_ref(),
            "model_version": meta.model_version.as_ref(),
            "model_registry": meta.model_registry.as_ref(),
            "error_code": meta.error_code.as_ref(),
            "stream": stream,
        });
        let _ = logger.append(&entry);
    }

    fn attach_response_meta_headers(&self, resp: &mut HttpResponseData, meta: &HttpRequestMeta) {
        resp.headers
            .entry("x-enkai-request-id".to_string())
            .or_insert_with(|| meta.id.to_string());
        resp.headers
            .entry("x-enkai-correlation-id".to_string())
            .or_insert_with(|| meta.correlation_id.clone());
        resp.headers
            .entry("x-enkai-queue-ms".to_string())
            .or_insert_with(|| meta.queue_ms.to_string());
        resp.headers
            .entry("x-enkai-latency-ms".to_string())
            .or_insert_with(|| meta.start.elapsed().as_millis().to_string());
        resp.headers
            .entry("x-enkai-inflight".to_string())
            .or_insert_with(|| meta.inflight_at_start.to_string());
        if let Some(tenant) = &meta.tenant {
            resp.headers
                .entry("x-enkai-tenant".to_string())
                .or_insert_with(|| tenant.clone());
        }
        if let Some(model_name) = &meta.model_name {
            resp.headers
                .entry("x-enkai-model-name".to_string())
                .or_insert_with(|| model_name.clone());
        }
        if let Some(model_version) = &meta.model_version {
            resp.headers
                .entry("x-enkai-model-version".to_string())
                .or_insert_with(|| model_version.clone());
        }
        if let Some(model_registry) = &meta.model_registry {
            resp.headers
                .entry("x-enkai-model-registry".to_string())
                .or_insert_with(|| model_registry.clone());
        }
        if let Some(code) = &meta.error_code {
            resp.headers
                .entry("x-enkai-error-code".to_string())
                .or_insert_with(|| code.clone());
        }
    }

    fn write_http_error(&self, stream: TcpStream, message: &str) {
        let response = error_response(500, "server_error", message);
        std::thread::spawn(move || {
            let _ = write_http_response(stream, response);
        });
    }

    fn response_from_value(&self, value: Value) -> Result<HttpResponseData, RuntimeError> {
        match value {
            Value::Null => Ok(HttpResponseData {
                status: 200,
                headers: HashMap::new(),
                body: Vec::new(),
            }),
            Value::Obj(obj) => match obj.as_obj() {
                Obj::Record(map) => {
                    let map = map.borrow();
                    let status = match map.get("status") {
                        Some(Value::Int(i)) => (*i).clamp(100, 599) as u16,
                        _ => 200,
                    };
                    let body = match map.get("body") {
                        Some(Value::Obj(obj)) => match obj.as_obj() {
                            Obj::Buffer(bytes) => bytes.clone(),
                            Obj::String(s) => s.as_bytes().to_vec(),
                            _ => Vec::new(),
                        },
                        Some(Value::Null) | None => Vec::new(),
                        _ => Vec::new(),
                    };
                    let mut headers = HashMap::new();
                    if let Some(Value::Obj(obj)) = map.get("headers") {
                        if let Obj::Record(hmap) = obj.as_obj() {
                            let hmap = hmap.borrow();
                            for (k, v) in hmap.iter() {
                                if let Value::Obj(vobj) = v {
                                    if let Obj::String(s) = vobj.as_obj() {
                                        headers.insert(k.to_lowercase(), s.clone());
                                    }
                                }
                            }
                        }
                    }
                    Ok(HttpResponseData {
                        status,
                        headers,
                        body,
                    })
                }
                Obj::String(s) => Ok(HttpResponseData {
                    status: 200,
                    headers: HashMap::new(),
                    body: s.as_bytes().to_vec(),
                }),
                Obj::Buffer(bytes) => Ok(HttpResponseData {
                    status: 200,
                    headers: HashMap::new(),
                    body: bytes.clone(),
                }),
                _ => Err(RuntimeError::new("Invalid http response value")),
            },
            _ => Err(RuntimeError::new("Invalid http response value")),
        }
    }

    fn json_parse(&self, text: Value) -> Result<Value, RuntimeError> {
        let text = match text {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::String(s) => s.clone(),
                _ => return Err(RuntimeError::new("json.parse expects String")),
            },
            _ => return Err(RuntimeError::new("json.parse expects String")),
        };
        let json: serde_json::Value =
            serde_json::from_str(&text).map_err(|err| RuntimeError::new(&err.to_string()))?;
        Ok(json_to_value(json))
    }

    fn json_stringify(&self, value: Value) -> Result<Value, RuntimeError> {
        let json = self.value_to_json(value)?;
        let text =
            serde_json::to_string(&json).map_err(|err| RuntimeError::new(&err.to_string()))?;
        Ok(string_value(&text))
    }

    fn json_parse_many(&self, values: Value) -> Result<Value, RuntimeError> {
        match values {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::List(list) => {
                    let items = list.borrow().clone();
                    let mut out = Vec::with_capacity(items.len());
                    for value in items {
                        out.push(self.json_parse(value)?);
                    }
                    Ok(Value::Obj(ObjRef::new(Obj::List(RefCell::new(out)))))
                }
                Obj::Record(map) => {
                    let map = map.borrow();
                    let value = map
                        .get("value")
                        .cloned()
                        .ok_or_else(|| RuntimeError::new("json.parse_many record missing value"))?;
                    let count = map
                        .get("count")
                        .ok_or_else(|| RuntimeError::new("json.parse_many record missing count"))?;
                    let count = match count {
                        Value::Int(i) if *i >= 0 => *i as usize,
                        _ => {
                            return Err(RuntimeError::new(
                                "json.parse_many record count must be Int >= 0",
                            ));
                        }
                    };
                    let mut out = Vec::with_capacity(count);
                    for _ in 0..count {
                        out.push(self.json_parse(value.clone())?);
                    }
                    Ok(Value::Obj(ObjRef::new(Obj::List(RefCell::new(out)))))
                }
                _ => Err(RuntimeError::new(
                    "json.parse_many expects List[String] or { value, count } record",
                )),
            },
            _ => Err(RuntimeError::new(
                "json.parse_many expects List[String] or { value, count } record",
            )),
        }
    }

    fn json_stringify_many(&self, values: Value) -> Result<Value, RuntimeError> {
        match values {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::List(list) => {
                    let items = list.borrow().clone();
                    let mut out = Vec::with_capacity(items.len());
                    for value in items {
                        out.push(self.json_stringify(value)?);
                    }
                    Ok(Value::Obj(ObjRef::new(Obj::List(RefCell::new(out)))))
                }
                Obj::Record(map) => {
                    let map = map.borrow();
                    let value = map.get("value").cloned().ok_or_else(|| {
                        RuntimeError::new("json.stringify_many record missing value")
                    })?;
                    let count = map.get("count").ok_or_else(|| {
                        RuntimeError::new("json.stringify_many record missing count")
                    })?;
                    let count = match count {
                        Value::Int(i) if *i >= 0 => *i as usize,
                        _ => {
                            return Err(RuntimeError::new(
                                "json.stringify_many record count must be Int >= 0",
                            ));
                        }
                    };
                    let mut out = Vec::with_capacity(count);
                    for _ in 0..count {
                        out.push(self.json_stringify(value.clone())?);
                    }
                    Ok(Value::Obj(ObjRef::new(Obj::List(RefCell::new(out)))))
                }
                _ => Err(RuntimeError::new(
                    "json.stringify_many expects List or { value, count } record",
                )),
            },
            _ => Err(RuntimeError::new(
                "json.stringify_many expects List or { value, count } record",
            )),
        }
    }

    fn tool_invoke(&self, tool_name: Value, tool_args: Value) -> Result<Value, RuntimeError> {
        let tool_name = value_as_string(&tool_name)?;
        let tool_args_json = self.value_to_json(tool_args).map_err(|err| {
            if let Some(code) = err.code() {
                RuntimeError::with_code(code, &err.message)
            } else {
                RuntimeError::with_code("E_TOOL_PAYLOAD", &err.message)
            }
        })?;
        let payload = serde_json::json!({
            "tool": tool_name,
            "args": tool_args_json,
        });
        let payload_bytes = serde_json::to_vec(&payload).map_err(|err| {
            RuntimeError::with_code(
                "E_TOOL_PAYLOAD",
                &format!("tool payload encode failed: {}", err),
            )
        })?;
        let tool_command = resolve_tool_command(&tool_name)?;
        let timeout_ms = tool_timeout_ms();
        let (stdout, stderr, code) = run_tool_process(&tool_command, &payload_bytes, timeout_ms)?;
        if code.unwrap_or(1) != 0 {
            let stderr_text = String::from_utf8_lossy(&stderr).trim().to_string();
            let detail = if stderr_text.is_empty() {
                "tool process failed without stderr output".to_string()
            } else {
                stderr_text
            };
            return Err(RuntimeError::with_code(
                "E_TOOL_EXIT",
                &format!(
                    "Tool invocation failed for {} (exit={}): {}",
                    tool_name,
                    code.unwrap_or(-1),
                    detail
                ),
            ));
        }
        if stdout.is_empty() {
            return Ok(Value::Null);
        }
        let stdout_text = String::from_utf8(stdout).map_err(|_| {
            RuntimeError::with_code(
                "E_TOOL_OUTPUT_FORMAT",
                "tool output is not valid UTF-8; expected JSON or text",
            )
        })?;
        let trimmed = stdout_text.trim();
        if trimmed.is_empty() {
            return Ok(Value::Null);
        }
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(trimmed) {
            return Ok(json_to_value(json));
        }
        Ok(string_value(trimmed))
    }

    fn bootstrap_format(&self, source: Value) -> Result<Value, RuntimeError> {
        let source = value_as_string(&source)?;
        let formatted = format_source(&source)
            .map_err(|err| RuntimeError::new(&format!("format failed: {}", err)))?;
        Ok(string_value(&formatted))
    }

    fn bootstrap_check(&self, source: Value) -> Result<Value, RuntimeError> {
        let source = value_as_string(&source)?;
        Ok(Value::Bool(check_format(&source).is_ok()))
    }

    fn bootstrap_lint(&self, source: Value) -> Result<Value, RuntimeError> {
        let source = value_as_string(&source)?;
        let items = collect_lint_issues(&source);
        let values: Vec<Value> = items
            .iter()
            .map(|item| lint_issue(item.line, item.code, item.message))
            .collect();
        Ok(Value::Obj(ObjRef::new(Obj::List(RefCell::new(values)))))
    }

    fn bootstrap_lint_count(&self, source: Value) -> Result<Value, RuntimeError> {
        let source = value_as_string(&source)?;
        Ok(Value::Int(collect_lint_issues(&source).len() as i64))
    }

    fn bootstrap_lint_json(&self, file: Value, source: Value) -> Result<Value, RuntimeError> {
        let file = value_as_string(&file)?;
        let source = value_as_string(&source)?;
        let entries: Vec<serde_json::Value> = collect_lint_issues(&source)
            .iter()
            .map(|item| {
                serde_json::json!({
                    "file": file.as_str(),
                    "line": item.line,
                    "code": item.code,
                    "message": item.message,
                })
            })
            .collect();
        let text = serde_json::to_string(&entries)
            .map_err(|err| RuntimeError::new(&format!("lint_json failed: {}", err)))?;
        Ok(string_value(&text))
    }

    fn compiler_parse_subset(&self, source: Value) -> Result<Value, RuntimeError> {
        let source = value_as_string(&source)?;
        let module = parse_subset_module(&source)?;
        validate_bootstrap_subset(&module)?;
        let mut summary = HashMap::new();
        summary.insert("items".to_string(), Value::Int(module.items.len() as i64));
        summary.insert(
            "functions".to_string(),
            Value::Int(
                module
                    .items
                    .iter()
                    .filter(|item| matches!(item, Item::Fn(_)))
                    .count() as i64,
            ),
        );
        Ok(record_value(summary))
    }

    fn compiler_check_subset(&self, source: Value) -> Result<Value, RuntimeError> {
        let source = value_as_string(&source)?;
        let module = parse_subset_module(&source)?;
        validate_bootstrap_subset(&module)?;
        let mut checker = TypeChecker::new();
        checker
            .check_module(&module)
            .map_err(|err| RuntimeError::new(&format_type_error(&err)))?;
        Ok(Value::Bool(true))
    }

    fn compiler_emit_subset(&self, source: Value, output: Value) -> Result<Value, RuntimeError> {
        let source = value_as_string(&source)?;
        let output = value_as_string(&output)?;
        let program = compile_subset_program(&source)?;
        let bytes = bincode::serialize(&program)
            .map_err(|err| RuntimeError::new(&format!("emit_subset serialize failed: {}", err)))?;
        let write_context = CapabilityContext::for_path(&output);
        self.check_capability(
            &["fs".to_string(), "write".to_string()],
            Some(&write_context),
        )?;
        std::fs::write(&output, bytes).map_err(|err| {
            RuntimeError::new(&format!("emit_subset failed to write {}: {}", output, err))
        })?;
        Ok(Value::Bool(true))
    }

    fn tokenizer_train(&self, config: Value) -> Result<Value, RuntimeError> {
        let mut train_cfg = TrainConfig::default();
        let mut save_path: Option<String> = None;
        let path = match config {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::String(s) => s.clone(),
                Obj::Record(map) => {
                    let map = map.borrow();
                    let path_value = map
                        .get("path")
                        .ok_or_else(|| RuntimeError::new("tokenizer.train config missing path"))?;
                    let path = value_as_string(path_value)
                        .map_err(|_| RuntimeError::new("tokenizer.train path must be string"))?;
                    if let Some(value) = map.get("vocab_size") {
                        let size = value_as_int(value)?;
                        if size > 0 {
                            train_cfg.vocab_size = size as usize;
                        }
                    }
                    if let Some(value) = map.get("lowercase") {
                        train_cfg.lowercase = value_as_bool(value)?;
                    }
                    if let Some(value) = map.get("min_freq") {
                        let min = value_as_int(value)?;
                        if min > 0 {
                            train_cfg.min_freq = min as usize;
                        }
                    }
                    if let Some(value) = map.get("seed") {
                        let seed = value_as_int(value)?;
                        if seed < 0 {
                            return Err(RuntimeError::new("tokenizer.train seed must be >= 0"));
                        }
                        train_cfg.seed = Some(seed as u64);
                    }
                    if let Some(value) = map.get("save_path") {
                        save_path = Some(value_as_string(value).map_err(|_| {
                            RuntimeError::new("tokenizer.train save_path must be string")
                        })?);
                    }
                    path
                }
                _ => return Err(RuntimeError::new("tokenizer.train expects config record")),
            },
            _ => return Err(RuntimeError::new("tokenizer.train expects config record")),
        };
        let tokenizer = Tokenizer::train_from_path(Path::new(&path), &train_cfg)
            .map_err(|err| RuntimeError::new(&format!("tokenizer.train failed: {}", err)))?;
        if let Some(save_path) = save_path {
            let save_context = CapabilityContext::for_path(&save_path);
            self.check_capability(
                &["fs".to_string(), "write".to_string()],
                Some(&save_context),
            )?;
            tokenizer
                .save(Path::new(&save_path))
                .map_err(|err| RuntimeError::new(&err))?;
        }
        Ok(Value::Obj(ObjRef::new(Obj::Tokenizer(tokenizer))))
    }

    fn tokenizer_load(&self, path: Value) -> Result<Value, RuntimeError> {
        let path = value_as_string(&path)?;
        let tokenizer = Tokenizer::load(Path::new(&path)).map_err(|err| RuntimeError::new(&err))?;
        Ok(Value::Obj(ObjRef::new(Obj::Tokenizer(tokenizer))))
    }

    fn tokenizer_save(&self, tokenizer: Value, path: Value) -> Result<(), RuntimeError> {
        let obj = match tokenizer {
            Value::Obj(obj) => obj,
            _ => return Err(RuntimeError::new("tokenizer.save expects Tokenizer")),
        };
        let tokenizer = match obj.as_obj() {
            Obj::Tokenizer(tok) => tok.clone(),
            _ => return Err(RuntimeError::new("tokenizer.save expects Tokenizer")),
        };
        let path = value_as_string(&path)?;
        tokenizer
            .save(Path::new(&path))
            .map_err(|err| RuntimeError::new(&err))?;
        Ok(())
    }

    fn tokenizer_encode(&self, tokenizer: Value, text: Value) -> Result<Value, RuntimeError> {
        let obj = match tokenizer {
            Value::Obj(obj) => obj,
            _ => return Err(RuntimeError::new("tokenizer.encode expects Tokenizer")),
        };
        let tokenizer = match obj.as_obj() {
            Obj::Tokenizer(tok) => tok.clone(),
            _ => return Err(RuntimeError::new("tokenizer.encode expects Tokenizer")),
        };
        let text = value_as_string(&text)?;
        let ids = tokenizer.encode(&text, false);
        let bytes = ids_to_bytes(&ids);
        Ok(Value::Obj(ObjRef::new(Obj::Buffer(bytes))))
    }

    fn tokenizer_decode(&self, tokenizer: Value, tokens: Value) -> Result<Value, RuntimeError> {
        let obj = match tokenizer {
            Value::Obj(obj) => obj,
            _ => return Err(RuntimeError::new("tokenizer.decode expects Tokenizer")),
        };
        let tokenizer = match obj.as_obj() {
            Obj::Tokenizer(tok) => tok.clone(),
            _ => return Err(RuntimeError::new("tokenizer.decode expects Tokenizer")),
        };
        let ids = value_to_token_ids(&tokens)?;
        let text = tokenizer.decode(&ids);
        Ok(string_value(&text))
    }

    fn dataset_open(
        &self,
        path: Value,
        tokenizer: Value,
        config: Value,
    ) -> Result<Value, RuntimeError> {
        let path = value_as_string(&path)?;
        let tokenizer = match tokenizer {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::Tokenizer(tok) => tok.clone(),
                _ => return Err(RuntimeError::new("dataset.open expects Tokenizer")),
            },
            _ => return Err(RuntimeError::new("dataset.open expects Tokenizer")),
        };
        let cfg = match config {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::Record(map) => {
                    let map = map.borrow();
                    let seq_len = map
                        .get("seq_len")
                        .ok_or_else(|| RuntimeError::new("dataset config missing seq_len"))
                        .and_then(value_as_int)?;
                    let batch_size = map
                        .get("batch_size")
                        .ok_or_else(|| RuntimeError::new("dataset config missing batch_size"))
                        .and_then(value_as_int)?;
                    if seq_len <= 0 || batch_size <= 0 {
                        return Err(RuntimeError::new("seq_len and batch_size must be > 0"));
                    }
                    let mut cfg = DatasetConfig::new(seq_len as usize, batch_size as usize);
                    if let Some(value) = map.get("add_eos") {
                        cfg.add_eos = value_as_bool(value)?;
                    }
                    if let Some(value) = map.get("drop_remainder") {
                        cfg.drop_remainder = value_as_bool(value)?;
                    }
                    if let Some(value) = map.get("pad_id") {
                        let id = value_as_int(value)?;
                        if id >= 0 {
                            cfg.pad_id = id as u32;
                        }
                    }
                    if let Some(value) = map.get("seed") {
                        let seed = value_as_int(value)?;
                        if seed < 0 {
                            return Err(RuntimeError::new("dataset seed must be >= 0"));
                        }
                        cfg.seed = Some(seed as u64);
                    }
                    if let Some(value) = map.get("shuffle") {
                        cfg.shuffle = value_as_bool(value)?;
                    }
                    if let Some(value) = map.get("prefetch_batches") {
                        let count = value_as_int(value)?;
                        if count < 0 {
                            return Err(RuntimeError::new("dataset prefetch_batches must be >= 0"));
                        }
                        cfg.prefetch_batches = count as usize;
                    }
                    cfg
                }
                _ => return Err(RuntimeError::new("dataset.open expects config record")),
            },
            _ => return Err(RuntimeError::new("dataset.open expects config record")),
        };
        let paths = resolve_dataset_paths(&path)
            .map_err(|err| RuntimeError::new(&format!("dataset.open failed: {}", err)))?;
        let stream = DatasetStream::new(paths, tokenizer, cfg)
            .map_err(|err| RuntimeError::new(&format!("dataset.open failed: {}", err)))?;
        Ok(Value::Obj(ObjRef::new(Obj::DatasetStream(Box::new(
            std::cell::RefCell::new(stream),
        )))))
    }

    fn dataset_next_batch(&self, stream: Value) -> Result<Option<Value>, RuntimeError> {
        let obj = match stream {
            Value::Obj(obj) => obj,
            _ => return Err(RuntimeError::new("next_batch expects DatasetStream")),
        };
        let stream = match obj.as_obj() {
            Obj::DatasetStream(inner) => inner,
            _ => return Err(RuntimeError::new("next_batch expects DatasetStream")),
        };
        let batch = stream
            .borrow_mut()
            .next_batch()
            .map_err(|err| RuntimeError::new(&err))?;
        Ok(batch.map(batch_to_value))
    }

    fn checkpoint_save(&self, dir: Value, state: Value) -> Result<(), RuntimeError> {
        let dir = value_as_string(&dir)?;
        let obj = match state {
            Value::Obj(obj) => obj,
            _ => return Err(RuntimeError::new("checkpoint.save expects record state")),
        };
        let state = match obj.as_obj() {
            Obj::Record(map) => map.borrow().clone(),
            _ => return Err(RuntimeError::new("checkpoint.save expects record state")),
        };
        let weights = match state.get("weights") {
            Some(Value::Obj(obj)) => match obj.as_obj() {
                Obj::Buffer(bytes) => buffer_to_f32(bytes)?,
                _ => return Err(RuntimeError::new("checkpoint weights must be Buffer")),
            },
            _ => return Err(RuntimeError::new("checkpoint state missing weights")),
        };
        let optimizer = match state.get("optimizer") {
            Some(Value::Obj(obj)) => match obj.as_obj() {
                Obj::Buffer(bytes) => buffer_to_f32(bytes)?,
                _ => return Err(RuntimeError::new("checkpoint optimizer must be Buffer")),
            },
            _ => Vec::new(),
        };
        let step = match state.get("step") {
            Some(value) => value_as_int(value)? as u64,
            None => return Err(RuntimeError::new("checkpoint state missing step")),
        };
        let tokens = match state.get("tokens") {
            Some(value) => value_as_int(value)? as u64,
            None => 0,
        };
        let loss = match state.get("loss") {
            Some(Value::Float(f)) => *f,
            Some(Value::Int(i)) => *i as f64,
            Some(_) => return Err(RuntimeError::new("checkpoint loss must be Float")),
            None => 0.0,
        };
        let format_version = match state.get("format_version") {
            Some(Value::Int(i)) if *i >= 0 => *i as u32,
            Some(_) => {
                return Err(RuntimeError::new(
                    "checkpoint format_version must be Int >= 0",
                ))
            }
            None => 1,
        };
        let config_hash = match state.get("config_hash") {
            Some(value) => value_as_string(value)?,
            None => "".to_string(),
        };
        let model_sig = match state.get("model_sig") {
            Some(value) => value_as_string(value)?,
            None => "".to_string(),
        };
        let dtype = match state.get("dtype") {
            Some(value) => value_as_string(value)?,
            None => "".to_string(),
        };
        let device = match state.get("device") {
            Some(value) => value_as_string(value)?,
            None => "".to_string(),
        };
        let world_size = match state.get("world_size") {
            Some(value) => value_as_int(value)? as usize,
            None => 1,
        };
        let rank = match state.get("rank") {
            Some(value) => value_as_int(value)? as usize,
            None => 0,
        };
        let grad_accum_steps = match state.get("grad_accum_steps") {
            Some(value) => value_as_int(value)? as usize,
            None => 1,
        };
        let grad_clip_norm = match state.get("grad_clip_norm") {
            Some(Value::Float(f)) if *f > 0.0 => Some(*f),
            Some(Value::Int(i)) if *i > 0 => Some(*i as f64),
            Some(Value::Float(_)) | Some(Value::Int(_)) => None,
            Some(_) => return Err(RuntimeError::new("checkpoint grad_clip_norm must be Float")),
            None => None,
        };
        if format_version > 1 {
            return Err(RuntimeError::new("unsupported checkpoint format version"));
        }
        let meta = CheckpointMeta {
            format_version,
            step,
            tokens,
            loss,
            config_hash,
            model_sig,
            dtype,
            device,
            world_size,
            rank,
            grad_accum_steps,
            grad_clip_norm,
            amp: None,
        };
        let state = CheckpointState {
            weights,
            optimizer,
            meta,
        };
        save_checkpoint(Path::new(&dir), &state).map_err(|err| RuntimeError::new(&err))?;
        Ok(())
    }

    fn checkpoint_load(&self, path: Value) -> Result<Value, RuntimeError> {
        let path = value_as_string(&path)?;
        let state = load_checkpoint(Path::new(&path)).map_err(|err| RuntimeError::new(&err))?;
        let mut map = HashMap::new();
        map.insert(
            "weights".to_string(),
            Value::Obj(ObjRef::new(Obj::Buffer(f32_to_bytes(&state.weights)))),
        );
        map.insert(
            "optimizer".to_string(),
            Value::Obj(ObjRef::new(Obj::Buffer(f32_to_bytes(&state.optimizer)))),
        );
        map.insert("step".to_string(), Value::Int(state.meta.step as i64));
        map.insert("tokens".to_string(), Value::Int(state.meta.tokens as i64));
        map.insert("loss".to_string(), Value::Float(state.meta.loss));
        map.insert(
            "format_version".to_string(),
            Value::Int(state.meta.format_version as i64),
        );
        map.insert(
            "config_hash".to_string(),
            string_value(&state.meta.config_hash),
        );
        map.insert("model_sig".to_string(), string_value(&state.meta.model_sig));
        map.insert("dtype".to_string(), string_value(&state.meta.dtype));
        map.insert("device".to_string(), string_value(&state.meta.device));
        map.insert(
            "world_size".to_string(),
            Value::Int(state.meta.world_size as i64),
        );
        map.insert("rank".to_string(), Value::Int(state.meta.rank as i64));
        map.insert(
            "grad_accum_steps".to_string(),
            Value::Int(state.meta.grad_accum_steps as i64),
        );
        if let Some(norm) = state.meta.grad_clip_norm {
            map.insert("grad_clip_norm".to_string(), Value::Float(norm));
        } else {
            map.insert("grad_clip_norm".to_string(), Value::Null);
        }
        Ok(record_value(map))
    }

    fn checkpoint_latest(&self, dir: Value) -> Result<Value, RuntimeError> {
        let dir = value_as_string(&dir)?;
        let latest = latest_checkpoint(Path::new(&dir)).map_err(|err| RuntimeError::new(&err))?;
        Ok(match latest {
            Some(path) => string_value(&path.to_string_lossy()),
            None => Value::Null,
        })
    }

    fn checkpoint_rotate(&self, dir: Value, keep: Value) -> Result<(), RuntimeError> {
        let dir = value_as_string(&dir)?;
        let keep = value_as_int(&keep)?;
        if keep < 0 {
            return Err(RuntimeError::new("checkpoint.rotate expects keep >= 0"));
        }
        rotate_checkpoints(Path::new(&dir), keep as usize)
            .map_err(|err| RuntimeError::new(&err))?;
        Ok(())
    }

    fn sparse_get(&mut self, matrix: Value, row: Value, col: Value) -> Result<Value, RuntimeError> {
        let row = value_as_non_negative_int(&row, "sparse.get expects row >= 0")?;
        let col = value_as_non_negative_int(&col, "sparse.get expects col >= 0")?;
        match matrix {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::SparseMatrix(inner) => Ok(inner
                    .borrow()
                    .data
                    .get(&(row, col))
                    .copied()
                    .map(Value::Float)
                    .unwrap_or(Value::Null)),
                _ => Err(RuntimeError::new("sparse.get expects SparseMatrix")),
            },
            _ => Err(RuntimeError::new("sparse.get expects SparseMatrix")),
        }
    }

    fn sparse_set(
        &mut self,
        matrix: Value,
        row: Value,
        col: Value,
        value: Value,
    ) -> Result<(), RuntimeError> {
        let row = value_as_non_negative_int(&row, "sparse.set expects row >= 0")?;
        let col = value_as_non_negative_int(&col, "sparse.set expects col >= 0")?;
        let value = value_as_float_like(&value)?;
        if !value.is_finite() {
            return Err(RuntimeError::new("sparse.set expects finite value"));
        }
        match matrix {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::SparseMatrix(inner) => {
                    let mut inner = inner.borrow_mut();
                    if value == 0.0 {
                        inner.data.remove(&(row, col));
                    } else {
                        inner.data.insert((row, col), value);
                    }
                    if let (Some(handle), Some(bindings)) =
                        (inner.native.clone(), self.sim_accel_bindings())
                    {
                        let _ = bindings.sparse_matrix_set.call(&[
                            handle,
                            Value::Int(row),
                            Value::Int(col),
                            Value::Float(value),
                        ]);
                    }
                }
                _ => return Err(RuntimeError::new("sparse.set expects SparseMatrix")),
            },
            _ => return Err(RuntimeError::new("sparse.set expects SparseMatrix")),
        }
        Ok(())
    }

    fn sparse_vector_get(&mut self, vector: Value, index: Value) -> Result<Value, RuntimeError> {
        let index = value_as_non_negative_int(&index, "sparse.get_vector expects index >= 0")?;
        match vector {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::SparseVector(inner) => Ok(inner
                    .borrow()
                    .data
                    .get(&index)
                    .copied()
                    .map(Value::Float)
                    .unwrap_or(Value::Null)),
                _ => Err(RuntimeError::new("sparse.get_vector expects SparseVector")),
            },
            _ => Err(RuntimeError::new("sparse.get_vector expects SparseVector")),
        }
    }

    fn sparse_vector_set(
        &mut self,
        vector: Value,
        index: Value,
        value: Value,
    ) -> Result<(), RuntimeError> {
        let index = value_as_non_negative_int(&index, "sparse.set_vector expects index >= 0")?;
        let value = value_as_float_like(&value)?;
        if !value.is_finite() {
            return Err(RuntimeError::new("sparse.set_vector expects finite value"));
        }
        match vector {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::SparseVector(inner) => {
                    let mut inner = inner.borrow_mut();
                    if value == 0.0 {
                        inner.data.remove(&index);
                    } else {
                        inner.data.insert(index, value);
                    }
                    if let (Some(handle), Some(bindings)) =
                        (inner.native.clone(), self.sim_accel_bindings())
                    {
                        let _ = bindings.sparse_vector_set.call(&[
                            handle,
                            Value::Int(index),
                            Value::Float(value),
                        ]);
                    }
                }
                _ => return Err(RuntimeError::new("sparse.set_vector expects SparseVector")),
            },
            _ => return Err(RuntimeError::new("sparse.set_vector expects SparseVector")),
        }
        Ok(())
    }

    fn sparse_nonzero(&mut self, matrix: Value) -> Result<Value, RuntimeError> {
        match matrix {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::SparseMatrix(inner) => {
                    let inner = inner.borrow();
                    let mut values = Vec::with_capacity(inner.data.len());
                    for ((row, col), value) in inner.data.iter() {
                        values.push(record_value(HashMap::from([
                            ("row".to_string(), Value::Int(*row)),
                            ("col".to_string(), Value::Int(*col)),
                            ("value".to_string(), Value::Float(*value)),
                        ])));
                    }
                    Ok(Value::Obj(ObjRef::new(Obj::List(RefCell::new(values)))))
                }
                _ => Err(RuntimeError::new("sparse.nonzero expects SparseMatrix")),
            },
            _ => Err(RuntimeError::new("sparse.nonzero expects SparseMatrix")),
        }
    }

    fn sparse_vector_nonzero(&mut self, vector: Value) -> Result<Value, RuntimeError> {
        match vector {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::SparseVector(inner) => {
                    let inner = inner.borrow();
                    let mut values = Vec::with_capacity(inner.data.len());
                    for (index, value) in inner.data.iter() {
                        values.push(record_value(HashMap::from([
                            ("index".to_string(), Value::Int(*index)),
                            ("value".to_string(), Value::Float(*value)),
                        ])));
                    }
                    Ok(Value::Obj(ObjRef::new(Obj::List(RefCell::new(values)))))
                }
                _ => Err(RuntimeError::new(
                    "sparse.nonzero_vector expects SparseVector",
                )),
            },
            _ => Err(RuntimeError::new(
                "sparse.nonzero_vector expects SparseVector",
            )),
        }
    }

    fn sparse_dot(&mut self, vector: Value, dense: Value) -> Result<f64, RuntimeError> {
        let dense = value_as_dense_f64(&dense)?;
        match vector {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::SparseVector(inner) => {
                    let inner = inner.borrow();
                    if let (Some(handle), Some(bindings)) =
                        (inner.native.clone(), self.sim_accel_bindings())
                    {
                        let encoded = encode_f64_buffer(&dense);
                        if let Ok(Value::Float(out)) =
                            bindings.sparse_vector_dot.call(&[handle, encoded])
                        {
                            if out.is_finite() {
                                return Ok(out);
                            }
                        }
                    }
                    let mut out = 0.0;
                    for (index, value) in inner.data.iter() {
                        let idx = *index as usize;
                        if let Some(dense_value) = dense.get(idx) {
                            out += value * dense_value;
                        }
                    }
                    Ok(out)
                }
                _ => Err(RuntimeError::new("sparse.dot expects SparseVector")),
            },
            _ => Err(RuntimeError::new("sparse.dot expects SparseVector")),
        }
    }

    fn sparse_matvec(&mut self, matrix: Value, dense: Value) -> Result<Value, RuntimeError> {
        let dense = value_as_dense_f64(&dense)?;
        match matrix {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::SparseMatrix(inner) => {
                    let inner = inner.borrow();
                    if let (Some(handle), Some(bindings)) =
                        (inner.native.clone(), self.sim_accel_bindings())
                    {
                        let encoded = encode_f64_buffer(&dense);
                        if let Ok(buffer) = bindings.sparse_matrix_matvec.call(&[handle, encoded]) {
                            if let Ok(out) = decode_f64_buffer(buffer) {
                                return Ok(Value::Obj(ObjRef::new(Obj::List(RefCell::new(
                                    out.into_iter().map(Value::Float).collect(),
                                )))));
                            }
                        }
                    }
                    let max_row = inner
                        .data
                        .keys()
                        .map(|(row, _)| *row as usize)
                        .max()
                        .unwrap_or(0);
                    let mut out = if inner.data.is_empty() {
                        Vec::new()
                    } else {
                        vec![0.0; max_row + 1]
                    };
                    for ((row, col), value) in inner.data.iter() {
                        let col = *col as usize;
                        if let Some(dense_value) = dense.get(col) {
                            out[*row as usize] += value * dense_value;
                        }
                    }
                    Ok(Value::Obj(ObjRef::new(Obj::List(RefCell::new(
                        out.into_iter().map(Value::Float).collect(),
                    )))))
                }
                _ => Err(RuntimeError::new("sparse.matvec expects SparseMatrix")),
            },
            _ => Err(RuntimeError::new("sparse.matvec expects SparseMatrix")),
        }
    }

    fn sparse_nnz(&mut self, value: Value) -> Result<usize, RuntimeError> {
        match value {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::SparseMatrix(inner) => Ok(inner.borrow().data.len()),
                Obj::SparseVector(inner) => Ok(inner.borrow().data.len()),
                _ => Err(RuntimeError::new(
                    "sparse.nnz expects SparseMatrix or SparseVector",
                )),
            },
            _ => Err(RuntimeError::new(
                "sparse.nnz expects SparseMatrix or SparseVector",
            )),
        }
    }

    fn event_push(&mut self, queue: Value, time: Value, event: Value) -> Result<(), RuntimeError> {
        let time = value_as_float_like(&time)?;
        if !time.is_finite() {
            return Err(RuntimeError::new("event.push expects finite time"));
        }
        match queue {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::EventQueue(inner) => {
                    let mut inner = inner.borrow_mut();
                    let seq = inner.next_seq;
                    inner.next_seq = inner.next_seq.saturating_add(1);
                    inner.items.push(crate::object::ScheduledEvent {
                        time,
                        seq,
                        event: event.clone(),
                    });
                    inner.payloads.insert(seq, event);
                    if let (Some(handle), Some(bindings)) =
                        (inner.native.clone(), self.sim_accel_bindings())
                    {
                        let _ = bindings.event_queue_push.call(&[
                            handle,
                            Value::Float(time),
                            Value::Int(seq as i64),
                        ]);
                    }
                    Ok(())
                }
                _ => Err(RuntimeError::new("event.push expects EventQueue")),
            },
            _ => Err(RuntimeError::new("event.push expects EventQueue")),
        }
    }

    fn event_pop(&mut self, queue: Value) -> Result<Value, RuntimeError> {
        match queue {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::EventQueue(inner) => {
                    let mut inner = inner.borrow_mut();
                    if let (Some(handle), Some(bindings)) =
                        (inner.native.clone(), self.sim_accel_bindings())
                    {
                        if let Ok(Some((time, seq))) = decode_event_meta(
                            bindings.event_queue_pop.call(&[handle])?,
                            "sim_event_queue_pop",
                        ) {
                            let event = inner.payloads.remove(&seq).unwrap_or(Value::Null);
                            let _ = inner.items.pop();
                            return Ok(event_record_value(crate::object::ScheduledEvent {
                                time,
                                seq,
                                event,
                            }));
                        }
                    }
                    Ok(match inner.items.pop() {
                        Some(item) => {
                            inner.payloads.remove(&item.seq);
                            event_record_value(item)
                        }
                        None => Value::Null,
                    })
                }
                _ => Err(RuntimeError::new("event.pop expects EventQueue")),
            },
            _ => Err(RuntimeError::new("event.pop expects EventQueue")),
        }
    }

    fn event_peek(&mut self, queue: Value) -> Result<Value, RuntimeError> {
        match queue {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::EventQueue(inner) => {
                    let inner = inner.borrow();
                    if let (Some(handle), Some(bindings)) =
                        (inner.native.clone(), self.sim_accel_bindings())
                    {
                        if let Ok(Some((time, seq))) = decode_event_meta(
                            bindings.event_queue_peek.call(&[handle])?,
                            "sim_event_queue_peek",
                        ) {
                            let event = inner.payloads.get(&seq).cloned().unwrap_or(Value::Null);
                            return Ok(event_record_value(crate::object::ScheduledEvent {
                                time,
                                seq,
                                event,
                            }));
                        }
                    }
                    Ok(inner
                        .items
                        .peek()
                        .cloned()
                        .map(event_record_value)
                        .unwrap_or(Value::Null))
                }
                _ => Err(RuntimeError::new("event.peek expects EventQueue")),
            },
            _ => Err(RuntimeError::new("event.peek expects EventQueue")),
        }
    }

    fn event_len(&mut self, queue: Value) -> Result<usize, RuntimeError> {
        match queue {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::EventQueue(inner) => {
                    let inner = inner.borrow();
                    if let (Some(handle), Some(bindings)) =
                        (inner.native.clone(), self.sim_accel_bindings())
                    {
                        if let Ok(Value::Int(len)) = bindings.event_queue_len.call(&[handle]) {
                            if len >= 0 {
                                return Ok(len as usize);
                            }
                        }
                    }
                    Ok(inner.items.len())
                }
                _ => Err(RuntimeError::new("event.len expects EventQueue")),
            },
            _ => Err(RuntimeError::new("event.len expects EventQueue")),
        }
    }

    fn pool_make(&mut self, capacity: Value, growable: bool) -> Result<Value, RuntimeError> {
        let capacity = value_as_non_negative_int(
            &capacity,
            if growable {
                "pool.make_growable expects capacity >= 0"
            } else {
                "pool.make expects capacity >= 0"
            },
        )?;
        let capacity = capacity as usize;
        let native = self.sim_accel_bindings().and_then(|bindings| {
            bindings
                .pool_new
                .call(&[Value::Int(capacity as i64), Value::Bool(growable)])
                .ok()
        });
        Ok(pool_value_with_native(capacity, growable, native))
    }

    fn pool_acquire(&mut self, pool: Value) -> Result<Value, RuntimeError> {
        match pool {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::Pool(inner) => {
                    let mut inner = inner.borrow_mut();
                    if let (Some(handle), Some(bindings)) =
                        (inner.native.clone(), self.sim_accel_bindings())
                    {
                        let _ = bindings.pool_acquire.call(&[handle]);
                    }
                    if let Some(value) = inner.items.pop() {
                        inner.acquire_hits = inner.acquire_hits.saturating_add(1);
                        Ok(value)
                    } else {
                        inner.acquire_misses = inner.acquire_misses.saturating_add(1);
                        Ok(Value::Null)
                    }
                }
                _ => Err(RuntimeError::new("pool.acquire expects Pool")),
            },
            _ => Err(RuntimeError::new("pool.acquire expects Pool")),
        }
    }

    fn pool_release(&mut self, pool: Value, value: Value) -> Result<bool, RuntimeError> {
        match pool {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::Pool(inner) => {
                    let mut inner = inner.borrow_mut();
                    if let (Some(handle), Some(bindings)) =
                        (inner.native.clone(), self.sim_accel_bindings())
                    {
                        if let Ok(Value::Bool(accepted)) =
                            bindings.pool_release.call(std::slice::from_ref(&handle))
                        {
                            if !accepted {
                                inner.dropped_on_full = inner.dropped_on_full.saturating_add(1);
                                return Ok(false);
                            }
                            if let Ok(Value::Int(capacity)) = bindings.pool_capacity.call(&[handle])
                            {
                                if capacity >= 0 {
                                    inner.capacity = capacity as usize;
                                }
                            }
                        }
                    }
                    if inner.items.len() >= inner.capacity {
                        if inner.growable {
                            inner.capacity = inner
                                .capacity
                                .max(1)
                                .saturating_mul(2)
                                .max(inner.items.len().saturating_add(1));
                        } else {
                            inner.dropped_on_full = inner.dropped_on_full.saturating_add(1);
                            return Ok(false);
                        }
                    }
                    inner.releases = inner.releases.saturating_add(1);
                    inner.items.push(value);
                    inner.high_watermark = inner.high_watermark.max(inner.items.len());
                    Ok(true)
                }
                _ => Err(RuntimeError::new("pool.release expects Pool")),
            },
            _ => Err(RuntimeError::new("pool.release expects Pool")),
        }
    }

    fn pool_reset(&mut self, pool: Value) -> Result<(), RuntimeError> {
        match pool {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::Pool(inner) => {
                    let mut inner = inner.borrow_mut();
                    if let (Some(handle), Some(bindings)) =
                        (inner.native.clone(), self.sim_accel_bindings())
                    {
                        let _ = bindings.pool_reset.call(&[handle]);
                    }
                    inner.items.clear();
                    Ok(())
                }
                _ => Err(RuntimeError::new("pool.reset expects Pool")),
            },
            _ => Err(RuntimeError::new("pool.reset expects Pool")),
        }
    }

    fn pool_available(&mut self, pool: Value) -> Result<usize, RuntimeError> {
        match pool {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::Pool(inner) => {
                    let inner = inner.borrow();
                    if let (Some(handle), Some(bindings)) =
                        (inner.native.clone(), self.sim_accel_bindings())
                    {
                        if let Ok(Value::Int(available)) = bindings.pool_available.call(&[handle]) {
                            if available >= 0 {
                                return Ok(available as usize);
                            }
                        }
                    }
                    Ok(inner.items.len())
                }
                _ => Err(RuntimeError::new("pool.available expects Pool")),
            },
            _ => Err(RuntimeError::new("pool.available expects Pool")),
        }
    }

    fn pool_capacity(&mut self, pool: Value) -> Result<usize, RuntimeError> {
        match pool {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::Pool(inner) => {
                    let inner = inner.borrow();
                    if let (Some(handle), Some(bindings)) =
                        (inner.native.clone(), self.sim_accel_bindings())
                    {
                        if let Ok(Value::Int(capacity)) = bindings.pool_capacity.call(&[handle]) {
                            if capacity >= 0 {
                                return Ok(capacity as usize);
                            }
                        }
                    }
                    Ok(inner.capacity)
                }
                _ => Err(RuntimeError::new("pool.capacity expects Pool")),
            },
            _ => Err(RuntimeError::new("pool.capacity expects Pool")),
        }
    }

    fn pool_stats(&mut self, pool: Value) -> Result<Value, RuntimeError> {
        match pool {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::Pool(inner) => {
                    let inner = inner.borrow();
                    let mut available = inner.items.len() as i64;
                    let mut capacity = inner.capacity as i64;
                    let mut acquire_hits = inner.acquire_hits as i64;
                    let mut acquire_misses = inner.acquire_misses as i64;
                    let mut releases = inner.releases as i64;
                    let mut dropped_on_full = inner.dropped_on_full as i64;
                    let mut high_watermark = inner.high_watermark as i64;
                    if let (Some(handle), Some(bindings)) =
                        (inner.native.clone(), self.sim_accel_bindings())
                    {
                        if let Ok(stats) = bindings.pool_stats.call(std::slice::from_ref(&handle)) {
                            if let Ok(decoded) = decode_pool_stats_buffer(stats) {
                                available = decoded[0];
                                capacity = decoded[1];
                                acquire_hits = decoded[2];
                                acquire_misses = decoded[3];
                                releases = decoded[4];
                            }
                        }
                        if let Ok(Value::Int(value)) =
                            bindings.pool_capacity.call(std::slice::from_ref(&handle))
                        {
                            capacity = value;
                        }
                        if let Ok(Value::Int(value)) = bindings.pool_available.call(&[handle]) {
                            available = value;
                        }
                        dropped_on_full = inner.dropped_on_full as i64;
                        high_watermark = inner.high_watermark as i64;
                    }
                    Ok(record_value(HashMap::from([
                        ("available".to_string(), Value::Int(available)),
                        ("capacity".to_string(), Value::Int(capacity)),
                        ("growable".to_string(), Value::Bool(inner.growable)),
                        ("acquire_hits".to_string(), Value::Int(acquire_hits)),
                        ("acquire_misses".to_string(), Value::Int(acquire_misses)),
                        ("releases".to_string(), Value::Int(releases)),
                        ("dropped_on_full".to_string(), Value::Int(dropped_on_full)),
                        ("high_watermark".to_string(), Value::Int(high_watermark)),
                    ])))
                }
                _ => Err(RuntimeError::new("pool.stats expects Pool")),
            },
            _ => Err(RuntimeError::new("pool.stats expects Pool")),
        }
    }

    fn spatial_make(&mut self) -> Value {
        let native = self
            .sim_accel_bindings()
            .and_then(|bindings| bindings.spatial_index_new.call(&[]).ok());
        spatial_index_value_with_native(native)
    }

    fn spatial_upsert(
        &mut self,
        spatial: Value,
        entity_id: Value,
        x: Value,
        y: Value,
    ) -> Result<(), RuntimeError> {
        let entity_id = value_as_int(&entity_id)?;
        let x = value_as_float_like(&x)?;
        let y = value_as_float_like(&y)?;
        match spatial {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::SpatialIndex(inner) => {
                    let mut inner = inner.borrow_mut();
                    inner.positions.insert(entity_id, (x, y));
                    if let (Some(handle), Some(bindings)) =
                        (inner.native.clone(), self.sim_accel_bindings())
                    {
                        let _ = bindings.spatial_upsert.call(&[
                            handle,
                            Value::Int(entity_id),
                            Value::Float(x),
                            Value::Float(y),
                        ]);
                    }
                    Ok(())
                }
                _ => Err(RuntimeError::new("spatial.upsert expects SpatialIndex")),
            },
            _ => Err(RuntimeError::new("spatial.upsert expects SpatialIndex")),
        }
    }

    fn spatial_remove(&mut self, spatial: Value, entity_id: Value) -> Result<bool, RuntimeError> {
        let entity_id = value_as_int(&entity_id)?;
        match spatial {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::SpatialIndex(inner) => {
                    let mut inner = inner.borrow_mut();
                    let removed = inner.positions.remove(&entity_id).is_some();
                    if let (Some(handle), Some(bindings)) =
                        (inner.native.clone(), self.sim_accel_bindings())
                    {
                        if let Ok(Value::Bool(native_removed)) = bindings
                            .spatial_remove
                            .call(&[handle, Value::Int(entity_id)])
                        {
                            return Ok(native_removed || removed);
                        }
                    }
                    Ok(removed)
                }
                _ => Err(RuntimeError::new("spatial.remove expects SpatialIndex")),
            },
            _ => Err(RuntimeError::new("spatial.remove expects SpatialIndex")),
        }
    }

    fn spatial_radius(
        &mut self,
        spatial: Value,
        x: Value,
        y: Value,
        radius: Value,
    ) -> Result<Value, RuntimeError> {
        let x = value_as_float_like(&x)?;
        let y = value_as_float_like(&y)?;
        let radius = value_as_float_like(&radius)?;
        if radius < 0.0 || !radius.is_finite() {
            return Err(RuntimeError::new(
                "spatial.radius expects finite radius >= 0",
            ));
        }
        match spatial {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::SpatialIndex(inner) => {
                    let inner = inner.borrow();
                    if let (Some(handle), Some(bindings)) =
                        (inner.native.clone(), self.sim_accel_bindings())
                    {
                        if let Ok(buffer) = bindings.spatial_radius.call(&[
                            handle,
                            Value::Float(x),
                            Value::Float(y),
                            Value::Float(radius),
                        ]) {
                            if let Ok(ids) = decode_i64_buffer(buffer, "sim_spatial_radius") {
                                return Ok(Value::Obj(ObjRef::new(Obj::List(RefCell::new(
                                    ids.into_iter().map(Value::Int).collect(),
                                )))));
                            }
                        }
                    }
                    let radius_sq = radius * radius;
                    let mut ids: Vec<(f64, i64)> = inner
                        .positions
                        .iter()
                        .filter_map(|(id, (px, py))| {
                            let dx = px - x;
                            let dy = py - y;
                            let dist_sq = dx * dx + dy * dy;
                            (dist_sq <= radius_sq).then_some((dist_sq, *id))
                        })
                        .collect();
                    ids.sort_by(|left, right| {
                        left.0
                            .partial_cmp(&right.0)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then_with(|| left.1.cmp(&right.1))
                    });
                    Ok(Value::Obj(ObjRef::new(Obj::List(RefCell::new(
                        ids.into_iter().map(|(_, id)| Value::Int(id)).collect(),
                    )))))
                }
                _ => Err(RuntimeError::new("spatial.radius expects SpatialIndex")),
            },
            _ => Err(RuntimeError::new("spatial.radius expects SpatialIndex")),
        }
    }

    fn spatial_nearest(
        &mut self,
        spatial: Value,
        x: Value,
        y: Value,
    ) -> Result<Value, RuntimeError> {
        let x = value_as_float_like(&x)?;
        let y = value_as_float_like(&y)?;
        match spatial {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::SpatialIndex(inner) => {
                    let inner = inner.borrow();
                    if let (Some(handle), Some(bindings)) =
                        (inner.native.clone(), self.sim_accel_bindings())
                    {
                        if let Ok(Value::Int(id)) = bindings.spatial_nearest.call(&[
                            handle,
                            Value::Float(x),
                            Value::Float(y),
                        ]) {
                            return Ok(if id >= 0 { Value::Int(id) } else { Value::Null });
                        }
                    }
                    let nearest = inner
                        .positions
                        .iter()
                        .map(|(id, (px, py))| {
                            let dx = px - x;
                            let dy = py - y;
                            (dx * dx + dy * dy, *id)
                        })
                        .min_by(|left, right| {
                            left.0
                                .partial_cmp(&right.0)
                                .unwrap_or(std::cmp::Ordering::Equal)
                                .then_with(|| left.1.cmp(&right.1))
                        });
                    Ok(nearest.map(|(_, id)| Value::Int(id)).unwrap_or(Value::Null))
                }
                _ => Err(RuntimeError::new("spatial.nearest expects SpatialIndex")),
            },
            _ => Err(RuntimeError::new("spatial.nearest expects SpatialIndex")),
        }
    }

    fn spatial_occupancy(
        &mut self,
        spatial: Value,
        min_x: Value,
        min_y: Value,
        max_x: Value,
        max_y: Value,
    ) -> Result<i64, RuntimeError> {
        let min_x = value_as_float_like(&min_x)?;
        let min_y = value_as_float_like(&min_y)?;
        let max_x = value_as_float_like(&max_x)?;
        let max_y = value_as_float_like(&max_y)?;
        match spatial {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::SpatialIndex(inner) => {
                    let inner = inner.borrow();
                    if let (Some(handle), Some(bindings)) =
                        (inner.native.clone(), self.sim_accel_bindings())
                    {
                        if let Ok(Value::Int(count)) = bindings.spatial_occupancy.call(&[
                            handle,
                            Value::Float(min_x),
                            Value::Float(min_y),
                            Value::Float(max_x),
                            Value::Float(max_y),
                        ]) {
                            return Ok(count.max(0));
                        }
                    }
                    Ok(inner
                        .positions
                        .values()
                        .filter(|(x, y)| *x >= min_x && *x <= max_x && *y >= min_y && *y <= max_y)
                        .count() as i64)
                }
                _ => Err(RuntimeError::new("spatial.occupancy expects SpatialIndex")),
            },
            _ => Err(RuntimeError::new("spatial.occupancy expects SpatialIndex")),
        }
    }

    fn snn_make(&mut self, neuron_count: Value) -> Result<Value, RuntimeError> {
        let neuron_count =
            value_as_non_negative_int(&neuron_count, "snn.make expects neuron_count >= 0")?
                as usize;
        let sparse_native = self
            .sim_accel_bindings()
            .and_then(|bindings| bindings.sparse_matrix_new.call(&[]).ok());
        let network_native = self.sim_accel_bindings().and_then(|bindings| {
            bindings
                .snn_network_new
                .call(&[Value::Int(neuron_count as i64)])
                .ok()
        });
        let synapses = sparse_matrix_value_with_native(sparse_native);
        Ok(snn_network_value(neuron_count, synapses, network_native))
    }

    fn snn_connect(
        &mut self,
        network: Value,
        from: Value,
        to: Value,
        weight: Value,
    ) -> Result<(), RuntimeError> {
        let from_idx = value_as_non_negative_int(&from, "snn.connect expects from >= 0")?;
        let to_idx = value_as_non_negative_int(&to, "snn.connect expects to >= 0")?;
        let weight = value_as_float_like(&weight)?;
        match network {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::SnnNetwork(inner) => {
                    let handle = {
                        let inner = inner.borrow();
                        if from_idx as usize >= inner.neuron_count
                            || to_idx as usize >= inner.neuron_count
                        {
                            return Err(RuntimeError::new("snn.connect index out of range"));
                        }
                        inner.native.clone()
                    };
                    let synapses = {
                        let inner = inner.borrow();
                        inner.synapses.clone()
                    };
                    self.sparse_set(
                        synapses,
                        Value::Int(from_idx),
                        Value::Int(to_idx),
                        Value::Float(weight),
                    )?;
                    if let (Some(handle), Some(bindings)) = (handle, self.sim_accel_bindings()) {
                        let _ = bindings.snn_connect.call(&[
                            handle,
                            Value::Int(from_idx),
                            Value::Int(to_idx),
                            Value::Float(weight),
                        ]);
                    }
                    Ok(())
                }
                _ => Err(RuntimeError::new("snn.connect expects SnnNetwork")),
            },
            _ => Err(RuntimeError::new("snn.connect expects SnnNetwork")),
        }
    }

    fn snn_set_potential(
        &mut self,
        network: Value,
        index: Value,
        value: Value,
    ) -> Result<(), RuntimeError> {
        let index = value_as_non_negative_int(&index, "snn.set_potential expects index >= 0")?;
        let value = value_as_float_like(&value)?;
        match network {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::SnnNetwork(inner) => {
                    let mut inner = inner.borrow_mut();
                    let slot = inner
                        .potentials
                        .get_mut(index as usize)
                        .ok_or_else(|| RuntimeError::new("snn.set_potential index out of range"))?;
                    *slot = value;
                    if let (Some(handle), Some(bindings)) =
                        (inner.native.clone(), self.sim_accel_bindings())
                    {
                        let _ = bindings.snn_set_potential.call(&[
                            handle,
                            Value::Int(index),
                            Value::Float(value),
                        ]);
                    }
                    Ok(())
                }
                _ => Err(RuntimeError::new("snn.set_potential expects SnnNetwork")),
            },
            _ => Err(RuntimeError::new("snn.set_potential expects SnnNetwork")),
        }
    }

    fn snn_get_potential(&mut self, network: Value, index: Value) -> Result<f64, RuntimeError> {
        let index = value_as_non_negative_int(&index, "snn.get_potential expects index >= 0")?;
        match network {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::SnnNetwork(inner) => {
                    let inner = inner.borrow();
                    if let (Some(handle), Some(bindings)) =
                        (inner.native.clone(), self.sim_accel_bindings())
                    {
                        if let Ok(Value::Float(value)) = bindings
                            .snn_get_potential
                            .call(&[handle, Value::Int(index)])
                        {
                            return Ok(value);
                        }
                    }
                    inner
                        .potentials
                        .get(index as usize)
                        .copied()
                        .ok_or_else(|| RuntimeError::new("snn.get_potential index out of range"))
                }
                _ => Err(RuntimeError::new("snn.get_potential expects SnnNetwork")),
            },
            _ => Err(RuntimeError::new("snn.get_potential expects SnnNetwork")),
        }
    }

    fn snn_set_threshold(
        &mut self,
        network: Value,
        index: Value,
        value: Value,
    ) -> Result<(), RuntimeError> {
        let index = value_as_non_negative_int(&index, "snn.set_threshold expects index >= 0")?;
        let value = value_as_float_like(&value)?;
        match network {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::SnnNetwork(inner) => {
                    let mut inner = inner.borrow_mut();
                    let slot = inner
                        .thresholds
                        .get_mut(index as usize)
                        .ok_or_else(|| RuntimeError::new("snn.set_threshold index out of range"))?;
                    *slot = value;
                    if let (Some(handle), Some(bindings)) =
                        (inner.native.clone(), self.sim_accel_bindings())
                    {
                        let _ = bindings.snn_set_threshold.call(&[
                            handle,
                            Value::Int(index),
                            Value::Float(value),
                        ]);
                    }
                    Ok(())
                }
                _ => Err(RuntimeError::new("snn.set_threshold expects SnnNetwork")),
            },
            _ => Err(RuntimeError::new("snn.set_threshold expects SnnNetwork")),
        }
    }

    fn snn_get_threshold(&mut self, network: Value, index: Value) -> Result<f64, RuntimeError> {
        let index = value_as_non_negative_int(&index, "snn.get_threshold expects index >= 0")?;
        match network {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::SnnNetwork(inner) => {
                    let inner = inner.borrow();
                    if let (Some(handle), Some(bindings)) =
                        (inner.native.clone(), self.sim_accel_bindings())
                    {
                        if let Ok(Value::Float(value)) = bindings
                            .snn_get_threshold
                            .call(&[handle, Value::Int(index)])
                        {
                            return Ok(value);
                        }
                    }
                    inner
                        .thresholds
                        .get(index as usize)
                        .copied()
                        .ok_or_else(|| RuntimeError::new("snn.get_threshold index out of range"))
                }
                _ => Err(RuntimeError::new("snn.get_threshold expects SnnNetwork")),
            },
            _ => Err(RuntimeError::new("snn.get_threshold expects SnnNetwork")),
        }
    }

    fn snn_set_decay(&mut self, network: Value, value: Value) -> Result<(), RuntimeError> {
        let value = value_as_float_like(&value)?;
        match network {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::SnnNetwork(inner) => {
                    let mut inner = inner.borrow_mut();
                    inner.decay = value;
                    if let (Some(handle), Some(bindings)) =
                        (inner.native.clone(), self.sim_accel_bindings())
                    {
                        let _ = bindings.snn_set_decay.call(&[handle, Value::Float(value)]);
                    }
                    Ok(())
                }
                _ => Err(RuntimeError::new("snn.set_decay expects SnnNetwork")),
            },
            _ => Err(RuntimeError::new("snn.set_decay expects SnnNetwork")),
        }
    }

    fn snn_get_decay(&mut self, network: Value) -> Result<f64, RuntimeError> {
        match network {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::SnnNetwork(inner) => {
                    let inner = inner.borrow();
                    if let (Some(handle), Some(bindings)) =
                        (inner.native.clone(), self.sim_accel_bindings())
                    {
                        if let Ok(Value::Float(value)) = bindings.snn_get_decay.call(&[handle]) {
                            return Ok(value);
                        }
                    }
                    Ok(inner.decay)
                }
                _ => Err(RuntimeError::new("snn.get_decay expects SnnNetwork")),
            },
            _ => Err(RuntimeError::new("snn.get_decay expects SnnNetwork")),
        }
    }

    fn snn_step(&mut self, network: Value, input: Value) -> Result<Value, RuntimeError> {
        let dense = value_as_dense_f64(&input)?;
        match network {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::SnnNetwork(inner) => {
                    let mut inner = inner.borrow_mut();
                    let neuron_count = inner.neuron_count;
                    let mut input_vec = vec![0.0; neuron_count];
                    for (idx, value) in dense.into_iter().take(neuron_count).enumerate() {
                        input_vec[idx] = value;
                    }
                    if let (Some(handle), Some(bindings)) =
                        (inner.native.clone(), self.sim_accel_bindings())
                    {
                        let encoded = encode_f64_buffer(&input_vec);
                        if let Ok(buffer) = bindings.snn_step.call(&[handle, encoded]) {
                            if let Ok((potentials, spikes)) =
                                decode_snn_step_buffer(buffer, neuron_count)
                            {
                                inner.potentials = potentials;
                                inner.last_spikes = spikes.clone();
                                return Ok(Value::Obj(ObjRef::new(Obj::List(RefCell::new(
                                    spikes
                                        .into_iter()
                                        .enumerate()
                                        .filter_map(|(idx, fired)| {
                                            fired.then_some(Value::Int(idx as i64))
                                        })
                                        .collect(),
                                )))));
                            }
                        }
                    }
                    let mut recurrent = vec![0.0; neuron_count];
                    let synapses = inner.synapses.clone();
                    if let Value::Obj(syn_obj) = synapses {
                        if let Obj::SparseMatrix(matrix) = syn_obj.as_obj() {
                            for ((from, to), weight) in matrix.borrow().data.iter() {
                                let from_idx = *from as usize;
                                let to_idx = *to as usize;
                                if from_idx < inner.last_spikes.len()
                                    && to_idx < recurrent.len()
                                    && inner.last_spikes[from_idx]
                                {
                                    recurrent[to_idx] += *weight;
                                }
                            }
                        }
                    }
                    let mut spikes = vec![false; neuron_count];
                    for idx in 0..neuron_count {
                        let mut next = inner.potentials[idx] * inner.decay + input_vec[idx];
                        next += recurrent[idx];
                        let fired = next >= inner.thresholds[idx];
                        if fired {
                            next = 0.0;
                        }
                        inner.potentials[idx] = next;
                        spikes[idx] = fired;
                    }
                    inner.last_spikes = spikes.clone();
                    Ok(Value::Obj(ObjRef::new(Obj::List(RefCell::new(
                        spikes
                            .into_iter()
                            .enumerate()
                            .filter_map(|(idx, fired)| fired.then_some(Value::Int(idx as i64)))
                            .collect(),
                    )))))
                }
                _ => Err(RuntimeError::new("snn.step expects SnnNetwork")),
            },
            _ => Err(RuntimeError::new("snn.step expects SnnNetwork")),
        }
    }

    fn snn_spikes(&self, network: Value) -> Result<Value, RuntimeError> {
        match network {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::SnnNetwork(inner) => Ok(Value::Obj(ObjRef::new(Obj::List(RefCell::new(
                    inner
                        .borrow()
                        .last_spikes
                        .iter()
                        .enumerate()
                        .filter_map(|(idx, fired)| fired.then_some(Value::Int(idx as i64)))
                        .collect(),
                ))))),
                _ => Err(RuntimeError::new("snn.spikes expects SnnNetwork")),
            },
            _ => Err(RuntimeError::new("snn.spikes expects SnnNetwork")),
        }
    }

    fn snn_potentials(&self, network: Value) -> Result<Value, RuntimeError> {
        match network {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::SnnNetwork(inner) => Ok(Value::Obj(ObjRef::new(Obj::List(RefCell::new(
                    inner
                        .borrow()
                        .potentials
                        .iter()
                        .copied()
                        .map(Value::Float)
                        .collect(),
                ))))),
                _ => Err(RuntimeError::new("snn.potentials expects SnnNetwork")),
            },
            _ => Err(RuntimeError::new("snn.potentials expects SnnNetwork")),
        }
    }

    fn snn_synapses(&self, network: Value) -> Result<Value, RuntimeError> {
        match network {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::SnnNetwork(inner) => Ok(inner.borrow().synapses.clone()),
                _ => Err(RuntimeError::new("snn.synapses expects SnnNetwork")),
            },
            _ => Err(RuntimeError::new("snn.synapses expects SnnNetwork")),
        }
    }

    fn agent_make(&self, world: Value, spatial: Value) -> Result<Value, RuntimeError> {
        match (&world, &spatial) {
            (Value::Obj(world_obj), Value::Obj(spatial_obj))
                if matches!(world_obj.as_obj(), Obj::SimWorld(_))
                    && matches!(spatial_obj.as_obj(), Obj::SpatialIndex(_)) =>
            {
                Ok(agent_env_value(world, spatial))
            }
            _ => Err(RuntimeError::new(
                "agent.make expects (SimWorld, SpatialIndex)",
            )),
        }
    }

    fn agent_register(
        &mut self,
        env: Value,
        agent_id: Value,
        body: Value,
        memory: Value,
        x: Value,
        y: Value,
    ) -> Result<(), RuntimeError> {
        let agent_id = value_as_int(&agent_id)?;
        let x = value_as_float_like(&x)?;
        let y = value_as_float_like(&y)?;
        let snapshot = record_value(HashMap::from([
            ("body".to_string(), body.clone()),
            ("memory".to_string(), memory.clone()),
            ("reward".to_string(), Value::Float(0.0)),
            (
                "position".to_string(),
                record_value(HashMap::from([
                    ("x".to_string(), Value::Float(x)),
                    ("y".to_string(), Value::Float(y)),
                ])),
            ),
        ]));
        let (world, spatial) = match env {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::AgentEnv(inner) => {
                    let mut inner = inner.borrow_mut();
                    inner.agents.insert(
                        agent_id,
                        crate::object::AgentRecordState {
                            body,
                            memory,
                            reward: 0.0,
                            sensors: VecDeque::new(),
                            actions: VecDeque::new(),
                            x,
                            y,
                        },
                    );
                    (inner.world.clone(), inner.spatial.clone())
                }
                _ => return Err(RuntimeError::new("agent.register expects AgentEnv")),
            },
            _ => return Err(RuntimeError::new("agent.register expects AgentEnv")),
        };
        self.spatial_upsert(
            spatial,
            Value::Int(agent_id),
            Value::Float(x),
            Value::Float(y),
        )?;
        self.sim_entity_set(world, Value::Int(agent_id), snapshot)?;
        Ok(())
    }

    fn agent_state(&self, env: Value, agent_id: Value) -> Result<Value, RuntimeError> {
        let agent_id = value_as_int(&agent_id)?;
        match env {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::AgentEnv(inner) => {
                    let inner = inner.borrow();
                    let record = inner.agents.get(&agent_id).ok_or_else(|| {
                        RuntimeError::new("agent.state expects a registered agent")
                    })?;
                    Ok(record_value(HashMap::from([
                        ("body".to_string(), record.body.clone()),
                        ("memory".to_string(), record.memory.clone()),
                        ("reward".to_string(), Value::Float(record.reward)),
                        (
                            "position".to_string(),
                            record_value(HashMap::from([
                                ("x".to_string(), Value::Float(record.x)),
                                ("y".to_string(), Value::Float(record.y)),
                            ])),
                        ),
                        (
                            "senses_pending".to_string(),
                            Value::Int(record.sensors.len() as i64),
                        ),
                        (
                            "actions_pending".to_string(),
                            Value::Int(record.actions.len() as i64),
                        ),
                    ])))
                }
                _ => Err(RuntimeError::new("agent.state expects AgentEnv")),
            },
            _ => Err(RuntimeError::new("agent.state expects AgentEnv")),
        }
    }

    fn agent_body(&self, env: Value, agent_id: Value) -> Result<Value, RuntimeError> {
        let agent_id = value_as_int(&agent_id)?;
        match env {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::AgentEnv(inner) => inner
                    .borrow()
                    .agents
                    .get(&agent_id)
                    .map(|record| record.body.clone())
                    .ok_or_else(|| RuntimeError::new("agent.body expects a registered agent")),
                _ => Err(RuntimeError::new("agent.body expects AgentEnv")),
            },
            _ => Err(RuntimeError::new("agent.body expects AgentEnv")),
        }
    }

    fn agent_memory(&self, env: Value, agent_id: Value) -> Result<Value, RuntimeError> {
        let agent_id = value_as_int(&agent_id)?;
        match env {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::AgentEnv(inner) => inner
                    .borrow()
                    .agents
                    .get(&agent_id)
                    .map(|record| record.memory.clone())
                    .ok_or_else(|| RuntimeError::new("agent.memory expects a registered agent")),
                _ => Err(RuntimeError::new("agent.memory expects AgentEnv")),
            },
            _ => Err(RuntimeError::new("agent.memory expects AgentEnv")),
        }
    }

    fn agent_set_body(
        &mut self,
        env: Value,
        agent_id: Value,
        body: Value,
    ) -> Result<(), RuntimeError> {
        let agent_id_num = value_as_int(&agent_id)?;
        let world = match &env {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::AgentEnv(inner) => {
                    let mut inner = inner.borrow_mut();
                    let record = inner.agents.get_mut(&agent_id_num).ok_or_else(|| {
                        RuntimeError::new("agent.set_body expects a registered agent")
                    })?;
                    record.body = body;
                    inner.world.clone()
                }
                _ => return Err(RuntimeError::new("agent.set_body expects AgentEnv")),
            },
            _ => return Err(RuntimeError::new("agent.set_body expects AgentEnv")),
        };
        let snapshot = self.agent_state(env, Value::Int(agent_id_num))?;
        self.sim_entity_set(world, Value::Int(agent_id_num), snapshot)?;
        Ok(())
    }

    fn agent_set_memory(
        &mut self,
        env: Value,
        agent_id: Value,
        memory: Value,
    ) -> Result<(), RuntimeError> {
        let agent_id_num = value_as_int(&agent_id)?;
        let world = match &env {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::AgentEnv(inner) => {
                    let mut inner = inner.borrow_mut();
                    let record = inner.agents.get_mut(&agent_id_num).ok_or_else(|| {
                        RuntimeError::new("agent.set_memory expects a registered agent")
                    })?;
                    record.memory = memory;
                    inner.world.clone()
                }
                _ => return Err(RuntimeError::new("agent.set_memory expects AgentEnv")),
            },
            _ => return Err(RuntimeError::new("agent.set_memory expects AgentEnv")),
        };
        let snapshot = self.agent_state(env, Value::Int(agent_id_num))?;
        self.sim_entity_set(world, Value::Int(agent_id_num), snapshot)?;
        Ok(())
    }

    fn agent_position(&self, env: Value, agent_id: Value) -> Result<Value, RuntimeError> {
        let agent_id = value_as_int(&agent_id)?;
        match env {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::AgentEnv(inner) => {
                    let inner = inner.borrow();
                    let record = inner.agents.get(&agent_id).ok_or_else(|| {
                        RuntimeError::new("agent.position expects a registered agent")
                    })?;
                    Ok(record_value(HashMap::from([
                        ("x".to_string(), Value::Float(record.x)),
                        ("y".to_string(), Value::Float(record.y)),
                    ])))
                }
                _ => Err(RuntimeError::new("agent.position expects AgentEnv")),
            },
            _ => Err(RuntimeError::new("agent.position expects AgentEnv")),
        }
    }

    fn agent_set_position(
        &mut self,
        env: Value,
        agent_id: Value,
        x: Value,
        y: Value,
    ) -> Result<(), RuntimeError> {
        let agent_id_num = value_as_int(&agent_id)?;
        let x = value_as_float_like(&x)?;
        let y = value_as_float_like(&y)?;
        let (world, spatial) = match &env {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::AgentEnv(inner) => {
                    let mut inner = inner.borrow_mut();
                    let record = inner.agents.get_mut(&agent_id_num).ok_or_else(|| {
                        RuntimeError::new("agent.set_position expects a registered agent")
                    })?;
                    record.x = x;
                    record.y = y;
                    (inner.world.clone(), inner.spatial.clone())
                }
                _ => return Err(RuntimeError::new("agent.set_position expects AgentEnv")),
            },
            _ => return Err(RuntimeError::new("agent.set_position expects AgentEnv")),
        };
        self.spatial_upsert(
            spatial,
            Value::Int(agent_id_num),
            Value::Float(x),
            Value::Float(y),
        )?;
        let snapshot = self.agent_state(env, Value::Int(agent_id_num))?;
        self.sim_entity_set(world, Value::Int(agent_id_num), snapshot)?;
        Ok(())
    }

    fn agent_neighbors(
        &mut self,
        env: Value,
        agent_id: Value,
        radius: Value,
    ) -> Result<Value, RuntimeError> {
        let agent_id_num = value_as_int(&agent_id)?;
        let radius_f = value_as_float_like(&radius)?;
        let (x, y, spatial) = match env {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::AgentEnv(inner) => {
                    let inner = inner.borrow();
                    let record = inner.agents.get(&agent_id_num).ok_or_else(|| {
                        RuntimeError::new("agent.neighbors expects a registered agent")
                    })?;
                    (record.x, record.y, inner.spatial.clone())
                }
                _ => return Err(RuntimeError::new("agent.neighbors expects AgentEnv")),
            },
            _ => return Err(RuntimeError::new("agent.neighbors expects AgentEnv")),
        };
        let neighbors = self.spatial_radius(
            spatial,
            Value::Float(x),
            Value::Float(y),
            Value::Float(radius_f),
        )?;
        let list = value_as_list(&neighbors)?
            .into_iter()
            .filter(|value| !matches!(value, Value::Int(id) if *id == agent_id_num))
            .collect();
        Ok(Value::Obj(ObjRef::new(Obj::List(RefCell::new(list)))))
    }

    fn agent_reward_add(
        &mut self,
        env: Value,
        agent_id: Value,
        delta: Value,
    ) -> Result<(), RuntimeError> {
        let agent_id = value_as_int(&agent_id)?;
        let delta = value_as_float_like(&delta)?;
        match env {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::AgentEnv(inner) => {
                    let mut inner = inner.borrow_mut();
                    let record = inner.agents.get_mut(&agent_id).ok_or_else(|| {
                        RuntimeError::new("agent.reward_add expects a registered agent")
                    })?;
                    record.reward += delta;
                    Ok(())
                }
                _ => Err(RuntimeError::new("agent.reward_add expects AgentEnv")),
            },
            _ => Err(RuntimeError::new("agent.reward_add expects AgentEnv")),
        }
    }

    fn agent_reward_get(&self, env: Value, agent_id: Value) -> Result<f64, RuntimeError> {
        let agent_id = value_as_int(&agent_id)?;
        match env {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::AgentEnv(inner) => inner
                    .borrow()
                    .agents
                    .get(&agent_id)
                    .map(|record| record.reward)
                    .ok_or_else(|| {
                        RuntimeError::new("agent.reward_get expects a registered agent")
                    }),
                _ => Err(RuntimeError::new("agent.reward_get expects AgentEnv")),
            },
            _ => Err(RuntimeError::new("agent.reward_get expects AgentEnv")),
        }
    }

    fn agent_reward_take(&mut self, env: Value, agent_id: Value) -> Result<f64, RuntimeError> {
        let agent_id = value_as_int(&agent_id)?;
        match env {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::AgentEnv(inner) => {
                    let mut inner = inner.borrow_mut();
                    let record = inner.agents.get_mut(&agent_id).ok_or_else(|| {
                        RuntimeError::new("agent.reward_take expects a registered agent")
                    })?;
                    let reward = record.reward;
                    record.reward = 0.0;
                    Ok(reward)
                }
                _ => Err(RuntimeError::new("agent.reward_take expects AgentEnv")),
            },
            _ => Err(RuntimeError::new("agent.reward_take expects AgentEnv")),
        }
    }

    fn agent_sense_push(
        &mut self,
        env: Value,
        agent_id: Value,
        value: Value,
    ) -> Result<(), RuntimeError> {
        let agent_id = value_as_int(&agent_id)?;
        match env {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::AgentEnv(inner) => {
                    let mut inner = inner.borrow_mut();
                    let record = inner.agents.get_mut(&agent_id).ok_or_else(|| {
                        RuntimeError::new("agent.sense_push expects a registered agent")
                    })?;
                    record.sensors.push_back(value);
                    Ok(())
                }
                _ => Err(RuntimeError::new("agent.sense_push expects AgentEnv")),
            },
            _ => Err(RuntimeError::new("agent.sense_push expects AgentEnv")),
        }
    }

    fn agent_sense_take(&mut self, env: Value, agent_id: Value) -> Result<Value, RuntimeError> {
        let agent_id = value_as_int(&agent_id)?;
        match env {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::AgentEnv(inner) => {
                    let mut inner = inner.borrow_mut();
                    let record = inner.agents.get_mut(&agent_id).ok_or_else(|| {
                        RuntimeError::new("agent.sense_take expects a registered agent")
                    })?;
                    Ok(record.sensors.pop_front().unwrap_or(Value::Null))
                }
                _ => Err(RuntimeError::new("agent.sense_take expects AgentEnv")),
            },
            _ => Err(RuntimeError::new("agent.sense_take expects AgentEnv")),
        }
    }

    fn agent_action_push(
        &mut self,
        env: Value,
        agent_id: Value,
        value: Value,
    ) -> Result<(), RuntimeError> {
        let agent_id = value_as_int(&agent_id)?;
        match env {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::AgentEnv(inner) => {
                    let mut inner = inner.borrow_mut();
                    let record = inner.agents.get_mut(&agent_id).ok_or_else(|| {
                        RuntimeError::new("agent.action_push expects a registered agent")
                    })?;
                    record.actions.push_back(value);
                    Ok(())
                }
                _ => Err(RuntimeError::new("agent.action_push expects AgentEnv")),
            },
            _ => Err(RuntimeError::new("agent.action_push expects AgentEnv")),
        }
    }

    fn agent_action_take(&mut self, env: Value, agent_id: Value) -> Result<Value, RuntimeError> {
        let agent_id = value_as_int(&agent_id)?;
        match env {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::AgentEnv(inner) => {
                    let mut inner = inner.borrow_mut();
                    let record = inner.agents.get_mut(&agent_id).ok_or_else(|| {
                        RuntimeError::new("agent.action_take expects a registered agent")
                    })?;
                    Ok(record.actions.pop_front().unwrap_or(Value::Null))
                }
                _ => Err(RuntimeError::new("agent.action_take expects AgentEnv")),
            },
            _ => Err(RuntimeError::new("agent.action_take expects AgentEnv")),
        }
    }

    fn agent_stream(
        &mut self,
        env: Value,
        agent_id: Value,
        domain: Value,
    ) -> Result<Value, RuntimeError> {
        let agent_id = value_as_int(&agent_id)?;
        let domain_name = value_as_string(&domain)?;
        let world_seed = match env {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::AgentEnv(inner) => self.sim_seed(inner.borrow().world.clone())?,
                _ => return Err(RuntimeError::new("agent.stream expects AgentEnv")),
            },
            _ => return Err(RuntimeError::new("agent.stream expects AgentEnv")),
        };
        let domain_id = stable_domain_hash(&domain_name);
        let state = mix_rng_seed(world_seed, agent_id, domain_id);
        let native = self.sim_accel_bindings().and_then(|bindings| {
            bindings
                .rng_stream_new
                .call(&[
                    Value::Int(world_seed),
                    Value::Int(agent_id),
                    Value::Int(domain_id),
                ])
                .ok()
        });
        Ok(rng_stream_value(
            world_seed, agent_id, domain_id, state, native,
        ))
    }

    fn agent_next_float(&mut self, stream: Value) -> Result<f64, RuntimeError> {
        match stream {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::RngStream(inner) => {
                    let mut inner = inner.borrow_mut();
                    if let (Some(handle), Some(bindings)) =
                        (inner.native.clone(), self.sim_accel_bindings())
                    {
                        if let Ok(Value::Float(value)) =
                            bindings.rng_stream_next_float.call(&[handle])
                        {
                            return Ok(value);
                        }
                    }
                    let raw = splitmix64_next(&mut inner.state) >> 11;
                    Ok((raw as f64) * (1.0 / ((1u64 << 53) as f64)))
                }
                _ => Err(RuntimeError::new("agent.next_float expects RngStream")),
            },
            _ => Err(RuntimeError::new("agent.next_float expects RngStream")),
        }
    }

    fn agent_next_int(&mut self, stream: Value, upper: Value) -> Result<i64, RuntimeError> {
        let upper = value_as_non_negative_int(&upper, "agent.next_int expects upper > 0")?;
        if upper <= 0 {
            return Err(RuntimeError::new("agent.next_int expects upper > 0"));
        }
        match stream {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::RngStream(inner) => {
                    let mut inner = inner.borrow_mut();
                    if let (Some(handle), Some(bindings)) =
                        (inner.native.clone(), self.sim_accel_bindings())
                    {
                        if let Ok(Value::Int(value)) = bindings
                            .rng_stream_next_int
                            .call(&[handle, Value::Int(upper)])
                        {
                            return Ok(value);
                        }
                    }
                    Ok((splitmix64_next(&mut inner.state) % upper as u64) as i64)
                }
                _ => Err(RuntimeError::new("agent.next_int expects RngStream")),
            },
            _ => Err(RuntimeError::new("agent.next_int expects RngStream")),
        }
    }

    fn sim_make(&self, max_events: Value, seed: Value) -> Result<Value, RuntimeError> {
        let max_events =
            value_as_non_negative_int(&max_events, "sim.make expects max_events >= 0")?;
        let seed = value_as_int(&seed)?;
        Ok(sim_world_value(max_events as usize, seed))
    }

    fn sim_time(&self, world: Value) -> Result<f64, RuntimeError> {
        match world {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::SimWorld(inner) => Ok(inner.borrow().now),
                _ => Err(RuntimeError::new("sim.time expects SimWorld")),
            },
            _ => Err(RuntimeError::new("sim.time expects SimWorld")),
        }
    }

    fn sim_seed(&self, world: Value) -> Result<i64, RuntimeError> {
        match world {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::SimWorld(inner) => Ok(inner.borrow().seed),
                _ => Err(RuntimeError::new("sim.seed expects SimWorld")),
            },
            _ => Err(RuntimeError::new("sim.seed expects SimWorld")),
        }
    }

    fn sim_pending(&self, world: Value) -> Result<usize, RuntimeError> {
        match world {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::SimWorld(inner) => Ok(inner.borrow().queue.len()),
                _ => Err(RuntimeError::new("sim.pending expects SimWorld")),
            },
            _ => Err(RuntimeError::new("sim.pending expects SimWorld")),
        }
    }

    fn sim_schedule(&self, world: Value, time: Value, event: Value) -> Result<(), RuntimeError> {
        let time = value_as_float_like(&time)?;
        if !time.is_finite() {
            return Err(RuntimeError::with_code(
                "E_SIM_TIME_ORDER",
                "sim.schedule expects finite event time",
            ));
        }
        match world {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::SimWorld(inner) => {
                    let mut inner = inner.borrow_mut();
                    if time < inner.now {
                        return Err(RuntimeError::with_code(
                            "E_SIM_TIME_ORDER",
                            "sim.schedule cannot insert an event before the current world time",
                        ));
                    }
                    if inner.queue.len() >= inner.max_events {
                        return Err(RuntimeError::with_code(
                            "E_SIM_EVENT_OVERFLOW",
                            "sim.schedule exceeded the configured event capacity",
                        ));
                    }
                    let seq = inner.next_seq;
                    inner.next_seq = inner.next_seq.saturating_add(1);
                    inner
                        .queue
                        .push(crate::object::ScheduledEvent { time, seq, event });
                    Ok(())
                }
                _ => Err(RuntimeError::new("sim.schedule expects SimWorld")),
            },
            _ => Err(RuntimeError::new("sim.schedule expects SimWorld")),
        }
    }

    fn sim_step(&self, world: Value) -> Result<Value, RuntimeError> {
        match world {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::SimWorld(inner) => {
                    let mut inner = inner.borrow_mut();
                    let Some(event) = inner.queue.pop() else {
                        return Ok(Value::Null);
                    };
                    if event.time < inner.now {
                        return Err(RuntimeError::with_code(
                            "E_SIM_TIME_ORDER",
                            "sim.step encountered an event earlier than the current world time",
                        ));
                    }
                    inner.now = event.time;
                    inner.log.push(event.clone());
                    Ok(sim_event_record_value(event))
                }
                _ => Err(RuntimeError::new("sim.step expects SimWorld")),
            },
            _ => Err(RuntimeError::new("sim.step expects SimWorld")),
        }
    }

    fn sim_run(&self, world: Value, max_steps: Value) -> Result<Value, RuntimeError> {
        let max_steps =
            value_as_non_negative_int(&max_steps, "sim.run expects max_steps >= 0")? as usize;
        let mut dispatched = Vec::new();
        for _ in 0..max_steps {
            let next = self.sim_step(world.clone())?;
            if matches!(next, Value::Null) {
                return Ok(Value::Obj(ObjRef::new(Obj::List(RefCell::new(dispatched)))));
            }
            dispatched.push(next);
        }
        if self.sim_pending(world)? > 0 {
            return Err(RuntimeError::with_code(
                "E_SIM_STARVATION",
                "sim.run exhausted its step budget before the event queue drained",
            ));
        }
        Ok(Value::Obj(ObjRef::new(Obj::List(RefCell::new(dispatched)))))
    }

    fn sim_snapshot(&self, world: Value) -> Result<Value, RuntimeError> {
        match world {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::SimWorld(inner) => {
                    let inner = inner.borrow();
                    let mut pending = inner.queue.clone().into_vec();
                    pending.sort_by(|left, right| {
                        left.time
                            .partial_cmp(&right.time)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then_with(|| left.seq.cmp(&right.seq))
                    });
                    let queue = pending
                        .into_iter()
                        .map(snapshot_event_record_value)
                        .collect::<Result<Vec<_>, RuntimeError>>()?;
                    let log = inner
                        .log
                        .iter()
                        .cloned()
                        .map(snapshot_event_record_value)
                        .collect::<Result<Vec<_>, RuntimeError>>()?;
                    let entities = inner
                        .entities
                        .iter()
                        .map(|(id, value)| {
                            let snapshot_value = clone_snapshot_value(value)?;
                            Ok(record_value(HashMap::from([
                                ("id".to_string(), Value::Int(*id)),
                                ("value".to_string(), snapshot_value),
                            ])))
                        })
                        .collect::<Result<Vec<_>, RuntimeError>>()?;
                    Ok(record_value(HashMap::from([
                        ("seed".to_string(), Value::Int(inner.seed)),
                        ("now".to_string(), Value::Float(inner.now)),
                        (
                            "max_events".to_string(),
                            Value::Int(inner.max_events as i64),
                        ),
                        ("next_seq".to_string(), Value::Int(inner.next_seq as i64)),
                        (
                            "queue".to_string(),
                            Value::Obj(ObjRef::new(Obj::List(RefCell::new(queue)))),
                        ),
                        (
                            "log".to_string(),
                            Value::Obj(ObjRef::new(Obj::List(RefCell::new(log)))),
                        ),
                        (
                            "entities".to_string(),
                            Value::Obj(ObjRef::new(Obj::List(RefCell::new(entities)))),
                        ),
                    ])))
                }
                _ => Err(RuntimeError::new("sim.snapshot expects SimWorld")),
            },
            _ => Err(RuntimeError::new("sim.snapshot expects SimWorld")),
        }
    }

    fn sim_restore(&self, snapshot: Value) -> Result<Value, RuntimeError> {
        let map = value_as_record(&snapshot).map_err(|_| {
            RuntimeError::with_code(
                "E_SIM_CORRUPTED_REPLAY",
                "sim.restore expects a snapshot record",
            )
        })?;
        let seed = map
            .get("seed")
            .ok_or_else(|| {
                RuntimeError::with_code(
                    "E_SIM_CORRUPTED_REPLAY",
                    "sim.restore snapshot is missing seed",
                )
            })
            .and_then(value_as_int)?;
        let now = map
            .get("now")
            .ok_or_else(|| {
                RuntimeError::with_code(
                    "E_SIM_CORRUPTED_REPLAY",
                    "sim.restore snapshot is missing now",
                )
            })
            .and_then(value_as_float_like)?;
        let max_events = map
            .get("max_events")
            .ok_or_else(|| {
                RuntimeError::with_code(
                    "E_SIM_CORRUPTED_REPLAY",
                    "sim.restore snapshot is missing max_events",
                )
            })
            .and_then(|value| {
                value_as_non_negative_int(value, "sim.restore expects max_events >= 0")
            })? as usize;
        let next_seq = map
            .get("next_seq")
            .ok_or_else(|| {
                RuntimeError::with_code(
                    "E_SIM_CORRUPTED_REPLAY",
                    "sim.restore snapshot is missing next_seq",
                )
            })
            .and_then(|value| {
                value_as_non_negative_int(value, "sim.restore expects next_seq >= 0")
            })? as u64;
        let queue = map.get("queue").ok_or_else(|| {
            RuntimeError::with_code(
                "E_SIM_CORRUPTED_REPLAY",
                "sim.restore snapshot is missing queue",
            )
        })?;
        let log = map.get("log").ok_or_else(|| {
            RuntimeError::with_code(
                "E_SIM_CORRUPTED_REPLAY",
                "sim.restore snapshot is missing log",
            )
        })?;
        let entities = map.get("entities").ok_or_else(|| {
            RuntimeError::with_code(
                "E_SIM_CORRUPTED_REPLAY",
                "sim.restore snapshot is missing entities",
            )
        })?;
        let queue_items = value_as_list(queue).map_err(|_| {
            RuntimeError::with_code("E_SIM_CORRUPTED_REPLAY", "sim.restore queue must be a list")
        })?;
        let log_items = value_as_list(log).map_err(|_| {
            RuntimeError::with_code("E_SIM_CORRUPTED_REPLAY", "sim.restore log must be a list")
        })?;
        let entity_items = value_as_list(entities).map_err(|_| {
            RuntimeError::with_code(
                "E_SIM_CORRUPTED_REPLAY",
                "sim.restore entities must be a list",
            )
        })?;
        let mut state = crate::object::SimWorldState::new(max_events, seed);
        state.now = now;
        let mut required_next_seq = 0u64;
        for item in queue_items {
            let event = scheduled_event_from_value(&item)?;
            if event.time < now {
                return Err(RuntimeError::with_code(
                    "E_SIM_CORRUPTED_REPLAY",
                    "sim.restore queue contains an event before the current world time",
                ));
            }
            required_next_seq = required_next_seq.max(event.seq.saturating_add(1));
            state.queue.push(event);
        }
        for item in log_items {
            let event = scheduled_event_from_value(&item)?;
            if event.time > now {
                return Err(RuntimeError::with_code(
                    "E_SIM_CORRUPTED_REPLAY",
                    "sim.restore log contains an event after the current world time",
                ));
            }
            required_next_seq = required_next_seq.max(event.seq.saturating_add(1));
            state.log.push(event);
        }
        if next_seq < required_next_seq {
            return Err(RuntimeError::with_code(
                "E_SIM_CORRUPTED_REPLAY",
                "sim.restore snapshot next_seq is lower than the highest event sequence",
            ));
        }
        state.next_seq = next_seq;
        for item in entity_items {
            let entity = value_as_record(&item).map_err(|_| {
                RuntimeError::with_code(
                    "E_SIM_CORRUPTED_REPLAY",
                    "sim.restore entity entries must be records",
                )
            })?;
            let id = entity.get("id").ok_or_else(|| {
                RuntimeError::with_code(
                    "E_SIM_CORRUPTED_REPLAY",
                    "sim.restore entity record is missing id",
                )
            })?;
            let value = entity.get("value").ok_or_else(|| {
                RuntimeError::with_code(
                    "E_SIM_CORRUPTED_REPLAY",
                    "sim.restore entity record is missing value",
                )
            })?;
            state
                .entities
                .insert(value_as_int(id)?, clone_snapshot_value(value)?);
        }
        Ok(Value::Obj(ObjRef::new(Obj::SimWorld(Box::new(
            RefCell::new(state),
        )))))
    }

    fn sim_replay(
        &self,
        log: Value,
        max_events: Value,
        seed: Value,
    ) -> Result<Value, RuntimeError> {
        let max_events =
            value_as_non_negative_int(&max_events, "sim.replay expects max_events >= 0")? as usize;
        let seed = value_as_int(&seed)?;
        let entries = value_as_list(&log).map_err(|_| {
            RuntimeError::with_code("E_SIM_CORRUPTED_REPLAY", "sim.replay expects a log list")
        })?;
        let world = sim_world_value(max_events, seed);
        let mut parsed = Vec::with_capacity(entries.len());
        for entry in entries {
            parsed.push(scheduled_event_from_value(&entry)?);
        }
        match &world {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::SimWorld(inner) => {
                    let mut inner = inner.borrow_mut();
                    for event in parsed {
                        if inner.queue.len() >= inner.max_events {
                            return Err(RuntimeError::with_code(
                                "E_SIM_EVENT_OVERFLOW",
                                "sim.replay exceeded the configured event capacity",
                            ));
                        }
                        inner.next_seq = inner.next_seq.max(event.seq.saturating_add(1));
                        inner.queue.push(event);
                    }
                }
                _ => unreachable!(),
            },
            _ => unreachable!(),
        }
        let pending = self.sim_pending(world.clone())?;
        self.sim_run(world.clone(), Value::Int(pending as i64))?;
        Ok(world)
    }

    fn sim_log(&self, world: Value) -> Result<Value, RuntimeError> {
        match world {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::SimWorld(inner) => {
                    let log = inner
                        .borrow()
                        .log
                        .iter()
                        .cloned()
                        .map(snapshot_event_record_value)
                        .collect::<Result<Vec<_>, RuntimeError>>()?;
                    Ok(Value::Obj(ObjRef::new(Obj::List(RefCell::new(log)))))
                }
                _ => Err(RuntimeError::new("sim.log expects SimWorld")),
            },
            _ => Err(RuntimeError::new("sim.log expects SimWorld")),
        }
    }

    fn sim_entity_set(&self, world: Value, id: Value, value: Value) -> Result<(), RuntimeError> {
        let id = value_as_int(&id)?;
        match world {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::SimWorld(inner) => {
                    inner.borrow_mut().entities.insert(id, value);
                    Ok(())
                }
                _ => Err(RuntimeError::new("sim.entity_set expects SimWorld")),
            },
            _ => Err(RuntimeError::new("sim.entity_set expects SimWorld")),
        }
    }

    fn sim_entity_get(&self, world: Value, id: Value) -> Result<Value, RuntimeError> {
        let id = value_as_int(&id)?;
        match world {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::SimWorld(inner) => Ok(inner
                    .borrow()
                    .entities
                    .get(&id)
                    .cloned()
                    .unwrap_or(Value::Null)),
                _ => Err(RuntimeError::new("sim.entity_get expects SimWorld")),
            },
            _ => Err(RuntimeError::new("sim.entity_get expects SimWorld")),
        }
    }

    fn sim_entity_remove(&self, world: Value, id: Value) -> Result<bool, RuntimeError> {
        let id = value_as_int(&id)?;
        match world {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::SimWorld(inner) => Ok(inner.borrow_mut().entities.remove(&id).is_some()),
                _ => Err(RuntimeError::new("sim.entity_remove expects SimWorld")),
            },
            _ => Err(RuntimeError::new("sim.entity_remove expects SimWorld")),
        }
    }

    fn sim_entity_ids(&self, world: Value) -> Result<Value, RuntimeError> {
        match world {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::SimWorld(inner) => {
                    let values = inner
                        .borrow()
                        .entities
                        .keys()
                        .copied()
                        .map(Value::Int)
                        .collect::<Vec<_>>();
                    Ok(Value::Obj(ObjRef::new(Obj::List(RefCell::new(values)))))
                }
                _ => Err(RuntimeError::new("sim.entity_ids expects SimWorld")),
            },
            _ => Err(RuntimeError::new("sim.entity_ids expects SimWorld")),
        }
    }

    fn sim_coroutine_spawn(
        &mut self,
        program: &Program,
        world: Value,
        func: Value,
        state: Option<Value>,
    ) -> Result<Value, RuntimeError> {
        match &world {
            Value::Obj(obj) if matches!(obj.as_obj(), Obj::SimWorld(_)) => {}
            _ => return Err(RuntimeError::new("sim.coroutine expects SimWorld")),
        }
        let expected_arity = if state.is_some() { 2 } else { 1 };
        let actual_arity = match &func {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::Function(f) => f.arity,
                Obj::BoundFunction(bf) => bf.arity,
                _ => {
                    return Err(RuntimeError::new(
                        "sim.coroutine expects a function or bound function",
                    ));
                }
            },
            _ => {
                return Err(RuntimeError::new(
                    "sim.coroutine expects a function or bound function",
                ));
            }
        };
        if actual_arity != expected_arity {
            return Err(RuntimeError::new(if state.is_some() {
                "sim.coroutine_with expects function arity 2"
            } else {
                "sim.coroutine expects function arity 1"
            }));
        }
        let predicted_task_id = self.next_task_id;
        let coroutine = sim_coroutine_value(world.clone(), state.clone(), predicted_task_id);
        let mut args = vec![coroutine.clone()];
        if let Some(state) = state {
            args.push(state);
        }
        let task_id = self.spawn_task_with_args(program, func, args, None)?;
        if self.sim_coroutines.len() <= task_id {
            self.sim_coroutines.resize(task_id + 1, None);
        }
        if let Value::Obj(obj) = &coroutine {
            self.sim_coroutines[task_id] = Some(obj.downgrade());
        }
        if let Some(task) = self.tasks.get(task_id).and_then(|entry| entry.as_ref()) {
            if task.result.is_some() {
                self.complete_sim_coroutine(task_id, task.result.clone())?;
            }
        }
        if let Some(profile) = self.bench_profile.as_mut() {
            profile.counters.sim_coroutines_spawned =
                profile.counters.sim_coroutines_spawned.saturating_add(1);
        }
        Ok(coroutine)
    }

    fn sim_coroutine_spawn_args(
        &mut self,
        program: &Program,
        world: Value,
        func: Value,
        args: Value,
    ) -> Result<Value, RuntimeError> {
        match &world {
            Value::Obj(obj) if matches!(obj.as_obj(), Obj::SimWorld(_)) => {}
            _ => return Err(RuntimeError::new("sim.coroutine_args expects SimWorld")),
        }
        let extra_args = value_as_list(&args)
            .map_err(|_| RuntimeError::new("sim.coroutine_args expects args list"))?;
        let expected_arity = extra_args.len() as u16 + 1;
        let actual_arity = match &func {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::Function(f) => f.arity,
                Obj::BoundFunction(bf) => bf.arity,
                _ => {
                    return Err(RuntimeError::new(
                        "sim.coroutine_args expects a function or bound function",
                    ));
                }
            },
            _ => {
                return Err(RuntimeError::new(
                    "sim.coroutine_args expects a function or bound function",
                ));
            }
        };
        if actual_arity != expected_arity {
            return Err(RuntimeError::new(
                "sim.coroutine_args arity must match args list length + 1",
            ));
        }
        let predicted_task_id = self.next_task_id;
        let coroutine = sim_coroutine_value(world.clone(), None, predicted_task_id);
        let mut spawn_args = Vec::with_capacity(extra_args.len() + 1);
        spawn_args.push(coroutine.clone());
        spawn_args.extend(extra_args);
        let task_id = self.spawn_task_with_args(program, func, spawn_args, None)?;
        if self.sim_coroutines.len() <= task_id {
            self.sim_coroutines.resize(task_id + 1, None);
        }
        if let Value::Obj(obj) = &coroutine {
            self.sim_coroutines[task_id] = Some(obj.downgrade());
        }
        if let Some(task) = self.tasks.get(task_id).and_then(|entry| entry.as_ref()) {
            if task.result.is_some() {
                self.complete_sim_coroutine(task_id, task.result.clone())?;
            }
        }
        if let Some(profile) = self.bench_profile.as_mut() {
            profile.counters.sim_coroutines_spawned =
                profile.counters.sim_coroutines_spawned.saturating_add(1);
        }
        Ok(coroutine)
    }

    fn sim_coroutine_world(&self, coroutine: Value) -> Result<Value, RuntimeError> {
        match coroutine {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::SimCoroutine(inner) => Ok(inner.borrow().world.clone()),
                _ => Err(RuntimeError::new("sim.world expects SimCoroutine")),
            },
            _ => Err(RuntimeError::new("sim.world expects SimCoroutine")),
        }
    }

    fn sim_coroutine_state(&self, coroutine: Value) -> Result<Value, RuntimeError> {
        match coroutine {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::SimCoroutine(inner) => Ok(inner.borrow().state.clone().unwrap_or(Value::Null)),
                _ => Err(RuntimeError::new("sim.state expects SimCoroutine")),
            },
            _ => Err(RuntimeError::new("sim.state expects SimCoroutine")),
        }
    }

    fn sim_coroutine_emit(&mut self, coroutine: Value, value: Value) -> Result<(), RuntimeError> {
        if matches!(value, Value::Null) {
            return Err(RuntimeError::new(
                "sim.emit does not allow none; use a concrete value",
            ));
        }
        match coroutine {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::SimCoroutine(inner) => {
                    let mut inner = inner.borrow_mut();
                    inner.emitted = inner.emitted.saturating_add(1);
                    if let Some(waiter) = inner.waiters.pop_front() {
                        if let Some(task) = self.tasks.get_mut(waiter).and_then(|t| t.as_mut()) {
                            if !matches!(task.state, TaskState::Finished) {
                                task.stack.push(value);
                                task.state = TaskState::Ready;
                                self.ready.push_back(waiter);
                            }
                        }
                    } else {
                        inner.outputs.push_back(value);
                    }
                    if let Some(profile) = self.bench_profile.as_mut() {
                        profile.counters.sim_coroutine_emits =
                            profile.counters.sim_coroutine_emits.saturating_add(1);
                    }
                    self.yield_now = true;
                    Ok(())
                }
                _ => Err(RuntimeError::new("sim.emit expects SimCoroutine")),
            },
            _ => Err(RuntimeError::new("sim.emit expects SimCoroutine")),
        }
    }

    fn sim_coroutine_next(&mut self, coroutine: Value) -> Result<Option<Value>, RuntimeError> {
        match coroutine {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::SimCoroutine(inner) => {
                    let mut inner = inner.borrow_mut();
                    if let Some(value) = inner.outputs.pop_front() {
                        return Ok(Some(value));
                    }
                    let result = self
                        .tasks
                        .get(inner.task_id)
                        .and_then(|entry| entry.as_ref())
                        .and_then(|task| task.result.clone());
                    if let Some(result) = result {
                        return match result {
                            Ok(_) => Ok(Some(Value::Null)),
                            Err(err) => Err(err),
                        };
                    }
                    let current_id = self
                        .current_task
                        .ok_or_else(|| RuntimeError::new("No current task"))?;
                    inner.waiters.push_back(current_id);
                    if let Some(profile) = self.bench_profile.as_mut() {
                        profile.counters.sim_coroutine_next_waits =
                            profile.counters.sim_coroutine_next_waits.saturating_add(1);
                    }
                    self.pending_state = Some(TaskState::BlockedChannel);
                    self.yield_now = true;
                    Ok(None)
                }
                _ => Err(RuntimeError::new("sim.next expects SimCoroutine")),
            },
            _ => Err(RuntimeError::new("sim.next expects SimCoroutine")),
        }
    }

    fn sim_coroutine_join(&mut self, coroutine: Value) -> Result<Option<Value>, RuntimeError> {
        let task_id = match coroutine {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::SimCoroutine(inner) => inner.borrow().task_id,
                _ => return Err(RuntimeError::new("sim.join expects SimCoroutine")),
            },
            _ => return Err(RuntimeError::new("sim.join expects SimCoroutine")),
        };
        self.task_join(task_handle_value(task_id))
    }

    fn sim_coroutine_done(&self, coroutine: Value) -> Result<bool, RuntimeError> {
        match coroutine {
            Value::Obj(obj) => match obj.as_obj() {
                Obj::SimCoroutine(inner) => {
                    let inner = inner.borrow();
                    let finished = self
                        .tasks
                        .get(inner.task_id)
                        .and_then(|entry| entry.as_ref())
                        .and_then(|task| task.result.clone())
                        .is_some();
                    Ok(finished && inner.outputs.is_empty())
                }
                _ => Err(RuntimeError::new("sim.done expects SimCoroutine")),
            },
            _ => Err(RuntimeError::new("sim.done expects SimCoroutine")),
        }
    }

    fn complete_sim_coroutine(
        &mut self,
        task_id: usize,
        result: Option<Result<Value, RuntimeError>>,
    ) -> Result<(), RuntimeError> {
        let Some(slot) = self
            .sim_coroutines
            .get(task_id)
            .and_then(|entry| entry.as_ref())
        else {
            return Ok(());
        };
        let Some(obj) = slot.upgrade() else {
            return Ok(());
        };
        let Obj::SimCoroutine(inner) = obj.as_ref() else {
            return Ok(());
        };
        let mut inner = inner.borrow_mut();
        inner.finished = true;
        while let Some(waiter) = inner.waiters.pop_front() {
            if let Some(task) = self.tasks.get_mut(waiter).and_then(|t| t.as_mut()) {
                if matches!(task.state, TaskState::Finished) {
                    continue;
                }
                match &result {
                    Some(Err(err)) => {
                        task.pending_error = Some(err.clone());
                    }
                    _ => task.stack.push(Value::Null),
                }
                task.state = TaskState::Ready;
                self.ready.push_back(waiter);
            }
        }
        Ok(())
    }

    fn value_to_json(&self, value: Value) -> Result<serde_json::Value, RuntimeError> {
        match value {
            Value::Null => Ok(serde_json::Value::Null),
            Value::Bool(b) => Ok(serde_json::Value::Bool(b)),
            Value::Int(i) => Ok(serde_json::Value::Number(serde_json::Number::from(i))),
            Value::Float(f) => serde_json::Number::from_f64(f)
                .map(serde_json::Value::Number)
                .ok_or_else(|| RuntimeError::new("json.stringify invalid float")),
            Value::Obj(obj) => match obj.as_obj() {
                Obj::String(s) => Ok(serde_json::Value::String(s.clone())),
                Obj::Json(value) => Ok(value.clone()),
                Obj::Buffer(bytes) => Ok(serde_json::Value::String(
                    String::from_utf8_lossy(bytes).to_string(),
                )),
                Obj::List(items) => {
                    let items = items.borrow();
                    let mut out = Vec::with_capacity(items.len());
                    for item in items.iter() {
                        out.push(self.value_to_json(item.clone())?);
                    }
                    Ok(serde_json::Value::Array(out))
                }
                Obj::Record(map) => {
                    let map = map.borrow();
                    let mut out = serde_json::Map::new();
                    for (k, v) in map.iter() {
                        let value = self.value_to_json(v.clone())?;
                        out.insert(k.clone(), value);
                    }
                    Ok(serde_json::Value::Object(out))
                }
                _ => Err(RuntimeError::new("json.stringify unsupported value")),
            },
        }
    }

    fn stack_trace(&self, program: &Program, ip: usize) -> Vec<RuntimeFrame> {
        let mut frames = Vec::new();
        let last_index = self.frames.len().saturating_sub(1);
        for (idx, frame) in self.frames.iter().enumerate().rev() {
            let func = &program.functions[frame.func_index as usize];
            let current_ip = if idx == last_index { ip } else { frame.ip };
            let line = func.chunk.lines.get(current_ip).copied();
            frames.push(RuntimeFrame {
                function: func.name.clone(),
                source: func.source_name.clone(),
                line,
            });
        }
        frames
    }

    fn constant_to_value(
        &mut self,
        c: &Constant,
        program: &Program,
    ) -> Result<Value, RuntimeError> {
        Ok(match c {
            Constant::Int(i) => Value::Int(*i),
            Constant::Float(f) => Value::Float(*f),
            Constant::Bool(b) => Value::Bool(*b),
            Constant::Null => Value::Null,
            Constant::String(s) => string_value(s),
            Constant::Function(idx) => function_value(*idx, program),
            Constant::NativeFunction(decl) => {
                let ffi = self.ffi_loader.bind(decl)?;
                Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                    name: decl.name.clone(),
                    arity: ffi.arity() as u16,
                    kind: NativeImpl::Ffi(Box::new(ffi)),
                    bound: None,
                })))
            }
        })
    }

    fn sim_accel_bindings(&mut self) -> Option<&mut SimAccelBindings> {
        match &self.sim_accel {
            SimAccelState::Ready(_) => {}
            SimAccelState::Disabled => return None,
            SimAccelState::Uninitialized => {
                let enabled = std::env::var("ENKAI_SIM_ACCEL")
                    .ok()
                    .map(|value| value.trim().to_ascii_lowercase())
                    .map(|value| !matches!(value.as_str(), "0" | "false" | "off" | "no"))
                    .unwrap_or(true);
                if !enabled {
                    self.sim_accel = SimAccelState::Disabled;
                    return None;
                }
                self.sim_accel = match SimAccelBindings::load(&mut self.ffi_loader) {
                    Ok(bindings) => SimAccelState::Ready(Box::new(bindings)),
                    Err(_) => SimAccelState::Disabled,
                };
            }
        }
        match &mut self.sim_accel {
            SimAccelState::Ready(bindings) => Some(bindings.as_mut()),
            SimAccelState::Disabled | SimAccelState::Uninitialized => None,
        }
    }
}

fn encode_f64_buffer(values: &[f64]) -> Value {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    buffer_value(bytes)
}

fn decode_f64_buffer(value: Value) -> Result<Vec<f64>, RuntimeError> {
    let bytes = value_as_buffer(&value)?;
    if bytes.len() % 8 != 0 {
        return Err(RuntimeError::new(
            "native sparse acceleration returned invalid f64 buffer length",
        ));
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| {
            let mut raw = [0u8; 8];
            raw.copy_from_slice(chunk);
            f64::from_le_bytes(raw)
        })
        .collect())
}

fn decode_event_meta(value: Value, name: &str) -> Result<Option<(f64, u64)>, RuntimeError> {
    let bytes = value_as_buffer(&value)?;
    if bytes.is_empty() {
        return Ok(None);
    }
    if bytes.len() != 16 {
        return Err(RuntimeError::new(&format!(
            "{} returned invalid event metadata length",
            name
        )));
    }
    let mut time_raw = [0u8; 8];
    let mut seq_raw = [0u8; 8];
    time_raw.copy_from_slice(&bytes[..8]);
    seq_raw.copy_from_slice(&bytes[8..]);
    Ok(Some((
        f64::from_le_bytes(time_raw),
        u64::from_le_bytes(seq_raw),
    )))
}

fn decode_pool_stats_buffer(value: Value) -> Result<[i64; 5], RuntimeError> {
    let bytes = value_as_buffer(&value)?;
    if bytes.len() != 40 {
        return Err(RuntimeError::new(
            "native pool stats returned invalid payload length",
        ));
    }
    let mut out = [0i64; 5];
    for (idx, chunk) in bytes.chunks_exact(8).enumerate() {
        let mut raw = [0u8; 8];
        raw.copy_from_slice(chunk);
        out[idx] = i64::from_le_bytes(raw);
    }
    Ok(out)
}

fn decode_i64_buffer(value: Value, name: &str) -> Result<Vec<i64>, RuntimeError> {
    let bytes = value_as_buffer(&value)?;
    if bytes.len() % 8 != 0 {
        return Err(RuntimeError::new(&format!(
            "{} returned invalid i64 buffer length",
            name
        )));
    }
    let mut out = Vec::with_capacity(bytes.len() / 8);
    for chunk in bytes.chunks_exact(8) {
        let mut raw = [0u8; 8];
        raw.copy_from_slice(chunk);
        out.push(i64::from_le_bytes(raw));
    }
    Ok(out)
}

fn decode_snn_step_buffer(
    value: Value,
    neuron_count: usize,
) -> Result<(Vec<f64>, Vec<bool>), RuntimeError> {
    let bytes = value_as_buffer(&value)?;
    let expected = neuron_count
        .checked_mul(9)
        .ok_or_else(|| RuntimeError::new("native SNN payload size overflow"))?;
    if bytes.len() != expected {
        return Err(RuntimeError::new(
            "native SNN step returned invalid payload length",
        ));
    }
    let potentials_bytes_len = neuron_count * 8;
    let mut potentials = Vec::with_capacity(neuron_count);
    for chunk in bytes[..potentials_bytes_len].chunks_exact(8) {
        let mut raw = [0u8; 8];
        raw.copy_from_slice(chunk);
        potentials.push(f64::from_le_bytes(raw));
    }
    let spikes = bytes[potentials_bytes_len..]
        .iter()
        .map(|flag| *flag != 0)
        .collect();
    Ok((potentials, spikes))
}

fn stable_domain_hash(value: &str) -> i64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in value.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    (hash & i64::MAX as u64) as i64
}

fn mix_rng_seed(world_seed: i64, stream_id: i64, domain: i64) -> u64 {
    let mut state = (world_seed as u64)
        ^ (stream_id as u64).rotate_left(21)
        ^ (domain as u64).rotate_left(42)
        ^ 0x9E37_79B9_7F4A_7C15;
    if state == 0 {
        state = 0xA076_1D64_78BD_642F;
    }
    state
}

fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

fn write_http_response(mut stream: TcpStream, resp: HttpResponseData) -> std::io::Result<()> {
    let bytes = format_http_response_bytes(&resp);
    stream.write_all(&bytes)?;
    let _ = stream.shutdown(std::net::Shutdown::Both);
    Ok(())
}

fn write_http_stream(
    mut stream: TcpStream,
    status: u16,
    mut headers: HashMap<String, String>,
    rx: mpsc::Receiver<StreamCommand>,
) -> std::io::Result<()> {
    headers
        .entry("transfer-encoding".to_string())
        .or_insert_with(|| "chunked".to_string());
    headers
        .entry("connection".to_string())
        .or_insert_with(|| "close".to_string());
    let reason = http_status_reason(status);
    let mut head = Vec::new();
    head.extend_from_slice(format!("HTTP/1.1 {} {}\r\n", status, reason).as_bytes());
    for (k, v) in headers {
        head.extend_from_slice(format!("{}: {}\r\n", header_key(&k), v).as_bytes());
    }
    head.extend_from_slice(b"\r\n");
    stream.write_all(&head)?;
    for cmd in rx {
        match cmd {
            StreamCommand::Data(bytes) => {
                if bytes.is_empty() {
                    continue;
                }
                let chunk_len = format!("{:X}\r\n", bytes.len());
                stream.write_all(chunk_len.as_bytes())?;
                stream.write_all(&bytes)?;
                stream.write_all(b"\r\n")?;
            }
            StreamCommand::Close => break,
        }
    }
    stream.write_all(b"0\r\n\r\n")?;
    let _ = stream.shutdown(std::net::Shutdown::Both);
    Ok(())
}

fn websocket_accept_key(client_key: &str) -> String {
    let mut hasher = Sha1::new();
    hasher.update(client_key.as_bytes());
    hasher.update(b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
    let digest = hasher.finalize();
    base64::engine::general_purpose::STANDARD.encode(digest)
}

fn write_websocket_session(
    mut stream: TcpStream,
    accept: &str,
    rx: mpsc::Receiver<WsCommand>,
    incoming: mpsc::Sender<WsIncoming>,
) -> std::io::Result<()> {
    let mut response = String::new();
    response.push_str("HTTP/1.1 101 Switching Protocols\r\n");
    response.push_str("Upgrade: websocket\r\n");
    response.push_str("Connection: Upgrade\r\n");
    response.push_str("Sec-WebSocket-Accept: ");
    response.push_str(accept);
    response.push_str("\r\n\r\n");
    stream.write_all(response.as_bytes())?;
    let _ = stream.set_read_timeout(Some(Duration::from_millis(25)));
    let mut closed = false;
    loop {
        while let Ok(cmd) = rx.try_recv() {
            match cmd {
                WsCommand::Text(text) => write_ws_frame(&mut stream, 0x1, text.as_bytes())?,
                WsCommand::Binary(bytes) => write_ws_frame(&mut stream, 0x2, &bytes)?,
                WsCommand::Close => {
                    let _ = write_ws_frame(&mut stream, 0x8, &[]);
                    closed = true;
                    break;
                }
            }
        }
        if closed {
            break;
        }
        match read_ws_frame(&mut stream) {
            Ok(Some(WsIncoming::Closed)) => {
                let _ = incoming.send(WsIncoming::Closed);
                let _ = write_ws_frame(&mut stream, 0x8, &[]);
                break;
            }
            Ok(Some(message)) => {
                let _ = incoming.send(message);
            }
            Ok(None) => {}
            Err(_) => {
                let _ = incoming.send(WsIncoming::Closed);
                let _ = write_ws_frame(&mut stream, 0x8, &[]);
                break;
            }
        }
    }
    let _ = incoming.send(WsIncoming::Closed);
    let _ = stream.shutdown(std::net::Shutdown::Both);
    Ok(())
}

fn read_ws_frame(stream: &mut TcpStream) -> std::io::Result<Option<WsIncoming>> {
    let mut header = [0u8; 2];
    match stream.read_exact(&mut header) {
        Ok(()) => {}
        Err(err)
            if err.kind() == std::io::ErrorKind::WouldBlock
                || err.kind() == std::io::ErrorKind::TimedOut =>
        {
            return Ok(None);
        }
        Err(err) => return Err(err),
    }
    let opcode = header[0] & 0x0f;
    let masked = (header[1] & 0x80) != 0;
    let mut payload_len = (header[1] & 0x7f) as usize;
    if payload_len == 126 {
        let mut ext = [0u8; 2];
        stream.read_exact(&mut ext)?;
        payload_len = u16::from_be_bytes(ext) as usize;
    } else if payload_len == 127 {
        let mut ext = [0u8; 8];
        stream.read_exact(&mut ext)?;
        payload_len = u64::from_be_bytes(ext) as usize;
    }
    let mut mask = [0u8; 4];
    if masked {
        stream.read_exact(&mut mask)?;
    }
    let mut payload = vec![0u8; payload_len];
    if payload_len > 0 {
        stream.read_exact(&mut payload)?;
    }
    if masked {
        for (idx, byte) in payload.iter_mut().enumerate() {
            *byte ^= mask[idx % 4];
        }
    }
    match opcode {
        0x1 => Ok(Some(WsIncoming::Text(
            String::from_utf8_lossy(&payload).to_string(),
        ))),
        0x2 => Ok(Some(WsIncoming::Binary(payload))),
        0x8 => Ok(Some(WsIncoming::Closed)),
        0x9 => {
            let _ = write_ws_frame(stream, 0xA, &payload);
            Ok(None)
        }
        0xA => Ok(None),
        _ => Ok(None),
    }
}

fn write_ws_frame(stream: &mut TcpStream, opcode: u8, payload: &[u8]) -> std::io::Result<()> {
    let mut header = Vec::with_capacity(14);
    header.push(0x80 | (opcode & 0x0f));
    let len = payload.len();
    if len <= 125 {
        header.push(len as u8);
    } else if len <= u16::MAX as usize {
        header.push(126);
        header.extend_from_slice(&(len as u16).to_be_bytes());
    } else {
        header.push(127);
        header.extend_from_slice(&(len as u64).to_be_bytes());
    }
    stream.write_all(&header)?;
    stream.write_all(payload)?;
    stream.flush()?;
    Ok(())
}

impl ServerLogger {
    fn append(&self, entry: &serde_json::Value) -> Result<(), RuntimeError> {
        let line =
            serde_json::to_string(entry).map_err(|e| RuntimeError::new(&e.to_string()))? + "\n";
        if let Some(parent) = self.path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|e| RuntimeError::new(&e.to_string()))?;
        file.write_all(line.as_bytes())
            .map_err(|e| RuntimeError::new(&e.to_string()))
    }
}

fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_millis() as u64
}

fn format_http_response_bytes(resp: &HttpResponseData) -> Vec<u8> {
    let reason = http_status_reason(resp.status);
    let mut headers = resp.headers.clone();
    headers
        .entry("content-length".to_string())
        .or_insert_with(|| resp.body.len().to_string());
    headers
        .entry("connection".to_string())
        .or_insert_with(|| "close".to_string());
    let mut out = Vec::new();
    out.extend_from_slice(format!("HTTP/1.1 {} {}\r\n", resp.status, reason).as_bytes());
    for (k, v) in headers {
        out.extend_from_slice(format!("{}: {}\r\n", header_key(&k), v).as_bytes());
    }
    out.extend_from_slice(b"\r\n");
    out.extend_from_slice(&resp.body);
    out
}

fn http_status_reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        201 => "Created",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "OK",
    }
}

fn header_key(key: &str) -> String {
    if key.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    let mut upper = true;
    for ch in key.chars() {
        if ch == '-' {
            upper = true;
            out.push(ch);
            continue;
        }
        if upper {
            out.extend(ch.to_uppercase());
            upper = false;
        } else {
            out.push(ch);
        }
    }
    out
}

fn read_http_request(stream: &mut TcpStream) -> Result<HttpRequestData, String> {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
    let mut buf = Vec::new();
    let mut temp = [0u8; 1024];
    let header_end = loop {
        let n = stream.read(&mut temp).map_err(|err| err.to_string())?;
        if n == 0 {
            return Err("connection closed".to_string());
        }
        buf.extend_from_slice(&temp[..n]);
        if let Some(pos) = find_header_end(&buf) {
            break pos;
        }
        if buf.len() > 1024 * 1024 {
            return Err("request too large".to_string());
        }
    };
    let header_bytes = &buf[..header_end];
    let mut body = buf[header_end + 4..].to_vec();
    let header_text = String::from_utf8_lossy(header_bytes);
    let mut lines = header_text.split("\r\n");
    let request_line = lines
        .next()
        .ok_or_else(|| "missing request line".to_string())?;
    let mut parts = request_line.split_whitespace();
    let method = parts
        .next()
        .ok_or_else(|| "missing method".to_string())?
        .to_string();
    let full_path = parts
        .next()
        .ok_or_else(|| "missing path".to_string())?
        .to_string();
    let mut headers = HashMap::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        if let Some((name, value)) = line.split_once(':') {
            headers.insert(name.trim().to_lowercase(), value.trim().to_string());
        }
    }
    let content_len = headers
        .get("content-length")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    while body.len() < content_len {
        let n = stream.read(&mut temp).map_err(|err| err.to_string())?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&temp[..n]);
    }
    let (path, query) = split_query(&full_path);
    Ok(HttpRequestData {
        method,
        path,
        query,
        headers,
        body,
        remote_addr: String::new(),
    })
}

fn find_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

fn split_query(path: &str) -> (String, String) {
    if let Some((p, q)) = path.split_once('?') {
        (p.to_string(), q.to_string())
    } else {
        (path.to_string(), String::new())
    }
}

fn http_request_thread(
    method: &str,
    url: &str,
    body: &[u8],
    opts: &HttpRequestOptions,
) -> Result<HttpResponseData, String> {
    let mut last_err = None;
    for attempt in 0..=opts.retries {
        match http_request_once(method, url, body, opts) {
            Ok(resp) => return Ok(resp),
            Err(err) => {
                last_err = Some(err);
                if attempt < opts.retries {
                    std::thread::sleep(Duration::from_millis(
                        opts.retry_backoff_ms.saturating_mul(attempt as u64 + 1),
                    ));
                }
            }
        }
    }
    Err(last_err.unwrap_or_else(|| "http request failed".to_string()))
}

fn http_request_once(
    method: &str,
    url: &str,
    body: &[u8],
    opts: &HttpRequestOptions,
) -> Result<HttpResponseData, String> {
    let (scheme, host, port, path) = parse_url(url)?;
    let request = build_http_request(method, &host, &path, body, &opts.headers);
    if scheme == "https" {
        return http_request_https(&host, port, &request, opts.timeout_ms);
    }
    let mut stream = connect_with_timeout(&host, port, opts.timeout_ms)?;
    if let Some(timeout) = opts.timeout_ms {
        let _ = stream.set_read_timeout(Some(Duration::from_millis(timeout)));
        let _ = stream.set_write_timeout(Some(Duration::from_millis(timeout)));
    }
    stream.write_all(&request).map_err(|err| err.to_string())?;
    let mut resp_buf = Vec::new();
    stream
        .read_to_end(&mut resp_buf)
        .map_err(|err| err.to_string())?;
    parse_http_response(&resp_buf)
}

fn connect_with_timeout(
    host: &str,
    port: u16,
    timeout_ms: Option<u64>,
) -> Result<TcpStream, String> {
    let addr = format!("{}:{}", host, port);
    if let Some(ms) = timeout_ms {
        let timeout = Duration::from_millis(ms);
        TcpStream::connect_timeout(&addr.parse().map_err(|_| "invalid address")?, timeout)
            .map_err(|err| err.to_string())
    } else {
        TcpStream::connect(addr).map_err(|err| err.to_string())
    }
}

fn http_request_https(
    host: &str,
    port: u16,
    request: &[u8],
    timeout_ms: Option<u64>,
) -> Result<HttpResponseData, String> {
    let addr = format!("{}:{}", host, port);
    let stream = if let Some(ms) = timeout_ms {
        let timeout = Duration::from_millis(ms);
        TcpStream::connect_timeout(&addr.parse().map_err(|_| "invalid address")?, timeout)
            .map_err(|err| err.to_string())?
    } else {
        TcpStream::connect(addr).map_err(|err| err.to_string())?
    };
    if let Some(timeout) = timeout_ms {
        let _ = stream.set_read_timeout(Some(Duration::from_millis(timeout)));
        let _ = stream.set_write_timeout(Some(Duration::from_millis(timeout)));
    }
    let connector = native_tls::TlsConnector::new().map_err(|err| err.to_string())?;
    let mut stream = connector
        .connect(host, stream)
        .map_err(|err| err.to_string())?;
    stream.write_all(request).map_err(|err| err.to_string())?;
    let mut resp_buf = Vec::new();
    stream
        .read_to_end(&mut resp_buf)
        .map_err(|err| err.to_string())?;
    parse_http_response(&resp_buf)
}

fn parse_url(url: &str) -> Result<(String, String, u16, String), String> {
    let (scheme, rest) = if let Some(rest) = url.strip_prefix("http://") {
        ("http", rest)
    } else if let Some(rest) = url.strip_prefix("https://") {
        ("https", rest)
    } else {
        ("http", url)
    };
    let (host_port, path) = if let Some((h, p)) = rest.split_once('/') {
        (h, format!("/{}", p))
    } else {
        (rest, "/".to_string())
    };
    if host_port.is_empty() {
        return Err("invalid url".to_string());
    }
    let (host, port) = if let Some((h, p)) = host_port.rsplit_once(':') {
        if let Ok(port) = p.parse::<u16>() {
            (h.to_string(), port)
        } else {
            (
                host_port.to_string(),
                if scheme == "https" { 443 } else { 80 },
            )
        }
    } else {
        (
            host_port.to_string(),
            if scheme == "https" { 443 } else { 80 },
        )
    };
    Ok((scheme.to_string(), host, port, path))
}

fn build_http_request(
    method: &str,
    host: &str,
    path: &str,
    body: &[u8],
    headers: &HashMap<String, String>,
) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(format!("{} {} HTTP/1.1\r\n", method, path).as_bytes());
    out.extend_from_slice(format!("Host: {}\r\n", host).as_bytes());
    out.extend_from_slice(b"Connection: close\r\n");
    for (k, v) in headers {
        out.extend_from_slice(format!("{}: {}\r\n", header_key(k), v).as_bytes());
    }
    if !body.is_empty() {
        out.extend_from_slice(format!("Content-Length: {}\r\n", body.len()).as_bytes());
    } else {
        out.extend_from_slice(b"Content-Length: 0\r\n");
    }
    out.extend_from_slice(b"\r\n");
    out.extend_from_slice(body);
    out
}

fn parse_http_response(buf: &[u8]) -> Result<HttpResponseData, String> {
    let header_end = find_header_end(buf).ok_or_else(|| "invalid http response".to_string())?;
    let header_text = String::from_utf8_lossy(&buf[..header_end]);
    let mut lines = header_text.split("\r\n");
    let status_line = lines
        .next()
        .ok_or_else(|| "missing status line".to_string())?;
    let mut parts = status_line.split_whitespace();
    let _version = parts.next().ok_or_else(|| "missing version".to_string())?;
    let status = parts
        .next()
        .ok_or_else(|| "missing status".to_string())?
        .parse::<u16>()
        .map_err(|_| "invalid status".to_string())?;
    let mut headers = HashMap::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        if let Some((name, value)) = line.split_once(':') {
            headers.insert(name.trim().to_lowercase(), value.trim().to_string());
        }
    }
    let body = buf[header_end + 4..].to_vec();
    Ok(HttpResponseData {
        status,
        headers,
        body,
    })
}

fn error_response(status: u16, code: &str, message: &str) -> HttpResponseData {
    let body = serde_json::json!({
        "error": {
            "code": code,
            "message": message
        }
    });
    let mut headers = HashMap::new();
    headers.insert("content-type".to_string(), "application/json".to_string());
    headers.insert("x-enkai-error-code".to_string(), code.to_string());
    HttpResponseData {
        status,
        headers,
        body: body.to_string().into_bytes(),
    }
}

fn route_method_match(route_method: &str, req_method: &str) -> bool {
    if route_method == "*" || route_method.eq_ignore_ascii_case("ANY") {
        return true;
    }
    route_method.eq_ignore_ascii_case(req_method)
}

fn match_route(pattern: &str, path: &str) -> Option<HashMap<String, String>> {
    if pattern == path {
        return Some(HashMap::new());
    }
    let pat = split_path(pattern);
    let actual = split_path(path);
    if pat.is_empty() && actual.is_empty() {
        return Some(HashMap::new());
    }
    if pat.len() != actual.len() {
        return None;
    }
    let mut params = HashMap::new();
    for (seg_pat, seg_val) in pat.iter().zip(actual.iter()) {
        if seg_pat.starts_with(':') {
            let key = seg_pat.trim_start_matches(':');
            if key.is_empty() {
                return None;
            }
            params.insert(key.to_string(), url_decode(seg_val));
        } else if seg_pat != seg_val {
            return None;
        }
    }
    Some(params)
}

fn split_path(path: &str) -> Vec<&str> {
    let trimmed = path.trim_matches('/');
    if trimmed.is_empty() {
        return Vec::new();
    }
    trimmed.split('/').filter(|s| !s.is_empty()).collect()
}

fn env_flag(name: &str) -> bool {
    match std::env::var(name) {
        Ok(value) => matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => false,
    }
}

fn model_selector_header(req: &HttpRequestData, key: &str) -> Option<String> {
    req.headers
        .get(key)
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .map(|value| value.to_string())
}

fn is_model_loaded_in_registry(
    registry_dir: &Path,
    name: &str,
    version: &str,
) -> Result<bool, String> {
    let path = registry_dir.join(".serve_state.json");
    if !path.is_file() {
        return Ok(false);
    }
    let text = std::fs::read_to_string(&path)
        .map_err(|err| format!("failed to read {}: {}", path.display(), err))?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .map_err(|err| format!("invalid {}: {}", path.display(), err))?;
    let Some(loaded_models) = value.get("loaded").and_then(|v| v.as_object()) else {
        return Err("model serve state missing `loaded` object".to_string());
    };
    let Some(loaded_versions) = loaded_models.get(name) else {
        return Ok(false);
    };
    if let Some(map) = loaded_versions.as_object() {
        return Ok(map.contains_key(version));
    }
    if let Some(list) = loaded_versions.as_array() {
        for item in list {
            if item.as_str().map(|v| v == version).unwrap_or(false) {
                return Ok(true);
            }
        }
        return Ok(false);
    }
    Err("model serve state has invalid version map".to_string())
}

fn parse_env_usize(name: &str) -> Option<usize> {
    let raw = std::env::var(name).ok()?;
    raw.trim().parse::<usize>().ok()
}

fn parse_non_negative_usize(value: &Value, error: &str) -> Result<usize, RuntimeError> {
    match value {
        Value::Int(i) if *i >= 0 => Ok(*i as usize),
        _ => Err(RuntimeError::new(error)),
    }
}

fn parse_bool_value(value: &Value, error: &str) -> Result<bool, RuntimeError> {
    match value {
        Value::Bool(v) => Ok(*v),
        _ => Err(RuntimeError::new(error)),
    }
}

fn resolve_correlation_id(req: &HttpRequestData, request_id: u64) -> String {
    let Some(raw) = req.headers.get("x-enkai-correlation-id") else {
        return format!("req-{}", request_id);
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed.len() > 128 {
        return format!("req-{}", request_id);
    }
    if trimmed
        .chars()
        .any(|ch| ch.is_control() || !(ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.')))
    {
        return format!("req-{}", request_id);
    }
    trimmed.to_string()
}

fn extract_auth_token(req: &HttpRequestData, auth: &ServerAuthConfig) -> Option<String> {
    let header = auth.header.to_lowercase();
    let value = req.headers.get(&header)?;
    let value = value.trim();
    if header == "authorization" {
        let lower = value.to_lowercase();
        if let Some(rest) = lower.strip_prefix("bearer ") {
            let offset = value.len().saturating_sub(rest.len());
            return Some(value[offset..].trim().to_string());
        }
    }
    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
}

fn remote_ip_only(remote_addr: &str) -> String {
    if let Ok(addr) = remote_addr.parse::<std::net::SocketAddr>() {
        return addr.ip().to_string();
    }
    remote_addr
        .rsplit_once(':')
        .map(|(host, _)| host.to_string())
        .unwrap_or_else(|| remote_addr.to_string())
}

fn rate_limit_key(
    key_kind: RateLimitKey,
    req: &HttpRequestData,
    token_value: Option<&str>,
    tenant: Option<&str>,
    model_name: Option<&str>,
    model_version: Option<&str>,
) -> String {
    let ip = remote_ip_only(&req.remote_addr);
    let token_or_ip = token_value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| ip.clone());
    let tenant_key = tenant
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "anonymous".to_string());
    let model_key = format!(
        "{}@{}",
        model_name.unwrap_or("unknown-model"),
        model_version.unwrap_or("unknown-version")
    );
    match key_kind {
        RateLimitKey::Ip => ip,
        RateLimitKey::Token => token_or_ip,
        RateLimitKey::Tenant => tenant_key,
        RateLimitKey::Model => model_key,
        RateLimitKey::TenantModel => format!("{}|{}", tenant_key, model_key),
    }
}

fn rate_limit_allow(
    state: &mut HashMap<String, RateLimitBucket>,
    key: &str,
    config: &RateLimitConfig,
) -> bool {
    let now = Instant::now();
    let bucket = state.entry(key.to_string()).or_insert(RateLimitBucket {
        tokens: config.capacity,
        last: now,
    });
    let elapsed = now.duration_since(bucket.last).as_secs_f64();
    if elapsed > 0.0 {
        bucket.tokens = (bucket.tokens + elapsed * config.refill_per_sec).min(config.capacity);
        bucket.last = now;
    }
    if bucket.tokens >= 1.0 {
        bucket.tokens -= 1.0;
        true
    } else {
        false
    }
}

fn tool_path_to_capability(tool_path: &str) -> Vec<String> {
    let mut out = Vec::new();
    for segment in tool_path.split('.') {
        let segment = segment.trim();
        if !segment.is_empty() {
            out.push(segment.to_string());
        }
    }
    if out.is_empty() {
        vec!["tool".to_string(), "invoke".to_string()]
    } else {
        out
    }
}

fn resolve_tool_command(tool_path: &str) -> Result<Vec<String>, RuntimeError> {
    let env_key = format!("ENKAI_TOOL_{}", sanitize_tool_env_key(tool_path));
    if let Ok(spec) = std::env::var(&env_key) {
        let command = parse_tool_command_spec(&spec)?;
        if command.is_empty() {
            return Err(RuntimeError::with_code(
                "E_TOOL_CONFIG",
                &format!("{} is set but empty", env_key),
            ));
        }
        return Ok(command);
    }
    if let Ok(spec) = std::env::var("ENKAI_TOOL_RUNNER") {
        let mut command = parse_tool_command_spec(&spec)?;
        if command.is_empty() {
            return Err(RuntimeError::with_code(
                "E_TOOL_CONFIG",
                "ENKAI_TOOL_RUNNER is set but empty",
            ));
        }
        command.push(tool_path.to_string());
        return Ok(command);
    }
    Err(RuntimeError::with_code(
        "E_TOOL_CONFIG",
        &format!(
            "Tool {} is not configured. Set {} or ENKAI_TOOL_RUNNER",
            tool_path, env_key
        ),
    ))
}

fn sanitize_tool_env_key(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_uppercase());
        } else {
            out.push('_');
        }
    }
    out
}

fn parse_tool_command_spec(spec: &str) -> Result<Vec<String>, RuntimeError> {
    let trimmed = spec.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    if trimmed.starts_with('[') {
        return serde_json::from_str::<Vec<String>>(trimmed).map_err(|err| {
            RuntimeError::with_code(
                "E_TOOL_CONFIG",
                &format!("Invalid tool command JSON array {}: {}", trimmed, err),
            )
        });
    }
    if matches!(
        std::env::var("ENKAI_TOOL_ALLOW_LEGACY_SPLIT")
            .ok()
            .as_deref()
            .map(|value| value.trim().to_ascii_lowercase()),
        Some(value) if matches!(value.as_str(), "1" | "true" | "yes" | "on")
    ) {
        return Ok(trimmed
            .split_whitespace()
            .map(|item| item.to_string())
            .collect());
    }
    Err(RuntimeError::with_code(
        "E_TOOL_CONFIG",
        "Tool command must be a JSON array (set ENKAI_TOOL_ALLOW_LEGACY_SPLIT=1 for legacy split mode)",
    ))
}

fn tool_timeout_ms() -> u64 {
    match std::env::var("ENKAI_TOOL_TIMEOUT_MS") {
        Ok(value) => value.trim().parse::<u64>().unwrap_or(30_000),
        Err(_) => 30_000,
    }
}

fn run_tool_process(
    command: &[String],
    payload: &[u8],
    timeout_ms: u64,
) -> Result<ToolProcessOutput, RuntimeError> {
    if command.is_empty() {
        return Err(RuntimeError::with_code(
            "E_TOOL_CONFIG",
            "tool command cannot be empty",
        ));
    }
    let stdout_path = tool_temp_path("stdout");
    let stderr_path = tool_temp_path("stderr");
    let stdout_file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&stdout_path)
        .map_err(|err| {
            RuntimeError::with_code("E_TOOL_IO", &format!("tool stdout open failed: {}", err))
        })?;
    let stderr_file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&stderr_path)
        .map_err(|err| {
            RuntimeError::with_code("E_TOOL_IO", &format!("tool stderr open failed: {}", err))
        })?;

    let mut child = Command::new(&command[0])
        .args(&command[1..])
        .stdin(Stdio::piped())
        .stdout(Stdio::from(stdout_file))
        .stderr(Stdio::from(stderr_file))
        .spawn()
        .map_err(|err| {
            RuntimeError::with_code("E_TOOL_SPAWN", &format!("tool spawn failed: {}", err))
        })?;

    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(payload).map_err(|err| {
            RuntimeError::with_code("E_TOOL_IO", &format!("tool stdin write failed: {}", err))
        })?;
    }

    let deadline = Instant::now() + Duration::from_millis(timeout_ms.max(1));
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    let _ = std::fs::remove_file(&stdout_path);
                    let _ = std::fs::remove_file(&stderr_path);
                    return Err(RuntimeError::with_code(
                        "E_TOOL_TIMEOUT",
                        &format!("tool timed out after {}ms", timeout_ms),
                    ));
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(err) => {
                let _ = std::fs::remove_file(&stdout_path);
                let _ = std::fs::remove_file(&stderr_path);
                return Err(RuntimeError::with_code(
                    "E_TOOL_WAIT",
                    &format!("tool wait failed: {}", err),
                ));
            }
        }
    };

    let stdout = std::fs::read(&stdout_path).map_err(|err| {
        RuntimeError::with_code("E_TOOL_IO", &format!("tool stdout read failed: {}", err))
    })?;
    let stderr = std::fs::read(&stderr_path).map_err(|err| {
        RuntimeError::with_code("E_TOOL_IO", &format!("tool stderr read failed: {}", err))
    })?;
    let _ = std::fs::remove_file(&stdout_path);
    let _ = std::fs::remove_file(&stderr_path);
    Ok((stdout, stderr, status.code()))
}

fn tool_temp_path(kind: &str) -> std::path::PathBuf {
    let mut path = std::env::temp_dir();
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_nanos();
    let seq = TOOL_IO_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
    path.push(format!(
        "enkai_tool_{}_{}_{}_{}.tmp",
        std::process::id(),
        now,
        seq,
        kind
    ));
    path
}

fn query_param(query: &str, name: &str) -> Option<String> {
    if query.is_empty() {
        return None;
    }
    for pair in query.split('&') {
        if pair.is_empty() {
            continue;
        }
        let mut it = pair.splitn(2, '=');
        let key = it.next().unwrap_or("");
        let val = it.next().unwrap_or("");
        if url_decode(key) == name {
            return Some(url_decode(val));
        }
    }
    None
}

fn url_decode(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut idx = 0usize;
    while idx < bytes.len() {
        match bytes[idx] {
            b'%' if idx + 2 < bytes.len() => {
                if let Ok(hex) = std::str::from_utf8(&bytes[idx + 1..idx + 3]) {
                    if let Ok(val) = u8::from_str_radix(hex, 16) {
                        out.push(val);
                        idx += 3;
                        continue;
                    }
                }
                out.push(bytes[idx]);
                idx += 1;
            }
            b'+' => {
                out.push(b' ');
                idx += 1;
            }
            b => {
                out.push(b);
                idx += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).to_string()
}

fn json_to_value(value: serde_json::Value) -> Value {
    match value {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Bool(b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int(i)
            } else if let Some(f) = n.as_f64() {
                Value::Float(f)
            } else {
                Value::Null
            }
        }
        serde_json::Value::String(s) => string_value(&s),
        serde_json::Value::Array(items) => {
            let values = items.into_iter().map(json_to_value).collect();
            Value::Obj(ObjRef::new(Obj::List(RefCell::new(values))))
        }
        serde_json::Value::Object(map) => {
            let mut out = HashMap::new();
            for (k, v) in map {
                out.insert(k, json_to_value(v));
            }
            record_value(out)
        }
    }
}

fn value_as_string(value: &Value) -> Result<String, RuntimeError> {
    match value {
        Value::Obj(obj) => match obj.as_obj() {
            Obj::String(s) => Ok(s.clone()),
            _ => Err(RuntimeError::new("Expected String value")),
        },
        _ => Err(RuntimeError::new("Expected String value")),
    }
}

fn value_as_int(value: &Value) -> Result<i64, RuntimeError> {
    match value {
        Value::Int(i) => Ok(*i),
        _ => Err(RuntimeError::new("Expected Int value")),
    }
}

fn value_as_non_negative_int(value: &Value, message: &str) -> Result<i64, RuntimeError> {
    let value = value_as_int(value)?;
    if value < 0 {
        return Err(RuntimeError::new(message));
    }
    Ok(value)
}

fn value_as_float_like(value: &Value) -> Result<f64, RuntimeError> {
    match value {
        Value::Int(i) => Ok(*i as f64),
        Value::Float(f) => Ok(*f),
        _ => Err(RuntimeError::new("Expected Float value")),
    }
}

fn value_as_bool(value: &Value) -> Result<bool, RuntimeError> {
    match value {
        Value::Bool(b) => Ok(*b),
        _ => Err(RuntimeError::new("Expected Bool value")),
    }
}

fn value_as_list(value: &Value) -> Result<Vec<Value>, RuntimeError> {
    match value {
        Value::Obj(obj) => match obj.as_obj() {
            Obj::List(values) => Ok(values.borrow().clone()),
            _ => Err(RuntimeError::new("Expected List value")),
        },
        _ => Err(RuntimeError::new("Expected List value")),
    }
}

fn value_as_buffer(value: &Value) -> Result<Vec<u8>, RuntimeError> {
    match value {
        Value::Obj(obj) => match obj.as_obj() {
            Obj::Buffer(bytes) => Ok(bytes.clone()),
            _ => Err(RuntimeError::new("Expected Buffer value")),
        },
        _ => Err(RuntimeError::new("Expected Buffer value")),
    }
}

fn value_as_record(
    value: &Value,
) -> Result<std::collections::HashMap<String, Value>, RuntimeError> {
    match value {
        Value::Obj(obj) => match obj.as_obj() {
            Obj::Record(map) => Ok(map.borrow().clone()),
            _ => Err(RuntimeError::new("Expected Record value")),
        },
        _ => Err(RuntimeError::new("Expected Record value")),
    }
}

fn value_as_dense_f64(value: &Value) -> Result<Vec<f64>, RuntimeError> {
    match value {
        Value::Obj(obj) => match obj.as_obj() {
            Obj::List(values) => values
                .borrow()
                .iter()
                .map(value_as_float_like)
                .collect::<Result<Vec<_>, _>>(),
            Obj::Buffer(bytes) => {
                if bytes.len() % 4 != 0 {
                    return Err(RuntimeError::new(
                        "Dense Buffer must be little-endian f32 bytes",
                    ));
                }
                let mut out = Vec::with_capacity(bytes.len() / 4);
                for chunk in bytes.chunks_exact(4) {
                    out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f64);
                }
                Ok(out)
            }
            _ => Err(RuntimeError::new("Expected dense List or Buffer value")),
        },
        _ => Err(RuntimeError::new("Expected dense List or Buffer value")),
    }
}

fn event_record_value(event: crate::object::ScheduledEvent) -> Value {
    sim_event_record_value(event)
}

fn sim_event_record_value(event: crate::object::ScheduledEvent) -> Value {
    record_value(HashMap::from([
        ("time".to_string(), Value::Float(event.time)),
        ("seq".to_string(), Value::Int(event.seq as i64)),
        ("event".to_string(), event.event),
    ]))
}

fn snapshot_event_record_value(
    event: crate::object::ScheduledEvent,
) -> Result<Value, RuntimeError> {
    Ok(record_value(HashMap::from([
        ("time".to_string(), Value::Float(event.time)),
        ("seq".to_string(), Value::Int(event.seq as i64)),
        ("event".to_string(), clone_snapshot_value(&event.event)?),
    ])))
}

fn clone_snapshot_value(value: &Value) -> Result<Value, RuntimeError> {
    match value {
        Value::Int(_) | Value::Float(_) | Value::Bool(_) | Value::Null => Ok(value.clone()),
        Value::Obj(obj) => match obj.as_obj() {
            Obj::String(text) => Ok(string_value(text)),
            Obj::Buffer(bytes) => Ok(buffer_value(bytes.clone())),
            Obj::Json(json) => Ok(Value::Obj(ObjRef::new(Obj::Json(json.clone())))),
            Obj::List(values) => Ok(Value::Obj(ObjRef::new(Obj::List(RefCell::new(
                values
                    .borrow()
                    .iter()
                    .map(clone_snapshot_value)
                    .collect::<Result<Vec<_>, _>>()?,
            ))))),
            Obj::Record(map) => {
                let map = map.borrow();
                let mut cloned = HashMap::with_capacity(map.len());
                for (key, value) in map.iter() {
                    cloned.insert(key.clone(), clone_snapshot_value(value)?);
                }
                Ok(record_value(cloned))
            }
            _ => Err(RuntimeError::with_code(
                "E_SIM_UNSNAPSHOTTABLE",
                &format!(
                    "simulation snapshots do not support runtime value type {}",
                    value.type_name()
                ),
            )),
        },
    }
}

fn scheduled_event_from_value(
    value: &Value,
) -> Result<crate::object::ScheduledEvent, RuntimeError> {
    let map = value_as_record(value).map_err(|_| {
        RuntimeError::with_code(
            "E_SIM_CORRUPTED_REPLAY",
            "simulation event entries must be records",
        )
    })?;
    let time = map
        .get("time")
        .ok_or_else(|| {
            RuntimeError::with_code(
                "E_SIM_CORRUPTED_REPLAY",
                "simulation event record is missing time",
            )
        })
        .and_then(value_as_float_like)?;
    if !time.is_finite() {
        return Err(RuntimeError::with_code(
            "E_SIM_CORRUPTED_REPLAY",
            "simulation event time must be finite",
        ));
    }
    let seq = map
        .get("seq")
        .ok_or_else(|| {
            RuntimeError::with_code(
                "E_SIM_CORRUPTED_REPLAY",
                "simulation event record is missing seq",
            )
        })
        .and_then(|value| value_as_non_negative_int(value, "simulation event seq must be >= 0"))?
        as u64;
    let event = map.get("event").ok_or_else(|| {
        RuntimeError::with_code(
            "E_SIM_CORRUPTED_REPLAY",
            "simulation event record is missing event payload",
        )
    })?;
    Ok(crate::object::ScheduledEvent {
        time,
        seq,
        event: clone_snapshot_value(event)?,
    })
}

fn value_to_token_ids(value: &Value) -> Result<Vec<u32>, RuntimeError> {
    match value {
        Value::Obj(obj) => match obj.as_obj() {
            Obj::Buffer(bytes) => bytes_to_ids(bytes)
                .map_err(|err| RuntimeError::new(&format!("Invalid token buffer: {}", err))),
            Obj::List(items) => {
                let items = items.borrow();
                let mut out = Vec::with_capacity(items.len());
                for item in items.iter() {
                    match item {
                        Value::Int(i) if *i >= 0 => out.push(*i as u32),
                        _ => return Err(RuntimeError::new("Token list must contain Int values")),
                    }
                }
                Ok(out)
            }
            _ => Err(RuntimeError::new("Tokens must be Buffer or List")),
        },
        _ => Err(RuntimeError::new("Tokens must be Buffer or List")),
    }
}

fn batch_to_value(batch: Batch) -> Value {
    let mut map = HashMap::new();
    map.insert(
        "input_ids".to_string(),
        Value::Obj(ObjRef::new(Obj::Buffer(ids_to_bytes(&batch.input_ids)))),
    );
    map.insert(
        "target_ids".to_string(),
        Value::Obj(ObjRef::new(Obj::Buffer(ids_to_bytes(&batch.target_ids)))),
    );
    map.insert(
        "attention_mask".to_string(),
        Value::Obj(ObjRef::new(Obj::Buffer(batch.attention_mask))),
    );
    map.insert(
        "batch_size".to_string(),
        Value::Int(batch.batch_size as i64),
    );
    map.insert("seq_len".to_string(), Value::Int(batch.seq_len as i64));
    map.insert(
        "token_count".to_string(),
        Value::Int(batch.token_count as i64),
    );
    map.insert(
        "packing_efficiency".to_string(),
        Value::Float(batch.packing_efficiency as f64),
    );
    record_value(map)
}

fn f32_to_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn buffer_to_f32(bytes: &[u8]) -> Result<Vec<f32>, RuntimeError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(RuntimeError::new("Invalid f32 buffer length"));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

fn bound_policy(bound: &Value) -> Option<String> {
    let obj = match bound {
        Value::Obj(obj) => obj,
        _ => return None,
    };
    let record = match obj.as_obj() {
        Obj::Record(map) => map.borrow(),
        _ => return None,
    };
    let kind = match record.get("__kind") {
        Some(Value::Obj(obj)) => match obj.as_obj() {
            Obj::String(s) => s,
            _ => return None,
        },
        _ => return None,
    };
    if kind != "agent" {
        return None;
    }
    match record.get("policy_name") {
        Some(Value::Obj(obj)) => match obj.as_obj() {
            Obj::String(s) => Some(s.clone()),
            _ => None,
        },
        _ => None,
    }
}

fn filters_match(filters: &[PolicyFilterRuntime], context: Option<&CapabilityContext>) -> bool {
    if filters.is_empty() {
        return true;
    }
    let ctx = match context {
        Some(ctx) => ctx,
        None => return false,
    };
    filters.iter().all(|filter| filter_matches(filter, ctx))
}

fn filter_matches(filter: &PolicyFilterRuntime, context: &CapabilityContext) -> bool {
    match (filter.name.as_str(), context) {
        ("path_prefix", CapabilityContext::Path(path)) => filter
            .values
            .iter()
            .any(|value| path_prefix_matches(value, path)),
        ("domain", CapabilityContext::Domain(domain)) => filter
            .values
            .iter()
            .any(|value| domain_matches(value, domain)),
        _ => false,
    }
}

fn parse_subset_module(source: &str) -> Result<Module, RuntimeError> {
    parse_module_named(source, Some("<enkai-lite>"))
        .map_err(|err| RuntimeError::new(&format!("subset parse failed: {}", err)))
}

fn compile_subset_program(source: &str) -> Result<Program, RuntimeError> {
    let module = parse_subset_module(source)?;
    validate_bootstrap_subset(&module)?;
    let mut checker = TypeChecker::new();
    checker
        .check_module(&module)
        .map_err(|err| RuntimeError::new(&format_type_error(&err)))?;

    // Compile through package loading to preserve import behavior (`std::*` and local imports)
    // while still enforcing subset rules on the entry source.
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    source.hash(&mut hasher);
    let source_hash = hasher.finish();
    let mut tmp = std::env::temp_dir();
    tmp.push(format!(
        "enkai_subset_{}_{}.enk",
        std::process::id(),
        source_hash
    ));
    std::fs::write(&tmp, source).map_err(|err| RuntimeError::new(&err.to_string()))?;
    let package = match load_package(&tmp) {
        Ok(package) => package,
        Err(err) => {
            let _ = std::fs::remove_file(&tmp);
            return Err(RuntimeError::new(&err.to_string()));
        }
    };
    let _ = std::fs::remove_file(&tmp);
    TypeChecker::check_package(&package)
        .map_err(|err| RuntimeError::new(&format_type_error(&err)))?;
    let mut program =
        compile_package(&package).map_err(|err| RuntimeError::new(&format_compile_error(&err)))?;
    for function in &mut program.functions {
        function.source_name = None;
    }
    Ok(program)
}

fn format_type_error(err: &enkaic::TypeError) -> String {
    if let Some(diagnostic) = err.diagnostic() {
        diagnostic.to_string()
    } else {
        format!(
            "Type error: {} at {}:{}",
            err.message, err.span.line, err.span.col
        )
    }
}

fn format_compile_error(err: &enkaic::compiler::CompileError) -> String {
    if let Some(diagnostic) = err.diagnostic() {
        diagnostic.to_string()
    } else if let Some(span) = &err.span {
        format!(
            "Compile error: {} at {}:{}",
            err.message, span.line, span.col
        )
    } else {
        format!("Compile error: {}", err.message)
    }
}

fn validate_bootstrap_subset(module: &Module) -> Result<(), RuntimeError> {
    for item in &module.items {
        match item {
            Item::Import(_) | Item::Use(_) | Item::Policy(_) | Item::Type(_) | Item::Enum(_) => {}
            Item::Fn(decl) => validate_subset_block(&decl.body)?,
            Item::Impl(decl) => {
                for method in &decl.methods {
                    validate_subset_block(&method.body)?;
                }
            }
            Item::Stmt(stmt) => validate_subset_stmt(stmt)?,
            Item::NativeImport(_) => {
                return Err(RuntimeError::new(
                    "subset disallows native::import declarations",
                ));
            }
            Item::Tool(_) | Item::Prompt(_) | Item::Model(_) | Item::Agent(_) => {
                return Err(RuntimeError::new(
                    "subset only supports import/use/policy/type/enum/impl/fn/stmts declarations",
                ));
            }
        }
    }
    Ok(())
}

fn validate_subset_block(block: &Block) -> Result<(), RuntimeError> {
    for stmt in &block.stmts {
        validate_subset_stmt(stmt)?;
    }
    Ok(())
}

fn validate_subset_stmt(stmt: &Stmt) -> Result<(), RuntimeError> {
    match stmt {
        Stmt::Let { expr, .. } => validate_subset_expr(expr),
        Stmt::Assign { target, expr } => {
            validate_subset_lvalue(target)?;
            validate_subset_expr(expr)
        }
        Stmt::Expr(expr) => validate_subset_expr(expr),
        Stmt::If {
            cond,
            then_block,
            else_branch,
        } => {
            validate_subset_expr(cond)?;
            validate_subset_block(then_block)?;
            if let Some(branch) = else_branch {
                match branch {
                    enkaic::ast::ElseBranch::Block(block) => validate_subset_block(block)?,
                    enkaic::ast::ElseBranch::If(stmt) => validate_subset_stmt(stmt)?,
                }
            }
            Ok(())
        }
        Stmt::While { cond, body } => {
            validate_subset_expr(cond)?;
            validate_subset_block(body)
        }
        Stmt::Return { expr } => {
            if let Some(expr) = expr {
                validate_subset_expr(expr)?;
            }
            Ok(())
        }
        Stmt::For { .. } => Err(RuntimeError::new("subset disallows for loops")),
        Stmt::Match { .. } => Err(RuntimeError::new("subset disallows match statements")),
        Stmt::Try { .. } => Err(RuntimeError::new("subset disallows try/catch statements")),
        Stmt::Break => Err(RuntimeError::new("subset disallows break statements")),
        Stmt::Continue => Err(RuntimeError::new("subset disallows continue statements")),
    }
}

fn validate_subset_lvalue(target: &LValue) -> Result<(), RuntimeError> {
    for access in &target.accesses {
        if let enkaic::ast::LValueAccess::Index(expr) = access {
            validate_subset_expr(expr)?;
        }
    }
    Ok(())
}

fn validate_subset_expr(expr: &Expr) -> Result<(), RuntimeError> {
    match expr {
        Expr::Literal { .. } | Expr::Ident { .. } => Ok(()),
        Expr::Binary { left, right, .. } => {
            validate_subset_expr(left)?;
            validate_subset_expr(right)
        }
        Expr::Unary { expr, .. } => validate_subset_expr(expr),
        Expr::Call { callee, args, .. } => {
            validate_subset_expr(callee)?;
            for arg in args {
                match arg {
                    Arg::Positional(expr) | Arg::Named(_, expr) => validate_subset_expr(expr)?,
                }
            }
            Ok(())
        }
        Expr::Index { target, index, .. } => {
            validate_subset_expr(target)?;
            validate_subset_expr(index)
        }
        Expr::Field { target, .. } => validate_subset_expr(target),
        Expr::List { items, .. } => {
            for item in items {
                validate_subset_expr(item)?;
            }
            Ok(())
        }
        Expr::Try { expr, .. } => validate_subset_expr(expr),
        Expr::Lambda { body, .. } => validate_subset_expr(body),
        Expr::Match { .. } => Err(RuntimeError::new("subset disallows match expressions")),
    }
}

fn capability_matches(rule: &[String], requested: &[String]) -> bool {
    if rule.len() > requested.len() {
        return false;
    }
    rule.iter().zip(requested.iter()).all(|(a, b)| a == b)
}

#[derive(Clone, Copy)]
struct LintIssue {
    line: i64,
    code: &'static str,
    message: &'static str,
}

fn collect_lint_issues(source: &str) -> Vec<LintIssue> {
    let mut items = Vec::new();
    for (idx, line) in source.lines().enumerate() {
        let line_no = (idx + 1) as i64;
        if line.ends_with(' ') || line.ends_with('\t') {
            items.push(LintIssue {
                line: line_no,
                code: "trailing_whitespace",
                message: "Line has trailing whitespace",
            });
        }
        if line.contains('\t') {
            items.push(LintIssue {
                line: line_no,
                code: "tab_indent",
                message: "Line uses tab indentation",
            });
        }
        if line.chars().count() > 120 {
            items.push(LintIssue {
                line: line_no,
                code: "line_too_long",
                message: "Line exceeds 120 characters",
            });
        }
        if line.contains("TODO") {
            items.push(LintIssue {
                line: line_no,
                code: "todo_marker",
                message: "TODO marker should be tracked as a task",
            });
        }
    }
    items
}

fn lint_issue(line: i64, code: &str, message: &str) -> Value {
    let mut map = HashMap::new();
    map.insert("line".to_string(), Value::Int(line));
    map.insert("code".to_string(), string_value(code));
    map.insert("message".to_string(), string_value(message));
    record_value(map)
}

fn domain_matches(pattern: &str, domain: &str) -> bool {
    let pattern = normalize_domain(pattern);
    let domain = normalize_domain(domain);
    domain.ends_with(&pattern)
}

fn normalize_domain(domain: &str) -> String {
    domain.trim().to_ascii_lowercase()
}

fn path_prefix_matches(prefix: &str, path: &str) -> bool {
    let prefix = normalize_path(prefix);
    let path = normalize_path(path);
    path.starts_with(&prefix)
}

fn normalize_path(path: &str) -> String {
    let mut parts = Vec::new();
    let mut prefix: Option<String> = None;
    let mut has_root = false;
    for component in Path::new(path).components() {
        match component {
            Component::Prefix(value) => {
                prefix = Some(value.as_os_str().to_string_lossy().replace('\\', "/"));
            }
            Component::RootDir => {
                has_root = true;
            }
            Component::CurDir => {}
            Component::ParentDir => {
                if let Some(last) = parts.last() {
                    if last != ".." {
                        parts.pop();
                        continue;
                    }
                }
                if !has_root {
                    parts.push("..".to_string());
                }
            }
            Component::Normal(value) => {
                parts.push(value.to_string_lossy().to_string());
            }
        }
    }
    let mut normalized = String::new();
    if let Some(prefix) = prefix {
        normalized.push_str(&prefix);
        if has_root {
            normalized.push('/');
        }
    } else if has_root {
        normalized.push('/');
    }
    if !normalized.ends_with('/') && !normalized.is_empty() && !parts.is_empty() {
        normalized.push('/');
    }
    normalized.push_str(&parts.join("/"));
    if cfg!(windows) {
        normalized = normalized.to_ascii_lowercase();
    }
    normalized
}

fn domain_from_url(url: &str) -> Option<String> {
    let url = url.trim();
    let without_scheme = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
        .unwrap_or(url);
    let host_port = without_scheme.split('/').next().unwrap_or(without_scheme);
    let host = host_port.split('@').next_back().unwrap_or(host_port);
    let host = host.split(':').next().unwrap_or(host);
    if host.is_empty() {
        None
    } else {
        Some(host.to_string())
    }
}

fn compare_lt(a: Value, b: Value) -> Result<Value, RuntimeError> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Ok(Value::Bool(x < y)),
        (Value::Int(x), Value::Float(y)) => Ok(Value::Bool((x as f64) < y)),
        (Value::Float(x), Value::Int(y)) => Ok(Value::Bool(x < (y as f64))),
        (Value::Float(x), Value::Float(y)) => Ok(Value::Bool(x < y)),
        _ => Err(RuntimeError::new("Operands must be numbers")),
    }
}

fn compare_gt(a: Value, b: Value) -> Result<Value, RuntimeError> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Ok(Value::Bool(x > y)),
        (Value::Int(x), Value::Float(y)) => Ok(Value::Bool((x as f64) > y)),
        (Value::Float(x), Value::Int(y)) => Ok(Value::Bool(x > (y as f64))),
        (Value::Float(x), Value::Float(y)) => Ok(Value::Bool(x > y)),
        _ => Err(RuntimeError::new("Operands must be numbers")),
    }
}

fn compare_le(a: Value, b: Value) -> Result<Value, RuntimeError> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Ok(Value::Bool(x <= y)),
        (Value::Int(x), Value::Float(y)) => Ok(Value::Bool((x as f64) <= y)),
        (Value::Float(x), Value::Int(y)) => Ok(Value::Bool(x <= (y as f64))),
        (Value::Float(x), Value::Float(y)) => Ok(Value::Bool(x <= y)),
        _ => Err(RuntimeError::new("Operands must be numbers")),
    }
}

fn compare_ge(a: Value, b: Value) -> Result<Value, RuntimeError> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Ok(Value::Bool(x >= y)),
        (Value::Int(x), Value::Float(y)) => Ok(Value::Bool((x as f64) >= y)),
        (Value::Float(x), Value::Int(y)) => Ok(Value::Bool(x >= (y as f64))),
        (Value::Float(x), Value::Float(y)) => Ok(Value::Bool(x >= y)),
        _ => Err(RuntimeError::new("Operands must be numbers")),
    }
}

fn display_value(v: &Value) -> String {
    match v {
        Value::Int(i) => i.to_string(),
        Value::Float(f) => f.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Null => "null".to_string(),
        Value::Obj(obj) => match obj.as_obj() {
            Obj::String(s) => s.clone(),
            Obj::Buffer(bytes) => format!("<buffer {} bytes>", bytes.len()),
            Obj::List(values) => format!("<list {} items>", values.borrow().len()),
            Obj::Json(_) => "<json>".to_string(),
            Obj::Function(f) => format!("<fn {}>", f.name.clone().unwrap_or_default()),
            Obj::BoundFunction(_) => "<bound_fn>".to_string(),
            Obj::NativeFunction(n) => format!("<native {}>", n.name),
            Obj::NativeHandle(_) => "<handle>".to_string(),
            Obj::SparseVector(inner) => format!("<sparse_vector {}>", inner.borrow().data.len()),
            Obj::SparseMatrix(inner) => format!("<sparse_matrix {}>", inner.borrow().data.len()),
            Obj::EventQueue(inner) => format!("<event_queue {}>", inner.borrow().items.len()),
            Obj::Pool(inner) => format!("<pool {}>", inner.borrow().items.len()),
            Obj::SimWorld(inner) => {
                let inner = inner.borrow();
                format!(
                    "<sim_world now={} pending={} entities={}>",
                    inner.now,
                    inner.queue.len(),
                    inner.entities.len()
                )
            }
            Obj::SimCoroutine(inner) => {
                let inner = inner.borrow();
                format!(
                    "<sim_coroutine task={} queued={} finished={}>",
                    inner.task_id,
                    inner.outputs.len(),
                    inner.finished
                )
            }
            Obj::SpatialIndex(inner) => {
                format!("<spatial_index {}>", inner.borrow().positions.len())
            }
            Obj::SnnNetwork(inner) => {
                let inner = inner.borrow();
                format!(
                    "<snn_network neurons={} synapses={}>",
                    inner.neuron_count,
                    self::display_snn_synapse_count(&inner.synapses)
                )
            }
            Obj::AgentEnv(inner) => format!("<agent_env {}>", inner.borrow().agents.len()),
            Obj::RngStream(inner) => {
                let inner = inner.borrow();
                format!("<rng_stream {}:{}>", inner.stream_id, inner.domain)
            }
            Obj::TaskHandle(id) => format!("<task {}>", id),
            Obj::Channel(_) => "<channel>".to_string(),
            Obj::TcpListener(_) => "<tcp_listener>".to_string(),
            Obj::TcpConnection(_) => "<tcp_connection>".to_string(),
            Obj::HttpStream(_) => "<http_stream>".to_string(),
            Obj::WebSocket(_) => "<websocket>".to_string(),
            Obj::Tokenizer(tok) => format!("<tokenizer {}>", tok.vocab_size()),
            Obj::DatasetStream(_) => "<dataset>".to_string(),
            Obj::Record(_) => "<record>".to_string(),
        },
    }
}

fn display_snn_synapse_count(value: &Value) -> usize {
    match value {
        Value::Obj(obj) => match obj.as_obj() {
            Obj::SparseMatrix(inner) => inner.borrow().data.len(),
            _ => 0,
        },
        _ => 0,
    }
}
