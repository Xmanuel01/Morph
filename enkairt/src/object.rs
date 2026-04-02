use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap, VecDeque};
use std::ffi::c_void;
use std::rc::Rc;

use enkaic::bytecode::Program;
use libffi::middle::{Arg, Cif, CodePtr};

use crate::dataset::DatasetStream;
use crate::ffi::native_fn::FfiFunction;
use crate::tokenizer::Tokenizer;
use crate::value::{ObjRef, Value};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

pub type NativeFunc =
    dyn Fn(&mut crate::vm::VM, &[Value]) -> Result<Value, crate::error::RuntimeError>;

#[derive(Debug)]
pub enum Obj {
    String(String),
    Buffer(Vec<u8>),
    List(RefCell<Vec<Value>>),
    Json(serde_json::Value),
    Function(FunctionObj),
    BoundFunction(BoundFunctionObj),
    NativeFunction(NativeFunction),
    NativeHandle(NativeHandle),
    SparseVector(Box<RefCell<SparseVectorState>>),
    SparseMatrix(Box<RefCell<SparseMatrixState>>),
    EventQueue(Box<RefCell<EventQueueState>>),
    Pool(Box<RefCell<ValuePoolState>>),
    SimWorld(Box<RefCell<SimWorldState>>),
    SimCoroutine(Box<RefCell<SimCoroutineState>>),
    TaskHandle(usize),
    Channel(RefCell<ChannelState>),
    TcpListener(RefCell<std::net::TcpListener>),
    TcpConnection(RefCell<std::net::TcpStream>),
    HttpStream(HttpStream),
    WebSocket(WebSocketHandle),
    Tokenizer(Tokenizer),
    DatasetStream(Box<RefCell<DatasetStream>>),
    Record(RefCell<std::collections::HashMap<String, Value>>),
}

impl Obj {
    pub fn type_name(&self) -> &'static str {
        match self {
            Obj::String(_) => "String",
            Obj::Buffer(_) => "Buffer",
            Obj::List(_) => "List",
            Obj::Json(_) => "Json",
            Obj::Function(_) => "Function",
            Obj::BoundFunction(_) => "BoundFunction",
            Obj::NativeFunction(_) => "NativeFunction",
            Obj::NativeHandle(_) => "Handle",
            Obj::SparseVector(_) => "SparseVector",
            Obj::SparseMatrix(_) => "SparseMatrix",
            Obj::EventQueue(_) => "EventQueue",
            Obj::Pool(_) => "Pool",
            Obj::SimWorld(_) => "SimWorld",
            Obj::SimCoroutine(_) => "SimCoroutine",
            Obj::TaskHandle(_) => "TaskHandle",
            Obj::Channel(_) => "Channel",
            Obj::TcpListener(_) => "TcpListener",
            Obj::TcpConnection(_) => "TcpConnection",
            Obj::HttpStream(_) => "HttpStream",
            Obj::WebSocket(_) => "WebSocket",
            Obj::Tokenizer(_) => "Tokenizer",
            Obj::DatasetStream(_) => "DatasetStream",
            Obj::Record(_) => "Record",
        }
    }
}

#[derive(Debug)]
pub struct FunctionObj {
    pub name: Option<String>,
    pub arity: u16,
    pub func_index: u16,
}

#[derive(Debug)]
pub struct BoundFunctionObj {
    pub func_index: u16,
    pub arity: u16,
    pub bound: Value,
}

#[derive(Clone)]
pub enum NativeImpl {
    Rust(Rc<NativeFunc>),
    Ffi(Box<FfiFunction>),
}

pub struct NativeFunction {
    pub name: String,
    pub arity: u16,
    pub kind: NativeImpl,
    pub bound: Option<Value>,
}

#[derive(Debug)]
pub struct NativeHandleDrop {
    pub _library: Arc<libloading::Library>,
    pub free_ptr: CodePtr,
    pub free_cif: Cif,
}

#[derive(Debug)]
pub struct NativeHandle {
    pub ptr: *mut c_void,
    pub dropper: Rc<NativeHandleDrop>,
}

impl Drop for NativeHandle {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        let ptr_args: Vec<*const c_void> = vec![self.ptr as *const c_void];
        let args = vec![Arg::new(ptr_args.last().unwrap())];
        let _: () = unsafe { self.dropper.free_cif.call(self.dropper.free_ptr, &args) };
        self.ptr = std::ptr::null_mut();
    }
}

#[derive(Debug, Default)]
pub struct SparseVectorState {
    pub data: BTreeMap<i64, f64>,
    pub native: Option<Value>,
}

#[derive(Debug, Default)]
pub struct SparseMatrixState {
    pub data: BTreeMap<(i64, i64), f64>,
    pub native: Option<Value>,
}

#[derive(Debug, Clone)]
pub struct ScheduledEvent {
    pub time: f64,
    pub seq: u64,
    pub event: Value,
}

impl PartialEq for ScheduledEvent {
    fn eq(&self, other: &Self) -> bool {
        self.time.to_bits() == other.time.to_bits() && self.seq == other.seq
    }
}

impl Eq for ScheduledEvent {}

impl PartialOrd for ScheduledEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScheduledEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .time
            .partial_cmp(&self.time)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

#[derive(Debug, Default)]
pub struct EventQueueState {
    pub next_seq: u64,
    pub items: BinaryHeap<ScheduledEvent>,
    pub native: Option<Value>,
    pub payloads: BTreeMap<u64, Value>,
}

#[derive(Debug)]
pub struct ValuePoolState {
    pub items: Vec<Value>,
    pub capacity: usize,
    pub growable: bool,
    pub acquire_hits: u64,
    pub acquire_misses: u64,
    pub releases: u64,
    pub dropped_on_full: u64,
    pub high_watermark: usize,
    pub native: Option<Value>,
}

impl ValuePoolState {
    pub fn new(capacity: usize, growable: bool) -> Self {
        Self {
            items: Vec::with_capacity(capacity),
            capacity,
            growable,
            acquire_hits: 0,
            acquire_misses: 0,
            releases: 0,
            dropped_on_full: 0,
            high_watermark: 0,
            native: None,
        }
    }
}

#[derive(Debug)]
pub struct SimWorldState {
    pub seed: i64,
    pub now: f64,
    pub max_events: usize,
    pub next_seq: u64,
    pub queue: BinaryHeap<ScheduledEvent>,
    pub log: Vec<ScheduledEvent>,
    pub entities: BTreeMap<i64, Value>,
}

impl SimWorldState {
    pub fn new(max_events: usize, seed: i64) -> Self {
        Self {
            seed,
            now: 0.0,
            max_events,
            next_seq: 0,
            queue: BinaryHeap::new(),
            log: Vec::new(),
            entities: BTreeMap::new(),
        }
    }
}

#[derive(Debug)]
pub struct SimCoroutineState {
    pub world: Value,
    pub state: Option<Value>,
    pub task_id: usize,
    pub outputs: VecDeque<Value>,
    pub waiters: VecDeque<usize>,
    pub finished: bool,
    pub emitted: u64,
}

impl SimCoroutineState {
    pub fn new(world: Value, state: Option<Value>, task_id: usize) -> Self {
        Self {
            world,
            state,
            task_id,
            outputs: VecDeque::new(),
            waiters: VecDeque::new(),
            finished: false,
            emitted: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub enum StreamCommand {
    Data(Vec<u8>),
    Close,
}

#[derive(Debug, Clone)]
pub struct HttpStream {
    pub sender: mpsc::Sender<StreamCommand>,
}

#[derive(Debug, Clone)]
pub enum WsCommand {
    Text(String),
    Binary(Vec<u8>),
    Close,
}

#[derive(Debug, Clone)]
pub enum WsIncoming {
    Text(String),
    Binary(Vec<u8>),
    Closed,
}

#[derive(Debug, Clone)]
pub struct WebSocketHandle {
    pub sender: mpsc::Sender<WsCommand>,
    pub incoming: Arc<Mutex<mpsc::Receiver<WsIncoming>>>,
}

impl std::fmt::Debug for NativeFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeFunction")
            .field("name", &self.name)
            .field("arity", &self.arity)
            .finish()
    }
}

pub fn function_value(func_index: u16, program: &Program) -> Value {
    let f = &program.functions[func_index as usize];
    Value::Obj(ObjRef::new(Obj::Function(FunctionObj {
        name: f.name.clone(),
        arity: f.arity,
        func_index,
    })))
}

pub fn string_value(s: &str) -> Value {
    Value::Obj(ObjRef::new(Obj::String(s.to_string())))
}

pub fn buffer_value(bytes: Vec<u8>) -> Value {
    Value::Obj(ObjRef::new(Obj::Buffer(bytes)))
}

pub fn record_value(map: std::collections::HashMap<String, Value>) -> Value {
    Value::Obj(ObjRef::new(Obj::Record(RefCell::new(map))))
}

pub fn task_handle_value(id: usize) -> Value {
    Value::Obj(ObjRef::new(Obj::TaskHandle(id)))
}

pub fn native_handle_value(ptr: *mut c_void, dropper: Rc<NativeHandleDrop>) -> Value {
    Value::Obj(ObjRef::new(Obj::NativeHandle(NativeHandle {
        ptr,
        dropper,
    })))
}

pub fn sparse_vector_value() -> Value {
    sparse_vector_value_with_native(None)
}

pub fn sparse_vector_value_with_native(native: Option<Value>) -> Value {
    Value::Obj(ObjRef::new(Obj::SparseVector(Box::new(RefCell::new(
        SparseVectorState {
            native,
            ..SparseVectorState::default()
        },
    )))))
}

pub fn sparse_matrix_value() -> Value {
    sparse_matrix_value_with_native(None)
}

pub fn sparse_matrix_value_with_native(native: Option<Value>) -> Value {
    Value::Obj(ObjRef::new(Obj::SparseMatrix(Box::new(RefCell::new(
        SparseMatrixState {
            native,
            ..SparseMatrixState::default()
        },
    )))))
}

pub fn event_queue_value() -> Value {
    event_queue_value_with_native(None)
}

pub fn event_queue_value_with_native(native: Option<Value>) -> Value {
    Value::Obj(ObjRef::new(Obj::EventQueue(Box::new(RefCell::new(
        EventQueueState {
            native,
            ..EventQueueState::default()
        },
    )))))
}

pub fn pool_value(capacity: usize, growable: bool) -> Value {
    pool_value_with_native(capacity, growable, None)
}

pub fn pool_value_with_native(capacity: usize, growable: bool, native: Option<Value>) -> Value {
    Value::Obj(ObjRef::new(Obj::Pool(Box::new(RefCell::new(
        ValuePoolState {
            native,
            ..ValuePoolState::new(capacity, growable)
        },
    )))))
}

pub fn sim_world_value(max_events: usize, seed: i64) -> Value {
    Value::Obj(ObjRef::new(Obj::SimWorld(Box::new(RefCell::new(
        SimWorldState::new(max_events, seed),
    )))))
}

pub fn sim_coroutine_value(world: Value, state: Option<Value>, task_id: usize) -> Value {
    Value::Obj(ObjRef::new(Obj::SimCoroutine(Box::new(RefCell::new(
        SimCoroutineState::new(world, state, task_id),
    )))))
}

#[derive(Debug)]
pub struct ChannelState {
    pub queue: VecDeque<Value>,
    pub waiters: VecDeque<usize>,
}

impl ChannelState {
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            waiters: VecDeque::new(),
        }
    }
}

impl Default for ChannelState {
    fn default() -> Self {
        Self::new()
    }
}

pub fn channel_value() -> Value {
    Value::Obj(ObjRef::new(Obj::Channel(RefCell::new(ChannelState::new()))))
}
