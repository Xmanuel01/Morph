use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Component, Path};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use enkaic::bytecode::{Constant, Instruction, Program};

use crate::checkpoint::{
    latest_checkpoint, load_checkpoint, rotate_checkpoints, save_checkpoint, CheckpointMeta,
    CheckpointState,
};
use crate::dataset::{resolve_dataset_paths, Batch, DatasetConfig, DatasetStream};
use crate::error::{RuntimeError, RuntimeFrame};
use crate::ffi::FfiLoader;
use crate::object::{
    channel_value, function_value, record_value, string_value, task_handle_value, BoundFunctionObj,
    NativeFunction, NativeImpl, Obj,
};
use crate::tokenizer::{bytes_to_ids, ids_to_bytes, Tokenizer, TrainConfig};
use crate::value::{ObjRef, Value};

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

struct IoEvent {
    task_id: usize,
    result: IoResult,
}

struct ServerEvent {
    server_id: usize,
    request: HttpRequestData,
    stream: TcpStream,
}

struct HttpServer {
    handler: Value,
    stop: mpsc::Sender<()>,
}

#[derive(Clone)]
struct HttpRequestData {
    method: String,
    path: String,
    query: String,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}

#[derive(Clone)]
struct HttpResponseData {
    status: u16,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}

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
    policies: HashMap<String, Policy>,
    active_policy: Option<String>,
}

impl VM {
    pub fn new(trace: bool, disasm: bool, trace_task: bool, trace_net: bool) -> Self {
        let (io_sender, io_receiver) = mpsc::channel();
        let (server_sender, server_receiver) = mpsc::channel();
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
            policies: HashMap::new(),
            active_policy: None,
        }
    }

    pub fn run(&mut self, program: &Program) -> Result<Value, RuntimeError> {
        self.install_globals(program)?;
        if self.disasm {
            println!("{}", program.disassemble());
        }
        let main_func = function_value(program.main, program);
        let main_id = self.spawn_task_internal(program, main_func)?;
        self.scheduler_loop(program, main_id)
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
            "response".to_string(),
            Value::Obj(ObjRef::new(Obj::NativeFunction(NativeFunction {
                name: "http.response".to_string(),
                arity: 2,
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
        let json_value = record_value(json_record);
        if let Some(idx) = self.globals_map.get("json").copied() {
            self.globals[idx as usize] = json_value;
        } else {
            self.globals_map
                .insert("json".to_string(), self.globals.len() as u16);
            self.globals.push(json_value);
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
        std::mem::swap(&mut self.stack, &mut task.stack);
        std::mem::swap(&mut self.frames, &mut task.frames);
        let outcome = self.execute(program, budget);
        std::mem::swap(&mut self.stack, &mut task.stack);
        std::mem::swap(&mut self.frames, &mut task.frames);
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
        if let Some(task) = self.tasks.get_mut(task_id).and_then(|t| t.as_mut()) {
            task.state = TaskState::Finished;
            task.result = Some(result.clone());
            joiners.append(&mut task.join_waiters);
            http_conn = task.http_conn.take();
        }
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
            self.respond_http_task(stream, &result);
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
            let handler = match self.servers.get(event.server_id) {
                Some(server) => server.handler.clone(),
                None => {
                    self.write_http_error(stream, "Unknown server");
                    continue;
                }
            };
            let request = self.http_request_value(event.request);
            if let Err(err) =
                self.spawn_task_with_args(program, handler, vec![request], Some(stream))
            {
                eprintln!("Failed to spawn http handler: {}", err);
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
            let trace_frames = self.stack_trace(program, ip);
            let trace = |err: RuntimeError| err.with_frames(trace_frames.clone());
            if ip >= func.chunk.code.len() {
                return TaskRunOutcome::Errored(trace(RuntimeError::new(
                    "Instruction pointer out of bounds",
                )));
            }
            let instr = func.chunk.code[ip].clone();
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
                    let v = match self
                        .constant_to_value(&func.chunk.constants[idx as usize], program)
                    {
                        Ok(v) => v,
                        Err(err) => return TaskRunOutcome::Errored(trace(err)),
                    };
                    self.stack.push(v);
                }
                Instruction::Pop => {
                    self.stack.pop();
                }
                Instruction::DefineGlobal(idx) => {
                    let val = match self.stack.pop() {
                        Some(val) => val,
                        None => {
                            return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                "Stack underflow",
                            )))
                        }
                    };
                    let slot = idx as usize;
                    if slot >= self.globals.len() {
                        return TaskRunOutcome::Errored(trace(RuntimeError::new(
                            "Global not found",
                        )));
                    }
                    self.globals[slot] = val;
                }
                Instruction::LoadLocal(idx) => {
                    let val = match self.stack.get(base + idx as usize).cloned() {
                        Some(val) => val,
                        None => {
                            return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                "LoadLocal out of range",
                            )))
                        }
                    };
                    self.stack.push(val);
                }
                Instruction::StoreLocal(idx) => {
                    let val = match self.stack.pop() {
                        Some(val) => val,
                        None => {
                            return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                "Stack underflow",
                            )))
                        }
                    };
                    let slot = base + idx as usize;
                    if slot >= self.stack.len() {
                        self.stack.resize(slot + 1, Value::Null);
                    }
                    self.stack[slot] = val;
                }
                Instruction::LoadGlobal(idx) => {
                    let v = match self.globals.get(idx as usize).cloned() {
                        Some(v) => v,
                        None => {
                            return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                "Global not found",
                            )))
                        }
                    };
                    self.stack.push(v);
                }
                Instruction::StoreGlobal(idx) => {
                    let val = match self.stack.pop() {
                        Some(val) => val,
                        None => {
                            return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                "Stack underflow",
                            )))
                        }
                    };
                    let slot = idx as usize;
                    if slot >= self.globals.len() {
                        return TaskRunOutcome::Errored(trace(RuntimeError::new(
                            "Global not found",
                        )));
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
                            return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                "Stack underflow",
                            )))
                        }
                    };
                    let a = match self.stack.pop() {
                        Some(val) => val,
                        None => {
                            return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                "Stack underflow",
                            )))
                        }
                    };
                    let result = match instr {
                        Instruction::Add => match numeric_op(a, b, |x, y| x + y) {
                            Ok(v) => v,
                            Err(err) => return TaskRunOutcome::Errored(trace(err)),
                        },
                        Instruction::Sub => match numeric_op(a, b, |x, y| x - y) {
                            Ok(v) => v,
                            Err(err) => return TaskRunOutcome::Errored(trace(err)),
                        },
                        Instruction::Mul => match numeric_op(a, b, |x, y| x * y) {
                            Ok(v) => v,
                            Err(err) => return TaskRunOutcome::Errored(trace(err)),
                        },
                        Instruction::Div => match numeric_op(a, b, |x, y| x / y) {
                            Ok(v) => v,
                            Err(err) => return TaskRunOutcome::Errored(trace(err)),
                        },
                        Instruction::Mod => match numeric_mod(a, b) {
                            Ok(v) => v,
                            Err(err) => return TaskRunOutcome::Errored(trace(err)),
                        },
                        Instruction::Eq => Value::Bool(a == b),
                        Instruction::Neq => Value::Bool(a != b),
                        Instruction::Lt => match compare_op(a, b, |x, y| x < y) {
                            Ok(v) => v,
                            Err(err) => return TaskRunOutcome::Errored(trace(err)),
                        },
                        Instruction::Gt => match compare_op(a, b, |x, y| x > y) {
                            Ok(v) => v,
                            Err(err) => return TaskRunOutcome::Errored(trace(err)),
                        },
                        Instruction::Le => match compare_op(a, b, |x, y| x <= y) {
                            Ok(v) => v,
                            Err(err) => return TaskRunOutcome::Errored(trace(err)),
                        },
                        Instruction::Ge => match compare_op(a, b, |x, y| x >= y) {
                            Ok(v) => v,
                            Err(err) => return TaskRunOutcome::Errored(trace(err)),
                        },
                        _ => unreachable!(),
                    };
                    self.stack.push(result);
                }
                Instruction::Neg => {
                    let v = match self.stack.pop() {
                        Some(val) => val,
                        None => {
                            return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                "Stack underflow",
                            )))
                        }
                    };
                    let result = match v {
                        Value::Int(i) => Value::Int(-i),
                        Value::Float(f) => Value::Float(-f),
                        _ => {
                            return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                "Neg expects number",
                            )))
                        }
                    };
                    self.stack.push(result);
                }
                Instruction::Not => {
                    let v = match self.stack.pop() {
                        Some(val) => val,
                        None => {
                            return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                "Stack underflow",
                            )))
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
                            return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                "Stack underflow",
                            )))
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
                                return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                    "Stack underflow",
                                )))
                            }
                        };
                        let key = match self.stack.pop() {
                            Some(val) => val,
                            None => {
                                return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                    "Stack underflow",
                                )))
                            }
                        };
                        let key = match key {
                            Value::Obj(obj) => match obj.as_obj() {
                                Obj::String(s) => s.clone(),
                                _ => {
                                    return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                        "Record key must be string",
                                    )))
                                }
                            },
                            _ => {
                                return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                    "Record key must be string",
                                )))
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
                                return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                    "Stack underflow",
                                )))
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
                            return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                "Stack underflow",
                            )))
                        }
                    };
                    let name = match &func.chunk.constants[name_idx as usize] {
                        Constant::String(s) => s.clone(),
                        _ => {
                            return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                "Field name must be string constant",
                            )))
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
                                                RuntimeError::new("Unknown field"),
                                            ));
                                        }
                                    } else {
                                        return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                            "Unknown field",
                                        )));
                                    }
                                } else {
                                    return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                        "Unknown field",
                                    )));
                                }
                            }
                            Obj::TcpListener(_) => match name.as_str() {
                                "accept" => self.bound_native("net.accept", 0, target),
                                "port" => self.bound_native("net.listener.port", 0, target),
                                "close" => self.bound_native("net.listener.close", 0, target),
                                _ => {
                                    return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                        "Unknown field",
                                    )))
                                }
                            },
                            Obj::TcpConnection(_) => match name.as_str() {
                                "read" => self.bound_native("net.read", 1, target),
                                "read_all" => self.bound_native("net.read_all", 0, target),
                                "write" => self.bound_native("net.write", 1, target),
                                "close" => self.bound_native("net.close", 0, target),
                                _ => {
                                    return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                        "Unknown field",
                                    )))
                                }
                            },
                            Obj::Tokenizer(_) => match name.as_str() {
                                "encode" => self.bound_native("tokenizer.encode", 1, target),
                                "decode" => self.bound_native("tokenizer.decode", 1, target),
                                "save" => self.bound_native("tokenizer.save", 1, target),
                                _ => {
                                    return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                        "Unknown field",
                                    )))
                                }
                            },
                            Obj::DatasetStream(_) => match name.as_str() {
                                "next_batch" => self.bound_native("dataset.next_batch", 0, target),
                                _ => {
                                    return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                        "Unknown field",
                                    )))
                                }
                            },
                            _ => {
                                return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                    "Field access expects record",
                                )))
                            }
                        },
                        _ => {
                            return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                "Field access expects record",
                            )))
                        }
                    };
                    self.stack.push(value);
                }
                Instruction::GetIndex => {
                    let index = match self.stack.pop() {
                        Some(val) => val,
                        None => {
                            return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                "Stack underflow",
                            )))
                        }
                    };
                    let target = match self.stack.pop() {
                        Some(val) => val,
                        None => {
                            return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                "Stack underflow",
                            )))
                        }
                    };
                    let idx = match index {
                        Value::Int(i) => i,
                        _ => {
                            return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                "Index expects Int",
                            )))
                        }
                    };
                    let value = match target {
                        Value::Obj(obj) => match obj.as_obj() {
                            Obj::List(items) => {
                                if idx < 0 {
                                    return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                        "Index out of range",
                                    )));
                                }
                                let items = items.borrow();
                                let idx = idx as usize;
                                match items.get(idx).cloned() {
                                    Some(val) => val,
                                    None => {
                                        return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                            "Index out of range",
                                        )))
                                    }
                                }
                            }
                            _ => {
                                return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                    "Indexing expects a list",
                                )))
                            }
                        },
                        _ => {
                            return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                "Indexing expects a list",
                            )))
                        }
                    };
                    self.stack.push(value);
                }
                Instruction::SetField(name_idx) => {
                    let value = match self.stack.pop() {
                        Some(val) => val,
                        None => {
                            return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                "Stack underflow",
                            )))
                        }
                    };
                    let target = match self.stack.pop() {
                        Some(val) => val,
                        None => {
                            return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                "Stack underflow",
                            )))
                        }
                    };
                    let name = match &func.chunk.constants[name_idx as usize] {
                        Constant::String(s) => s.clone(),
                        _ => {
                            return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                "Field name must be string constant",
                            )))
                        }
                    };
                    match target {
                        Value::Obj(obj) => match obj.as_obj() {
                            Obj::Record(map) => {
                                map.borrow_mut().insert(name, value);
                            }
                            _ => {
                                return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                    "Field assignment expects record",
                                )))
                            }
                        },
                        _ => {
                            return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                "Field assignment expects record",
                            )))
                        }
                    }
                }
                Instruction::SetIndex => {
                    let value = match self.stack.pop() {
                        Some(val) => val,
                        None => {
                            return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                "Stack underflow",
                            )))
                        }
                    };
                    let index = match self.stack.pop() {
                        Some(val) => val,
                        None => {
                            return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                "Stack underflow",
                            )))
                        }
                    };
                    let target = match self.stack.pop() {
                        Some(val) => val,
                        None => {
                            return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                "Stack underflow",
                            )))
                        }
                    };
                    let idx = match index {
                        Value::Int(i) => i,
                        _ => {
                            return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                "Index expects Int",
                            )))
                        }
                    };
                    match target {
                        Value::Obj(obj) => match obj.as_obj() {
                            Obj::List(items) => {
                                if idx < 0 {
                                    return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                        "Index out of range",
                                    )));
                                }
                                let mut items = items.borrow_mut();
                                let idx = idx as usize;
                                if idx >= items.len() {
                                    return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                        "Index out of range",
                                    )));
                                }
                                items[idx] = value;
                            }
                            _ => {
                                return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                    "Index assignment expects list",
                                )))
                            }
                        },
                        _ => {
                            return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                "Index assignment expects list",
                            )))
                        }
                    }
                }
                Instruction::Call(argc) => {
                    if let Some(caller) = self.frames.last_mut() {
                        caller.ip = next_ip;
                    }
                    if let Err(err) = self.call_value(program, argc as usize) {
                        return TaskRunOutcome::Errored(trace(err));
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
                            return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                "Stack underflow",
                            )))
                        }
                    };
                    match value {
                        Value::Null => {
                            return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                "Tried to unwrap none",
                            )))
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
    ) -> Result<Option<(Vec<String>, Option<CapabilityContext>)>, RuntimeError> {
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
                if let Some(path) = args.get(0) {
                    context = Some(CapabilityContext::for_path(&value_as_string(path)?));
                }
            }
            "fsx_write_bytes" => {
                capability = Some(vec!["fs".to_string(), "write".to_string()]);
                if let Some(path) = args.get(0) {
                    context = Some(CapabilityContext::for_path(&value_as_string(path)?));
                }
            }
            "env_get" | "env_cwd" => {
                capability = Some(vec!["env".to_string(), "read".to_string()]);
            }
            "env_set" | "env_remove" | "env_set_cwd" => {
                capability = Some(vec!["env".to_string(), "write".to_string()]);
                if name == "env_set_cwd" {
                    if let Some(path) = args.get(0) {
                        context = Some(CapabilityContext::for_path(&value_as_string(path)?));
                    }
                }
            }
            "process_spawn" | "process_run" => {
                capability = Some(vec!["process".to_string(), "spawn".to_string()]);
                if let Some(cmd) = args.get(0) {
                    context = Some(CapabilityContext::for_path(&value_as_string(cmd)?));
                }
            }
            "process_wait" | "process_kill" => {
                capability = Some(vec!["process".to_string(), "control".to_string()]);
            }
            "process_exit" => {
                capability = Some(vec!["process".to_string(), "exit".to_string()]);
            }
            "net.bind" => {
                capability = Some(vec!["net".to_string(), "listen".to_string()]);
                if let Some(host) = args.get(0) {
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
                if let Some(url) = args.get(0) {
                    if let Some(domain) = domain_from_url(&value_as_string(url)?) {
                        context = Some(CapabilityContext::for_domain(&domain));
                    }
                }
            }
            "http.serve" => {
                capability = Some(vec!["net".to_string(), "serve".to_string()]);
                if let Some(host) = args.get(0) {
                    context = Some(CapabilityContext::for_domain(&value_as_string(host)?));
                }
            }
            "tokenizer.train" | "tokenizer.load" => {
                capability = Some(vec!["fs".to_string(), "read".to_string()]);
                if let Some(path) = args.get(0) {
                    context = Some(CapabilityContext::for_path(&value_as_string(path)?));
                }
            }
            "tokenizer.save" => {
                capability = Some(vec!["fs".to_string(), "write".to_string()]);
                if let Some(path) = args.get(0) {
                    context = Some(CapabilityContext::for_path(&value_as_string(path)?));
                }
            }
            "dataset.open" => {
                capability = Some(vec!["fs".to_string(), "read".to_string()]);
                if let Some(path) = args.get(0) {
                    context = Some(CapabilityContext::for_path(&value_as_string(path)?));
                }
            }
            "checkpoint.save" => {
                capability = Some(vec!["fs".to_string(), "write".to_string()]);
                if let Some(dir) = args.get(0) {
                    context = Some(CapabilityContext::for_path(&value_as_string(dir)?));
                }
            }
            "checkpoint.load" | "checkpoint.latest" => {
                capability = Some(vec!["fs".to_string(), "read".to_string()]);
                if let Some(dir) = args.get(0) {
                    context = Some(CapabilityContext::for_path(&value_as_string(dir)?));
                }
            }
            "checkpoint.rotate" => {
                capability = Some(vec!["fs".to_string(), "write".to_string()]);
                if let Some(dir) = args.get(0) {
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
            RuntimeError::new(&format!("Policy denied: {}", capability.join(".")))
        })?;
        let policy = self
            .policies
            .get(&policy_name)
            .ok_or_else(|| RuntimeError::new(&format!("Unknown policy: {}", policy_name)))?;
        if policy.is_allowed(capability, context) {
            Ok(())
        } else {
            Err(RuntimeError::new(&format!(
                "Policy denied: {}",
                capability.join(".")
            )))
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
        self.validate_http_handler(&handler)?;
        let addr = format!("{}:{}", host, port);
        let listener = TcpListener::bind(addr)
            .map_err(|err| RuntimeError::new(&format!("http.serve bind failed: {}", err)))?;
        listener
            .set_nonblocking(true)
            .map_err(|err| RuntimeError::new(&format!("http.serve failed: {}", err)))?;
        let server_id = self.servers.len();
        let (stop_sender, stop_receiver) = mpsc::channel();
        self.servers.push(HttpServer {
            handler,
            stop: stop_sender,
        });
        let sender = self.server_sender.clone();
        std::thread::spawn(move || loop {
            if stop_receiver.try_recv().is_ok() {
                break;
            }
            match listener.accept() {
                Ok((mut stream, _)) => {
                    if let Ok(req) = read_http_request(&mut stream) {
                        let _ = sender.send(ServerEvent {
                            server_id,
                            request: req,
                            stream,
                        });
                    } else {
                        let _ = write_http_response(
                            stream,
                            HttpResponseData {
                                status: 400,
                                headers: HashMap::new(),
                                body: b"Bad Request".to_vec(),
                            },
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

    fn http_get(&mut self, url: Value) -> Result<Option<Value>, RuntimeError> {
        self.http_request("GET", url, None)
    }

    fn http_post(&mut self, url: Value, body: Value) -> Result<Option<Value>, RuntimeError> {
        self.http_request("POST", url, Some(body))
    }

    fn http_request(
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
        let task_id = self
            .current_task
            .ok_or_else(|| RuntimeError::new("No current task"))?;
        let sender = self.io_sender.clone();
        let method = method.to_string();
        std::thread::spawn(move || {
            let result = http_request_thread(&method, &url, &body_bytes);
            let _ = sender.send(IoEvent {
                task_id,
                result: IoResult::HttpResponse(result),
            });
        });
        self.pending_state = Some(TaskState::BlockedIo);
        self.yield_now = true;
        Ok(None)
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

    fn respond_http_task(&mut self, stream: TcpStream, result: &Result<Value, RuntimeError>) {
        let response = match result {
            Ok(value) => match self.response_from_value(value.clone()) {
                Ok(resp) => resp,
                Err(err) => HttpResponseData {
                    status: 500,
                    headers: HashMap::new(),
                    body: err.message.into_bytes(),
                },
            },
            Err(err) => HttpResponseData {
                status: 500,
                headers: HashMap::new(),
                body: err.to_string().into_bytes(),
            },
        };
        std::thread::spawn(move || {
            let _ = write_http_response(stream, response);
        });
    }

    fn write_http_error(&self, stream: TcpStream, message: &str) {
        let response = HttpResponseData {
            status: 500,
            headers: HashMap::new(),
            body: message.as_bytes().to_vec(),
        };
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
                    let path = value_as_string(path_value)?;
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
                        save_path = Some(value_as_string(value)?);
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
        Ok(Value::Obj(ObjRef::new(Obj::DatasetStream(
            std::cell::RefCell::new(stream),
        ))))
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
                    kind: NativeImpl::Ffi(ffi),
                    bound: None,
                })))
            }
        })
    }
}

fn write_http_response(mut stream: TcpStream, resp: HttpResponseData) -> std::io::Result<()> {
    let bytes = format_http_response_bytes(&resp);
    stream.write_all(&bytes)?;
    let _ = stream.shutdown(std::net::Shutdown::Both);
    Ok(())
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

fn http_request_thread(method: &str, url: &str, body: &[u8]) -> Result<HttpResponseData, String> {
    let (host, port, path) = parse_url(url)?;
    let addr = format!("{}:{}", host, port);
    let mut stream = TcpStream::connect(addr).map_err(|err| err.to_string())?;
    let request = build_http_request(method, &host, &path, body);
    stream.write_all(&request).map_err(|err| err.to_string())?;
    let mut resp_buf = Vec::new();
    stream
        .read_to_end(&mut resp_buf)
        .map_err(|err| err.to_string())?;
    parse_http_response(&resp_buf)
}

fn parse_url(url: &str) -> Result<(String, u16, String), String> {
    let url = if let Some(rest) = url.strip_prefix("http://") {
        rest
    } else if url.starts_with("https://") {
        return Err("https not supported".to_string());
    } else {
        url
    };
    let (host_port, path) = if let Some((h, p)) = url.split_once('/') {
        (h, format!("/{}", p))
    } else {
        (url, "/".to_string())
    };
    if host_port.is_empty() {
        return Err("invalid url".to_string());
    }
    let (host, port) = if let Some((h, p)) = host_port.rsplit_once(':') {
        if let Ok(port) = p.parse::<u16>() {
            (h.to_string(), port)
        } else {
            (host_port.to_string(), 80)
        }
    } else {
        (host_port.to_string(), 80)
    };
    Ok((host, port, path))
}

fn build_http_request(method: &str, host: &str, path: &str, body: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(format!("{} {} HTTP/1.1\r\n", method, path).as_bytes());
    out.extend_from_slice(format!("Host: {}\r\n", host).as_bytes());
    out.extend_from_slice(b"Connection: close\r\n");
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

fn capability_matches(rule: &[String], requested: &[String]) -> bool {
    if rule.len() > requested.len() {
        return false;
    }
    rule.iter().zip(requested.iter()).all(|(a, b)| a == b)
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
    let host = host_port.split('@').last().unwrap_or(host_port);
    let host = host.split(':').next().unwrap_or(host);
    if host.is_empty() {
        None
    } else {
        Some(host.to_string())
    }
}

fn numeric_op<F>(a: Value, b: Value, f: F) -> Result<Value, RuntimeError>
where
    F: Fn(f64, f64) -> f64,
{
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Ok(Value::Int(f(x as f64, y as f64) as i64)),
        (Value::Int(x), Value::Float(y)) => Ok(Value::Float(f(x as f64, y))),
        (Value::Float(x), Value::Int(y)) => Ok(Value::Float(f(x, y as f64))),
        (Value::Float(x), Value::Float(y)) => Ok(Value::Float(f(x, y))),
        _ => Err(RuntimeError::new("Operands must be numbers")),
    }
}

fn numeric_mod(a: Value, b: Value) -> Result<Value, RuntimeError> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Ok(Value::Int(x % y)),
        _ => Err(RuntimeError::new("Modulo expects integers")),
    }
}

fn compare_op<F>(a: Value, b: Value, f: F) -> Result<Value, RuntimeError>
where
    F: Fn(f64, f64) -> bool,
{
    let result = match (a, b) {
        (Value::Int(x), Value::Int(y)) => f(x as f64, y as f64),
        (Value::Int(x), Value::Float(y)) => f(x as f64, y),
        (Value::Float(x), Value::Int(y)) => f(x, y as f64),
        (Value::Float(x), Value::Float(y)) => f(x, y),
        _ => return Err(RuntimeError::new("Operands must be numbers")),
    };
    Ok(Value::Bool(result))
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
            Obj::TaskHandle(id) => format!("<task {}>", id),
            Obj::Channel(_) => "<channel>".to_string(),
            Obj::TcpListener(_) => "<tcp_listener>".to_string(),
            Obj::TcpConnection(_) => "<tcp_connection>".to_string(),
            Obj::Tokenizer(tok) => format!("<tokenizer {}>", tok.vocab_size()),
            Obj::DatasetStream(_) => "<dataset>".to_string(),
            Obj::Record(_) => "<record>".to_string(),
        },
    }
}
