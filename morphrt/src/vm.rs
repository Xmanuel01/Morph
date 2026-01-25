use std::collections::{HashMap, VecDeque};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use morphc::bytecode::{Constant, Instruction, Program};

use crate::error::{RuntimeError, RuntimeFrame};
use crate::ffi::FfiLoader;
use crate::object::{
    channel_value, function_value, record_value, string_value, task_handle_value, NativeFunction,
    NativeImpl, Obj,
};
use crate::value::{ObjRef, Value};

#[derive(Debug)]
struct CallFrame {
    func_index: u16,
    ip: usize,
    base: usize,      // start of locals/args
    caller_sp: usize, // stack height where callee was placed
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
        std::mem::swap(&mut self.stack, &mut task.stack);
        std::mem::swap(&mut self.frames, &mut task.frames);
        let outcome = self.execute(program, budget);
        std::mem::swap(&mut self.stack, &mut task.stack);
        std::mem::swap(&mut self.frames, &mut task.frames);
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
                            Obj::Record(map) => match map.get(&name).cloned() {
                                Some(val) => val,
                                None => {
                                    return TaskRunOutcome::Errored(trace(RuntimeError::new(
                                        "Unknown field",
                                    )))
                                }
                            },
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
                    self.frames.pop().unwrap();
                    self.stack.truncate(caller_sp);
                    self.stack.push(ret);
                    continue;
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
            Value::Obj(obj) => match obj.as_obj() {
                Obj::Function(f) => {
                    if argc as u16 != f.arity {
                        return Err(RuntimeError::new("Arity mismatch"));
                    }
                    self.frames.push(CallFrame {
                        func_index: f.func_index,
                        ip: 0,
                        base: callee_index + 1, // locals start at first arg
                        caller_sp: callee_index,
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
                    let result = match &nf.kind {
                        NativeImpl::Rust(func) => (func)(self, &args)?,
                        NativeImpl::Ffi(func) => func.call(&args)?,
                    };
                    self.stack.truncate(callee_index);
                    self.stack.push(result);
                }
                _ => return Err(RuntimeError::new("Callee is not callable")),
            },
            _ => return Err(RuntimeError::new("Callee is not callable")),
        }
        Ok(())
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
                            for (k, v) in hmap {
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
                    let mut out = Vec::with_capacity(items.len());
                    for item in items {
                        out.push(self.value_to_json(item.clone())?);
                    }
                    Ok(serde_json::Value::Array(out))
                }
                Obj::Record(map) => {
                    let mut out = serde_json::Map::new();
                    for (k, v) in map {
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
        serde_json::Value::Array(items) => Value::Obj(ObjRef::new(Obj::List(
            items.into_iter().map(json_to_value).collect(),
        ))),
        serde_json::Value::Object(map) => {
            let mut out = HashMap::new();
            for (k, v) in map {
                out.insert(k, json_to_value(v));
            }
            record_value(out)
        }
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
            Obj::List(values) => format!("<list {} items>", values.len()),
            Obj::Json(_) => "<json>".to_string(),
            Obj::Function(f) => format!("<fn {}>", f.name.clone().unwrap_or_default()),
            Obj::NativeFunction(n) => format!("<native {}>", n.name),
            Obj::TaskHandle(id) => format!("<task {}>", id),
            Obj::Channel(_) => "<channel>".to_string(),
            Obj::TcpListener(_) => "<tcp_listener>".to_string(),
            Obj::TcpConnection(_) => "<tcp_connection>".to_string(),
            Obj::Record(_) => "<record>".to_string(),
        },
    }
}
