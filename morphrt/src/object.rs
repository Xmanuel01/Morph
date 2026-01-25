use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use morphc::bytecode::Program;

use crate::ffi::native_fn::FfiFunction;
use crate::value::{ObjRef, Value};

pub type NativeFunc =
    dyn Fn(&mut crate::vm::VM, &[Value]) -> Result<Value, crate::error::RuntimeError>;

#[derive(Debug)]
pub enum Obj {
    String(String),
    Buffer(Vec<u8>),
    List(Vec<Value>),
    Json(serde_json::Value),
    Function(FunctionObj),
    NativeFunction(NativeFunction),
    TaskHandle(usize),
    Channel(RefCell<ChannelState>),
    TcpListener(RefCell<std::net::TcpListener>),
    TcpConnection(RefCell<std::net::TcpStream>),
    Record(std::collections::HashMap<String, Value>),
}

impl Obj {
    pub fn type_name(&self) -> &'static str {
        match self {
            Obj::String(_) => "String",
            Obj::Buffer(_) => "Buffer",
            Obj::List(_) => "List",
            Obj::Json(_) => "Json",
            Obj::Function(_) => "Function",
            Obj::NativeFunction(_) => "NativeFunction",
            Obj::TaskHandle(_) => "TaskHandle",
            Obj::Channel(_) => "Channel",
            Obj::TcpListener(_) => "TcpListener",
            Obj::TcpConnection(_) => "TcpConnection",
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

#[derive(Clone)]
pub enum NativeImpl {
    Rust(Rc<NativeFunc>),
    Ffi(FfiFunction),
}

pub struct NativeFunction {
    pub name: String,
    pub arity: u16,
    pub kind: NativeImpl,
    pub bound: Option<Value>,
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
    Value::Obj(ObjRef::new(Obj::Record(map)))
}

pub fn task_handle_value(id: usize) -> Value {
    Value::Obj(ObjRef::new(Obj::TaskHandle(id)))
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
