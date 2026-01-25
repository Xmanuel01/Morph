use std::collections::HashMap;

use morphc::bytecode::{Constant, Instruction, Program};

use crate::error::{RuntimeError, RuntimeFrame};
use crate::object::{function_value, record_value, string_value, NativeFunction, Obj};
use crate::value::{ObjRef, Value};

#[derive(Debug)]
struct CallFrame {
    func_index: u16,
    ip: usize,
    base: usize,      // start of locals/args
    caller_sp: usize, // stack height where callee was placed
}

pub struct VM {
    stack: Vec<Value>,
    frames: Vec<CallFrame>,
    globals: Vec<Value>,
    globals_map: HashMap<String, u16>,
    trace: bool,
    disasm: bool,
}

impl VM {
    pub fn new(trace: bool, disasm: bool) -> Self {
        Self {
            stack: Vec::new(),
            frames: Vec::new(),
            globals: Vec::new(),
            globals_map: HashMap::new(),
            trace,
            disasm,
        }
    }

    pub fn run(&mut self, program: &Program) -> Result<Value, RuntimeError> {
        self.install_globals(program)?;
        if self.disasm {
            println!("{}", program.disassemble());
        }
        let main_func = function_value(program.main, program);
        self.stack.push(main_func);
        self.call_value(program, 0)
            .map_err(|err| self.with_trace(err, program, 0))?;
        self.execute(program)
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
            func: std::rc::Rc::new(|_, args| {
                println!("{}", display_value(&args[0]));
                Ok(Value::Null)
            }),
        })));
        if let Some(idx) = self.globals_map.get("print").copied() {
            self.globals[idx as usize] = print;
        } else {
            self.globals_map
                .insert("print".to_string(), self.globals.len() as u16);
            self.globals.push(print);
        }
        Ok(())
    }

    fn execute(&mut self, program: &Program) -> Result<Value, RuntimeError> {
        while let Some(frame_view) = self.frames.last() {
            let func_index = frame_view.func_index;
            let ip = frame_view.ip;
            let base = frame_view.base;
            let caller_sp = frame_view.caller_sp;
            let func = &program.functions[func_index as usize];
            let trace_frames = self.stack_trace(program, ip);
            let trace = |err: RuntimeError| err.with_frames(trace_frames.clone());
            if ip >= func.chunk.code.len() {
                return Err(trace(RuntimeError::new(
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
                    let v = self
                        .constant_to_value(&func.chunk.constants[idx as usize], program)
                        .map_err(trace)?;
                    self.stack.push(v);
                }
                Instruction::Pop => {
                    self.stack.pop();
                }
                Instruction::DefineGlobal(idx) => {
                    let val = self
                        .stack
                        .pop()
                        .ok_or_else(|| trace(RuntimeError::new("Stack underflow")))?;
                    let slot = idx as usize;
                    if slot >= self.globals.len() {
                        return Err(trace(RuntimeError::new("Global not found")));
                    }
                    self.globals[slot] = val;
                }
                Instruction::LoadLocal(idx) => {
                    let val = self
                        .stack
                        .get(base + idx as usize)
                        .cloned()
                        .ok_or_else(|| trace(RuntimeError::new("LoadLocal out of range")))?;
                    self.stack.push(val);
                }
                Instruction::StoreLocal(idx) => {
                    let val = self
                        .stack
                        .pop()
                        .ok_or_else(|| trace(RuntimeError::new("Stack underflow")))?;
                    let slot = base + idx as usize;
                    if slot >= self.stack.len() {
                        self.stack.resize(slot + 1, Value::Null);
                    }
                    self.stack[slot] = val;
                }
                Instruction::LoadGlobal(idx) => {
                    let v = self
                        .globals
                        .get(idx as usize)
                        .cloned()
                        .ok_or_else(|| trace(RuntimeError::new("Global not found")))?;
                    self.stack.push(v);
                }
                Instruction::StoreGlobal(idx) => {
                    let val = self
                        .stack
                        .pop()
                        .ok_or_else(|| trace(RuntimeError::new("Stack underflow")))?;
                    let slot = idx as usize;
                    if slot >= self.globals.len() {
                        return Err(trace(RuntimeError::new("Global not found")));
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
                    let b = self
                        .stack
                        .pop()
                        .ok_or_else(|| trace(RuntimeError::new("Stack underflow")))?;
                    let a = self
                        .stack
                        .pop()
                        .ok_or_else(|| trace(RuntimeError::new("Stack underflow")))?;
                    let result = match instr {
                        Instruction::Add => numeric_op(a, b, |x, y| x + y).map_err(trace)?,
                        Instruction::Sub => numeric_op(a, b, |x, y| x - y).map_err(trace)?,
                        Instruction::Mul => numeric_op(a, b, |x, y| x * y).map_err(trace)?,
                        Instruction::Div => numeric_op(a, b, |x, y| x / y).map_err(trace)?,
                        Instruction::Mod => numeric_mod(a, b).map_err(trace)?,
                        Instruction::Eq => Value::Bool(a == b),
                        Instruction::Neq => Value::Bool(a != b),
                        Instruction::Lt => compare_op(a, b, |x, y| x < y).map_err(trace)?,
                        Instruction::Gt => compare_op(a, b, |x, y| x > y).map_err(trace)?,
                        Instruction::Le => compare_op(a, b, |x, y| x <= y).map_err(trace)?,
                        Instruction::Ge => compare_op(a, b, |x, y| x >= y).map_err(trace)?,
                        _ => unreachable!(),
                    };
                    self.stack.push(result);
                }
                Instruction::Neg => {
                    let v = self
                        .stack
                        .pop()
                        .ok_or_else(|| trace(RuntimeError::new("Stack underflow")))?;
                    let result = match v {
                        Value::Int(i) => Value::Int(-i),
                        Value::Float(f) => Value::Float(-f),
                        _ => return Err(trace(RuntimeError::new("Neg expects number"))),
                    };
                    self.stack.push(result);
                }
                Instruction::Not => {
                    let v = self
                        .stack
                        .pop()
                        .ok_or_else(|| trace(RuntimeError::new("Stack underflow")))?;
                    let b = v.is_truthy();
                    self.stack.push(Value::Bool(!b));
                }
                Instruction::Jump(target) => {
                    next_ip = target;
                }
                Instruction::JumpIfFalse(target) => {
                    let cond = self
                        .stack
                        .pop()
                        .ok_or_else(|| trace(RuntimeError::new("Stack underflow")))?;
                    if !cond.is_truthy() {
                        next_ip = target;
                    }
                }
                Instruction::MakeRecord(count) => {
                    let mut map = std::collections::HashMap::new();
                    for _ in 0..count {
                        let value = self
                            .stack
                            .pop()
                            .ok_or_else(|| trace(RuntimeError::new("Stack underflow")))?;
                        let key = self
                            .stack
                            .pop()
                            .ok_or_else(|| trace(RuntimeError::new("Stack underflow")))?;
                        let key = match key {
                            Value::Obj(obj) => match obj.as_obj() {
                                Obj::String(s) => s.clone(),
                                _ => {
                                    return Err(trace(RuntimeError::new(
                                        "Record key must be string",
                                    )))
                                }
                            },
                            _ => return Err(trace(RuntimeError::new("Record key must be string"))),
                        };
                        map.insert(key, value);
                    }
                    self.stack.push(record_value(map));
                }
                Instruction::GetField(name_idx) => {
                    let target = self
                        .stack
                        .pop()
                        .ok_or_else(|| trace(RuntimeError::new("Stack underflow")))?;
                    let name = match &func.chunk.constants[name_idx as usize] {
                        Constant::String(s) => s.clone(),
                        _ => {
                            return Err(trace(RuntimeError::new(
                                "Field name must be string constant",
                            )))
                        }
                    };
                    let value = match target {
                        Value::Obj(obj) => match obj.as_obj() {
                            Obj::Record(map) => map
                                .get(&name)
                                .cloned()
                                .ok_or_else(|| trace(RuntimeError::new("Unknown field")))?,
                            _ => {
                                return Err(trace(RuntimeError::new("Field access expects record")))
                            }
                        },
                        _ => return Err(trace(RuntimeError::new("Field access expects record"))),
                    };
                    self.stack.push(value);
                }
                Instruction::Call(argc) => {
                    if let Some(caller) = self.frames.last_mut() {
                        caller.ip = next_ip;
                    }
                    self.call_value(program, argc as usize).map_err(trace)?;
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
            if update_ip {
                if let Some(frame_mut) = self.frames.last_mut() {
                    frame_mut.ip = next_ip;
                }
            }
        }
        Ok(self.stack.pop().unwrap_or(Value::Null))
    }

    fn call_value(&mut self, _program: &Program, argc: usize) -> Result<(), RuntimeError> {
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
                    let args: Vec<Value> = self.stack[args_start..].to_vec();
                    let result = (nf.func)(self, &args)?;
                    self.stack.truncate(callee_index);
                    self.stack.push(result);
                }
                _ => return Err(RuntimeError::new("Callee is not callable")),
            },
            _ => return Err(RuntimeError::new("Callee is not callable")),
        }
        Ok(())
    }

    fn with_trace(&self, mut err: RuntimeError, program: &Program, ip: usize) -> RuntimeError {
        if err.frames.is_empty() {
            err = err.with_frames(self.stack_trace(program, ip));
        }
        err
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

    fn constant_to_value(&self, c: &Constant, program: &Program) -> Result<Value, RuntimeError> {
        Ok(match c {
            Constant::Int(i) => Value::Int(*i),
            Constant::Float(f) => Value::Float(*f),
            Constant::Bool(b) => Value::Bool(*b),
            Constant::Null => Value::Null,
            Constant::String(s) => string_value(s),
            Constant::Function(idx) => function_value(*idx, program),
        })
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
            Obj::Function(f) => format!("<fn {}>", f.name.clone().unwrap_or_default()),
            Obj::NativeFunction(n) => format!("<native {}>", n.name),
            Obj::Record(_) => "<record>".to_string(),
        },
    }
}
