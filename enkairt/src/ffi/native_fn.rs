use std::ffi::c_void;
use std::sync::Arc;

use libffi::middle::{Arg, Cif, CodePtr, Type};

use enkaic::bytecode::{FfiSignature, FfiType};

use crate::error::RuntimeError;
use crate::object::buffer_value;
use crate::value::Value;

const MAX_FFI_BUFFER: usize = 128 * 1024 * 1024;

#[derive(Clone)]
pub struct FfiFunction {
    name: String,
    signature: FfiSignature,
    _library: Arc<libloading::Library>,
    code_ptr: CodePtr,
    cif: Cif,
    free_ptr: Option<CodePtr>,
    free_cif: Option<Cif>,
}

impl std::fmt::Debug for FfiFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FfiFunction")
            .field("name", &self.name)
            .field("signature", &self.signature)
            .finish()
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FfiSlice {
    ptr: *mut u8,
    len: usize,
}

impl FfiFunction {
    pub fn new(
        name: String,
        signature: FfiSignature,
        library: Arc<libloading::Library>,
        symbol: *const c_void,
        free_symbol: Option<*const c_void>,
    ) -> Result<Self, RuntimeError> {
        validate_signature(&signature)?;
        let arg_types = expand_param_types(&signature.params);
        let ret_type = ffi_return_type(&signature.ret);
        let cif = Cif::new(arg_types, ret_type);
        let free_ptr = free_symbol.map(CodePtr::from_ptr);
        let free_cif =
            free_ptr.map(|_| Cif::new(vec![Type::pointer(), Type::usize()], Type::void()));
        Ok(Self {
            name,
            signature,
            _library: library,
            code_ptr: CodePtr::from_ptr(symbol),
            cif,
            free_ptr,
            free_cif,
        })
    }

    pub fn arity(&self) -> usize {
        self.signature.params.len()
    }

    pub fn call(&self, args: &[Value]) -> Result<Value, RuntimeError> {
        if args.len() != self.signature.params.len() {
            return Err(RuntimeError::new("Arity mismatch"));
        }
        let mut i64_args: Vec<i64> = Vec::new();
        let mut f64_args: Vec<f64> = Vec::new();
        let mut u8_args: Vec<u8> = Vec::new();
        let mut usize_args: Vec<usize> = Vec::new();
        let mut ptr_args: Vec<*const c_void> = Vec::new();
        let mut keep_alive: Vec<Value> = Vec::new();
        let mut ffi_args: Vec<Arg> = Vec::new();

        for (value, param) in args.iter().zip(self.signature.params.iter()) {
            match param {
                FfiType::Int => {
                    let v = match value {
                        Value::Int(i) => *i,
                        _ => return Err(RuntimeError::new("FFI arg expects Int")),
                    };
                    i64_args.push(v);
                    ffi_args.push(Arg::new(i64_args.last().unwrap()));
                }
                FfiType::Float => {
                    let v = match value {
                        Value::Float(f) => *f,
                        _ => return Err(RuntimeError::new("FFI arg expects Float")),
                    };
                    f64_args.push(v);
                    ffi_args.push(Arg::new(f64_args.last().unwrap()));
                }
                FfiType::Bool => {
                    let v = match value {
                        Value::Bool(b) => {
                            if *b {
                                1u8
                            } else {
                                0u8
                            }
                        }
                        _ => return Err(RuntimeError::new("FFI arg expects Bool")),
                    };
                    u8_args.push(v);
                    ffi_args.push(Arg::new(u8_args.last().unwrap()));
                }
                FfiType::String => {
                    let (ptr, len) = string_arg(value, &mut keep_alive)?;
                    ptr_args.push(ptr);
                    usize_args.push(len);
                    ffi_args.push(Arg::new(ptr_args.last().unwrap()));
                    ffi_args.push(Arg::new(usize_args.last().unwrap()));
                }
                FfiType::Buffer => {
                    let (ptr, len) = buffer_arg(value, &mut keep_alive)?;
                    ptr_args.push(ptr);
                    usize_args.push(len);
                    ffi_args.push(Arg::new(ptr_args.last().unwrap()));
                    ffi_args.push(Arg::new(usize_args.last().unwrap()));
                }
                FfiType::Void => {
                    return Err(RuntimeError::new("FFI arg cannot be Void"));
                }
                FfiType::Optional(inner) => {
                    if matches!(value, Value::Null) {
                        match inner.as_ref() {
                            FfiType::String | FfiType::Buffer => {
                                ptr_args.push(std::ptr::null());
                                usize_args.push(0);
                                ffi_args.push(Arg::new(ptr_args.last().unwrap()));
                                ffi_args.push(Arg::new(usize_args.last().unwrap()));
                            }
                            _ => {
                                return Err(RuntimeError::new("FFI optional expects String/Buffer"))
                            }
                        }
                    } else {
                        match inner.as_ref() {
                            FfiType::String => {
                                let (ptr, len) = string_arg(value, &mut keep_alive)?;
                                ptr_args.push(ptr);
                                usize_args.push(len);
                                ffi_args.push(Arg::new(ptr_args.last().unwrap()));
                                ffi_args.push(Arg::new(usize_args.last().unwrap()));
                            }
                            FfiType::Buffer => {
                                let (ptr, len) = buffer_arg(value, &mut keep_alive)?;
                                ptr_args.push(ptr);
                                usize_args.push(len);
                                ffi_args.push(Arg::new(ptr_args.last().unwrap()));
                                ffi_args.push(Arg::new(usize_args.last().unwrap()));
                            }
                            _ => {
                                return Err(RuntimeError::new("FFI optional expects String/Buffer"))
                            }
                        }
                    }
                }
            }
        }

        let ret = match &self.signature.ret {
            FfiType::Int => {
                let value: i64 = unsafe { self.cif.call(self.code_ptr, &ffi_args) };
                Value::Int(value)
            }
            FfiType::Float => {
                let value: f64 = unsafe { self.cif.call(self.code_ptr, &ffi_args) };
                Value::Float(value)
            }
            FfiType::Bool => {
                let value: u8 = unsafe { self.cif.call(self.code_ptr, &ffi_args) };
                Value::Bool(value != 0)
            }
            FfiType::Void => {
                let _: () = unsafe { self.cif.call(self.code_ptr, &ffi_args) };
                Value::Null
            }
            FfiType::String => self.call_slice(&mut ffi_args, true)?,
            FfiType::Buffer => self.call_slice(&mut ffi_args, false)?,
            FfiType::Optional(inner) => match inner.as_ref() {
                FfiType::String => self.call_slice_optional(&mut ffi_args, true)?,
                FfiType::Buffer => self.call_slice_optional(&mut ffi_args, false)?,
                _ => {
                    return Err(RuntimeError::new(
                        "FFI optional return expects String/Buffer",
                    ))
                }
            },
        };
        Ok(ret)
    }

    fn call_slice(&self, args: &mut [Arg], as_string: bool) -> Result<Value, RuntimeError> {
        let slice: FfiSlice = unsafe { self.cif.call(self.code_ptr, args) };
        if slice.ptr.is_null() {
            return Err(RuntimeError::new("FFI returned null pointer"));
        }
        if slice.len > MAX_FFI_BUFFER {
            return Err(RuntimeError::new("FFI returned oversized buffer"));
        }
        let bytes = unsafe { std::slice::from_raw_parts(slice.ptr, slice.len) }.to_vec();
        self.free_buffer(slice.ptr, slice.len)?;
        if as_string {
            let string = String::from_utf8(bytes)
                .map_err(|_| RuntimeError::new("FFI returned invalid UTF-8"))?;
            Ok(crate::object::string_value(&string))
        } else {
            Ok(buffer_value(bytes))
        }
    }

    fn call_slice_optional(
        &self,
        args: &mut [Arg],
        as_string: bool,
    ) -> Result<Value, RuntimeError> {
        let slice: FfiSlice = unsafe { self.cif.call(self.code_ptr, args) };
        if slice.ptr.is_null() {
            return Ok(Value::Null);
        }
        if slice.len > MAX_FFI_BUFFER {
            return Err(RuntimeError::new("FFI returned oversized buffer"));
        }
        let bytes = unsafe { std::slice::from_raw_parts(slice.ptr, slice.len) }.to_vec();
        self.free_buffer(slice.ptr, slice.len)?;
        if as_string {
            let string = String::from_utf8(bytes)
                .map_err(|_| RuntimeError::new("FFI returned invalid UTF-8"))?;
            Ok(crate::object::string_value(&string))
        } else {
            Ok(buffer_value(bytes))
        }
    }

    fn free_buffer(&self, ptr: *mut u8, len: usize) -> Result<(), RuntimeError> {
        let free_ptr = self
            .free_ptr
            .ok_or_else(|| RuntimeError::new("enkai_free symbol missing"))?;
        let free_cif = self
            .free_cif
            .as_ref()
            .ok_or_else(|| RuntimeError::new("enkai_free symbol missing"))?;
        let ptr_args: Vec<*const c_void> = vec![ptr as *const c_void];
        let len_args: Vec<usize> = vec![len];
        let args = vec![
            Arg::new(ptr_args.last().unwrap()),
            Arg::new(len_args.last().unwrap()),
        ];
        let _: () = unsafe { free_cif.call(free_ptr, &args) };
        Ok(())
    }
}

fn validate_signature(signature: &FfiSignature) -> Result<(), RuntimeError> {
    for param in &signature.params {
        if let FfiType::Optional(inner) = param {
            if !matches!(inner.as_ref(), FfiType::String | FfiType::Buffer) {
                return Err(RuntimeError::new(
                    "Unsupported optional FFI parameter type; only String?/Buffer? are allowed",
                ));
            }
        }
    }
    if let FfiType::Optional(inner) = &signature.ret {
        if !matches!(inner.as_ref(), FfiType::String | FfiType::Buffer) {
            return Err(RuntimeError::new(
                "Unsupported optional FFI return type; only String?/Buffer? are allowed",
            ));
        }
    }
    Ok(())
}

pub fn requires_free(signature: &FfiSignature) -> bool {
    match &signature.ret {
        FfiType::String | FfiType::Buffer => true,
        FfiType::Optional(inner) => matches!(inner.as_ref(), FfiType::String | FfiType::Buffer),
        _ => false,
    }
}

fn expand_param_types(params: &[FfiType]) -> Vec<Type> {
    let mut types = Vec::new();
    for param in params {
        match param {
            FfiType::Int => types.push(Type::i64()),
            FfiType::Float => types.push(Type::f64()),
            FfiType::Bool => types.push(Type::u8()),
            FfiType::String | FfiType::Buffer => {
                types.push(Type::pointer());
                types.push(Type::usize());
            }
            FfiType::Void => types.push(Type::void()),
            FfiType::Optional(inner) => match inner.as_ref() {
                FfiType::String | FfiType::Buffer => {
                    types.push(Type::pointer());
                    types.push(Type::usize());
                }
                FfiType::Int => types.push(Type::i64()),
                FfiType::Float => types.push(Type::f64()),
                FfiType::Bool => types.push(Type::u8()),
                _ => types.push(Type::void()),
            },
        }
    }
    types
}

fn ffi_return_type(ret: &FfiType) -> Type {
    match ret {
        FfiType::Int => Type::i64(),
        FfiType::Float => Type::f64(),
        FfiType::Bool => Type::u8(),
        FfiType::String | FfiType::Buffer => Type::structure(vec![Type::pointer(), Type::usize()]),
        FfiType::Void => Type::void(),
        FfiType::Optional(inner) => match inner.as_ref() {
            FfiType::String | FfiType::Buffer => {
                Type::structure(vec![Type::pointer(), Type::usize()])
            }
            FfiType::Int => Type::i64(),
            FfiType::Float => Type::f64(),
            FfiType::Bool => Type::u8(),
            _ => Type::void(),
        },
    }
}

fn string_arg(
    value: &Value,
    keep_alive: &mut Vec<Value>,
) -> Result<(*const c_void, usize), RuntimeError> {
    match value {
        Value::Obj(obj) => match obj.as_obj() {
            crate::object::Obj::String(s) => {
                keep_alive.push(value.clone());
                Ok((s.as_ptr() as *const c_void, s.len()))
            }
            _ => Err(RuntimeError::new("FFI arg expects String")),
        },
        _ => Err(RuntimeError::new("FFI arg expects String")),
    }
}

fn buffer_arg(
    value: &Value,
    keep_alive: &mut Vec<Value>,
) -> Result<(*const c_void, usize), RuntimeError> {
    match value {
        Value::Obj(obj) => match obj.as_obj() {
            crate::object::Obj::Buffer(bytes) => {
                keep_alive.push(value.clone());
                Ok((bytes.as_ptr() as *const c_void, bytes.len()))
            }
            _ => Err(RuntimeError::new("FFI arg expects Buffer")),
        },
        _ => Err(RuntimeError::new("FFI arg expects Buffer")),
    }
}
