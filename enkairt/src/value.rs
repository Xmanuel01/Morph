use std::rc::Rc;

use crate::object::Obj;

#[derive(Clone, Debug)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    Null,
    Obj(ObjRef),
}

#[derive(Clone, Debug)]
pub struct ObjRef(Rc<Obj>);

impl ObjRef {
    pub fn new(obj: Obj) -> Self {
        Self(Rc::new(obj))
    }

    pub fn downgrade(&self) -> std::rc::Weak<Obj> {
        Rc::downgrade(&self.0)
    }

    pub fn strong_count(&self) -> usize {
        Rc::strong_count(&self.0)
    }

    pub fn as_obj(&self) -> &Obj {
        &self.0
    }
}

impl From<ObjRef> for Value {
    fn from(obj: ObjRef) -> Self {
        Value::Obj(obj)
    }
}

impl Value {
    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            Value::Null => false,
            Value::Int(i) => *i != 0,
            Value::Float(f) => *f != 0.0,
            Value::Obj(_) => true,
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Int(_) => "Int",
            Value::Float(_) => "Float",
            Value::Bool(_) => "Bool",
            Value::Null => "Null",
            Value::Obj(obj) => obj.as_obj().type_name(),
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => a == b,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Null, Value::Null) => true,
            (Value::Obj(a), Value::Obj(b)) => Rc::ptr_eq(&a.0, &b.0),
            _ => false,
        }
    }
}
