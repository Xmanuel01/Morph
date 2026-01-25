#[derive(Debug, Clone)]
pub enum Constant {
    Int(i64),
    Float(f64),
    Bool(bool),
    Null,
    String(String),
    Function(u16), // index into Program.functions
}

#[derive(Debug, Clone)]
pub enum Instruction {
    Const(u16),
    Pop,
    DefineGlobal(u16),
    LoadLocal(u16),
    StoreLocal(u16),
    LoadGlobal(u16),
    StoreGlobal(u16),
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Neg,
    Not,
    Eq,
    Neq,
    Lt,
    Gt,
    Le,
    Ge,
    Jump(usize),
    JumpIfFalse(usize),
    Call(u16),
    Return,
}

#[derive(Debug, Clone)]
pub struct Chunk {
    pub code: Vec<Instruction>,
    pub constants: Vec<Constant>,
    pub lines: Vec<usize>,
}

impl Chunk {
    pub fn new() -> Self {
        Self {
            code: Vec::new(),
            constants: Vec::new(),
            lines: Vec::new(),
        }
    }

    pub fn add_constant(&mut self, constant: Constant) -> u16 {
        self.constants.push(constant);
        (self.constants.len() - 1) as u16
    }

    pub fn write(&mut self, instr: Instruction, line: usize) {
        self.code.push(instr);
        self.lines.push(line);
    }
}

impl Default for Chunk {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct ByteFunction {
    pub name: Option<String>,
    pub arity: u16,
    pub chunk: Chunk,
}

#[derive(Debug, Clone)]
pub struct Program {
    pub functions: Vec<ByteFunction>,
    pub main: u16,
    pub globals: Vec<String>,
    /// Optional initial values for globals (aligned with globals vec).
    pub global_inits: Vec<Option<Constant>>,
}

impl Program {
    pub fn disassemble(&self) -> String {
        let mut out = String::new();
        for (idx, func) in self.functions.iter().enumerate() {
            out.push_str(&format!(
                "== fn {} ({}) ==\n",
                func.name.clone().unwrap_or_else(|| format!("#{}", idx)),
                func.arity
            ));
            for (i, instr) in func.chunk.code.iter().enumerate() {
                let line = func.chunk.lines.get(i).copied().unwrap_or(0);
                out.push_str(&format!("{:04} {:>4} {:?}\n", i, line, instr));
            }
        }
        out
    }
}
