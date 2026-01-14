#[derive(Debug, Clone, PartialEq)]
pub struct Module {
    pub items: Vec<Item>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Item {
    Use(UseDecl),
    Fn(FnDecl),
    Type(TypeDecl),
    Enum(EnumDecl),
    Impl(ImplDecl),
    Tool(ToolDecl),
    Policy(PolicyDecl),
    Prompt(PromptDecl),
    Model(ModelDecl),
    Agent(AgentDecl),
    Stmt(Stmt),
}

#[derive(Debug, Clone, PartialEq)]
pub struct UseDecl {
    pub path: Vec<String>,
    pub alias: Option<String>,
    pub is_pub: bool,
    pub spans: Vec<Span>,
    pub symbols: Vec<UseSymbol>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UseSymbol {
    pub name: String,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FnDecl {
    pub name: String,
    pub params: Vec<Param>,
    pub return_type: Option<TypeRef>,
    pub body: Block,
    pub is_pub: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub type_ann: Option<TypeRef>,
    pub default: Option<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypeDecl {
    pub name: String,
    pub fields: Vec<TypeField>,
    pub is_pub: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypeField {
    pub name: String,
    pub field_type: TypeRef,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnumDecl {
    pub name: String,
    pub variants: Vec<String>,
    pub is_pub: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImplDecl {
    pub name: String,
    pub methods: Vec<FnDecl>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolDecl {
    pub path: Vec<String>,
    pub params: Vec<Param>,
    pub return_type: Option<TypeRef>,
    pub is_pub: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PolicyDecl {
    pub name: String,
    pub rules: Vec<PolicyRule>,
    pub is_pub: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PolicyRule {
    pub allow: bool,
    pub capability: Vec<String>,
    pub filters: Vec<PolicyFilter>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PolicyFilter {
    pub name: String,
    pub value: LiteralOrList,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LiteralOrList {
    Literal(Literal),
    List(Vec<Literal>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct PromptDecl {
    pub name: String,
    pub input_fields: Vec<TypeField>,
    pub output_type: Option<TypeRef>,
    pub template: Option<String>,
    pub is_pub: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelDecl {
    pub name: String,
    pub expr: Expr,
    pub is_pub: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AgentDecl {
    pub name: String,
    pub items: Vec<AgentItem>,
    pub is_pub: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AgentItem {
    PolicyUse(String),
    Memory(MemoryDecl),
    Fn(FnDecl),
    Stmt(Stmt),
}

#[derive(Debug, Clone, PartialEq)]
pub enum MemoryDecl {
    Path { name: String, path: String },
    Expr { name: String, expr: Expr },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub stmts: Vec<Stmt>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Let {
        name: String,
        type_ann: Option<TypeRef>,
        expr: Expr,
    },
    Assign {
        target: LValue,
        expr: Expr,
    },
    Expr(Expr),
    If {
        cond: Expr,
        then_block: Block,
        else_branch: Option<ElseBranch>,
    },
    While {
        cond: Expr,
        body: Block,
    },
    For {
        var: String,
        iter: Expr,
        body: Block,
    },
    Match {
        expr: Expr,
        arms: Vec<MatchArm>,
    },
    Try {
        body: Block,
        catch_name: String,
        catch_body: Block,
    },
    Return {
        expr: Option<Expr>,
    },
    Break,
    Continue,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ElseBranch {
    Block(Block),
    If(Box<Stmt>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Literal(Literal),
    Ident(String),
    Binary {
        left: Box<Expr>,
        op: BinaryOp,
        right: Box<Expr>,
    },
    Unary {
        op: UnaryOp,
        expr: Box<Expr>,
    },
    Call {
        callee: Box<Expr>,
        args: Vec<Arg>,
    },
    Index {
        target: Box<Expr>,
        index: Box<Expr>,
    },
    Field {
        target: Box<Expr>,
        name: String,
    },
    List(Vec<Expr>),
    Lambda {
        params: Vec<Param>,
        return_type: Option<TypeRef>,
        body: Box<Expr>,
    },
    Match {
        expr: Box<Expr>,
        arms: Vec<MatchArm>,
    },
    Try(Box<Expr>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Arg {
    Positional(Expr),
    Named(String, Expr),
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    And,
    Or,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOp {
    Negate,
    Not,
    Await,
    Spawn,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    Wildcard,
    Literal(Literal),
    Ident(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub body: ArmBody,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ArmBody {
    Expr(Expr),
    Block(Block),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    None,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeRef {
    Named {
        path: Vec<String>,
        args: Vec<TypeRef>,
        optional: bool,
    },
    Function {
        params: Vec<TypeRef>,
        ret: Box<TypeRef>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct LValue {
    pub base: String,
    pub accesses: Vec<LValueAccess>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LValueAccess {
    Field(String),
    Index(Expr),
}
use crate::diagnostic::Span;
