#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Int,
    Float,
    Bool,
    String,
    Buffer,
    Tokenizer,
    DataStream,
    Batch,
    Tensor,
    Device,
    DType,
    Shape,
    OptimizerState,
    Channel,
    TcpListener,
    TcpConnection,
    HttpRequest,
    HttpResponse,
    HttpStream,
    Void,
    Optional(Box<Type>),
    Function(Vec<Type>, Box<Type>),
    Unknown,
}

impl Type {
    pub fn display(&self) -> String {
        match self {
            Type::Int => "Int".to_string(),
            Type::Float => "Float".to_string(),
            Type::Bool => "Bool".to_string(),
            Type::String => "String".to_string(),
            Type::Buffer => "Buffer".to_string(),
            Type::Tokenizer => "Tokenizer".to_string(),
            Type::DataStream => "DataStream".to_string(),
            Type::Batch => "Batch".to_string(),
            Type::Tensor => "Tensor".to_string(),
            Type::Device => "Device".to_string(),
            Type::DType => "DType".to_string(),
            Type::Shape => "Shape".to_string(),
            Type::OptimizerState => "OptimizerState".to_string(),
            Type::Channel => "Channel".to_string(),
            Type::TcpListener => "TcpListener".to_string(),
            Type::TcpConnection => "TcpConnection".to_string(),
            Type::HttpRequest => "Request".to_string(),
            Type::HttpResponse => "Response".to_string(),
            Type::HttpStream => "HttpStream".to_string(),
            Type::Void => "Void".to_string(),
            Type::Optional(inner) => format!("{}?", inner.display()),
            Type::Function(params, ret) => {
                let args = params
                    .iter()
                    .map(|t| t.display())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("fn({}) -> {}", args, ret.display())
            }
            Type::Unknown => "Unknown".to_string(),
        }
    }
}
