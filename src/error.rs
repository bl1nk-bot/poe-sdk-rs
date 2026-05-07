use crate::span::Span;

/// ประเภทของข้อผิดพลาด
#[derive(Debug, Clone, PartialEq)]
pub enum ErrorKind {
    LexError,
    ParseError,
    EvalError,
    TypeError,
    FunctionError,
    ContextError,
}

/// ข้อผิดพลาดที่เกิดขึ้นใน engine
#[derive(Debug, Clone)]
pub struct FormulaError {
    pub kind: ErrorKind,
    pub code: String,       // เช่น "E001"
    pub message: String,    // คำอธิบายสั้น ๆ
    pub span: Option<Span>, // ตำแหน่งที่เกิด error
}

impl FormulaError {
    /// สร้างข้อผิดพลาดใหม่
    pub fn new(kind: ErrorKind, code: &str, message: &str, span: Option<Span>) -> Self {
        Self {
            kind,
            code: code.to_string(),
            message: message.to_string(),
            span,
        }
    }
}

impl std::fmt::Display for FormulaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.code, self.message)
    }
}

impl std::error::Error for FormulaError {}
