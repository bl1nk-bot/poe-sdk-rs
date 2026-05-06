/// ตัวดำเนินการสองตัว (binary)
#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOp {
    Add,       // +
    Sub,       // -
    Mul,       // *
    Div,       // /
    Eq,        // ==
    NotEq,     // !=
    Lt,        // <
    Gt,        // >
    LtEq,      // <=
    GtEq,      // >=
    And,       // &&
    Or,        // ||
}

/// ตัวดำเนินการเดี่ยว (unary)
#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOp {
    Neg,       // -
    Not,       // !
}

/// โหนดนิพจน์ใน AST
/// ไม่มี logic การคำนวณใด ๆ
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Literal(crate::value::Value),            // ค่าคงที่ เช่น 42, "hello", true
    Variable(String),                        // ตัวแปร เช่น score
    UnaryExpr {
        op: UnaryOp,
        expr: Box<Expr>,
    },
    BinaryExpr {
        left: Box<Expr>,
        op: BinaryOp,
        right: Box<Expr>,
    },
    FunctionCall {
        name: String,
        args: Vec<Expr>,
    },
    Grouping(Box<Expr>),                     // วงเล็บ (...)
}