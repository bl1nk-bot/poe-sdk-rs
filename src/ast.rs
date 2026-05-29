use crate::span::Span;
use crate::value::Value;

/// ตัวดำเนินการทวิภาค (Binary Operators)
#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOp {
    /// การบวก `+`
    Add,
    /// การลบ `-`
    Sub,
    /// การคูณ `*`
    Mul,
    /// การหาร `/`
    Div,
    /// การเปรียบเทียบเท่ากับ `==`
    Eq,
    /// การเปรียบเทียบไม่เท่ากับ `!=`
    NotEq,
    /// น้อยกว่า `<`
    Lt,
    /// มากกว่า `>`
    Gt,
    /// น้อยกว่าหรือเท่ากับ `<=`
    LtEq,
    /// มากกว่าหรือเท่ากับ `>=`
    GtEq,
    /// ตรรกะ AND `&&`
    And,
    /// ตรรกะ OR `||`
    Or,
}

/// ตัวดำเนินการเอกภาค (Unary Operators)
#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOp {
    /// ค่าติดลบ `-`
    Neg,
    /// ตรรกะ NOT `!`
    Not,
}

/// ข้อมูล Meta สำหรับนิพจน์ (เช่น ตำแหน่งใน Source Code)
#[derive(Debug, Clone, PartialEq)]
pub struct ExprMeta {
    /// ขอบเขตตำแหน่ง (Line, Column) ของนิพจน์
    pub span: Span,
}

/// นิพจน์ที่มาพร้อมกับข้อมูลตำแหน่ง (Metadata)
#[derive(Debug, Clone, PartialEq)]
pub struct SpannedExpr {
    /// ตัวนิพจน์ (Expression Node)
    pub expr: Expr,
    /// ข้อมูล Metadata (เช่น Span)
    pub meta: ExprMeta,
}

impl SpannedExpr {
    /// สร้างนิพจน์ใหม่พร้อมตำแหน่ง
    pub fn new(expr: Expr, span: Span) -> Self {
        Self {
            expr,
            meta: ExprMeta { span },
        }
    }
}

/// โครงสร้างต้นไม้ของนิพจน์ (AST Nodes)
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// ค่าคงที่ (Literal) เช่น `123`, `"hello"`, `true`
    Literal(Value),
    /// ตัวแปร (Variable) เช่น `score`, `user.name`
    Variable(String),
    /// นิพจน์ที่มีตัวดำเนินการเอกภาค (Unary Expression) เช่น `-5`, `!true`
    UnaryExpr {
        /// ตัวดำเนินการ (Neg, Not)
        op: UnaryOp,
        /// นิพจน์ที่ถูกกระทำ
        expr: Box<SpannedExpr>,
    },
    /// นิพจน์ที่มีตัวดำเนินการทวิภาค (Binary Expression) เช่น `1 + 2`, `a && b`
    BinaryExpr {
        /// นิพจน์ด้านซ้าย
        left: Box<SpannedExpr>,
        /// ตัวดำเนินการ
        op: BinaryOp,
        /// นิพจน์ด้านขวา
        right: Box<SpannedExpr>,
    },
    /// การเรียกใช้ฟังก์ชัน (Function Call) เช่น `sum(1, 2, 3)`
    FunctionCall {
        /// ชื่อฟังก์ชัน
        name: String,
        /// รายการอาร์กิวเมนต์
        args: Vec<SpannedExpr>,
    },
    /// นิพจน์ที่อยู่ในวงเล็บ (Grouping)
    Grouping(Box<SpannedExpr>),
    /// อาร์เรย์ (Array Literal) เช่น `[1, 2, 3]`
    ArrayLiteral(Vec<SpannedExpr>),
    /// แมพหรือออบเจกต์ (Map Literal) เช่น `{a: 1, b: 2}`
    MapLiteral(Vec<(String, SpannedExpr)>),

    /// แลมบ์ดา (Lambda Expression) เช่น `(x) => x * 2`
    LambdaExpr {
        /// รายชื่อพารามิเตอร์
        params: Vec<String>,
        /// นิพจน์ที่เป็นตัวแลมบ์ดา
        body: Box<SpannedExpr>,
    },

    /// การนิยามฟังก์ชันโดยผู้ใช้ (User-defined Function) เช่น `fn double(x) = x * 2`
    FunctionDef {
        /// ชื่อฟังก์ชัน
        name: String,
        /// รายชื่อพารามิเตอร์
        params: Vec<String>,
        /// นิพจน์ที่เป็นตัวฟังก์ชัน
        body: Box<SpannedExpr>,
    },
}
