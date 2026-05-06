use crate::span::{Position, Span};

/// รูปแบบ token ที่ lexer สร้าง
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // ตัวอักษร/คำ
    Identifier,      // เช่น score, if
    // ค่าคงที่
    Number,          // ตัวเลข เช่น 42, 3.14
    String,          // ข้อความในเครื่องหมายคำพูด "..."
    // วงเล็บและเครื่องหมาย
    LParen,          // (
    RParen,          // )
    Comma,           // ,
    // ตัวดำเนินการ
    Plus, Minus, Star, Slash, // + - * /
    Bang,            // !
    AndAnd,          // &&
    OrOr,            // ||
    EqEq, NotEq,     // == !=
    Lt, Gt, LtEq, GtEq, // < > <= >=
    // คำสงวน
    True, False,     // true, false
    // สิ้นสุด
    Eof,
}

/// Token ที่ประกอบด้วยชนิดและช่วงตำแหน่ง
#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
    pub lexeme: String, // ข้อความจริงของ token (เช่น "123", "score")
}

/// แปลงข้อความสูตรเป็น Vec<Token>
/// ถ้าเจออักขระที่ไม่รู้จักจะคืน Result::Err
pub fn tokenize(source: &str) -> Result<Vec<Token>, crate::error::FormulaError> {
    // TODO: implement lexer logic
    // - เดินอักขระ สร้าง token ตามชนิด
    // - เก็บตำแหน่งบรรทัด/คอลัมน์ใน Span
    // - ถ้าพบอักขระที่ไม่คาดคิดให้คืน LexError พร้อม span
    todo!("tokenize ยังไม่ได้ implement")
}