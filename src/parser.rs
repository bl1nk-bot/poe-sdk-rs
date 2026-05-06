use crate::ast::Expr;
use crate::error::FormulaError;
use crate::lexer::Token;

/// แปลง token stream เป็น AST
/// ใช้ recursive descent parser ทำงานตาม grammar ที่กำหนด
/// ถ้ามี syntax error จะคืน FormulaError พร้อม span
pub fn parse(tokens: &[Token]) -> Result<Expr, FormulaError> {
    // TODO: สร้าง parser ที่มี internal state (cursor)
    // - expression -> logical_or -> logical_and -> ... ตามลำดับความสำคัญ
    // - รองรับ grouping และ function call
    // - ถ้าพบ token ไม่คาดฝันให้สร้าง ParseError
    todo!("parser ยังไม่ได้ implement");
}