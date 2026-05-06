//! ไลบรารี Formula Engine
//!
//! ใช้สำหรับแยกส่วน (parse) ประมวลผล (evaluate) สูตรแบบ Notion-like
//! สถาปัตยกรรมแบ่งเป็นชั้น: Lexer -> Parser -> Evaluator
//! รองรับการขยายฟังก์ชันและชนิดข้อมูลผ่าน registry

pub mod ast;
pub mod lexer;
pub mod parser;
pub mod value;
pub mod eval;
pub mod context;
pub mod functions;
pub mod error;
pub mod span;
pub mod diagnostics;
pub mod builtins;

// re-export สิ่งที่ผู้ใช้ต้องการ
pub use ast::Expr;
pub use context::Context;
pub use error::FormulaError;
pub use eval::evaluate;
pub use functions::FunctionRegistry;
pub use lexer::tokenize;
pub use parser::parse;
pub use value::Value;