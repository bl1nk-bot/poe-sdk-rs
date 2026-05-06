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

#[cfg(test)]
mod integration_tests {
    use super::*;
    use context::Context;
    use functions::FunctionRegistry;
    use lexer::tokenize;
    use parser::parse;
    use eval::evaluate;

    // สร้าง registry พร้อมฟังก์ชันพื้นฐาน
    fn prepared_registry() -> FunctionRegistry {
        let mut reg = FunctionRegistry::new();
        builtins::register_all(&mut reg);
        reg
    }

    #[test]
    fn test_simple_math() {
        let tokens = tokenize("1 + 2 * 3").unwrap();
        let ast = parse(&tokens).unwrap();
        let ctx = Context::new();
        let reg = prepared_registry();
        let result = evaluate(&ast, &ctx, &reg).unwrap();
        assert_eq!(result, Value::Number(7.0));
    }

    #[test]
    fn test_comparison() {
        let tokens = tokenize("5 > 3 && 2 == 2").unwrap();
        let ast = parse(&tokens).unwrap();
        let ctx = Context::new();
        let reg = prepared_registry();
        let result = evaluate(&ast, &ctx, &reg).unwrap();
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn test_if_function() {
        let tokens = tokenize("if(true, 100, 200)").unwrap();
        let ast = parse(&tokens).unwrap();
        let ctx = Context::new();
        let reg = prepared_registry();
        let result = evaluate(&ast, &ctx, &reg).unwrap();
        assert_eq!(result, Value::Number(100.0));
    }

    #[test]
    fn test_string_concat() {
        let tokens = tokenize("\"hello\" + \" world\"").unwrap();
        let ast = parse(&tokens).unwrap();
        let ctx = Context::new();
        let reg = prepared_registry();
        let result = evaluate(&ast, &ctx, &reg).unwrap();
        assert_eq!(result, Value::String("hello world".to_string()));
    }

    #[test]
    fn test_variable_lookup() {
        let tokens = tokenize("score + 10").unwrap();
        let ast = parse(&tokens).unwrap();
        let mut ctx = Context::new();
        ctx.set("score", Value::Number(40.0));
        let reg = prepared_registry();
        let result = evaluate(&ast, &ctx, &reg).unwrap();
        assert_eq!(result, Value::Number(50.0));
    }

    #[test]
    fn test_division_by_zero_error() {
        let tokens = tokenize("1 / 0").unwrap();
        let ast = parse(&tokens).unwrap();
        let ctx = Context::new();
        let reg = prepared_registry();
        let result = evaluate(&ast, &ctx, &reg);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.code, "E010");
    }
}