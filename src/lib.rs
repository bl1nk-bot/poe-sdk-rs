//! ไลบรารี Formula Engine
//!
//! ใช้สำหรับแยกส่วน (parse) ประมวลผล (evaluate) สูตรแบบ Notion-like
//! สถาปัตยกรรมแบ่งเป็นชั้น: Lexer -> Parser -> Evaluator
//! รองรับการขยายฟังก์ชันและชนิดข้อมูลผ่าน registry
//!
//! # ตัวอย่างการใช้งาน
//!
//! ```
//! use formula_engine::{tokenize, parse, evaluate, Context, FunctionRegistry};
//! use formula_engine::builtins;
//!
//! // สร้าง registry พร้อมฟังก์ชันพื้นฐาน
//! let mut registry = FunctionRegistry::new();
//! builtins::register_all(&mut registry);
//!
//! // Parse และ evaluate สูตร
//! let tokens = tokenize("1 + 2 * len(\"hello\")").unwrap();
//! let ast = parse(&tokens).unwrap();
//! let ctx = Context::new();
//! let result = evaluate(&ast, &ctx, &registry).unwrap();
//!
//! println!("Result: {:?}", result); // Number(11.0)
//! ```

/// ```
/// use formula_engine::{tokenize, parse, evaluate, Context, FunctionRegistry};
/// use formula_engine::builtins;
///
/// let mut registry = FunctionRegistry::new();
/// builtins::register_all(&mut registry);
///
/// let tokens = tokenize("if(true, \"pass\", \"fail\")").unwrap();
/// let ast = parse(&tokens).unwrap();
/// let ctx = Context::new();
/// let result = evaluate(&ast, &ctx, &registry).unwrap();
/// assert_eq!(result, formula_engine::Value::String("pass".to_string()));
/// ```

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
    use error::ErrorKind;
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
    fn test_precedence_multiplication_over_addition() {
        let tokens = tokenize("1 + 2 * 3").unwrap();
        let ast = parse(&tokens).unwrap();
        let ctx = Context::new();
        let reg = prepared_registry();
        let result = evaluate(&ast, &ctx, &reg).unwrap();
        assert_eq!(result, Value::Number(7.0)); // 1 + (2 * 3) = 7
    }

    #[test]
    fn test_precedence_parentheses_override() {
        let tokens = tokenize("(1 + 2) * 3").unwrap();
        let ast = parse(&tokens).unwrap();
        let ctx = Context::new();
        let reg = prepared_registry();
        let result = evaluate(&ast, &ctx, &reg).unwrap();
        assert_eq!(result, Value::Number(9.0)); // (1 + 2) * 3 = 9
    }

    #[test]
    fn test_unary_negation() {
        let tokens = tokenize("-5 + 3").unwrap();
        let ast = parse(&tokens).unwrap();
        let ctx = Context::new();
        let reg = prepared_registry();
        let result = evaluate(&ast, &ctx, &reg).unwrap();
        assert_eq!(result, Value::Number(-2.0));
    }

    #[test]
    fn test_unary_not() {
        let tokens = tokenize("!true && false").unwrap();
        let ast = parse(&tokens).unwrap();
        let ctx = Context::new();
        let reg = prepared_registry();
        let result = evaluate(&ast, &ctx, &reg).unwrap();
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn test_complex_expression() {
        let tokens = tokenize("if(len(\"hello\") > 3 && 5 >= 5, upper(\"test\"), \"fail\")").unwrap();
        let ast = parse(&tokens).unwrap();
        let ctx = Context::new();
        let reg = prepared_registry();
        let result = evaluate(&ast, &ctx, &reg).unwrap();
        assert_eq!(result, Value::String("TEST".to_string()));
    }

    #[test]
    fn test_nested_function_calls() {
        let tokens = tokenize("len(upper(\"hello\"))").unwrap();
        let ast = parse(&tokens).unwrap();
        let ctx = Context::new();
        let reg = prepared_registry();
        let result = evaluate(&ast, &ctx, &reg).unwrap();
        assert_eq!(result, Value::Number(5.0)); // "HELLO".len() = 5
    }

    #[test]
    fn test_function_call_with_multiple_args() {
        let tokens = tokenize("if(true, \"pass\", \"fail\")").unwrap();
        let ast = parse(&tokens).unwrap();
        let ctx = Context::new();
        let reg = prepared_registry();
        let result = evaluate(&ast, &ctx, &reg).unwrap();
        assert_eq!(result, Value::String("pass".to_string()));
    }

    #[test]
    fn test_function_call_wrong_arg_count() {
        let tokens = tokenize("len(\"hello\", \"world\")").unwrap();
        let ast = parse(&tokens).unwrap();
        let ctx = Context::new();
        let reg = prepared_registry();
        let result = evaluate(&ast, &ctx, &reg);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind, ErrorKind::FunctionError);
        assert_eq!(err.code, "E008"); // Wrong arity
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

    #[test]
    fn test_undefined_variable_error() {
        let tokens = tokenize("undefined_var + 1").unwrap();
        let ast = parse(&tokens).unwrap();
        let ctx = Context::new();
        let reg = prepared_registry();
        let result = evaluate(&ast, &ctx, &reg);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind, ErrorKind::ContextError);
        assert_eq!(err.code, "E005");
    }

    #[test]
    fn test_undefined_function_error() {
        let tokens = tokenize("nonexistent(123)").unwrap();
        let ast = parse(&tokens).unwrap();
        let ctx = Context::new();
        let reg = prepared_registry();
        let result = evaluate(&ast, &ctx, &reg);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind, ErrorKind::FunctionError);
        assert_eq!(err.code, "E007");
    }

    #[test]
    fn test_wrong_arity_error() {
        let tokens = tokenize("len(\"hello\", \"world\")").unwrap();
        let ast = parse(&tokens).unwrap();
        let ctx = Context::new();
        let reg = prepared_registry();
        let result = evaluate(&ast, &ctx, &reg);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind, ErrorKind::FunctionError);
        assert_eq!(err.code, "E008");
    }

    #[test]
    fn test_type_error_add_string_number() {
        let tokens = tokenize("\"hello\" + 123").unwrap();
        let ast = parse(&tokens).unwrap();
        let ctx = Context::new();
        let reg = prepared_registry();
        let result = evaluate(&ast, &ctx, &reg);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind, ErrorKind::TypeError);
        assert_eq!(err.code, "E006");
    }
}