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
pub mod builtins;
pub mod context;
pub mod diagnostics;
pub mod error;
pub mod eval;
pub mod functions;
pub mod lexer;
pub mod parser;
pub mod span;
pub mod value;

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
    use eval::evaluate;
    use functions::FunctionRegistry;
    use lexer::tokenize;
    use parser::parse;

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
        let tokens =
            tokenize("if(len(\"hello\") > 3 && 5 >= 5, upper(\"test\"), \"fail\")").unwrap();
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

    #[test]
    fn test_array_literal_parsing() {
        let tokens = tokenize("[1, 2, 3]").unwrap();
        let ast = parse(&tokens).unwrap();
        match ast.expr {
            Expr::ArrayLiteral(elems) => assert_eq!(elems.len(), 3),
            _ => panic!("expect ArrayLiteral"),
        }
    }

    #[test]
    fn test_sum_array() {
        let tokens = tokenize("sum([1, 2, 3, 4])").unwrap();
        let ast = parse(&tokens).unwrap();
        let ctx = Context::new();
        let reg = prepared_registry();
        let result = evaluate(&ast, &ctx, &reg).unwrap();
        assert_eq!(result, Value::Number(10.0));
    }

    #[test]
    fn test_len_array() {
        let tokens = tokenize("len([1, 2, 3])").unwrap();
        let ast = parse(&tokens).unwrap();
        let ctx = Context::new();
        let reg = prepared_registry();
        let result = evaluate(&ast, &ctx, &reg).unwrap();
        assert_eq!(result, Value::Number(3.0));
    }

    #[test]
    fn test_avg_array() {
        let tokens = tokenize("avg([10, 20])").unwrap();
        let ast = parse(&tokens).unwrap();
        let ctx = Context::new();
        let reg = prepared_registry();
        let result = evaluate(&ast, &ctx, &reg).unwrap();
        assert_eq!(result, Value::Number(15.0));
    }

    #[test]
    fn test_min_max_array() {
        let reg = prepared_registry();
        // min
        let tokens = tokenize("min([5, 2, 8])").unwrap();
        let ast = parse(&tokens).unwrap();
        assert_eq!(
            evaluate(&ast, &Context::new(), &reg).unwrap(),
            Value::Number(2.0)
        );
        // max
        let tokens = tokenize("max([5, 2, 8])").unwrap();
        let ast = parse(&tokens).unwrap();
        assert_eq!(
            evaluate(&ast, &Context::new(), &reg).unwrap(),
            Value::Number(8.0)
        );
    }

    #[test]
    fn test_join_array() {
        let tokens = tokenize("join([\"a\", \"b\"], \", \")").unwrap();
        let ast = parse(&tokens).unwrap();
        let reg = prepared_registry();
        assert_eq!(
            evaluate(&ast, &Context::new(), &reg).unwrap(),
            Value::String("a, b".to_string())
        );
    }

    #[test]
    fn test_count_array() {
        let tokens = tokenize("count([1, 2, 3, 4, 5])").unwrap();
        let ast = parse(&tokens).unwrap();
        let reg = prepared_registry();
        assert_eq!(
            evaluate(&ast, &Context::new(), &reg).unwrap(),
            Value::Number(5.0)
        );
    }

    #[test]
    fn test_empty_array() {
        let tokens = tokenize("[]").unwrap();
        let ast = parse(&tokens).unwrap();
        let reg = prepared_registry();
        assert_eq!(
            evaluate(&ast, &Context::new(), &reg).unwrap(),
            Value::Array(vec![])
        );
    }

    #[test]
    fn test_nested_array() {
        let tokens = tokenize("[[1,2], [3,4]]").unwrap();
        let ast = parse(&tokens).unwrap();
        let reg = prepared_registry();
        let result = evaluate(&ast, &Context::new(), &reg).unwrap();
        // result should be Array([Array([1,2]), Array([3,4])])
        if let Value::Array(outer) = result {
            assert_eq!(outer.len(), 2);
        } else {
            panic!("expected array");
        }
    }

    // -- Additional edge case tests for array features (Phase 6.1) --

    #[test]
    fn test_len_empty_array() {
        let tokens = tokenize("len([])").unwrap();
        let ast = parse(&tokens).unwrap();
        let reg = prepared_registry();
        let result = evaluate(&ast, &Context::new(), &reg).unwrap();
        assert_eq!(result, Value::Number(0.0));
    }

    #[test]
    fn test_len_non_array_non_string_error() {
        let tokens = tokenize("len(42)").unwrap();
        let ast = parse(&tokens).unwrap();
        let reg = prepared_registry();
        let result = evaluate(&ast, &Context::new(), &reg);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind, ErrorKind::FunctionError);
        assert_eq!(err.code, "E006");
    }

    #[test]
    fn test_sum_empty_array_is_zero() {
        let tokens = tokenize("sum([])").unwrap();
        let ast = parse(&tokens).unwrap();
        let reg = prepared_registry();
        let result = evaluate(&ast, &Context::new(), &reg).unwrap();
        assert_eq!(result, Value::Number(0.0));
    }

    #[test]
    fn test_sum_non_number_element_error() {
        let tokens = tokenize("sum([\"a\", \"b\"])").unwrap();
        let ast = parse(&tokens).unwrap();
        let reg = prepared_registry();
        let result = evaluate(&ast, &Context::new(), &reg);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind, ErrorKind::FunctionError);
        assert_eq!(err.code, "E006");
    }

    #[test]
    fn test_avg_empty_array_error() {
        let tokens = tokenize("avg([])").unwrap();
        let ast = parse(&tokens).unwrap();
        let reg = prepared_registry();
        let result = evaluate(&ast, &Context::new(), &reg);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind, ErrorKind::FunctionError);
        assert_eq!(err.code, "E011");
    }

    #[test]
    fn test_min_empty_array_error() {
        let tokens = tokenize("min([])").unwrap();
        let ast = parse(&tokens).unwrap();
        let reg = prepared_registry();
        let result = evaluate(&ast, &Context::new(), &reg);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind, ErrorKind::FunctionError);
        assert_eq!(err.code, "E011");
    }

    #[test]
    fn test_max_empty_array_error() {
        let tokens = tokenize("max([])").unwrap();
        let ast = parse(&tokens).unwrap();
        let reg = prepared_registry();
        let result = evaluate(&ast, &Context::new(), &reg);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind, ErrorKind::FunctionError);
        assert_eq!(err.code, "E011");
    }

    #[test]
    fn test_join_empty_separator() {
        let tokens = tokenize("join([\"x\", \"y\", \"z\"], \"\")").unwrap();
        let ast = parse(&tokens).unwrap();
        let reg = prepared_registry();
        let result = evaluate(&ast, &Context::new(), &reg).unwrap();
        assert_eq!(result, Value::String("xyz".to_string()));
    }

    #[test]
    fn test_sum_with_float_values() {
        let tokens = tokenize("sum([1.5, 2.5, 3.0])").unwrap();
        let ast = parse(&tokens).unwrap();
        let reg = prepared_registry();
        let result = evaluate(&ast, &Context::new(), &reg).unwrap();
        assert_eq!(result, Value::Number(7.0));
    }

    #[test]
    fn test_sum_with_negative_values() {
        let tokens = tokenize("sum([-1, -2, 3])").unwrap();
        let ast = parse(&tokens).unwrap();
        let reg = prepared_registry();
        let result = evaluate(&ast, &Context::new(), &reg).unwrap();
        assert_eq!(result, Value::Number(0.0));
    }

    #[test]
    fn test_min_with_negative_values() {
        let tokens = tokenize("min([0, -5, 3])").unwrap();
        let ast = parse(&tokens).unwrap();
        let reg = prepared_registry();
        let result = evaluate(&ast, &Context::new(), &reg).unwrap();
        assert_eq!(result, Value::Number(-5.0));
    }

    #[test]
    fn test_max_with_negative_values() {
        let tokens = tokenize("max([-1, -2, -3])").unwrap();
        let ast = parse(&tokens).unwrap();
        let reg = prepared_registry();
        let result = evaluate(&ast, &Context::new(), &reg).unwrap();
        assert_eq!(result, Value::Number(-1.0));
    }

    #[test]
    fn test_count_empty_array() {
        let tokens = tokenize("count([])").unwrap();
        let ast = parse(&tokens).unwrap();
        let reg = prepared_registry();
        let result = evaluate(&ast, &Context::new(), &reg).unwrap();
        assert_eq!(result, Value::Number(0.0));
    }

    #[test]
    fn test_nested_array_inner_values() {
        let tokens = tokenize("[[1,2], [3,4]]").unwrap();
        let ast = parse(&tokens).unwrap();
        let reg = prepared_registry();
        let result = evaluate(&ast, &Context::new(), &reg).unwrap();
        if let Value::Array(outer) = result {
            assert_eq!(outer.len(), 2);
            if let Value::Array(inner) = &outer[0] {
                assert_eq!(inner[0], Value::Number(1.0));
                assert_eq!(inner[1], Value::Number(2.0));
            } else {
                panic!("expected inner array");
            }
        } else {
            panic!("expected outer array");
        }
    }

    #[test]
    fn test_array_in_if_condition() {
        // Array can be created and passed, len used in condition
        let tokens = tokenize("if(len([1,2,3]) == 3, \"yes\", \"no\")").unwrap();
        let ast = parse(&tokens).unwrap();
        let reg = prepared_registry();
        let result = evaluate(&ast, &Context::new(), &reg).unwrap();
        assert_eq!(result, Value::String("yes".to_string()));
    }

    #[test]
    fn test_sum_passed_non_array_error() {
        // sum() expects array, not a plain number
        let tokens = tokenize("sum(42)").unwrap();
        let ast = parse(&tokens).unwrap();
        let reg = prepared_registry();
        let result = evaluate(&ast, &Context::new(), &reg);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind, ErrorKind::FunctionError);
        assert_eq!(err.code, "E006");
    }

    #[test]
    fn test_join_non_string_element_error() {
        // join() requires string elements
        let tokens = tokenize("join([1, 2], \"-\")").unwrap();
        let ast = parse(&tokens).unwrap();
        let reg = prepared_registry();
        let result = evaluate(&ast, &Context::new(), &reg);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind, ErrorKind::FunctionError);
        assert_eq!(err.code, "E006");
    }

    #[test]
    fn test_array_equality() {
        // Value::Array supports PartialEq
        assert_eq!(
            Value::Array(vec![Value::Number(1.0), Value::Number(2.0)]),
            Value::Array(vec![Value::Number(1.0), Value::Number(2.0)])
        );
        assert_ne!(
            Value::Array(vec![Value::Number(1.0)]),
            Value::Array(vec![Value::Number(2.0)])
        );
    }

    #[test]
    fn test_array_with_bool_elements() {
        // Arrays may hold booleans
        let tokens = tokenize("[true, false]").unwrap();
        let ast = parse(&tokens).unwrap();
        let reg = prepared_registry();
        let result = evaluate(&ast, &Context::new(), &reg).unwrap();
        assert_eq!(
            result,
            Value::Array(vec![Value::Bool(true), Value::Bool(false)])
        );
    }

    #[test]
    fn test_count_equals_len_for_array() {
        // count and len should agree on array size
        let reg = prepared_registry();
        let tokens_count = tokenize("count([1, 2, 3])").unwrap();
        let ast_count = parse(&tokens_count).unwrap();
        let tokens_len = tokenize("len([1, 2, 3])").unwrap();
        let ast_len = parse(&tokens_len).unwrap();
        let result_count = evaluate(&ast_count, &Context::new(), &reg).unwrap();
        let result_len = evaluate(&ast_len, &Context::new(), &reg).unwrap();
        assert_eq!(result_count, result_len);
    }
}
