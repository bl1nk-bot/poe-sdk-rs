// src/builtins/string.rs
use crate::functions::BuiltinFunction;
use crate::value::Value;
use crate::error::{ErrorKind, FormulaError};

pub fn len() -> BuiltinFunction {
    BuiltinFunction {
        name: "len".to_string(),
        arity: 1,
        call: |args| {
            match &args[0] {
                Value::String(s) => Ok(Value::Number(s.len() as f64)),
                _ => Err(FormulaError::new(
                    ErrorKind::FunctionError,
                    "E006",
                    "len ต้องการข้อความ",
                    None,
                )),
            }
        },
    }
}

pub fn upper() -> BuiltinFunction {
    BuiltinFunction {
        name: "upper".to_string(),
        arity: 1,
        call: |args| {
            if let Value::String(s) = &args[0] {
                Ok(Value::String(s.to_uppercase()))
            } else {
                Err(FormulaError::new(ErrorKind::FunctionError, "E006", "upper ต้องการข้อความ", None))
            }
        },
    }
}

pub fn lower() -> BuiltinFunction {
    BuiltinFunction {
        name: "lower".to_string(),
        arity: 1,
        call: |args| {
            if let Value::String(s) = &args[0] {
                Ok(Value::String(s.to_lowercase()))
            } else {
                Err(FormulaError::new(ErrorKind::FunctionError, "E006", "lower ต้องการข้อความ", None))
            }
        },
    }
}