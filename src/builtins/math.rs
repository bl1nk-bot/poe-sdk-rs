//! ฟังก์ชันทางคณิตศาสตร์

use crate::error::{ErrorKind, FormulaError};
use crate::functions::BuiltinFunction;
use crate::value::Value;

pub fn abs() -> BuiltinFunction {
    BuiltinFunction {
        name: "abs".to_string(),
        arity: 1,
        call: |args| {
            if let Value::Number(n) = args[0] {
                Ok(Value::Number(n.abs()))
            } else {
                Err(FormulaError::new(
                    ErrorKind::FunctionError,
                    "E006",
                    "abs ต้องการตัวเลข",
                    None,
                ))
            }
        },
    }
}

pub fn min() -> BuiltinFunction {
    BuiltinFunction {
        name: "min".to_string(),
        arity: 2,
        call: |args| {
            match (&args[0], &args[1]) {
                (Value::Number(a), Value::Number(b)) => Ok(Value::Number(a.min(*b))),
                _ => Err(FormulaError::new(
                    ErrorKind::FunctionError,
                    "E006",
                    "min ต้องการตัวเลขสองตัว",
                    None,
                )),
            }
        },
    }
}

pub fn max() -> BuiltinFunction {
    BuiltinFunction {
        name: "max".to_string(),
        arity: 2,
        call: |args| {
            match (&args[0], &args[1]) {
                (Value::Number(a), Value::Number(b)) => Ok(Value::Number(a.max(*b))),
                _ => Err(FormulaError::new(
                    ErrorKind::FunctionError,
                    "E006",
                    "max ต้องการตัวเลขสองตัว",
                    None,
                )),
            }
        },
    }
}