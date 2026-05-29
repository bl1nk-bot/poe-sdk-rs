use crate::error::{ErrorKind, FormulaError};
use crate::functions::BuiltinFunction;
use crate::value::Value;
use std::sync::Arc;

pub fn abs() -> BuiltinFunction {
    BuiltinFunction {
        name: "abs".to_string(),
        arity: 1,
        call: Arc::new(|args: &[Value]| {
            if let Value::Number(n) = args[0] {
                Ok(Value::Number(n.abs()))
            } else {
                Err(FormulaError::new(
                    ErrorKind::FunctionError,
                    "E501",
                    "abs ต้องการตัวเลข",
                    None,
                ))
            }
        }),
    }
}
