use crate::error::{ErrorKind, FormulaError};
use crate::functions::BuiltinFunction;
use crate::value::Value;
use std::sync::Arc;

pub fn if_fn() -> BuiltinFunction {
    BuiltinFunction {
        name: "if".to_string(),
        arity: 3,
        call: Arc::new(|args: &[Value]| {
            let cond = &args[0];
            let is_true = match cond {
                Value::Bool(b) => *b,
                _ => {
                    return Err(FormulaError::new(
                        ErrorKind::TypeError,
                        "E401",
                        "เงื่อนไขของ IF ต้องเป็น boolean",
                        None,
                    ))
                }
            };
            if is_true {
                Ok(args[1].clone())
            } else {
                Ok(args[2].clone())
            }
        }),
    }
}
