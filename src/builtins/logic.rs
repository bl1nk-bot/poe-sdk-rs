use crate::functions::BuiltinFunction;
use crate::value::Value;

/// ฟังก์ชัน if(cond, true_value, false_value) -> Value
pub fn if_fn() -> BuiltinFunction {
    BuiltinFunction {
        name: "if".to_string(),
        arity: 3,
        call: |args| {
            let cond = &args[0];
            let is_true = match cond {
                Value::Bool(b) => *b,
                _ => return Err("if เงื่อนไขต้องเป็น boolean".to_string()),
            };
            if is_true {
                Ok(args[1].clone())
            } else {
                Ok(args[2].clone())
            }
        },
    }
}