use crate::functions::BuiltinFunction;
use crate::value::Value;

/// ฟังก์ชัน len(text) -> Number
/// คืนความยาวของข้อความ
pub fn len() -> BuiltinFunction {
    BuiltinFunction {
        name: "len".to_string(),
        arity: 1,
        call: |args| {
            // คาดว่า args[0] เป็น Value::String
            match &args[0] {
                Value::String(s) => Ok(Value::Number(s.len() as f64)),
                _ => Err("len ต้องการข้อความ".to_string()),
            }
        },
    }
}

/// ฟังก์ชัน upper(text) -> String
pub fn upper() -> BuiltinFunction {
    BuiltinFunction {
        name: "upper".to_string(),
        arity: 1,
        call: |args| {
            if let Value::String(s) = &args[0] {
                Ok(Value::String(s.to_uppercase()))
            } else {
                Err("upper ต้องการข้อความ".to_string())
            }
        },
    }
}

/// ฟังก์ชัน lower(text) -> String
pub fn lower() -> BuiltinFunction {
    BuiltinFunction {
        name: "lower".to_string(),
        arity: 1,
        call: |args| {
            if let Value::String(s) = &args[0] {
                Ok(Value::String(s.to_lowercase()))
            } else {
                Err("lower ต้องการข้อความ".to_string())
            }
        },
    }
}