//! ฟังก์ชันเกี่ยวกับ array (Phase 6.1)

use crate::error::{ErrorKind, FormulaError};
use crate::functions::BuiltinFunction;
use crate::value::Value;

/// sum(array) -> Number
/// หาผลรวมของตัวเลขใน array
pub fn sum() -> BuiltinFunction {
    BuiltinFunction {
        name: "sum".to_string(),
        arity: 1,
        call: |args| {
            let arr = require_array(&args[0])?;
            let total: f64 = arr
                .iter()
                .map(|v| require_number(v))
                .sum::<Result<f64, FormulaError>>()?;
            Ok(Value::Number(total))
        },
    }
}

/// avg(array) -> Number
/// หาค่าเฉลี่ยของตัวเลขใน array
pub fn avg() -> BuiltinFunction {
    BuiltinFunction {
        name: "avg".to_string(),
        arity: 1,
        call: |args| {
            let arr = require_array(&args[0])?;
            if arr.is_empty() {
                return Err(FormulaError::new(
                    ErrorKind::FunctionError,
                    "E011",
                    "avg ไม่สามารถใช้กับ array ว่าง",
                    None,
                ));
            }
            let total: f64 = arr
                .iter()
                .map(|v| require_number(v))
                .sum::<Result<f64, FormulaError>>()?;
            Ok(Value::Number(total / arr.len() as f64))
        },
    }
}

/// min(array) -> Number
/// คืนค่าต่ำสุดใน array
pub fn min_arr() -> BuiltinFunction {
    BuiltinFunction {
        name: "min".to_string(),
        arity: 1,
        call: |args| {
            let arr = require_array(&args[0])?;
            if arr.is_empty() {
                return Err(FormulaError::new(
                    ErrorKind::FunctionError,
                    "E011",
                    "min ไม่สามารถใช้กับ array ว่าง",
                    None,
                ));
            }
            let mut min_val: Option<f64> = None;
            for v in arr {
                let n = require_number(v)?;
                min_val = Some(match min_val {
                    None => n,
                    Some(cur) => if n < cur { n } else { cur },
                });
            }
            Ok(Value::Number(min_val.unwrap()))
        },
    }
}

/// max(array) -> Number
/// คืนค่าสูงสุดใน array
pub fn max_arr() -> BuiltinFunction {
    BuiltinFunction {
        name: "max".to_string(),
        arity: 1,
        call: |args| {
            let arr = require_array(&args[0])?;
            if arr.is_empty() {
                return Err(FormulaError::new(
                    ErrorKind::FunctionError,
                    "E011",
                    "max ไม่สามารถใช้กับ array ว่าง",
                    None,
                ));
            }
            let mut max_val: Option<f64> = None;
            for v in arr {
                let n = require_number(v)?;
                max_val = Some(match max_val {
                    None => n,
                    Some(cur) => if n > cur { n } else { cur },
                });
            }
            Ok(Value::Number(max_val.unwrap()))
        },
    }
}

/// join(array, separator) -> String
/// ต่อสมาชิก array ที่เป็น string ด้วยตัวคั่น
pub fn join() -> BuiltinFunction {
    BuiltinFunction {
        name: "join".to_string(),
        arity: 2,
        call: |args| {
            let arr = require_array(&args[0])?;
            let sep = require_string(&args[1])?;
            let parts: Result<Vec<String>, FormulaError> = arr
                .iter()
                .map(|v| require_string(v))
                .collect();
            Ok(Value::String(parts?.join(&sep)))
        },
    }
}

/// count(array) -> Number
/// นับจำนวนสมาชิกใน array (เหมือน len)
pub fn count() -> BuiltinFunction {
    BuiltinFunction {
        name: "count".to_string(),
        arity: 1,
        call: |args| {
            let arr = require_array(&args[0])?;
            Ok(Value::Number(arr.len() as f64))
        },
    }
}

// -- Helpers --

fn require_array(value: &Value) -> Result<&Vec<Value>, FormulaError> {
    match value {
        Value::Array(arr) => Ok(arr),
        _ => Err(FormulaError::new(
            ErrorKind::FunctionError,
            "E006",
            "ต้องการ array",
            None,
        )),
    }
}

fn require_number(value: &Value) -> Result<f64, FormulaError> {
    match value {
        Value::Number(n) => Ok(*n),
        _ => Err(FormulaError::new(
            ErrorKind::FunctionError,
            "E006",
            "ต้องการตัวเลข",
            None,
        )),
    }
}

fn require_string(value: &Value) -> Result<String, FormulaError> {
    match value {
        Value::String(s) => Ok(s.clone()),
        _ => Err(FormulaError::new(
            ErrorKind::FunctionError,
            "E006",
            "ต้องการข้อความ",
            None,
        )),
    }
}
