//! ฟังก์ชันเกี่ยวกับวันที่และเวลา (Phase 6.3)

use crate::error::{ErrorKind, FormulaError};
use crate::functions::BuiltinFunction;
use crate::value::Value;
use chrono::{DateTime, Duration, NaiveDate, NaiveDateTime, Utc};

/// now() -> String
/// คืนวันที่และเวลาปัจจุบันในรูปแบบ ISO 8601
pub fn now() -> BuiltinFunction {
    BuiltinFunction {
        name: "now".to_string(),
        arity: 0,
        call: |_| {
            let now: DateTime<Utc> = Utc::now();
            Ok(Value::String(now.to_rfc3339()))
        },
    }
}

/// date_add(date_str, days) -> String
/// เพิ่มจำนวนวันให้กับวันที่ (date_str ต้องเป็น ISO 8601)
pub fn date_add() -> BuiltinFunction {
    BuiltinFunction {
        name: "date_add".to_string(),
        arity: 2,
        call: |args| {
            let date_str = require_string(&args[0])?;
            let days = require_number(&args[1])? as i64;

            let dt = DateTime::parse_from_rfc3339(&date_str)
                .or_else(|_| {
                    // ลอง parse เป็น date เฉยๆ (YYYY-MM-DD)
                    NaiveDate::parse_from_str(&date_str, "%Y-%m-%d")
                        .map(|d| d.and_hms(0, 0, 0).and_utc())
                })
                .map_err(|_| {
                    FormulaError::new(
                        ErrorKind::FunctionError,
                        "E012",
                        "รูปแบบวันที่ไม่ถูกต้อง ต้องเป็น ISO 8601 หรือ YYYY-MM-DD",
                        None,
                    )
                })?;

            let new_dt = dt + Duration::days(days);
            Ok(Value::String(new_dt.to_rfc3339()))
        },
    }
}

/// date_diff(date1, date2) -> Number
/// คืนจำนวนวันระหว่างสองวันที่ (date1 - date2)
pub fn date_diff() -> BuiltinFunction {
    BuiltinFunction {
        name: "date_diff".to_string(),
        arity: 2,
        call: |args| {
            let date1_str = require_string(&args[0])?;
            let date2_str = require_string(&args[1])?;

            let dt1 = parse_date(&date1_str)?;
            let dt2 = parse_date(&date2_str)?;

            let duration = dt1.signed_duration_since(dt2);
            Ok(Value::Number(duration.num_days() as f64))
        },
    }
}

/// year(date_str) -> Number
/// คืนปีจากวันที่
pub fn year() -> BuiltinFunction {
    BuiltinFunction {
        name: "year".to_string(),
        arity: 1,
        call: |args| {
            let date_str = require_string(&args[0])?;
            let dt = parse_date(&date_str)?;
            Ok(Value::Number(dt.year() as f64))
        },
    }
}

/// month(date_str) -> Number
/// คืนเดือนจากวันที่ (1-12)
pub fn month() -> BuiltinFunction {
    BuiltinFunction {
        name: "month".to_string(),
        arity: 1,
        call: |args| {
            let date_str = require_string(&args[0])?;
            let dt = parse_date(&date_str)?;
            Ok(Value::Number(dt.month() as f64))
        },
    }
}

/// day(date_str) -> Number
/// คืนวันที่จากวันที่ (1-31)
pub fn day() -> BuiltinFunction {
    BuiltinFunction {
        name: "day".to_string(),
        arity: 1,
        call: |args| {
            let date_str = require_string(&args[0])?;
            let dt = parse_date(&date_str)?;
            Ok(Value::Number(dt.day() as f64))
        },
    }
}

// -- Helpers --

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

fn parse_date(date_str: &str) -> Result<DateTime<Utc>, FormulaError> {
    DateTime::parse_from_rfc3339(date_str)
        .or_else(|_| {
            NaiveDate::parse_from_str(date_str, "%Y-%m-%d").map(|d| d.and_hms(0, 0, 0).and_utc())
        })
        .map_err(|_| {
            FormulaError::new(
                ErrorKind::FunctionError,
                "E012",
                "รูปแบบวันที่ไม่ถูกต้อง ต้องเป็น ISO 8601 หรือ YYYY-MM-DD",
                None,
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call_fn(f: BuiltinFunction, args: Vec<Value>) -> Result<Value, FormulaError> {
        (f.call)(&args)
    }

    // -- now() tests --

    #[test]
    fn test_now() {
        let result = call_fn(now(), vec![]).unwrap();
        match result {
            Value::String(s) => {
                // ตรวจว่าเป็น ISO 8601 format คร่าวๆ
                assert!(s.contains('T'));
                assert!(s.contains('Z') || s.contains('+'));
            }
            _ => panic!("expected string"),
        }
    }

    // -- date_add() tests --

    #[test]
    fn test_date_add_basic() {
        let result = call_fn(
            date_add(),
            vec![
                Value::String("2023-01-01T00:00:00Z".to_string()),
                Value::Number(5.0),
            ],
        )
        .unwrap();
        match result {
            Value::String(s) => {
                assert!(s.starts_with("2023-01-06"));
            }
            _ => panic!("expected string"),
        }
    }

    #[test]
    fn test_date_add_negative() {
        let result = call_fn(
            date_add(),
            vec![
                Value::String("2023-01-10T00:00:00Z".to_string()),
                Value::Number(-3.0),
            ],
        )
        .unwrap();
        match result {
            Value::String(s) => {
                assert!(s.starts_with("2023-01-07"));
            }
            _ => panic!("expected string"),
        }
    }

    #[test]
    fn test_date_add_date_only() {
        let result = call_fn(
            date_add(),
            vec![Value::String("2023-01-01".to_string()), Value::Number(1.0)],
        )
        .unwrap();
        match result {
            Value::String(s) => {
                assert!(s.starts_with("2023-01-02"));
            }
            _ => panic!("expected string"),
        }
    }

    #[test]
    fn test_date_add_invalid_date() {
        let result = call_fn(
            date_add(),
            vec![Value::String("invalid".to_string()), Value::Number(1.0)],
        );
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.code, "E012");
    }

    #[test]
    fn test_date_add_non_string_arg() {
        let result = call_fn(date_add(), vec![Value::Number(123.0), Value::Number(1.0)]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.code, "E006");
    }

    // -- date_diff() tests --

    #[test]
    fn test_date_diff_basic() {
        let result = call_fn(
            date_diff(),
            vec![
                Value::String("2023-01-05T00:00:00Z".to_string()),
                Value::String("2023-01-01T00:00:00Z".to_string()),
            ],
        )
        .unwrap();
        assert_eq!(result, Value::Number(4.0));
    }

    #[test]
    fn test_date_diff_reverse() {
        let result = call_fn(
            date_diff(),
            vec![
                Value::String("2023-01-01T00:00:00Z".to_string()),
                Value::String("2023-01-05T00:00:00Z".to_string()),
            ],
        )
        .unwrap();
        assert_eq!(result, Value::Number(-4.0));
    }

    // -- year() tests --

    #[test]
    fn test_year_basic() {
        let result = call_fn(
            year(),
            vec![Value::String("2023-05-15T10:30:00Z".to_string())],
        )
        .unwrap();
        assert_eq!(result, Value::Number(2023.0));
    }

    // -- month() tests --

    #[test]
    fn test_month_basic() {
        let result = call_fn(
            month(),
            vec![Value::String("2023-05-15T10:30:00Z".to_string())],
        )
        .unwrap();
        assert_eq!(result, Value::Number(5.0));
    }

    // -- day() tests --

    #[test]
    fn test_day_basic() {
        let result = call_fn(
            day(),
            vec![Value::String("2023-05-15T10:30:00Z".to_string())],
        )
        .unwrap();
        assert_eq!(result, Value::Number(15.0));
    }
}
