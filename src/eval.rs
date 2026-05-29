use crate::ast::{Expr, SpannedExpr};
use crate::context::Context;
use crate::error::{ErrorKind, FormulaError};
use crate::functions::FunctionRegistry;
use crate::span::Span;
use crate::value::Value;

/// ประเมินผลนิพจน์ (SpannedExpr) และคืนค่าผลลัพธ์ (Value)
pub fn evaluate(
    expr: &SpannedExpr,
    ctx: &Context,
    registry: &FunctionRegistry,
) -> Result<Value, FormulaError> {
    evaluate_recursive(expr, ctx, registry, 0)
}

pub(crate) fn evaluate_recursive(
    expr: &SpannedExpr,
    ctx: &Context,
    registry: &FunctionRegistry,
    depth: usize,
) -> Result<Value, FormulaError> {
    if depth > 100 {
        return Err(FormulaError::new(
            ErrorKind::EvalError,
            "E303",
            "Recursion limit exceeded",
            Some(expr.meta.span),
        ));
    }
    let span = expr.meta.span;
    match &expr.expr {
        Expr::Literal(val) => Ok(val.clone()),
        Expr::Variable(name) => {
            let parts: Vec<&str> = name.split('.').collect();
            let mut current = ctx.get(parts[0]).cloned();
            if current.is_none() {
                return Err(FormulaError::new(
                    ErrorKind::ContextError,
                    "E601",
                    &format!("ไม่พบตัวแปร '{}'", parts[0]),
                    Some(span),
                ));
            }
            for i in 1..parts.len() {
                let part = parts[i];
                match &current {
                    Some(Value::Map(map)) => {
                        current = map.get(part).cloned();
                        if current.is_none() {
                            return Err(FormulaError::new(
                                ErrorKind::ContextError,
                                "E601",
                                &format!("ไม่พบตัวแปร '{}'", part),
                                Some(span),
                            ));
                        }
                    }
                    _ => {
                        return Err(FormulaError::new(
                            ErrorKind::TypeError,
                            "E401",
                            &format!("คาดหวังแผนที่ แต่ได้ชนิดอื่นที่ '{}'", parts[i - 1]),
                            Some(span),
                        ))
                    }
                }
            }
            current.ok_or_else(|| {
                FormulaError::new(
                    ErrorKind::ContextError,
                    "E601",
                    &format!("ไม่พบตัวแปร '{}'", name),
                    Some(span),
                )
            })
        }
        Expr::Grouping(inner) => evaluate_recursive(inner, ctx, registry, depth + 1),
        Expr::UnaryExpr { op, expr } => {
            let val = evaluate_recursive(expr, ctx, registry, depth + 1)?;
            match op {
                crate::ast::UnaryOp::Neg => {
                    if let Value::Number(n) = val {
                        Ok(Value::Number(-n))
                    } else {
                        Err(FormulaError::new(
                            ErrorKind::TypeError,
                            "E401",
                            "ตัวดำเนินการลบใช้ได้กับตัวเลขเท่านั้น",
                            Some(span),
                        ))
                    }
                }
                crate::ast::UnaryOp::Not => {
                    if let Value::Bool(b) = val {
                        Ok(Value::Bool(!b))
                    } else {
                        Err(FormulaError::new(
                            ErrorKind::TypeError,
                            "E401",
                            "ตัวดำเนินการ NOT ใช้ได้กับ boolean เท่านั้น",
                            Some(span),
                        ))
                    }
                }
            }
        }
        Expr::BinaryExpr { left, op, right } => {
            let l = evaluate_recursive(left, ctx, registry, depth + 1)?;
            let r = evaluate_recursive(right, ctx, registry, depth + 1)?;
            match op {
                crate::ast::BinaryOp::Add => add_values(l, r, span),
                crate::ast::BinaryOp::Sub => sub_values(l, r, span),
                crate::ast::BinaryOp::Mul => mul_values(l, r, span),
                crate::ast::BinaryOp::Div => div_values(l, r, span),
                crate::ast::BinaryOp::Eq => Ok(Value::Bool(l == r)),
                crate::ast::BinaryOp::NotEq => Ok(Value::Bool(l != r)),
                crate::ast::BinaryOp::Lt => compare_values(l, r, span, |a, b| a < b),
                crate::ast::BinaryOp::Gt => compare_values(l, r, span, |a, b| a > b),
                crate::ast::BinaryOp::LtEq => compare_values(l, r, span, |a, b| a <= b),
                crate::ast::BinaryOp::GtEq => compare_values(l, r, span, |a, b| a >= b),
                crate::ast::BinaryOp::And => logic_and(l, r, span),
                crate::ast::BinaryOp::Or => logic_or(l, r, span),
            }
        }
        Expr::ArrayLiteral(elements) => {
            let values: Result<Vec<Value>, _> = elements
                .iter()
                .map(|e| evaluate_recursive(e, ctx, registry, depth + 1))
                .collect();
            Ok(Value::Array(values?))
        }
        Expr::MapLiteral(pairs) => {
            let mut map = std::collections::HashMap::new();
            for (key, expr) in pairs {
                let value = evaluate_recursive(expr, ctx, registry, depth + 1)?;
                map.insert(key.clone(), value);
            }
            Ok(Value::Map(map))
        }
        Expr::FunctionCall { name, args } => {
            let evaluated_args: Vec<Value> = args
                .iter()
                .map(|a| evaluate_recursive(a, ctx, registry, depth + 1))
                .collect::<Result<_, _>>()?;
            if let Some((params, body)) = ctx.get_function(name) {
                if params.len() != evaluated_args.len() {
                    return Err(FormulaError::new(
                        ErrorKind::FunctionError,
                        "E503",
                        &format!(
                            "ฟังก์ชัน '{}' ต้องการ {} อาร์กิวเมนต์ แต่ได้ {}",
                            name,
                            params.len(),
                            evaluated_args.len()
                        ),
                        Some(span),
                    ));
                }
                let mut local_ctx = ctx.clone();
                for (param, val) in params.iter().zip(evaluated_args) {
                    local_ctx.set(param, val);
                }
                return evaluate_recursive(body, &local_ctx, registry, depth + 1);
            }
            let func = registry.find(name).ok_or_else(|| {
                FormulaError::new(
                    ErrorKind::FunctionError,
                    "E502",
                    &format!("ไม่พบฟังก์ชัน '{}'", name),
                    Some(span),
                )
            })?;
            if func.arity != args.len() {
                return Err(FormulaError::new(
                    ErrorKind::FunctionError,
                    "E503",
                    &format!(
                        "ฟังก์ชัน '{}' ต้องการ {} อาร์กิวเมนต์ แต่ได้ {}",
                        name,
                        func.arity,
                        args.len()
                    ),
                    Some(span),
                ));
            }
            (func.call)(&evaluated_args)
        }
        Expr::LambdaExpr { params, body } => Ok(Value::Closure {
            params: params.clone(),
            body: body.clone(),
            env: ctx.clone(),
        }),
        Expr::FunctionDef { .. } => Err(FormulaError::new(
            ErrorKind::EvalError,
            "E302",
            "Function definition ไม่รองรับใน evaluate โดยตรง",
            Some(span),
        )),
    }
}

fn add_values(l: Value, r: Value, span: Span) -> Result<Value, FormulaError> {
    match (l, r) {
        (Value::Number(a), Value::Number(b)) => Ok(Value::Number(a + b)),
        (Value::String(a), Value::String(b)) => Ok(Value::String(format!("{}{}", a, b))),
        _ => Err(FormulaError::new(
            ErrorKind::TypeError,
            "E401",
            "การบวกใช้ได้กับ number+number หรือ string+string เท่านั้น",
            Some(span),
        )),
    }
}
fn sub_values(l: Value, r: Value, span: Span) -> Result<Value, FormulaError> {
    if let (Value::Number(a), Value::Number(b)) = (l, r) {
        Ok(Value::Number(a - b))
    } else {
        Err(FormulaError::new(
            ErrorKind::TypeError,
            "E401",
            "การลบใช้ได้กับตัวเลขเท่านั้น",
            Some(span),
        ))
    }
}
fn mul_values(l: Value, r: Value, span: Span) -> Result<Value, FormulaError> {
    if let (Value::Number(a), Value::Number(b)) = (l, r) {
        Ok(Value::Number(a * b))
    } else {
        Err(FormulaError::new(
            ErrorKind::TypeError,
            "E401",
            "ใช้กับตัวเลข",
            Some(span),
        ))
    }
}
fn div_values(l: Value, r: Value, span: Span) -> Result<Value, FormulaError> {
    if let (Value::Number(a), Value::Number(b)) = (l, r) {
        if b == 0.0 {
            Err(FormulaError::new(
                ErrorKind::EvalError,
                "E301",
                "หารด้วยศูนย์",
                Some(span),
            ))
        } else {
            Ok(Value::Number(a / b))
        }
    } else {
        Err(FormulaError::new(
            ErrorKind::TypeError,
            "E401",
            "ใช้กับตัวเลข",
            Some(span),
        ))
    }
}
fn compare_values<F>(l: Value, r: Value, span: Span, f: F) -> Result<Value, FormulaError>
where
    F: Fn(f64, f64) -> bool,
{
    if let (Value::Number(a), Value::Number(b)) = (l, r) {
        Ok(Value::Bool(f(a, b)))
    } else {
        Err(FormulaError::new(
            ErrorKind::TypeError,
            "E401",
            "การเปรียบเทียบใช้ได้กับตัวเลขเท่านั้น",
            Some(span),
        ))
    }
}
fn logic_and(l: Value, r: Value, span: Span) -> Result<Value, FormulaError> {
    if let (Value::Bool(a), Value::Bool(b)) = (l, r) {
        Ok(Value::Bool(a && b))
    } else {
        Err(FormulaError::new(
            ErrorKind::TypeError,
            "E401",
            "AND ใช้ได้กับ boolean เท่านั้น",
            Some(span),
        ))
    }
}
fn logic_or(l: Value, r: Value, span: Span) -> Result<Value, FormulaError> {
    if let (Value::Bool(a), Value::Bool(b)) = (l, r) {
        Ok(Value::Bool(a || b))
    } else {
        Err(FormulaError::new(
            ErrorKind::TypeError,
            "E401",
            "OR ใช้ได้กับ boolean เท่านั้น",
            Some(span),
        ))
    }
}
