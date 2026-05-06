use crate::error::FormulaError;

/// แสดงข้อความ error สำหรับผู้ใช้
/// โดยนำ span มาช่วยบอกว่าผิดที่ไหน
pub fn format_error(source: &str, error: &FormulaError) -> String {
    match &error.span {
        Some(span) => {
            format!(
                "[{}] ที่บรรทัด {} คอลัมน์ {}-{}: {}",
                error.code,
                span.start.line,
                span.start.column,
                span.end.column,
                error.message
            )
        }
        None => format!("[{}] {}", error.code, error.message),
    }
}