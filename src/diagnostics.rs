use crate::error::FormulaError;

/// แสดงข้อความ error สำหรับผู้ใช้
/// โดยนำ span มาช่วยบอกว่าผิดที่ไหน พร้อมแสดง source code และ caret
pub fn format_error(source: &str, error: &FormulaError) -> String {
    let mut output = format!("[{}] {}\n", error.code, error.message);

    if let Some(span) = &error.span {
        // แยก source เป็นบรรทัด
        let lines: Vec<&str> = source.lines().collect();

        if span.start.line > 0 && (span.start.line as usize) <= lines.len() {
            let line_idx = span.start.line as usize - 1;
            let line = lines[line_idx];

            // แสดงบรรทัดที่มี error
            output.push_str(&format!(" {:4} | {}\n", span.start.line, line));

            // แสดง caret ชี้ตำแหน่ง
            let caret_col = span.start.column as usize;
            if caret_col <= line.len() {
                let spaces = " ".repeat(7 + caret_col.saturating_sub(1)); // 7 คือ "XXXX | "
                let carets = "^".repeat((span.end.column - span.start.column + 1).max(1) as usize);
                output.push_str(&format!("{}{}\n", spaces, carets));
            }
        }
    }

    output.trim_end().to_string()
}
