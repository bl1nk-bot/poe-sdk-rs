/// ชนิดข้อมูลที่ใช้ใน engine
/// เริ่มต้นมี Number, String, Bool, Null
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Number(f64),
    String(String),
    Bool(bool),
    Null,
    // Phase 2: Array(Vec<Value>), DateTime(...), Object(...)
}