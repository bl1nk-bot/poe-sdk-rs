/// ชนิดข้อมูลที่ใช้ใน engine
/// Phase 6.1: เพิ่ม Array
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Number(f64),
    String(String),
    Bool(bool),
    Null,
    Array(Vec<Value>),   // เพิ่มสำหรับ Array
}
