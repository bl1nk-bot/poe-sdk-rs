use std::collections::HashMap;
use crate::value::Value;

/// พิมพ์เขียวของฟังก์ชัน built-in
pub struct BuiltinFunction {
    /// ชื่อฟังก์ชัน (เก็บซ้ำกับ key ใน registry ก็ได้)
    pub name: String,
    /// จำนวนอาร์กิวเมนต์ที่คาดหวัง (arity)
    pub arity: usize,
    /// ฟังก์ชันคำนวณจริง
    pub call: fn(&[Value]) -> Result<Value, String>,
}

/// ลงทะเบียน built-in functions และจัดการการเรียก
pub struct FunctionRegistry {
    functions: HashMap<String, BuiltinFunction>,
}

impl FunctionRegistry {
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
        }
    }

    /// เพิ่มฟังก์ชันเข้า registry
    pub fn register(&mut self, func: BuiltinFunction) {
        self.functions.insert(func.name.clone(), func);
    }

    /// ค้นหาฟังก์ชันตามชื่อ
    pub fn find(&self, name: &str) -> Option<&BuiltinFunction> {
        self.functions.get(name)
    }
}