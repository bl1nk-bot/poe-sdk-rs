use crate::value::Value;
use std::collections::HashMap;

/// บริบทการทำงาน (Execution Context) สำหรับเก็บตัวแปรและฟังก์ชัน
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Context {
    variables: HashMap<String, Value>,
    functions: HashMap<String, (Vec<String>, crate::ast::SpannedExpr)>,
}

impl Context {
    /// สร้างบริบทใหม่ที่ว่างเปล่า
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            functions: HashMap::new(),
        }
    }

    /// กำหนดค่าให้กับตัวแปรในบริบท
    pub fn set(&mut self, name: &str, value: Value) {
        self.variables.insert(name.to_string(), value);
    }

    /// ดึงค่าของตัวแปรจากบริบท
    pub fn get(&self, name: &str) -> Option<&Value> {
        self.variables.get(name)
    }

    /// กำหนดฟังก์ชันที่ผู้ใช้สร้างเองลงในบริบท
    pub fn set_function(&mut self, name: &str, params: Vec<String>, body: crate::ast::SpannedExpr) {
        self.functions.insert(name.to_string(), (params, body));
    }

    /// ดึงข้อมูลฟังก์ชันที่ผู้ใช้สร้างเองจากบริบท
    pub fn get_function(&self, name: &str) -> Option<&(Vec<String>, crate::ast::SpannedExpr)> {
        self.functions.get(name)
    }
}
