use crate::value::Value;
use std::collections::HashMap;

/// สภาพแวดล้อมสำหรับเก็บตัวแปรและค่าคงที่
#[derive(Debug, Clone)]
pub struct Context {
    variables: HashMap<String, Value>,
}

impl Context {
    /// สร้าง context ใหม่
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
        }
    }

    /// ตั้งค่าตัวแปร
    pub fn set(&mut self, name: &str, value: Value) {
        self.variables.insert(name.to_string(), value);
    }

    /// รับค่าตัวแปร
    pub fn get(&self, name: &str) -> Option<&Value> {
        self.variables.get(name)
    }
}
