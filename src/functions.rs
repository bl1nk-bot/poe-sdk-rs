use crate::error::FormulaError;
use crate::value::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// สัญญาของฟังก์ชันที่ลงทะเบียนในระบบ
pub type FunctionCall = Arc<dyn Fn(&[Value]) -> Result<Value, FormulaError> + Send + Sync>;

/// ตัวแทนของฟังก์ชัน Built-in
#[derive(Clone)]
pub struct BuiltinFunction {
    pub name: String,
    pub arity: usize,
    pub call: FunctionCall,
}

/// ทะเบียนสำหรับเก็บและค้นหาฟังก์ชัน Built-in
#[derive(Default, Clone)]
pub struct FunctionRegistry {
    functions: HashMap<String, BuiltinFunction>,
}

impl FunctionRegistry {
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
        }
    }
    pub fn register(&mut self, func: BuiltinFunction) {
        self.functions.insert(func.name.clone(), func);
    }
    pub fn find(&self, name: &str) -> Option<&BuiltinFunction> {
        self.functions.get(name)
    }
    pub(crate) fn clone_box(&self) -> Self {
        self.clone()
    }
}
