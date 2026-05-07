use crate::error::FormulaError;
use crate::value::Value;
use std::collections::HashMap;

pub struct BuiltinFunction {
    pub name: String,
    pub arity: usize,
    pub call: fn(&[Value]) -> Result<Value, FormulaError>,
}

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
}
