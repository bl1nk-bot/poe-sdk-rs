//! ฟังก์ชันพื้นฐาน (built-in functions)
//! แต่ละกลุ่มแยกเป็นไฟล์: string, math, logic, date

pub mod string;
pub mod math;
pub mod logic;
pub mod date;

use crate::functions::{BuiltinFunction, FunctionRegistry};

/// ลงทะเบียน built-in ทั้งหมดจาก Phase 1 ลงใน registry
pub fn register_all(registry: &mut FunctionRegistry) {
    // ฟังก์ชัน string
    registry.register(string::len());
    registry.register(string::upper());
    registry.register(string::lower());
    registry.register(string::contains());
    registry.register(string::starts_with());
    registry.register(string::ends_with());

    // ฟังก์ชัน logic
    registry.register(logic::if_fn()); // fn เป็นคำสงวน ใช้ชื่อ if_fn

    // math
    registry.register(math::abs());
    registry.register(math::min());
    registry.register(math::max());

    // date (phase 2)
}