//! ฟังก์ชันพื้นฐาน (built-in functions)
//! แต่ละกลุ่มแยกเป็นไฟล์: string, math, logic, date

pub mod string;
pub mod math;
pub mod logic;
pub mod date;
pub mod collection;   // เพิ่ม

use crate::functions::FunctionRegistry;

/// ลงทะเบียน built-in ทั้งหมด (Phase 1 + Phase 6.1)
pub fn register_all(registry: &mut FunctionRegistry) {
    // ฟังก์ชัน string
    registry.register(string::len());
    registry.register(string::upper());
    registry.register(string::lower());
    registry.register(string::contains());
    registry.register(string::starts_with());
    registry.register(string::ends_with());

    // ฟังก์ชัน math (min/max ถูกย้ายไป collection)
    registry.register(math::abs());

    // ฟังก์ชัน logic
    registry.register(logic::if_fn());

    // ฟังก์ชัน array (Phase 6.1)
    registry.register(collection::sum());
    registry.register(collection::avg());
    registry.register(collection::min_arr());
    registry.register(collection::max_arr());
    registry.register(collection::join());
    registry.register(collection::count());
}