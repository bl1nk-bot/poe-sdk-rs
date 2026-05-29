pub mod collection;
pub mod date;
pub mod functional;
pub mod logic;
pub mod math;
pub mod string;

use crate::functions::FunctionRegistry;

pub fn register_all(registry: &mut FunctionRegistry) {
    registry.register(string::len());
    registry.register(string::upper());
    registry.register(string::lower());
    registry.register(string::contains());
    registry.register(string::starts_with());
    registry.register(string::ends_with());
    registry.register(math::abs());
    registry.register(logic::if_fn());
    registry.register(collection::sum());
    registry.register(collection::avg());
    registry.register(collection::min_arr());
    registry.register(collection::max_arr());
    registry.register(collection::join());
    registry.register(collection::count());
    let reg_clone = registry.clone_box();
    registry.register(functional::map(&reg_clone));
    registry.register(functional::filter(&reg_clone));
    registry.register(functional::reduce(&reg_clone));
    registry.register(date::now());
    registry.register(date::date_add());
    registry.register(date::date());
    registry.register(date::date_diff());
    registry.register(date::year());
    registry.register(date::month());
    registry.register(date::day());
}
