/// Represents a value in the formula engine.
///
/// Values are the result of evaluating expressions and can be used
/// as function arguments or stored in contexts.
///
/// # Supported Types
///
/// - **Number**: 64-bit floating point numbers for mathematical operations
/// - **String**: UTF-8 strings for text operations
/// - **Bool**: Boolean values for logical operations
/// - **Null**: Represents absence of value or uninitialized state
/// - **Array**: Ordered collection of values, supports nesting
/// - **Map**: Key-value dictionary with string keys
///
/// # Type Safety
///
/// All operations perform runtime type checking. Attempting to perform
/// incompatible operations (e.g., adding string to number) will result
/// in a `TypeError` with detailed error messages.
///
/// # Examples
///
/// ```
/// use formula_engine::Value;
/// use std::collections::HashMap;
///
/// // Basic values
/// let num = Value::Number(42.0);
/// let str = Value::String("hello".to_string());
/// let bool_val = Value::Bool(true);
/// let null_val = Value::Null;
///
/// // Arrays
/// let arr = Value::Array(vec![
///     Value::Number(1.0),
///     Value::Number(2.0),
///     Value::String("three".to_string())
/// ]);
///
/// // Maps
/// let mut map = HashMap::new();
/// map.insert("name".to_string(), Value::String("John".to_string()));
/// map.insert("age".to_string(), Value::Number(30.0));
/// let map_val = Value::Map(map);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// 64-bit floating point number for mathematical calculations.
    ///
    /// Supports all standard arithmetic operations and comparisons.
    /// Uses `f64` for maximum precision and compatibility.
    Number(f64),

    /// UTF-8 string for text operations and concatenation.
    ///
    /// Supports string functions like `len()`, `upper()`, `contains()`, etc.
    /// Can be concatenated with `+` operator.
    String(String),

    /// Boolean value for logical operations.
    ///
    /// Used in conditional expressions and logical operators (`&&`, `||`, `!`).
    Bool(bool),

    /// Represents null/absent value.
    ///
    /// Useful for optional values and uninitialized states.
    /// Cannot participate in most operations (will cause TypeError).
    Null,

    /// Ordered collection of values with O(1) indexing.
    ///
    /// Supports nesting (arrays of arrays) and heterogeneous types.
    /// Used with collection functions like `sum()`, `avg()`, `count()`, `join()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use formula_engine::Value;
    /// let numbers = Value::Array(vec![
    ///     Value::Number(1.0),
    ///     Value::Number(2.0),
    ///     Value::Number(3.0)
    /// ]);
    ///
    /// let nested = Value::Array(vec![
    ///     Value::Array(vec![Value::Number(1.0), Value::Number(2.0)]),
    ///     Value::Array(vec![Value::Number(3.0), Value::Number(4.0)])
    /// ]);
    /// ```
    Array(Vec<Value>),

    /// Key-value dictionary with string keys.
    ///
    /// Keys must be valid identifiers. Values can be any type including nested maps.
    /// Currently used for object literals but can be extended for more complex data structures.
    ///
    /// # Examples
    ///
    /// ```
    /// use formula_engine::Value;
    /// use std::collections::HashMap;
    ///
    /// let mut person = HashMap::new();
    /// person.insert("name".to_string(), Value::String("Alice".to_string()));
    /// person.insert("age".to_string(), Value::Number(25.0));
    /// person.insert("active".to_string(), Value::Bool(true));
    ///
    /// let person_val = Value::Map(person);
    /// ```
    Map(std::collections::HashMap<String, Value>),
}
