use crate::value::Value;
use std::collections::HashMap;

/// Execution context for storing variables and constants.
///
/// The Context provides variable resolution during formula evaluation.
/// Variables can be set before evaluation and accessed by name during execution.
/// This allows formulas to reference external data and configuration.
///
/// # Thread Safety
///
/// Context is not thread-safe by design. If you need concurrent access,
/// wrap it in appropriate synchronization primitives (e.g., `Arc<Mutex<Context>>`).
///
/// # Examples
///
/// ```
/// use formula_engine::{Context, Value, tokenize, parse, evaluate, FunctionRegistry};
/// use formula_engine::builtins;
///
/// let mut registry = FunctionRegistry::new();
/// builtins::register_all(&mut registry);
///
/// // Create context and set variables
/// let mut ctx = Context::new();
/// ctx.set("score", Value::Number(85.0));
/// ctx.set("bonus", Value::Number(15.0));
/// ctx.set("name", Value::String("Alice".to_string()));
///
/// // Evaluate formula using context variables
/// let tokens = tokenize("score + bonus").unwrap();
/// let ast = parse(&tokens).unwrap();
/// let result = evaluate(&ast, &ctx, &registry).unwrap();
/// assert_eq!(result, Value::Number(100.0));
///
/// // Variables can be updated
/// ctx.set("bonus", Value::Number(20.0));
/// let result2 = evaluate(&ast, &ctx, &registry).unwrap();
/// assert_eq!(result2, Value::Number(105.0));
/// ```
///
/// # Variable Naming
///
/// Variable names follow identifier rules:
/// - Must start with letter or underscore
/// - Can contain letters, digits, and underscores
/// - Case sensitive
/// - Cannot be reserved words (true, false, null)
///
/// # Performance
///
/// Variable lookup is O(1) average case using HashMap.
/// Memory usage scales with number of variables stored.
#[derive(Debug, Clone, Default)]
pub struct Context {
    variables: HashMap<String, Value>,
}

impl Context {
    /// Creates a new empty context.
    ///
    /// The context starts with no variables defined.
    /// Use `set()` to add variables before evaluation.
    ///
    /// # Examples
    ///
    /// ```
    /// use formula_engine::Context;
    /// let ctx = Context::new();
    /// assert!(ctx.get("undefined").is_none());
    /// ```
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
        }
    }

    /// Sets a variable in the context.
    ///
    /// If the variable already exists, its value is updated.
    /// Variable names are case-sensitive and must be valid identifiers.
    ///
    /// # Arguments
    /// * `name` - Variable name (must be valid identifier)
    /// * `value` - Value to store
    ///
    /// # Examples
    ///
    /// ```
    /// use formula_engine::{Context, Value};
    /// let mut ctx = Context::new();
    ///
    /// ctx.set("count", Value::Number(42.0));
    /// ctx.set("message", Value::String("hello".to_string()));
    /// ctx.set("active", Value::Bool(true));
    ///
    /// assert_eq!(*ctx.get("count").unwrap(), Value::Number(42.0));
    /// ```
    pub fn set(&mut self, name: &str, value: Value) {
        self.variables.insert(name.to_string(), value);
    }

    /// Retrieves a variable from the context.
    ///
    /// Returns `None` if the variable is not defined.
    /// During evaluation, undefined variables cause a `ContextError`.
    ///
    /// # Arguments
    /// * `name` - Variable name to look up
    ///
    /// # Returns
    /// * `Some(&Value)` - Reference to the stored value
    /// * `None` - Variable not found
    ///
    /// # Examples
    ///
    /// ```
    /// use formula_engine::{Context, Value};
    /// let mut ctx = Context::new();
    ///
    /// ctx.set("score", Value::Number(100.0));
    ///
    /// assert_eq!(ctx.get("score"), Some(&Value::Number(100.0)));
    /// assert_eq!(ctx.get("undefined"), None);
    /// ```
    pub fn get(&self, name: &str) -> Option<&Value> {
        self.variables.get(name)
    }
}
