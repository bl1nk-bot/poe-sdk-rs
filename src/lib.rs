//! # Formula Engine
//!
//! A high-performance, extensible formula evaluation engine written in Rust.
//! Supports mathematical expressions, string operations, arrays, maps, and date/time functions
//! with comprehensive error reporting and type safety.
//!
//! ## Architecture
//!
//! The engine follows a layered architecture:
//!
//! 1. **Lexer**: Tokenizes input strings into tokens with span information
//! 2. **Parser**: Builds an Abstract Syntax Tree (AST) with proper operator precedence
//! 3. **Evaluator**: Executes the AST using a context and function registry
//! 4. **Built-ins**: Extensible collection of functions for various operations
//!
//! ## Features
//!
//! - **Type Safety**: Strong typing with runtime validation and clear error messages
//! - **Extensibility**: Easy to add new functions and data types
//! - **Performance**: Zero-cost abstractions and efficient evaluation
//! - **Error Reporting**: Detailed error messages with source location information
//! - **Advanced Types**: Support for arrays, maps, dates, and custom data types
//!
//! ## Supported Syntax
//!
//! ### Literals
//! - Numbers: `42`, `3.14`, `-5`
//! - Strings: `"hello"`, `"world"`
//! - Booleans: `true`, `false`
//! - Arrays: `[1, 2, 3]`, `["a", "b"]`
//! - Maps: `{key: "value", count: 42}`
//!
//! ### Operators
//! - Arithmetic: `+`, `-`, `*`, `/`
//! - Comparison: `<`, `>`, `<=`, `>=`, `==`, `!=`
//! - Logic: `&&`, `||`, `!`
//!
//! ### Functions
//! - String: `len()`, `upper()`, `lower()`, `contains()`
//! - Math: `abs()`, `min()`, `max()`
//! - Logic: `if()`
//! - Collections: `sum()`, `avg()`, `min()`, `max()`, `count()`, `join()`
//! - Date: `now()`, `date_add()`, `date_diff()`, `year()`, `month()`, `day()`
//!
//! ## Example
//!
//! ```
//! use bl1z::{tokenize, parse, evaluate, Context, FunctionRegistry};
//! use bl1z::builtins;
//!
//! // Create a function registry with built-in functions
//! let mut registry = FunctionRegistry::new();
//! builtins::register_all(&mut registry);
//!
//! // Parse and evaluate a formula
//! let tokens = tokenize("if(sum([1, 2, 3]) > 5, \"pass\", \"fail\")").unwrap();
//! let ast = parse(&tokens).unwrap();
//! let ctx = Context::new();
//! let result = evaluate(&ast, &ctx, &registry).unwrap();
//!
//! assert_eq!(result, bl1z::Value::String("pass".to_string()));
//! ```
//!
//! ## Error Handling
//!
//! The engine provides comprehensive error reporting with:
//!
//! - Error codes (e.g., "E101", "E401")
//! - Descriptive messages in Thai
//! - Source location information (line/column spans)
//! - Error categorization (LexError, ParseError, EvalError, etc.)
//!
//! ## Performance
//!
//! - Zero-copy string handling
//! - Efficient AST evaluation
//! - Minimal allocations during parsing
//! - Fast function lookups via HashMap
//!
//! ## Safety
//!
//! - Memory safe (no unsafe code)
//! - Type safe evaluation
//! - Comprehensive test coverage
//! - No panics in normal operation

/// ```
/// use bl1z::{tokenize, parse, evaluate, Context, FunctionRegistry};
/// use bl1z::builtins;
///
/// let mut registry = FunctionRegistry::new();
/// builtins::register_all(&mut registry);
///
/// let tokens = tokenize("if(true, \"pass\", \"fail\")").unwrap();
/// let ast = parse(&tokens).unwrap();
/// let ctx = Context::new();
/// let result = evaluate(&ast, &ctx, &registry).unwrap();
/// assert_eq!(result, bl1z::Value::String("pass".to_string()));
/// ```
pub mod ast;
pub mod builtins;
pub mod context;
pub mod diagnostics;
pub mod error;
pub mod eval;
pub mod functions;
pub mod lexer;
pub mod parser;
pub mod plugins;
pub mod profiling;
pub mod span;
pub mod value;

// re-export สิ่งที่ผู้ใช้ต้องการ
pub use ast::Expr;
pub use context::Context;
pub use error::FormulaError;
pub use eval::evaluate;
pub use eval::evaluate_mut;
pub use functions::FunctionRegistry;
pub use lexer::tokenize;
pub use parser::parse;
pub use plugins::{Plugin, PluginManager};
pub use value::Value;

/// Tokenizes a formula string into a sequence of tokens with span information.
///
/// This is the first step in formula processing. The lexer handles:
/// - Numbers, strings, booleans, null literals
/// - Identifiers for variables and functions
/// - Operators (+, -, *, /, etc.)
/// - Brackets and braces for arrays and maps
/// - Proper error reporting for invalid characters
///
/// # Arguments
/// * `source` - The formula string to tokenize
///
/// # Returns
/// * `Ok(Vec<Token>)` - Successfully tokenized sequence
/// * `Err(FormulaError)` - Lexical error with span information
///
/// # Example
/// ```
/// use bl1z::tokenize;
/// let tokens = tokenize("1 + 2 * 3").unwrap();
/// assert_eq!(tokens.len(), 5); // 3 numbers, 2 operators, 1 EOF
/// ```
///
/// Parses a sequence of tokens into an Abstract Syntax Tree (AST).
///
/// The parser builds a tree structure representing the formula's syntax,
/// handling operator precedence and associativity correctly.
///
/// # Arguments
/// * `tokens` - Token sequence from the lexer
///
/// # Returns
/// * `Ok(SpannedExpr)` - Successfully parsed AST with span information
/// * `Err(FormulaError)` - Parse error with location details
///
/// # Example
/// ```
/// use bl1z::{tokenize, parse};
/// let tokens = tokenize("1 + 2 * 3").unwrap();
/// let ast = parse(&tokens).unwrap();
/// // AST represents (1 + (2 * 3))
///
/// Evaluates an AST using the provided context and function registry.
/// This executes the formula, resolving variables and calling functions
/// as needed. Type checking occurs during evaluation.
///
/// # Arguments
/// * `expr` - The AST to evaluate
/// * `ctx` - Variable context for name resolution
/// * `registry` - Function registry for built-in and custom functions
///
/// # Returns
/// * `Ok(Value)` - Successfully computed result
/// * `Err(FormulaError)` - Evaluation error with context
///
/// # Example
/// ```
/// use bl1z::{tokenize, parse, evaluate, Context, FunctionRegistry, Value};
/// use bl1z::builtins;
///
/// let mut registry = FunctionRegistry::new();
/// builtins::register_all(&mut registry);
///
/// let tokens = tokenize("score + 10").unwrap();
/// let ast = parse(&tokens).unwrap();
/// let mut ctx = Context::new();
/// ctx.set("score", Value::Number(85.0));
///
/// let result = evaluate(&ast, &ctx, &registry).unwrap();
/// assert_eq!(result, Value::Number(95.0));
/// ```
#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;
