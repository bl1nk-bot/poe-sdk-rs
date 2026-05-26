# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-05-18

### Added
- **Access Chaining**: Dot notation (`obj.prop`) and bracket indexing (`arr[0]`) for nested data access
- **Chained Access**: Full support for mixed chains like `users[0].name`, `config.db.host`, `a.b[0].c.d`
- **Context Scoping**: Parent chain variable resolution — child scopes inherit parent variables and can shadow them without mutation
- **Context Utilities**: `Context::with_parent()`, `Context::get_all()`, `Context::depth()` for nested scope management
- **Error Codes**: `E207` (PropertyNotFound), `E208` (IndexOutOfBounds) for access errors
- **Test Infrastructure**: Extracted 169 tests to `src/lib_tests.rs`, renamed to `{subject}_{scenario}_{expected_outcome}` convention
- **pretty_assertions**: Added dev-dependency for diff output on test failures

### Changed
- **Parser**: Replaced string-concatenation dot hack with proper `parse_postfix()` method supporting recursive access chains
- **Evaluator**: Replaced `name.split('.')` hack with dedicated `PropertyAccess`/`IndexAccess` evaluation
- **Context**: Internal storage changed from `HashMap` to `BTreeMap` for deterministic iteration
- **Context**: Now uses `Rc<Context>` for parent chain sharing (clone-safe, zero-copy parent references)
- **FormulaError**: Added `PartialEq` derive for test assertions

### Fixed
- 4 pre-existing test failures: error code mismatch (`E401` → `E501`) for function type errors
- 1 snapshot test: `array_function_wrong_type` snapshot updated to match actual error code

### Breaking Changes
- **Context**: `Context` no longer implements `Default` via derive (uses explicit impl). Clone behavior changed — clones share parent via `Rc`, not deep copy.
- **Internal**: `Expr::Variable` no longer supports dot notation (e.g., `user.score` is now `PropertyAccess`, not `Variable("user.score")`). This only affects code matching on `Expr` variants directly.

## [0.1.0] - 2026-05-07

### Added
- **Core bl1z**: Complete implementation of lexer, parser, AST, and evaluator
- **Data Types**: Support for Number, String, Bool, Null, Array, and Map types
- **Array Literals**: Syntax `[1, 2, 3]` with nested array support
- **Map Literals**: Syntax `{key: "value"}` for object-like data structures
- **Collection Functions**: `sum()`, `avg()`, `min()`, `max()`, `join()`, `count()` for arrays
- **Date/Time Functions**: `now()`, `date()`, `date_diff()`, `year()`, `month()`, `day()` using jiff library
- **String Functions**: `len()`, `upper()`, `lower()`, `contains()`, `starts_with()`, `ends_with()`
- **Math Functions**: `abs()`, `min()`, `max()` for numbers
- **Logic Functions**: `if()` conditional function
- **Performance Tools**: Benchmarking with Criterion, profiling utilities
- **Error Handling**: Comprehensive error reporting with span information and Thai language messages
- **Testing**: Extensive unit tests, integration tests, and snapshot tests for error formatting
- **CI/CD**: GitHub Actions workflow with automated testing, formatting, and linting

### Changed
- **Date Library**: Replaced `chrono` with pure Rust `jiff` library (v0.2) for better performance and no C dependencies
- **Build Optimization**: Added release profile with LTO, codegen-units=1, and stripping for optimal binary size

### Technical Details
- **Zero-cost abstractions** with efficient AST evaluation
- **Extensible architecture** via function registry system
- **Strong type safety** with runtime validation
- **Memory safe** - no unsafe code
- **Performance focused** - <10μs for typical evaluations

### Dependencies
- `jiff = "0.2"` - Pure Rust date/time library
- `criterion = "0.5"` - Benchmarking framework
- `insta = "1.34"` - Snapshot testing

### Compatibility
- **Rust**: 1.70+ (MSRV - Minimum Supported Rust Version)
- **Platforms**: All platforms supported by Rust toolchain
- **No C dependencies** - Pure Rust implementation

---

## Types of changes
- `Added` for new features
- `Changed` for changes in existing functionality
- `Deprecated` for soon-to-be removed features
- `Removed` for now removed features
- `Fixed` for any bug fixes
- `Security` in case of vulnerabilities