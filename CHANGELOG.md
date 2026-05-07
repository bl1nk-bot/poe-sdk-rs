# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05-07

### Added
- **Core Formula Engine**: Complete implementation of lexer, parser, AST, and evaluator
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