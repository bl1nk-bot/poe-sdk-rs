# AGENTS.md — bl1z

## Project

- **Crate name**: `bl1z` — a Notion-like formula evaluation engine in Rust
- **Single crate** (no workspace). Entry: `src/lib.rs`
- **Edition**: 2021 | **MSRV**: 1.90.0 (enforced in CI)
- **Dependencies**: `jiff` (date/time). Dev-deps: `criterion`, `insta`
- **No `unsafe` code** anywhere

## Architecture

Pipeline: **Lexer** (`src/lexer.rs`) → **Parser** (`src/parser.rs`) → **Evaluator** (`src/eval.rs`)

Core modules:
- `src/ast.rs` — AST nodes including `PropertyAccess` and `IndexAccess` (Phase 8)
- `src/value.rs` — `Value` enum (Number, String, Bool, Null, Array, Map)
- `src/context.rs` — variable storage with parent chain scoping (`Rc<Context>`)
- `src/functions.rs` — `FunctionRegistry`
- `src/error.rs` — `FormulaError` with error codes (E1xx–E6xx) and Thai messages
- `src/builtins/` — built-in functions: string, math, logic, date, collection
- `src/diagnostics.rs` — error formatting with span info
- `src/profiling.rs` — performance profiling

Public re-exports in `lib.rs`: `tokenize`, `parse`, `evaluate`, `Context`, `FunctionRegistry`, `Value`, `Expr`, `FormulaError`

## Developer Commands

```bash
cargo check                          # type-check only
cargo fmt --all -- --check           # format check (CI gate)
cargo clippy --all-targets --all-features -- -D warnings  # lint (CI gate, zero warnings allowed)
cargo test --verbose                 # all unit + integration + doc tests
cargo test --doc                     # doc tests only
cargo test --test error_snapshots    # insta snapshot tests only
cargo bench                          # benchmarks (criterion, harness = false)
cargo doc --no-deps --document-private-items  # generate docs
cargo run --example basic            # run basic example
cargo run --example advanced         # run advanced example
```

**CI order**: fmt → clippy → test → doc-test → bench → doc

## Testing

- **Unit tests**: `src/lib_tests.rs` (extracted from inline module) + per-module `#[cfg(test)]` blocks
- **Integration tests**: `tests/error_snapshots.rs` uses `insta` for snapshot testing
- **Snapshot updates**: `INSTA_UPDATE=1 cargo test --test error_snapshots` (CI runs this on `main` branch only)
- Snapshots live in `tests/snapshots/` — 11 error snapshot files
- When changing error messages or format, run with `INSTA_UPDATE=1` and review diffs
- **Test naming**: `{subject}_{scenario}_{expected_outcome}` (e.g., `evaluate_sum_empty_array_returns_zero`)
- **Assertions**: `pretty_assertions::assert_eq` for diff output on failures
- **Helper functions**: `eval_formula(formula)` and `eval_with_ctx(formula, ctx)` reduce boilerplate

## Release Profile

`[profile.release]` in `Cargo.toml`: `opt-level = 3`, `lto = true`, `codegen-units = 1`, `panic = "abort"`, `strip = true`

## Conventions

- **Commit messages**: conventional commits (`type(scope): description`)
- **Error codes**: E1xx=Lex, E2xx=Parse, E3xx=Eval, E4xx=Type, E5xx=Function, E6xx=Context
- **Error messages**: written in Thai
- **Doc comments**: required on all public APIs with `# Arguments`, `# Returns`, `# Examples`

## V2 Roadmap

- **Phase 8 ✅**: Access chaining (`obj.prop`, `arr[0]`) — DONE
- **Phase 8.5 ✅**: Context scoping (parent chain for lambda closures) — DONE
- **Phase 9**: Lambdas, higher-order functions (`map`, `filter`, `reduce`)
- **Phase 9.5**: `Function` trait refactor (stateful functions)
- **Phase 10**: User-defined functions (`def name(params) = expr`)
- **Phase 10.5**: Missing math/string builtins
- **Phase 11**: Native `DateTime`/`Duration` via `jiff`
- **Phase 12**: Serialization & caching
- **Phase 13**: Plugin SDK
- **Phase 14**: Performance optimization
- **Phase 15**: Error recovery + security limits

Do not implement future phases unless explicitly asked.

## Out of Scope (per SPEC.md)

WASM sandboxing, JIT compilation, async evaluation, complex static type system, null-safe navigation (`?.`)

## Key Files

| File | Purpose |
|------|---------|
| `Cargo.toml` | single-crate config, deps, profiles |
| `src/lib.rs` | re-exports (tests in `src/lib_tests.rs`) |
| `src/lib_tests.rs` | integration tests (extracted, 80+ tests) |
| `.github/workflows/ci.yml` | CI: stable/beta/1.90.0 matrix |
| `PLAN.md` | V2 phase-by-phase roadmap |
| `SPEC.md` | architecture spec and scope |
| `tests/error_snapshots.rs` | insta snapshot tests |
