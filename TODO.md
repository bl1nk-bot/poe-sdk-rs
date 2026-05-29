# 📋 Active TODO Checklist — bl1z V2

## Phase 10.5: Missing Math + String Builtins — IN PROGRESS

Duration: 2 weeks
> Goal: Complete features according to SPEC.md

**Math:**
- [x] `pi()` → 3.14159...
- [x] `round(n)`, `ceil(n)`, `floor(n)`
- [x] `sqrt(n)`, `pow(base, exp)`, `abs(n)` ✅ (Done)
- [ ] `sin(n)`, `cos(n)`, `tan(n)` (Use `libm` or pure Rust implementation)
- [ ] `random()` → random float 0-1

**String:**
- [x] `trim(s)`, `trim_start(s)`, `trim_end(s)`
- [x] `split(s, delimiter)` → Array of Strings
- [ ] `replace(s, from, to)`
- [ ] `substring(s, start, length)`

## Phase 11: Advanced Data Types — NEXT

Duration: 2 weeks
> Note: Refactor date builtins from string wrapping → native DateTime/Duration

- [ ] **11.1** Add `Value::DateTime(jiff::Timestamp)` and `Value::Duration(jiff::SignedDuration)`
- [ ] **11.2** Add `Value::Set(HashSet<Value>)` and `Value::Range { start, end, step }`
- [ ] **11.3** Refactor date builtins: `now()` → return `Value::DateTime`, `date()` → parse → `Value::DateTime`
- [ ] **11.4** Refactor `date_add()`, `date_diff()` → operate on native types
- [ ] **11.5** Add @ operator: `@2024-01-01` → DateTime literal
- [ ] **11.6** Set operations: `union`, `intersection`, `difference`, `in`
- [ ] **11.7** Range operations: `1..10`, iteration, `contains`
- [ ] **11.8** Test: type coercion rules, display formatting

## Phase 12: Serialization & Caching

Duration: 1.5 weeks

- [ ] **12.1** `#[derive(Serialize, Deserialize)]` on `Value`, `Expr` (behind `serde` feature gate)
- [ ] **12.2** Feature gate: `serialization` in Cargo.toml
- [ ] **12.3** `Evaluator::evaluate_cached()` — LRU cache for repeated expressions
- [ ] **12.4** `Context::to_json()` / `Context::from_json()` — serialize/deserialize variable store
- [ ] **12.5** Test: round-trip serialization, cache hit/miss

## Phase 14: Performance & Optimization

Duration: 2 weeks

- [ ] **14.1** Constant folding optimization pass: `1 + 2` → `3` at parse/compile time
- [ ] **14.2** AST optimization: `if(true, X, Y)` → `X`, `if(false, X, Y)` → `Y`
- [ ] **14.3** Add criterion benchmarks: comparison with V1 baseline
- [ ] **14.4** Memoization for pure functions (no side effects)
- [ ] **14.5** `#[bench]` for every builtin function
- [ ] **14.6** Profile-guided optimization documentation

## Phase 15: Error Recovery + Security Limits

Duration: 1 week

- [ ] **15.1** `parse_with_recovery()` — collect all errors instead of fail-fast
- [ ] **15.2** Error recovery strategies: skip to next delimiter, insert missing token
- [ ] **15.3** `EngineConfig { max_formula_length, max_depth, max_time }`
- [ ] **15.4** `Evaluator::with_config(config)` — enforce limits
- [ ] **15.5** Test: formula too long, recursion depth exceeded, timeout

---
**Note:** For the history of completed tasks, see [docs/DONE_DETAILED.md](./docs/DONE_DETAILED.md).
