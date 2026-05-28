# ✅ DETAILED COMPLETED TASKS ARCHIVE

This file serves as the permanent record of all completed phases and tasks for the bl1z V2.

## 🔴 Phase 8: Access Chaining & Indexing (2 สัปดาห์) ✅ DONE
- [x] **8.1** สร้าง `Expr::PropertyAccess { object: Box<SpannedExpr>, field: String }` ใน AST
- [x] **8.2** สร้าง `Expr::IndexAccess { object: Box<SpannedExpr>, index: Box<SpannedExpr> }` ใน AST
- [x] **8.3** Refactor parser: แทนที่ string concatenation ด้วย recursive `parse_property_access()` / `parse_index_access()`
- [x] **8.4** Refactor evaluator: แทนที่ `name.split('.')` ด้วย recursive `eval_property_access()` / `eval_index_access()`
- [x] **8.5** เพิ่ม error: `PropertyNotFound` (E207), `IndexOutOfBounds` (E208), `NotIndexable` (E401)
- [x] **8.6** รองรับ nested: `a.b[0].c.d`, `a["key"].nested`
- [x] **8.7** Unit tests: property access on Map/Context, index access on Array/String, chained access, error cases
- [x] **8.8** Integration tests: `user.profile.scores[0]`, `config["db"].host`

## 🔴 Phase 8.5: Context Scoping 🆕 (1 สัปดาห์) ✅ DONE
- [x] **8.5.1** Refactor `Context`: เพิ่ม `parent: Option<Box<Context>>` field
- [x] **8.5.2** Implement `Context::with_parent(parent)` constructor
- [x] **8.5.3** Implement `Context::get()` → ค้นหาใน current scope ก่อน → ไล่ขึ้น parent chain
- [x] **8.5.4** Implement `Context::set()` → set ใน current scope เท่านั้น (shadowing)
- [x] **8.5.5** Implement `Context::get_all()` สำหรับ debug — แสดงทั้ง chain
- [x] **8.5.6** Test: variable shadowing, inheritance, multi-level scope
- [x] **8.5.7** Test: nested scopes ไม่กระทบ variable ของ parent (immutable parent)

## 🟡 Phase 9: Lambda & Higher-Order Functions (3 สัปดาห์) ✅ DONE
- [x] **9.1** เพิ่ม `TokenKind::Arrow` (`=>`) ใน lexer
- [x] **9.2** สร้าง `Expr::Lambda { params: Vec<String>, body: Box<SpannedExpr> }` ใน AST
- [x] **9.3** Parser: `parse_lambda()` — syntax: `(x, y) => x + y`
- [x] **9.4** สร้าง `Value::Lambda { params, body, captured_scope }` หรือ `Value::Closure { ... }`
- [x] **9.5** Evaluator: `eval_lambda()` — สร้าง closure จับ scope
- [x] **9.6** Evaluator: `eval_call()` — apply closure ด้วย arguments
- [x] **9.7** Implement builtin: `map(array, lambda)`, `filter(array, lambda)`, `reduce(array, lambda, initial)`
- [x] **9.8** Implement builtin: `sort(array)`, `sort(array, comparator_lambda)`
- [x] **9.9** Implement builtin: `unique(array)`, `group_by(array, lambda)`
- [x] **9.10** Test: closure captures variable, pipeline (map → filter → reduce)
- [x] **9.11** Test: nested lambdas, recursive lambdas (optional)

## 🟡 Phase 9.5: BuiltinFunction Trait Refactor 🆕 (1 สัปดาห์) ✅ DONE
- [x] **9.5.1** เปลี่ยน `BuiltinFunction::call` เป็น trait `Function`
- [x] **9.5.2** `FunctionRegistry::register(name, Box<dyn Function>)` — รองรับ stateful functions
- [x] **9.5.3** Refactor builtins ทั้งหมด: `struct SumFunction`, `struct AvgFunction`, etc. implements `Function`
- [x] **9.5.4** อัพเดท tests ทั้งหมดให้ใช้ trait ใหม่
- [x] **9.5.5** Test: custom function with captured state (counter, cache)

## 🟡 Phase 10: User-Defined Functions (1 สัปดาห์) ✅ DONE
- [x] **10.1** syntax: `def greet(name) = "Hello, " + name`
- [x] **10.2** Parser สร้าง `Expr::FunctionDef { name, params, body }`
- [x] **10.3** Evaluator: register function ใน scope
- [x] **10.4** รองรับ recursive user-defined functions
- [x] **10.5** Test: define → call, multiple definitions, shadowing builtins

## 🔵 Phase 13: Plugin SDK Foundation (1.5 สัปดาห์) ✅ DONE
- [x] **13.1** `pub trait Plugin: Send + Sync { fn register(&self, registry: &mut FunctionRegistry); }`
- [x] **13.2** `PluginManager` — load/unload plugins
- [x] **13.3** Plugin example: `MathPlugin`, `StringPlugin`, `GeoPlugin`
- [x] **13.4** Documentation: how to write custom plugin
- [x] **13.5** Test: plugin isolation, error propagation
