# Architecture Rust สำหรับ Formula Engine (อัปเดต ณ commit 6d2529a)

โครงแบบสำหรับสร้าง **formula/calculation library** ด้วย Rust ให้โตแบบค่อยเป็นค่อยไป เหมาะกับแนว Notion-like formula engine  
สถานะโดยรวม: **V1 เสร็จสมบูรณ์** (number, string, bool, null + ตัวดำเนินการ + ฟังก์ชัน + context + error reporting)  
เฟส 6 (Advanced Features) ยังไม่เริ่ม เฟส 7 (Quality & Tooling) มีบางส่วนแล้ว

---

## 1) เป้าหมายของระบบ

ระบบนี้ควรทำได้ 3 อย่างหลัก:

1. **Parse** ข้อความสูตรเป็นโครงสร้างภายใน ✅
2. **Evaluate** สูตรให้ได้ค่า ✅
3. **Extend** เพิ่มฟังก์ชัน/ชนิดข้อมูล/บริบทได้ง่าย ✅

---

## 2) ขอบเขตของระบบ

### In scope (V1 – เสร็จแล้ว)
- นิพจน์คณิตศาสตร์ (+, -, *, /)
- การเปรียบเทียบ (<, >, <=, >=)
- equality (==, !=)
- logic boolean (&&, ||, !)
- string operations (ต่อสตริงด้วย +)
- function call (รวม nested calls)
- variable / context lookup
- error reporting (ทุกชั้น)
- extensible built-in functions (ผ่าน registry)

### Out of scope ในช่วงแรก (ไม่เปลี่ยนแปลง)
- compiler เต็มรูปแบบ
- optimization หนัก ๆ
- static type system ซับซ้อน
- user-defined function ภาษาเต็มรูปแบบ
- async evaluation
- sandbox execution แบบ script engine

---

# Architecture ระดับสูง (สถานะปัจจุบัน)

## Layer 1: Input Layer ✅
รับสูตรเป็น string เช่น:

```text
if(score > 50, "pass", "fail")
```

หน้าที่:

· รับข้อความ
· ส่งเข้า lexer/parser (ผ่าน tokenize ใน lexer.rs)

---

Layer 2: Lexing ✅

แปลง string → token stream

ตัวอย่าง token (เพิ่ม Null, True, False จากสเปกเดิม):

· Identifier, Number, String, LParen, RParen, Comma
· Operator: Plus, Minus, Star, Slash, Bang, AndAnd, OrOr, EqEq, NotEq, Lt, Gt, LtEq, GtEq
· Keyword: True, False, Null
· Eof

หน้าที่ของ lexer (ที่ implement แล้ว)

· ตัด whitespace
· อ่าน literal (number, string พร้อม escape)
· แยก operator (รวม &&, ||, <=, >=, !=, ==)
· จัดการตำแหน่งบรรทัด/คอลัมน์เพื่อสร้าง Span ให้ทุก token
· error เมื่อพบอักขระไม่รู้จัก

implement ใน src/lexer.rs ใช้ Chars iterator (zero-cost)

---

Layer 3: Parsing ✅

แปลง token → AST

AST เป็น tree ของ SpannedExpr (Expr + Span)

ตัวอย่างโหนด (Expr enum):

· Literal(Value)          // รองรับ Number, String, Bool, Null
· Variable(String)
· UnaryExpr { op, expr }  // op: Neg(-), Not(!)
· BinaryExpr { left, op, right } // op: +, -, *, /, ==, !=, <, >, <=, >=, &&, ||
· FunctionCall { name, args }
· Grouping(Box<SpannedExpr>)

หน้าที่ของ parser (ที่ implement แล้ว)

· ใช้ recursive descent พร้อม generic function parse_left_associative_binary จัดการ precedence
· สร้าง AST มี Span ทุกโหนด
· ทำ precedence / associativity ตามตาราง:
  1. Parentheses
  2. Unary -, !
  3. *, /
  4. +, -
  5. comparison (<, >, <=, >=)
  6. equality (==, !=)
  7. logical AND (&&)
  8. logical OR (||)
· รองรับ function call (identifier + ( args )) และ nested calls
· รายงาน syntax error พร้อม span

implement ใน src/parser.rs

---

Layer 4: Semantic / Validation 🟡 (runtime validation มีแล้ว)

ตรวจความหมายเชิงตรรกะบางส่วน (ทำงานตอน evaluate)

สิ่งที่ตรวจ:

· ฟังก์ชันมีอยู่จริงใน registry
· จำนวน arg ถูกต้อง (ตรวจ arity)
· ใช้ operator กับชนิดที่รองรับ (type validation ใน evaluator – สร้าง TypeError)
· variable มีใน context (สร้าง ContextError)

ณ ตอนนี้ยังเป็น runtime validation (ยังไม่มี static type checker) — เพียงพอสำหรับ V1

---

Layer 5: Evaluation ✅

เดิน AST แล้วคืนค่า Value

ตัวอย่าง Value (enum ที่ implement แล้ว):

· Number(f64)
· String(String)
· Bool(bool)
· Null
· (Phase 6: DateTime, Array, Object)

การทำงาน:

· รับ &SpannedExpr, &Context, &FunctionRegistry
· ประเมินแบบ recursive
· คืน Result<Value, FormulaError>
· รองรับทุก operator + function call + variable lookup
· ตรวจสอบ division by zero, type mismatch

implement ใน src/eval.rs, src/value.rs

---

Layer 6: Context ✅

เป็นที่เก็บข้อมูล runtime

· HashMap<String, Value>
· ฟังก์ชัน get(name) คืน Option<Value>
· ฟังก์ชัน set(name, value)

ใช้สำหรับ:

· score + 10 (เมื่อ context มี score)
· if(done, ...) (done เป็น Bool จาก context)

implement ใน src/context.rs

---

Layer 7: Built-in Function Registry ✅ (เกินแผน)

เก็บฟังก์ชันพร้อม signature และ implementation

ฟังก์ชันที่มี (10 ตัว):

· if(cond, a, b) – logic
· len(text) – string
· upper(text) – string
· lower(text) – string
· contains(text, pattern) – string
· starts_with(text, pattern) – string
· ends_with(text, pattern) – string
· abs(number) – math
· min(a, b) – math
· max(a, b) – math

Registry ใช้ HashMap<String, BuiltinFunction>

· BuiltinFunction มี name, arity, call (คืน Result<Value, FormulaError>)

implement ใน src/functions.rs, src/builtins/

---

Layer 8: Error System ✅

จัดการ error ทุกชั้น

ชนิดข้อผิดพลาด (ErrorKind):

· LexError
· ParseError
· EvalError
· TypeError
· FunctionError
· ContextError

ทุก error (FormulaError) ประกอบด้วย:

· kind: ErrorKind
· code: String (เช่น "E001", "E010")
· message: String (ภาษาไทย)
· span: Option<Span> (ตำแหน่งใน source)

การแสดงผล:

· diagnostics.rs สามารถแสดง source snippet พร้อม caret ชี้ตำแหน่ง

implement ใน src/error.rs, src/span.rs, src/diagnostics.rs

---

โครงสร้างไฟล์ Rust ที่ใช้จริง

```text
src/
  lib.rs              # re-export, doc-tests
  lexer.rs             # tokenization + tests
  parser.rs            # recursive descent parser + tests
  ast.rs               # SpannedExpr, Expr, BinaryOp, UnaryOp
  value.rs             # Value enum
  eval.rs              # evaluator + helper functions
  context.rs           # Context struct
  functions.rs         # BuiltinFunction, FunctionRegistry
  error.rs             # FormulaError, ErrorKind
  span.rs              # Position, Span
  diagnostics.rs       # format_error พร้อม snippet
  builtins/
    mod.rs             # register_all
    string.rs          # len, upper, lower, contains, starts_with, ends_with
    math.rs            # abs, min, max
    logic.rs           # if_fn
    date.rs            # (empty, รอ Phase 6)
```

---

สเปกที่อัปเดตตามสถานะจริง

A. Syntax Spec (grammar ที่ parser รองรับ)

Expression types

· literals: Number, String, Bool (true, false), Null (null)
· variables: identifiers (เช่น score)
· unary operators: - (Neg), ! (Not)
· binary operators: +, -, *, /, ==, !=, <, >, <=, >=, &&, ||
· function calls: identifier(expr, expr, ...)
· parentheses: (expr)

Operator precedence (implemented)

1. Parentheses
2. Unary -, !
3. *, /
4. +, -
5. comparison (<, >, <=, >=)
6. equality (==, !=)
7. logical AND (&&)
8. logical OR (||)

All binary operators are left-associative (except unary).

---

B. Value Spec (ชนิดข้อมูล)

Phase 1 (implemented)

· Number(f64)
· String(String)
· Bool(bool)
· Null

Phase 2 (ยังไม่ได้ทำ)

· Array(Vec<Value>)
· DateTime
· Object/Map

---

C. Function Spec (built-in functions)

Phase 1 (implemented – 4 ตัว)

· if(cond, a, b)
· len(text)
· upper(text)
· lower(text)

Phase 2 (implemented ไปบางส่วนแล้ว – 6 ตัวเพิ่ม)

· contains(text, pattern)
· starts_with(text, pattern)
· ends_with(text, pattern)
· abs(number)
· min(a, b)
· max(a, b)

ยังไม่ได้ทำ

· replace
· date functions (now, date_add, ...)

---

D. Error Spec

ทุก error มี:

· code: string code (เช่น "E001")
· message: ข้อความภาษาไทย
· span: ตำแหน่งใน source (optional ในบางกรณี)
· category: หนึ่งใน LexError, ParseError, EvalError, TypeError, FunctionError, ContextError

ตัวอย่าง error codes ที่ใช้แล้ว:

· E001 – Unexpected character (lexer)
· E002 – Unexpected token (parser)
· E003 – Invalid number literal (parser)
· E004 – Unexpected token (parser, fallback)
· E005 – Unknown variable (context)
· E006 – Type mismatch (eval/builtins)
· E007 – Function not found (eval)
· E008 – Argument count mismatch (eval)
· E009 – Built-in function error (eval)
· E010 – Division by zero (eval)

---

E. Context Spec (การ resolve variable)

· ใช้ HashMap<String, Value> ภายใน Context
· มองหา identifier ใน context โดยตรง
· ยังไม่รองรับ dot notation (user.name) – เลื่อนไป Phase 6
· ไม่มีการแยกชนิดของตัวแปร (local, record) ในตอนนี้

---

สรุปแผนงานที่เหลือ

· Phase 6: arrays, objects, date/time, access chaining, ฟังก์ชันเพิ่มเติม
· Phase 7: benchmarks, snapshot tests, API docs (rustdoc), CI/CD

V1 ถือว่า บรรลุเป้าหมาย ตามขอบเขตที่กำหนดไว้