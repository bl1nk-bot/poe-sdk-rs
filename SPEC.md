# Architecture Rust สำหรับ Formula Engine

>โครงแบบที่เหมาะสำหรับการสร้าง **formula/calculation library** ด้วย Rust ให้โตได้แบบค่อยเป็นค่อยไป เหมาะกับแนว Notion-like formula engine

---

## 1) เป้าหมายของระบบ

ระบบนี้ควรทำได้ 3 อย่างหลัก:

1. **Parse** ข้อความสูตรเป็นโครงสร้างภายใน
2. **Evaluate** สูตรให้ได้ค่า
3. **Extend** เพิ่มฟังก์ชัน/ชนิดข้อมูล/บริบทได้ง่าย

---

## 2) ขอบเขตของระบบ

### In scope
- นิพจน์คณิตศาสตร์
- การเปรียบเทียบ
- logic boolean
- string operations
- function call
- variable / context lookup
- error reporting
- extensible built-in functions

### Out of scope ในช่วงแรก
- compiler เต็มรูปแบบ
- optimization หนัก ๆ
- static type system ซับซ้อน
- user-defined function ภาษาเต็มรูปแบบ
- async evaluation
- sandbox execution แบบ script engine

---

# Architecture ระดับสูง

## Layer 1: Input Layer
รับสูตรเป็น string เช่น:

```text
if(score > 50, "pass", "fail")
```

หน้าที่:
- รับข้อความ
- ส่งเข้า lexer/parser

---

## Layer 2: Lexing
แปลง string → token stream

ตัวอย่าง token:
- Identifier
- Number
- String
- LParen
- RParen
- Comma
- Operator
- Keyword

### หน้าที่ของ lexer
- ตัด whitespace
- อ่าน literal
- แยก operator
- จัดการตำแหน่งบรรทัด/คอลัมน์เพื่อ error

---

## Layer 3: Parsing
แปลง token → AST

AST เป็น tree ของ expression

ตัวอย่างโหนด:
- Literal
- UnaryExpr
- BinaryExpr
- FunctionCall
- VariableRef
- Grouping

### หน้าที่ของ parser
- ตรวจ syntax
- สร้าง AST
- ทำ precedence / associativity
- รายงาน syntax error

---

## Layer 4: Semantic / Validation
ตรวจความหมายเชิงตรรกะบางส่วน

ตัวอย่าง:
- ฟังก์ชันมีอยู่จริงไหม
- จำนวน arg ถูกไหม
- ใช้ operator กับชนิดที่รองรับไหม
- variable มีใน context ไหม

> ชั้นนี้อาจเริ่มจาก validation บางส่วน แล้วค่อยแยกเป็น type checker ในอนาคต

---

## Layer 5: Evaluation
เดิน AST แล้วคืนค่า `Value`

ตัวอย่าง `Value`:
- Number
- String
- Bool
- Null
- DateTime
- Array
- Object

---

## Layer 6: Context
เป็นที่เก็บข้อมูล runtime

ตัวอย่าง:
- current record
- user input
- env variables
- external data

---

## Layer 7: Built-in Function Registry
เก็บฟังก์ชันพร้อม signature และ implementation

ตัวอย่าง:
- `if`
- `len`
- `upper`
- `lower`
- `contains`
- `date_add`

---

## Layer 8: Error System
จัดการ error ทุกชั้น

แนะนำแบ่ง:
- `LexError`
- `ParseError`
- `EvalError`
- `TypeError`
- `FunctionError`
- `ContextError`

---

# โครงสร้างไฟล์ Rust ที่แนะนำ

```text
src/
  lib.rs
  lexer.rs
  parser.rs
  ast.rs
  value.rs
  eval.rs
  context.rs
  functions.rs
  error.rs
  span.rs
  diagnostics.rs
  builtins/
    mod.rs
    string.rs
    math.rs
    logic.rs
    date.rs
```

---

# รายละเอียดแต่ละโมดูล

## 1) `ast.rs`
เก็บโครงสร้างของ expression

### ตัวอย่าง enums
- `Expr`
- `BinaryOp`
- `UnaryOp`

### หน้าที่
- เป็นกลาง ไม่ควรมี logic เยอะ
- ใช้ร่วมกันระหว่าง parser และ evaluator

---

## 2) `lexer.rs`
แปลงตัวอักษรเป็น token

### ควรมี
- `TokenKind`
- `Token`
- `Lexer`

### ความสามารถขั้นต่ำ
- number literal
- string literal
- identifiers
- operators
- parentheses
- comma

---

## 3) `parser.rs`
แปลง token เป็น AST

### ควรมี
- recursive descent parser
- precedence table
- error recovery เบื้องต้น

### ความสามารถขั้นต่ำ
- parse arithmetic
- parse function call
- parse variable ref
- parse comparison
- parse boolean logic

---

## 4) `value.rs`
นิยามค่าที่ evaluator คืน

### ควรมี enum เช่น:
- `Number(f64)`
- `String(String)`
- `Bool(bool)`
- `Null`
- `Array(Vec<Value>)`

### หมายเหตุ
ช่วงแรกอาจใช้ `f64` ก่อน  
ถ้าต้องการ precision ดีขึ้นค่อยเปลี่ยนเป็น decimal type ภายหลัง

---

## 5) `context.rs`
ตัวแปร runtime

### ตัวอย่าง
- `HashMap<String, Value>`
- ฟังก์ชัน `get(name)`
- ฟังก์ชัน `set(name, value)`

### ใช้สำหรับ
- `prop("score")`
- `x + y`

---

## 6) `functions.rs`
เก็บ registry ของ built-in function

### ควรมี
- function signature
- arity check
- call dispatcher

### รูปแบบ
- `HashMap<String, BuiltinFunction>`

---

## 7) `eval.rs`
ตัวคำนวณจริง

### หน้าที่
- รับ AST
- รับ context
- คืน `Result<Value, EvalError>`

### แนวทาง
- recursive evaluation
- match ตาม `Expr`

---

## 8) `error.rs`
ระบบ error แบบรวมศูนย์

### แนะนำ
- มี code
- message
- span

ตัวอย่าง:
- `UnexpectedToken`
- `UnknownIdentifier`
- `UnsupportedOperator`
- `DivisionByZero`
- `ArgumentCountMismatch`

---

## 9) `span.rs`
เก็บตำแหน่งใน source

### เพื่อ
- แสดง error ดีขึ้น
- ชี้ตำแหน่งสูตรที่ผิด

---

# สเปกที่ควรกำหนดตั้งแต่ต้น

---

## A. Syntax Spec
กำหนด grammar ให้ชัด

### Expression types
- literals
- variables
- unary operators
- binary operators
- function calls
- parentheses

### Operator precedence ตัวอย่าง
1. Parentheses
2. Unary `-`, `!`
3. `*`, `/`
4. `+`, `-`
5. comparison
6. equality
7. logical AND
8. logical OR

---

## B. Value Spec
กำหนดชนิดข้อมูลที่รองรับ

### Phase 1
- Number
- String
- Bool
- Null

### Phase 2
- Array
- DateTime
- Object/Map

---

## C. Function Spec
กำหนด built-in functions ที่รองรับ

### Phase 1
- `if(cond, a, b)`
- `len(text)`
- `upper(text)`
- `lower(text)`

### Phase 2
- `contains`
- `replace`
- `starts_with`
- `ends_with`
- date functions

---

## D. Error Spec
ทุก error ต้องมี:
- code
- message
- location/span
- category

---

## E. Context Spec
กำหนดวิธี resolve variable

ตัวอย่าง:
- local variables
- row/record data
- system variables