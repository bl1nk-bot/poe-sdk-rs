# เอกสารข้อกำหนดทางเทคนิคของ Formula Engine (ภาษาไทย)

## ภาพรวมและเป้าหมาย

>ระบบ Formula Engine เป็นไลบรารีการคำนวณที่ใช้ภายใน (คล้าย Notion formula) มีความสามารถหลัก 3 ด้าน

1. แยกโครงสร้าง (Parse) – แปลงข้อความสูตรให้อยู่ในรูป AST
2. ประมวลผล (Evaluate) – คำนวณ AST และคืนค่า
3. ขยายได้ง่าย (Extend) – เพิ่มชนิดข้อมูล ฟังก์ชัน และบริบทใหม่โดยกระทบโค้ดส่วนอื่นน้อยที่สุด

## ขอบเขตของระบบ (Scope)

In Scope (ภายในขอบเขต V1)

· นิพจน์คณิตศาสตร์ (บวก ลบ คูณ หาร)
· การเปรียบเทียบ (>, <, >=, <=, ==, !=)
· ตรรกศาสตร์ (AND, OR, NOT)
· การจัดการข้อความพื้นฐาน (ต่อสตริง)
· การเรียกฟังก์ชัน (function call)
· การอ้างอิงตัวแปร/บริบท (variable reference)
· การรายงานข้อผิดพลาด (lex, parse, eval)
· ฟังก์ชันในตัว (built-in) ที่ลงทะเบียนเพิ่มได้

Out of Scope (นอกขอบเขต V1)

· คอมไพเลอร์เต็มรูปแบบ
· การ optimize ขั้นสูง
· ระบบ type แบบ static ซับซ้อนเต็มตัว
· ฟังก์ชันที่ผู้ใช้กำหนดเองได้ (user‑defined functions)
· การประมวลผลแบบ async
· sandbox execution

### สถาปัตยกรรมระดับสูง (High-Level Architecture)

#### ระบบถูกออกแบบเป็นชั้น (layer) ตามลำดับการทำงาน

Layer ชื่อ หน้าที่
1 Input รับข้อความสูตร เช่น if(score > 50, "pass", "fail")
2 Lexer แปลงข้อความ → Token Stream
3 Parser แปลง Token Stream → AST (Abstract Syntax Tree)
4 Semantic / Validation ตรวจสอบความหมายพื้นฐาน (ฟังก์ชันมีจริง, จำนวนพารามิเตอร์, ชนิดข้อมูลเข้าได้)
5 Evaluator เดิน AST และสร้าง Value
6 Context เก็บข้อมูล runtime (ตัวแปร, ข้อมูล record, environment)
7 Function Registry จัดเก็บบิ้วท์อินฟังก์ชัน
8 Error System จัดการ error ที่มี span (ตำแหน่ง) ทุกตัว

รายละเอียดแต่ละโมดูล (Component Specification)

1. AST (ast.rs)

· เป็นโครงสร้างกลางที่ไม่มี logic การคำนวณ
· ประกอบด้วย enum Expr (Literal, UnaryExpr, BinaryExpr, FunctionCall, VariableRef, Grouping)
· มี BinaryOp และ UnaryOp แยกออกมา

2. Lexer (lexer.rs)

· รับข้อความ สร้าง token ทีละตัว
· Token ประกอบด้วย TokenKind (Identifier, Number, String, LParen, RParen, Comma, Operator, Keyword)
· เก็บตำแหน่ง (บรรทัด/คอลัมน์) เพื่อสร้าง Span
· ละเว้น whitespace, จัดการ literal, แยก operator

3. Parser (parser.rs)

· ใช้ recursive descent โดยมีตารางลำดับความสำคัญ (precedence) และกฎ associativity
· ลำดับความสำคัญของ operator:
  1. วงเล็บ ()
  2. Unary -, !
  3. *, /
  4. +, -
  5. เปรียบเทียบ (<, >, <=, >=)
  6. เท่ากัน (==, !=)
  7. Logical AND
  8. Logical OR
· รองรับ function call และ variable reference
· รายงาน syntax error พร้อม span
· มีระบบ error recovery ขั้นต้น (optional)

4. Value (value.rs)

· ชนิดข้อมูลที่ evaluator คืนค่า
· Enum Value:
  · Number(f64)
  · String(String)
  · Bool(bool)
  · Null
  · (Phase 2: Array(Vec<Value>), DateTime(...), Object(Map))
· ใช้ f64 ในช่วงแรก หากต้องการความแม่นยำสูงค่อยเปลี่ยนเป็น decimal type

5. Context (context.rs)

· เก็บตัวแปร runtime ใน HashMap<String, Value>
· ฟังก์ชัน get(name) -> Option<Value> และ set(name, value)
· ใช้สำหรับ resolve identifier เช่น score, user.name (อาจทำ nested lookup ภายหลัง)

6. Function Registry (functions.rs)

· เก็บ built-in functions ใน HashMap<String, BuiltinFunction>
· BuiltinFunction มี name, arity (จำนวนอาร์กิวเมนต์) และ call(args: &[Value]) -> Value
· ตรวจสอบจำนวนพารามิเตอร์ก่อนเรียก
· เพิ่มฟังก์ชันใหม่ได้โดยไม่ต้องแก้ evaluator

7. Evaluator (eval.rs)

· ฟังก์ชัน eval(expr: &Expr, ctx: &Context, functions: &FunctionRegistry) -> Result<Value, EvalError>
· ทำงานแบบ recursive: match ตาม Expr และประมวลผลตามชนิด
· ส่งคืน error สำหรับการดำเนินการที่ไม่ถูกต้อง (เช่น หารศูนย์, ชนิดข้อมูลเข้ากันไม่ได้)

8. Error System (error.rs, span.rs, diagnostics.rs)

· ทุก error ประกอบด้วย code, message, span (ตำแหน่งใน source), category
· Categories: LexError, ParseError, EvalError, TypeError, FunctionError, ContextError
· span.rs เก็บ Span { start: Position, end: Position } สำหรับใช้ชี้ตำแหน่งในข้อความสูตร
· diagnostics.rs ช่วยจัดรูปแบบข้อความ error ที่มีที่มา

9. Built-in Functions (builtins/)

จัดกลุ่มเป็นโมดูล:

· string.rs: len, upper, lower, contains, starts_with, ends_with, replace
· math.rs: abs, round, min, max, sqrt
· logic.rs: if, and, or, not
· date.rs: now, date_add, date_diff (เพิ่มภายหลัง)

รายละเอียดฟังก์ชันขั้นต่ำ V1

ฟังก์ชัน คำอธิบาย จำนวนพารามิเตอร์
if(cond, a, b) ถ้า cond เป็นจริง คืน a มิฉะนั้นคืน b 3
len(text) ความยาวสตริง 1
upper(text) แปลงเป็นตัวพิมพ์ใหญ่ 1
lower(text) แปลงเป็นตัวพิมพ์เล็ก 1

ข้อกำหนดของ Syntax (Grammar คร่าว ๆ)

```
expression → logical_or
logical_or → logical_and ('||' logical_and)*
logical_and → equality ('&&' equality)*
equality → comparison (('==' | '!=') comparison)*
comparison → term (('<' | '>' | '<=' | '>=') term)*
term → factor (('+' | '-') factor)*
factor → unary (('*' | '/') unary)*
unary → ('-' | '!')? primary
primary → NUMBER | STRING | IDENTIFIER | '(' expression ')' | function_call
function_call → IDENTIFIER '(' expression (',' expression)* ')'
```

ข้อกำหนด Error

Error ทุกตัวต้องมี:

· code (รหัสประจำตัว เช่น E001)
· message (ข้อความอธิบาย)
· span (ตำแหน่งเริ่มต้น-สิ้นสุดในข้อความ)
· category (กลุ่มข้อผิดพลาด)

ตัวอย่าง Error Codes:

· E001 – UnexpectedToken
· E002 – UnknownIdentifier
· E003 – UnsupportedOperator
· E004 – DivisionByZero
· E005 – ArgumentCountMismatch
· E006 – TypeMismatch

โครงสร้างไฟล์ Rust ที่แนะนำ

```
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

แผนการพัฒนาแบบเฟส (Roadmap)

Phase 0: Design & Scope Lock

· กำหนด use case, syntax, value types, operators, built‑ins, error format
· เอกสาร spec v1

Phase 1: Lexer + AST + Parser

· สร้าง tokenizer และ parser
· แปลง 1 + 2 * 3 ให้ได้ AST ที่ถูกต้อง

Phase 2: Basic Evaluator

· ใช้ Value และ evaluator
· ประมวลผลได้คณิตศาสตร์ เปรียบเทียบ ตรรกะ

Phase 3: Function System

· สร้าง registry, เรียก if, len, upper

Phase 4: Context Resolution

· รองรับตัวแปรจาก context

Phase 5: Type Handling & Validation (ภายหลัง)

· ตรวจสอบชนิดข้อมูลก่อน evaluation ให้ error ชัดเจนขึ้น

Phase 6: Advanced Features (Array, DateTime, Object, chaining)

Phase 7: Quality & Tooling (tests, docs, benchmarks)

Non-Functional Requirements (NFR)

· ประสิทธิภาพ: response ต่อโทเค็นนับร้อยภายในหลักมิลลิวินาที
· ความสามารถในการดูแลรักษา: เพิ่มฟังก์ชันใหม่โดยไม่ต้องแก้ไข core evaluator
· การรายงาน error: ระบุตำแหน่งที่แน่นอนได้
· ความเสถียร: ห้าม panic จาก input ที่ผิดปกติ (ใช้ Result ทุกจุด)

---