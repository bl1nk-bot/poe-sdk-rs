# Formula Engine

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)]()

## 📖 ภาพรวม

**Formula Engine** คือไลบรารีสำหรับแยกส่วน (parse) และประมวลผล (evaluate) สูตรทางคณิตศาสตร์และตรรกะแบบ Notion-like ที่เขียนด้วยภาษา Rust ออกแบบมาให้มีความยืดหยุ่นสูง สามารถขยายฟังก์ชันและชนิดข้อมูลได้ง่ายผ่านระบบ registry

### ✨ คุณสมบัติหลัก

- **Lexer & Parser**: แปลงข้อความสูตรเป็น AST (Abstract Syntax Tree) พร้อมข้อมูลตำแหน่ง (span) สำหรับรายงานข้อผิดพลาดที่แม่นยำ
- **Evaluator**: ประเมินค่า AST โดยรองรับชนิดข้อมูลพื้นฐาน 4 ประเภท:
  - `Number` (ตัวเลข)
  - `String` (ข้อความ)
  - `Bool` (ค่าความจริง)
  - `Null` (ค่าว่าง)
- **Function System**: ระบบฟังก์ชันที่ขยายได้ รองรับ built-in functions และผู้ใช้สามารถเพิ่มฟังก์ชันใหม่ได้เอง
- **Context Support**: รองรับตัวแปรและการอ้างอิงค่าจากภายนอก
- **Error Reporting**: รายงานข้อผิดพลาดอย่างละเอียดพร้อมตำแหน่งบรรทัดและคอลัมน์

---

## 🚀 การติดตั้ง

เพิ่ม dependency นี้ในไฟล์ `Cargo.toml` ของคุณ:

```toml
[dependencies]
formula_engine = "0.1.0"
```

หรือหากต้องการใช้จาก source code ในเครื่อง:

```toml
[dependencies]
formula_engine = { path = "./path/to/formula_engine" }
```

---

## 📖 วิธีใช้งาน

### ตัวอย่างพื้นฐาน

```rust
use formula_engine::{tokenize, parse, evaluate, Context, FunctionRegistry};
use formula_engine::builtins;

// สร้าง registry พร้อมฟังก์ชันพื้นฐาน
let mut registry = FunctionRegistry::new();
builtins::register_all(&mut registry);

// Parse และ evaluate สูตร
let tokens = tokenize("1 + 2 * 3").unwrap();
let ast = parse(&tokens).unwrap();
let ctx = Context::new();
let result = evaluate(&ast, &ctx, &registry).unwrap();

println!("ผลลัพธ์: {:?}", result); // Number(7.0)
```

### ใช้กับฟังก์ชัน Built-in

```rust
use formula_engine::{tokenize, parse, evaluate, Context, FunctionRegistry};
use formula_engine::builtins;

let mut registry = FunctionRegistry::new();
builtins::register_all(&mut registry);

// ตัวอย่าง: ฟังก์ชัน if
let tokens = tokenize("if(true, \"ผ่าน\", \"ไม่ผ่าน\")").unwrap();
let ast = parse(&tokens).unwrap();
let ctx = Context::new();
let result = evaluate(&ast, &ctx, &registry).unwrap();
assert_eq!(result, formula_engine::Value::String("ผ่าน".to_string()));

// ตัวอย่าง: ฟังก์ชัน string
let tokens = tokenize("len(\"hello\")").unwrap();
let ast = parse(&tokens).unwrap();
let result = evaluate(&ast, &ctx, &registry).unwrap();
assert_eq!(result, formula_engine::Value::Number(5.0));
```

### ใช้กับตัวแปร (Context)

```rust
use formula_engine::{tokenize, parse, evaluate, Context, FunctionRegistry, Value};
use formula_engine::builtins;

let mut registry = FunctionRegistry::new();
builtins::register_all(&mut registry);

let mut ctx = Context::new();
ctx.set_variable("score", Value::Number(85.0));
ctx.set_variable("name", Value::String("สมชาย".to_string()));

let tokens = tokenize("if(score > 50, name, \"ไม่มีใคร\")").unwrap();
let ast = parse(&tokens).unwrap();
let result = evaluate(&ast, &ctx, &registry).unwrap();
assert_eq!(result, formula_engine::Value::String("สมชาย".to_string()));
```

---

## 📋 ไวยากรณ์ที่รองรับ

### ตัวดำเนินการ (Operators)

| หมวดหมู่ | ตัวดำเนินการ | คำอธิบาย |
|----------|-------------|----------|
| คณิตศาสตร์ | `+`, `-`, `*`, `/` | บวก, ลบ, คูณ, หาร |
| เปรียบเทียบ | `<`, `>`, `<=`, `>=` | เปรียบเทียบค่า |
| เท่ากัน | `==`, `!=` | ตรวจสอบความเท่ากัน |
| ตรรกะ | `&&`, `\|\|`, `!` | AND, OR, NOT |
| ยูนารี | `-`, `!` | ลบเลข, นิเสธ |

### ชนิดข้อมูล (Data Types)

- **Number**: ตัวเลขทศนิยม เช่น `1`, `3.14`, `-5`
- **String**: ข้อความในเครื่องหมายคำพูด เช่น `"hello"`, `"สวัสดี"`
- **Bool**: ค่าความจริง `true` หรือ `false`
- **Null**: ค่าว่าง `null`

### ฟังก์ชัน Built-in

#### ฟังก์ชันจัดการข้อความ (String)
| ฟังก์ชัน | คำอธิบาย | ตัวอย่าง |
|----------|----------|----------|
| `len(str)` | หาความยาวข้อความ | `len("hello")` → `5` |
| `upper(str)` | แปลงเป็นตัวพิมพ์ใหญ่ | `upper("abc")` → `"ABC"` |
| `lower(str)` | แปลงเป็นตัวพิมพ์เล็ก | `lower("ABC")` → `"abc"` |
| `contains(str, substr)` | ตรวจสอบว่ามีข้อความย่อยหรือไม่ | `contains("hello", "ell")` → `true` |
| `starts_with(str, prefix)` | ตรวจสอบขึ้นต้นด้วย | `starts_with("hello", "he")` → `true` |
| `ends_with(str, suffix)` | ตรวจสอบลงท้ายด้วย | `ends_with("hello", "lo")` → `true` |

#### ฟังก์ชันคณิตศาสตร์ (Math)
| ฟังก์ชัน | คำอธิบาย | ตัวอย่าง |
|----------|----------|----------|
| `abs(num)` | ค่าสัมบูรณ์ | `abs(-5)` → `5` |

#### ฟังก์ชันตรรกะ (Logic)
| ฟังก์ชัน | คำอธิบาย | ตัวอย่าง |
|----------|----------|----------|
| `if(condition, true_val, false_val)` | เงื่อนไข | `if(true, 1, 0)` → `1` |

#### ฟังก์ชันคอลเลกชัน (Collection) - *กำลังพัฒนา*
| ฟังก์ชัน | คำอธิบาย | ตัวอย่าง |
|----------|----------|----------|
| `sum(arr)` | ผลรวมของอาร์เรย์ | `sum([1, 2, 3])` → `6` |
| `avg(arr)` | ค่าเฉลี่ยของอาร์เรย์ | `avg([1, 2, 3])` → `2` |
| `min(arr)` | ค่าน้อยที่สุดในอาร์เรย์ | `min([1, 2, 3])` → `1` |
| `max(arr)` | ค่ามากที่สุดในอาร์เรย์ | `max([1, 2, 3])` → `3` |
| `count(arr)` | นับจำนวนสมาชิก | `count([1, 2, 3])` → `3` |
| `join(arr, sep)` | ต่อข้อความในอาร์เรย์ | `join(["a","b"], ",")` → `"a,b"` |

---

## 🏗️ สถาปัตยกรรม

Formula Engine ใช้สถาปัตยกรรมแบบ layered architecture:

```
┌─────────────────┐
│   Input Layer   │  ← สูตร: "1 + 2 * if(x > 0, 3, 4)"
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Lexer       │  → Token Stream พร้อม Span
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Parser      │  → AST (Abstract Syntax Tree)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Evaluator    │  → Value (Number/String/Bool/Null)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Result      │
└─────────────────┘
```

### โครงสร้างโปรเจกต์

```
formula_engine/
├── src/
│   ├── lib.rs           # จุดเข้าใช้งานหลักและ re-export
│   ├── lexer.rs         # Lexer: แปลง string → tokens
│   ├── parser.rs        # Parser: แปลง tokens → AST
│   ├── ast.rs           # นิยามโครงสร้าง AST
│   ├── eval.rs          # Evaluator: ประเมินค่า AST
│   ├── value.rs         # นิยามชนิดข้อมูล Value
│   ├── context.rs       # Context: จัดการตัวแปร
│   ├── functions.rs     # Function Registry
│   ├── error.rs         # นิยามข้อผิดพลาด
│   ├── span.rs          # ข้อมูลตำแหน่ง (บรรทัด/คอลัมน์)
│   ├── diagnostics.rs   # ระบบวินิจฉัยข้อผิดพลาด
│   └── builtins/        # ฟังก์ชันพื้นฐาน
│       ├── mod.rs
│       ├── string.rs    # ฟังก์ชันข้อความ
│       ├── math.rs      # ฟังก์ชันคณิตศาสตร์
│       ├── logic.rs     # ฟังก์ชันตรรกะ
│       ├── date.rs      # ฟังก์ชันวันที่ (*กำลังพัฒนา*)
│       └── collection.rs # ฟังก์ชันคอลเลกชัน (*กำลังพัฒนา*)
├── docs/                # เอกสารประกอบ
├── Cargo.toml
├── LICENSE
├── README.md
├── SPEC.md              # ข้อมูลจำเพาะทางเทคนิค
└── PLAN.md              # แผนงานการพัฒนา
```

---

## 📊 สถานะการพัฒนา

### ✅ เสร็จสมบูรณ์ (V1)

- [x] Lexer: Tokenization พร้อม span tracking
- [x] Parser: Recursive descent parser พร้อม precedence handling
- [x] AST: โครงสร้างต้นไม้พร้อม span ทุกโหนด
- [x] Evaluator: รองรับ number, string, bool, null
- [x] Operators: คณิตศาสตร์, เปรียบเทียบ, equality, logic
- [x] Function System: Registry และ built-in functions 10+ ตัว
- [x] Context: ตัวแปรและการอ้างอิงค่า
- [x] Error Reporting: รายงานข้อผิดพลาดพร้อมตำแหน่ง
- [x] Documentation: Doc-tests และ integration tests

### 🚧 กำลังพัฒนา (Phase 6 - Advanced Features)

- [ ] Array syntax และ evaluation
- [ ] Map/Dictionary support
- [ ] Date/Time functions
- [ ] Additional collection functions
- [ ] Performance optimization

### 📋 แผนในอนาคต (Phase 7+)

- [ ] Static type checking
- [ ] Macro system
- [ ] User-defined functions
- [ ] Lazy evaluation
- [ ] Caching/Memoization

---

## 🧪 การทดสอบ

รันการทดสอบทั้งหมด:

```bash
cargo test
```

รันการทดสอบเฉพาะโมดูล:

```bash
cargo test lexer
cargo test parser
cargo test eval
```

รัน doc-tests:

```bash
cargo test --doc
```

---

## 📄 เอกสารเพิ่มเติม

- **[SPEC.md](SPEC.md)**: ข้อมูลจำเพาะทางเทคนิคและสถาปัตยกรรมโดยละเอียด
- **[PLAN.md](PLAN.md)**: แผนงานการพัฒนาแต่ละเฟส
- **[docs/PRD.md](docs/PRD.md)**: เอกสารความต้องการผลิตภัณฑ์
- **[docs/idea-extension-gemini.md](docs/idea-extension-gemini.md)**: ไอเดียการขยายระบบ
- **[docs/overview-extension-poe.md](docs/overview-extension-poe.md)**: ภาพรวมการขยายสำหรับ POE SDK

---

## 🤝 การมีส่วนร่วม

เราสนับสนุนการมีส่วนร่วมจากชุมชน! หากคุณพบปัญหาหรือมีข้อเสนอแนะ:

1. เปิด Issue เพื่อรายงานบั๊กหรือขอฟีเจอร์ใหม่
2. Fork โปรเจกต์และสร้าง Pull Request
3. เขียนทดสอบให้ครอบคลุมการเปลี่ยนแปลง
4. อัพเดตเอกสารหากมีการเปลี่ยนแปลง API

---

## 📝 ใบอนุญาต

โปรเจกต์นี้เผยแพร่ภายใต้ใบอนุญาต [MIT License](LICENSE)

---

## 🙏 ขอบคุณ

- สร้างด้วย ❤️ โดยใช้ [Rust](https://www.rust-lang.org/)
- ได้รับแรงบันดาลใจจาก Notion formula engine
- พัฒนาสำหรับ POE SDK ecosystem

---

## 📞 ติดต่อ

หากมีคำถามหรือต้องการความช่วยเหลือ กรุณาเปิด Issue ใน GitHub repository นี้