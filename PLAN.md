# แผนงานแบบเฟส

---

# Phase 0: Design & Scope Lock
## เป้าหมาย
กำหนดสิ่งที่จะสร้างให้ชัดก่อนเริ่มเขียนจริง

## งาน
- นิยาม use case หลัก
- นิยาม syntax
- นิยาม value types
- นิยาม operator set
- นิยาม built-in functions แรก
- นิยาม error format

## Milestone
- ได้เอกสาร spec v1
- ได้ grammar คร่าว ๆ
- ได้รายการฟังก์ชันขั้นต่ำ
- ได้ decision เรื่อง type model

## Done when
- ไม่มีคำถามใหญ่ค้างว่า “ภาษานี้หน้าตาแบบไหน”

---

# Phase 1: Lexer + AST + Parser
## เป้าหมาย
แปลงสูตร string เป็น AST ได้

## งาน
- implement tokenization
- implement AST enums
- implement parser
- support precedence
- support literals, grouping, arithmetic, comparison

## Milestone
- parse `1 + 2 * 3`
- parse `(1 + 2) * 3`
- parse `"hello"`
- parse `foo`
- parse `sum(1, 2)`

## Done when
- มี AST ที่ถูกต้องจาก input พื้นฐาน
- มี syntax error ที่อ่านรู้เรื่อง

---

# Phase 2: Basic Evaluator
## เป้าหมาย
รัน AST แล้วได้ผลลัพธ์

## งาน
- implement `Value`
- implement evaluator
- support arithmetic
- support comparison
- support boolean logic
- support unary ops

## Milestone
- evaluate expression พื้นฐานได้
- ผลลัพธ์ `Result<Value, Error>`
- error สำหรับ divide by zero / invalid op

## Done when
- `1 + 2 * 3 = 7`
- `"a" + "b"` ทำได้ตาม spec ที่กำหนด
- comparison และ logic ทำงาน

---

# Phase 3: Function System
## เป้าหมาย
เรียกฟังก์ชัน built-in ได้

## งาน
- implement registry
- implement signature validation
- implement argument count validation
- implement function calling
- add first built-ins

## Milestone
- `if(true, 1, 0)`
- `len("abc")`
- `upper("abc")`

## Done when
- เพิ่มฟังก์ชันใหม่ได้โดยไม่ต้องแก้ evaluator เยอะ

---

# Phase 4: Context Resolution
## เป้าหมาย
รองรับตัวแปรและข้อมูลภายนอก

## งาน
- implement `Context`
- resolve identifiers
- add nested lookup if needed
- support record/property access

## Milestone
- `score + 10`
- `if(done, "yes", "no")`
- `user.name`

## Done when
- สูตรเข้าถึงข้อมูล runtime ได้

---

# Phase 5: Type Handling & Validation
## เป้าหมาย
ลด runtime error และเพิ่มความแม่นยำ

## งาน
- type inference แบบพื้นฐาน
- argument type validation
- operator type validation
- clearer error messages

## Milestone
- error เมื่อเอา string ไปบวก number
- error เมื่อเรียกฟังก์ชันผิดชนิด
- diagnostics ดีขึ้น

## Done when
- error ส่วนใหญ่เข้าใจง่าย
- validation ก่อน evaluate ดีขึ้น

---

# Phase 6: Advanced Features
## เป้าหมาย
ให้ระบบใช้งานจริงได้มากขึ้น

## งาน
- arrays
- objects
- access chaining
- date/time
- more built-ins
- caching
- serialization

## Milestone
- `map`, `filter` เบื้องต้น
- date math
- nested context
- better performance

## Done when
- รองรับ use case ใกล้ของจริง

---

# Phase 7: Quality & Tooling
## เป้าหมาย
ทำให้ library พร้อมใช้งานในโปรดักชัน

## งาน
- unit tests
- parser tests
- eval tests
- snapshot tests
- benchmarks
- docs
- examples

## Milestone
- coverage ดี
- test cases ครบ
- API docs ชัด

---

# ขอบเขตที่แนะนำให้ล็อกไว้สำหรับ V1

## V1 ควรมี
- number, string, bool, null
- arithmetic
- comparison
- boolean logic
- function call
- variable/context lookup
- syntax/runtime errors
- span/diagnostics พื้นฐาน

## V1 ยังไม่ควรมี
- user-defined functions
- async
- full static typing
- plugin system
- module system
- optimizer ขั้นสูง

---

# ตัวอย่าง roadmap สั้นแบบใช้งานจริง

## Sprint 1
- grammar
- AST
- lexer
- parse arithmetic

## Sprint 2
- parser precedence
- evaluator
- basic errors

## Sprint 3
- functions
- context
- tests

## Sprint 4
- type checking
- diagnostics
- docs

---

# คำแนะนำเชิงสถาปัตยกรรม
## ใช้หลักนี้
- **AST ต้องไม่ผูกกับ evaluation**
- **Evaluator ต้องรับ Value/Context อย่างเดียว**
- **Functions ต้องแยกออกจาก core**
- **Error ต้องมี span เสมอ**