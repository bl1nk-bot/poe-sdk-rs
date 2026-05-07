# Session 2 Technical Specification: Advanced Formula Engine

## Overview

Session 2 builds upon Session 1's solid foundation to add advanced features that enable complex formula evaluation scenarios. The focus is on extensibility, performance, and advanced data manipulation capabilities.

## Architecture Extensions

### Core Components (Extended)

#### AST Extensions
```rust
pub enum Expr {
    // Session 1 expressions...
    PropertyAccess {
        object: Box<SpannedExpr>,
        property: String,
    },
    IndexAccess {
        array: Box<SpannedExpr>,
        index: Box<SpannedExpr>,
    },
    LambdaExpr {
        params: Vec<String>,
        body: Box<SpannedExpr>,
    },
    FunctionDef {
        name: String,
        params: Vec<String>,
        body: Box<SpannedExpr>,
    },
}
```

#### Value Extensions
```rust
pub enum Value {
    // Session 1 values...
    DateTime(DateTime<Utc>),
    Duration(Duration),
    Set(HashSet<Value>),
    Range { start: i64, end: i64 },
}
```

#### Context Extensions
```rust
pub struct Context {
    variables: HashMap<String, Value>,
    functions: HashMap<String, UserFunction>, // New
}

pub struct UserFunction {
    pub name: String,
    pub params: Vec<String>,
    pub body: Box<SpannedExpr>,
    pub metadata: FunctionMetadata,
}
```

### New Components

#### Plugin System
```rust
pub trait Plugin {
    fn name(&self) -> &str;
    fn version(&self) -> &str;
    fn functions(&self) -> Vec<BuiltinFunction>;
    fn types(&self) -> Vec<CustomType>;
}

pub struct PluginManager {
    plugins: HashMap<String, Box<dyn Plugin>>,
    sandbox: Sandbox,
}
```

#### Serialization
```rust
pub trait Serializable {
    fn serialize(&self) -> Result<String, SerializationError>;
    fn deserialize(data: &str) -> Result<Self, SerializationError>;
}

impl Serializable for SpannedExpr {
    // AST serialization implementation
}
```

## Syntax Extensions

### Access Chaining
```
# Property access
user.name
user.profile.email
data.items[0].price

# Null-safe navigation
user?.profile?.name

# Array indexing
array[0]
array[index + 1]
matrix[0][1]
```

### Lambda Expressions
```
# Anonymous functions
(x) => x * 2
(x, y) => x + y
(item) => item.price > 100

# Used in higher-order functions
map([1,2,3], (x) => x * 2)          # [2,4,6]
filter([1,2,3,4], (x) => x % 2 == 0) # [2,4]
reduce([1,2,3,4], (acc, x) => acc + x, 0) # 10
```

### User-Defined Functions
```
# Function definition
fn double(x) = x * 2
fn factorial(n) = if(n <= 1, 1, n * factorial(n - 1))
fn greet(name) = "Hello, " + name + "!"

# Usage
double(5)        # 10
factorial(5)     # 120
greet("World")   # "Hello, World!"
```

### Advanced Data Literals
```
# DateTime literals
@2023-12-25T10:30:00Z
@2023-01-01                   # Midnight UTC

# Duration literals
1h 30m                         # 1 hour 30 minutes
2d 3h 45m                      # 2 days, 3 hours, 45 minutes

# Range literals
1..10                          # inclusive range
'a'..'z'                       # character range

# Set literals
{1, 2, 3, 2}                  # {1, 2, 3} (duplicates removed)
```

## Function Categories

### Collection Functions (Extended)
```rust
// Session 1 functions + extensions
sum(array)
avg(array)
min(array)
max(array)
count(array)
join(array, separator)

// New higher-order functions
map(array, lambda)
filter(array, lambda)
reduce(array, lambda, initial)
group_by(array, key_lambda)
sort(array, compare_lambda)
unique(array)
```

### Date/Time Functions (Extended)
```rust
// Session 1 functions
now()
date_add(date, days)
date_diff(date1, date2)
year(date)
month(date)
day(date)

// Extended functions
hour(datetime)
minute(datetime)
second(datetime)
weekday(datetime)
format_date(datetime, format)
parse_date(string, format)
is_weekend(datetime)
date_between(datetime, start, end)
```

### String Functions (Extended)
```rust
// Session 1 functions
len(string)
upper(string)
lower(string)
contains(string, substr)
starts_with(string, prefix)
ends_with(string, suffix)

// Extended functions
trim(string)
split(string, separator)
replace(string, old, new)
substring(string, start, length)
regex_match(string, pattern)
regex_replace(string, pattern, replacement)
```

### Math Functions (Extended)
```rust
// Session 1 functions
abs(number)

// Extended functions
round(number, decimals)
ceil(number)
floor(number)
sqrt(number)
pow(base, exponent)
log(number, base)
sin(angle)
cos(angle)
tan(angle)
pi()
e()
random()
random_between(min, max)
```

## Performance Optimizations

### AST Optimizations
- Constant folding: `1 + 2` → `3`
- Dead code elimination
- Function inlining for small functions
- Loop unrolling for small arrays

### Evaluation Optimizations
- Lazy evaluation for boolean expressions
- Short-circuit evaluation
- Memoization for recursive functions
- Vectorized operations for arrays

### Caching Strategies
- AST caching for repeated formulas
- Function result caching
- Context snapshot caching
- Compiled bytecode caching

## Plugin Architecture

### Plugin Interface
```rust
#[derive(Serialize, Deserialize)]
pub struct PluginManifest {
    name: String,
    version: String,
    description: String,
    author: String,
    functions: Vec<FunctionSpec>,
    types: Vec<TypeSpec>,
}

pub trait Plugin {
    fn manifest(&self) -> PluginManifest;
    fn initialize(&mut self, context: &mut PluginContext) -> Result<(), PluginError>;
    fn execute(&self, name: &str, args: &[Value]) -> Result<Value, PluginError>;
}
```

### Sandboxing
- Resource limits (CPU, memory)
- Network access control
- File system restrictions
- Timeout enforcement
- Error isolation

## Error Handling Extensions

### Error Categories (Extended)
```rust
pub enum ErrorKind {
    // Session 1 errors...
    PropertyNotFound,
    IndexOutOfBounds,
    TypeMismatch,
    RecursionLimitExceeded,
    PluginError,
    SerializationError,
}
```

### Enhanced Diagnostics
- Stack traces for function calls
- Variable inspection
- Performance profiling data
- Plugin execution logs

## Serialization Format

### AST Serialization
```json
{
  "type": "BinaryExpr",
  "left": {
    "type": "Literal",
    "value": {"type": "Number", "value": 1}
  },
  "op": "Add",
  "right": {
    "type": "Literal",
    "value": {"type": "Number", "value": 2}
  }
}
```

### Context Serialization
```json
{
  "variables": {
    "user": {"type": "Map", "value": {"name": "Alice", "age": 30}},
    "items": {"type": "Array", "value": [1, 2, 3]}
  },
  "functions": {
    "double": {
      "params": ["x"],
      "body": {"type": "BinaryExpr", "left": "x", "op": "Mul", "right": 2}
    }
  }
}
```

## Testing Strategy

### Unit Tests
- AST parsing and serialization
- Function evaluation
- Plugin loading and execution
- Error handling edge cases

### Integration Tests
- Complex formula evaluation
- Plugin interactions
- Performance benchmarks
- Memory usage validation

### Fuzz Testing
- Random formula generation
- Invalid input handling
- Memory safety verification

## Performance Benchmarks

### Target Metrics
- **Parsing**: < 100μs for 1000-token formulas
- **Evaluation**: < 10μs for simple expressions
- **Memory**: < 1MB per concurrent evaluation
- **Plugin Load**: < 50ms for plugin initialization

### Benchmark Suite
```rust
// Example benchmark
#[bench]
fn bench_complex_formula(b: &mut Bencher) {
    let formula = r#"
        map(
            filter([1,2,3,4,5,6,7,8,9,10], (x) => x % 2 == 0),
            (x) => x * x
        )
    "#;
    // Benchmark evaluation...
}
```

## Migration Guide

### From Session 1
- All Session 1 APIs remain compatible
- New features are opt-in
- Performance improvements are automatic

### Breaking Changes
- Minimal breaking changes expected
- Deprecation warnings for legacy features
- Migration tools provided

## Future Compatibility

### Versioning Strategy
- Semantic versioning for API changes
- Plugin API versioning
- Backward compatibility guarantees

### Extensibility Points
- Custom type registration
- Operator overloading
- Custom evaluation strategies
- Third-party plugins