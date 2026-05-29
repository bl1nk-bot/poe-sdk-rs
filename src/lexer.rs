use crate::error::{ErrorKind, FormulaError};
use crate::span::{Position, Span};

/// ชนิดของ Token (Token Categories)
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    /// ชื่อตัวแปร หรือชื่อฟังก์ชัน (เช่น `score`, `sum`)
    Identifier,
    /// ตัวเลข (ทั้งจำนวนเต็มและทศนิยม เช่น `42`, `3.14`)
    Number,
    /// ข้อความที่อยู่ในเครื่องหมายอัญประกาศ (เช่น `"hello"`)
    String,
    /// วงเล็บเปิด `(`
    LParen,
    /// วงเล็บปิด `)`
    RParen,
    /// เครื่องหมายจุลภาค `,`
    Comma,
    /// เครื่องหมายบวก `+`
    Plus,
    /// เครื่องหมายลบ `-`
    Minus,
    /// เครื่องหมายคูณ `*`
    Star,
    /// เครื่องหมายหาร `/`
    Slash,
    /// เครื่องหมายตรรกะ NOT `!`
    Bang,
    /// เครื่องหมายตรรกะ AND `&&`
    AndAnd,
    /// เครื่องหมายตรรกะ OR `||`
    OrOr,
    /// เครื่องหมายเปรียบเทียบเท่ากับ `==`
    EqEq,
    /// เครื่องหมายเปรียบเทียบไม่เท่ากับ `!=`
    NotEq,
    /// เครื่องหมายน้อยกว่า `<`
    Lt,
    /// เครื่องหมายมากกว่า `>`
    Gt,
    /// เครื่องหมายน้อยกว่าหรือเท่ากับ `<=`
    LtEq,
    /// เครื่องหมายมากกว่าหรือเท่ากับ `>=`
    GtEq,
    /// ค่าความจริง `true`
    True,
    /// ค่าความจริง `false`
    False,
    /// ค่าว่าง `null`
    Null,
    /// เครื่องหมายจุด `.` (สำหรับ property access)
    Dot,
    /// เครื่องหมายก้ามปูเปิด `[`
    LBracket,
    /// เครื่องหมายก้ามปูปิด `]`
    RBracket,
    /// เครื่องหมายปีกกาเปิด `{`
    LBrace,
    /// เครื่องหมายปีกกาปิด `}`
    RBrace,
    /// เครื่องหมายทวิภาค `:`
    Colon,
    /// เครื่องหมายเท่ากับ `=`
    Equal,
    /// เครื่องหมายลูกศร `=>` (สำหรับ lambda)
    Arrow,
    /// คำสำคัญ `fn` (สำหรับ user-defined function)
    Fn,
    /// จุดสิ้นสุดของไฟล์หรือข้อความ
    Eof,
}

/// โครงสร้างข้อมูลที่เก็บรายละเอียดของแต่ละ Token
#[derive(Debug, Clone)]
pub struct Token {
    /// ชนิดของ Token
    pub kind: TokenKind,
    /// ตำแหน่งของ Token ใน Source Code
    pub span: Span,
    /// ข้อความต้นฉบับที่ประกอบเป็น Token นี้
    pub lexeme: String,
}

/// Lexer struct เก็บสถานะการอ่าน
struct Lexer<'a> {
    source: &'a str,
    pos: usize,
    line: usize,
    col: usize,
    tokens: Vec<Token>,
}

impl<'a> Lexer<'a> {
    fn new(source: &'a str) -> Self {
        Self {
            source,
            pos: 0,
            line: 1,
            col: 1,
            tokens: Vec::new(),
        }
    }

    fn peek(&self) -> Option<char> {
        self.source[self.pos..].chars().next()
    }

    fn advance(&mut self) -> Option<char> {
        let ch = self.peek();
        if let Some(c) = ch {
            if c == '\n' {
                self.line += 1;
                self.col = 1;
            } else {
                self.col += 1;
            }
            self.pos += c.len_utf8();
        }
        ch
    }

    fn make_span(&self, start_line: usize, start_col: usize) -> Span {
        Span {
            start: Position {
                line: start_line,
                column: start_col,
            },
            end: Position {
                line: self.line,
                column: self.col,
            },
        }
    }

    fn push_token(&mut self, kind: TokenKind, lexeme: String, start_line: usize, start_col: usize) {
        let span = self.make_span(start_line, start_col);
        self.tokens.push(Token { kind, span, lexeme });
    }

    fn scan_identifier(&mut self, first: char) {
        let start_line = self.line;
        let start_col = self.col;
        let mut lexeme = String::new();
        lexeme.push(first);
        self.advance();
        while let Some(c) = self.peek() {
            if c.is_alphanumeric() || c == '_' {
                lexeme.push(c);
                self.advance();
            } else {
                break;
            }
        }
        let kind = match lexeme.as_str() {
            "true" => TokenKind::True,
            "false" => TokenKind::False,
            "null" => TokenKind::Null,
            "fn" => TokenKind::Fn,
            _ => TokenKind::Identifier,
        };
        self.push_token(kind, lexeme, start_line, start_col);
    }

    fn scan_number(&mut self, first: char) {
        let start_line = self.line;
        let start_col = self.col;
        let mut lexeme = String::new();
        lexeme.push(first);
        self.advance();
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() || c == '.' {
                lexeme.push(c);
                self.advance();
            } else {
                break;
            }
        }
        self.push_token(TokenKind::Number, lexeme, start_line, start_col);
    }

    fn scan_string(&mut self) -> Result<(), FormulaError> {
        let start_line = self.line;
        let start_col = self.col;
        let mut lexeme = String::new();
        self.advance(); // consume "
        loop {
            match self.peek() {
                None => {
                    return Err(FormulaError::new(
                        ErrorKind::LexError,
                        "E101",
                        "ไม่พบเครื่องหมายปิดข้อความ",
                        Some(self.make_span(start_line, start_col)),
                    ));
                }
                Some('"') => {
                    self.advance();
                    break;
                }
                Some('\\') => {
                    self.advance();
                    if let Some(esc) = self.peek() {
                        lexeme.push(match esc {
                            '"' => '"',
                            '\\' => '\\',
                            'n' => '\n',
                            't' => '\t',
                            _ => esc,
                        });
                        self.advance();
                    }
                }
                Some(c) => {
                    lexeme.push(c);
                    self.advance();
                }
            }
        }
        self.push_token(TokenKind::String, lexeme, start_line, start_col);
        Ok(())
    }

    fn tokenize(mut self) -> Result<Vec<Token>, FormulaError> {
        while let Some(c) = self.peek() {
            match c {
                ' ' | '\t' | '\r' | '\n' => {
                    self.advance();
                }
                '(' => {
                    let sl = self.line;
                    let sc = self.col;
                    self.advance();
                    self.push_token(TokenKind::LParen, "(".to_string(), sl, sc);
                }
                ')' => {
                    let sl = self.line;
                    let sc = self.col;
                    self.advance();
                    self.push_token(TokenKind::RParen, ")".to_string(), sl, sc);
                }
                ',' => {
                    let sl = self.line;
                    let sc = self.col;
                    self.advance();
                    self.push_token(TokenKind::Comma, ",".to_string(), sl, sc);
                }
                '+' => {
                    let sl = self.line;
                    let sc = self.col;
                    self.advance();
                    self.push_token(TokenKind::Plus, "+".to_string(), sl, sc);
                }
                '-' => {
                    let sl = self.line;
                    let sc = self.col;
                    self.advance();
                    self.push_token(TokenKind::Minus, "-".to_string(), sl, sc);
                }
                '*' => {
                    let sl = self.line;
                    let sc = self.col;
                    self.advance();
                    self.push_token(TokenKind::Star, "*".to_string(), sl, sc);
                }
                '/' => {
                    let sl = self.line;
                    let sc = self.col;
                    self.advance();
                    self.push_token(TokenKind::Slash, "/".to_string(), sl, sc);
                }
                '!' => {
                    let sl = self.line;
                    let sc = self.col;
                    self.advance();
                    if self.peek() == Some('=') {
                        self.advance();
                        self.push_token(TokenKind::NotEq, "!=".to_string(), sl, sc);
                    } else {
                        self.push_token(TokenKind::Bang, "!".to_string(), sl, sc);
                    }
                }
                '=' => {
                    let sl = self.line;
                    let sc = self.col;
                    self.advance();
                    if self.peek() == Some('=') {
                        self.advance();
                        self.push_token(TokenKind::EqEq, "==".to_string(), sl, sc);
                    } else if self.peek() == Some('>') {
                        self.advance();
                        self.push_token(TokenKind::Arrow, "=>".to_string(), sl, sc);
                    } else {
                        self.push_token(TokenKind::Equal, "=".to_string(), sl, sc);
                    }
                }
                '<' => {
                    let sl = self.line;
                    let sc = self.col;
                    self.advance();
                    if self.peek() == Some('=') {
                        self.advance();
                        self.push_token(TokenKind::LtEq, "<=".to_string(), sl, sc);
                    } else {
                        self.push_token(TokenKind::Lt, "<".to_string(), sl, sc);
                    }
                }
                '>' => {
                    let sl = self.line;
                    let sc = self.col;
                    self.advance();
                    if self.peek() == Some('=') {
                        self.advance();
                        self.push_token(TokenKind::GtEq, ">=".to_string(), sl, sc);
                    } else {
                        self.push_token(TokenKind::Gt, ">".to_string(), sl, sc);
                    }
                }
                '&' => {
                    let sl = self.line;
                    let sc = self.col;
                    self.advance();
                    if self.peek() == Some('&') {
                        self.advance();
                        self.push_token(TokenKind::AndAnd, "&&".to_string(), sl, sc);
                    } else {
                        return Err(FormulaError::new(
                            ErrorKind::LexError,
                            "E101",
                            "พบ '&' ตัวเดียว ต้องใช้ '&&'",
                            Some(self.make_span(sl, sc)),
                        ));
                    }
                }
                '|' => {
                    let sl = self.line;
                    let sc = self.col;
                    self.advance();
                    if self.peek() == Some('|') {
                        self.advance();
                        self.push_token(TokenKind::OrOr, "||".to_string(), sl, sc);
                    } else {
                        return Err(FormulaError::new(
                            ErrorKind::LexError,
                            "E101",
                            "พบ '|' ตัวเดียว ต้องใช้ '||'",
                            Some(self.make_span(sl, sc)),
                        ));
                    }
                }
                '"' => {
                    self.scan_string()?;
                }
                '.' => {
                    let sl = self.line;
                    let sc = self.col;
                    self.advance();
                    self.push_token(TokenKind::Dot, ".".to_string(), sl, sc);
                }
                '[' => {
                    let sl = self.line;
                    let sc = self.col;
                    self.advance();
                    self.push_token(TokenKind::LBracket, "[".to_string(), sl, sc);
                }
                ']' => {
                    let sl = self.line;
                    let sc = self.col;
                    self.advance();
                    self.push_token(TokenKind::RBracket, "]".to_string(), sl, sc);
                }
                '{' => {
                    let sl = self.line;
                    let sc = self.col;
                    self.advance();
                    self.push_token(TokenKind::LBrace, "{".to_string(), sl, sc);
                }
                '}' => {
                    let sl = self.line;
                    let sc = self.col;
                    self.advance();
                    self.push_token(TokenKind::RBrace, "}".to_string(), sl, sc);
                }
                ':' => {
                    let sl = self.line;
                    let sc = self.col;
                    self.advance();
                    self.push_token(TokenKind::Colon, ":".to_string(), sl, sc);
                }
                c if c.is_alphabetic() || c == '_' => {
                    self.scan_identifier(c);
                }
                c if c.is_ascii_digit() => {
                    self.scan_number(c);
                }
                _ => {
                    let sl = self.line;
                    let sc = self.col;
                    return Err(FormulaError::new(
                        ErrorKind::LexError,
                        "E101",
                        &format!("อักขระไม่รู้จัก '{}'", c),
                        Some(self.make_span(sl, sc)),
                    ));
                }
            }
        }
        self.push_token(TokenKind::Eof, "".to_string(), self.line, self.col);
        Ok(self.tokens)
    }
}

/// แปลงสายอักขระของสูตรให้เป็นรายการ Token
pub fn tokenize(source: &str) -> Result<Vec<Token>, FormulaError> {
    Lexer::new(source).tokenize()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_arrow_token() {
        let tokens = tokenize("=>").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Arrow);
    }
}
