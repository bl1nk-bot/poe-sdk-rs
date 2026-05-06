use crate::ast::*;
use crate::error::{ErrorKind, FormulaError};
use crate::lexer::{Token, TokenKind};
use crate::span::Span;

/// สร้าง Span จาก token list โดยอ้างอิง index
fn token_span(tokens: &[Token], idx: usize) -> Span {
    tokens.get(idx).map(|t| t.span).unwrap_or(Span {
        start: crate::span::Position { line: 0, column: 0 },
        end: crate::span::Position { line: 0, column: 0 },
    })
}

pub struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
}

impl<'a> Parser<'a> {
    pub fn new(tokens: &'a [Token]) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> TokenKind {
        self.tokens.get(self.pos).map(|t| t.kind.clone()).unwrap_or(TokenKind::Eof)
    }

    fn peek_lexeme(&self) -> String {
        self.tokens.get(self.pos).map(|t| t.lexeme.clone()).unwrap_or_default()
    }

    fn advance(&mut self) -> &Token {
        let tok = &self.tokens[self.pos];
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
        tok
    }

    fn expect(&mut self, expected: TokenKind, err_msg: &str) -> Result<(), FormulaError> {
        if self.peek() != expected {
            return Err(FormulaError::new(
                ErrorKind::ParseError,
                "E002",
                err_msg,
                Some(token_span(self.tokens, self.pos)),
            ));
        }
        self.advance();
        Ok(())
    }

    /// expression = logical_or
    pub fn parse_expression(&mut self) -> Result<SpannedExpr, FormulaError> {
        self.parse_logical_or()
    }

    /// logical_or = logical_and ('||' logical_and)*
    fn parse_logical_or(&mut self) -> Result<SpannedExpr, FormulaError> {
        let mut left = self.parse_logical_and()?;
        while self.peek() == TokenKind::OrOr {
            let op_span = token_span(self.tokens, self.pos);
            self.advance();
            let right = self.parse_logical_and()?;
            let span = Span {
                start: left.meta.span.start,
                end: right.meta.span.end,
            };
            left = SpannedExpr::new(Expr::BinaryExpr {
                left: Box::new(left),
                op: BinaryOp::Or,
                right: Box::new(right),
            }, span);
        }
        Ok(left)
    }

    /// logical_and = equality ('&&' equality)*
    fn parse_logical_and(&mut self) -> Result<SpannedExpr, FormulaError> {
        let mut left = self.parse_equality()?;
        while self.peek() == TokenKind::AndAnd {
            let op_span = token_span(self.tokens, self.pos);
            self.advance();
            let right = self.parse_equality()?;
            let span = Span {
                start: left.meta.span.start,
                end: right.meta.span.end,
            };
            left = SpannedExpr::new(Expr::BinaryExpr {
                left: Box::new(left),
                op: BinaryOp::And,
                right: Box::new(right),
            }, span);
        }
        Ok(left)
    }

    /// equality = comparison (('==' | '!=') comparison)*
    fn parse_equality(&mut self) -> Result<SpannedExpr, FormulaError> {
        let mut left = self.parse_comparison()?;
        while self.peek() == TokenKind::EqEq || self.peek() == TokenKind::NotEq {
            let op = match self.peek() {
                TokenKind::EqEq => BinaryOp::Eq,
                TokenKind::NotEq => BinaryOp::NotEq,
                _ => unreachable!(),
            };
            self.advance();
            let right = self.parse_comparison()?;
            let span = Span {
                start: left.meta.span.start,
                end: right.meta.span.end,
            };
            left = SpannedExpr::new(Expr::BinaryExpr {
                left: Box::new(left),
                op,
                right: Box::new(right),
            }, span);
        }
        Ok(left)
    }

    /// comparison = term (('<' | '>' | '<=' | '>=') term)*
    fn parse_comparison(&mut self) -> Result<SpannedExpr, FormulaError> {
        let mut left = self.parse_term()?;
        while matches!(self.peek(), TokenKind::Lt | TokenKind::Gt | TokenKind::LtEq | TokenKind::GtEq) {
            let op = match self.peek() {
                TokenKind::Lt => BinaryOp::Lt,
                TokenKind::Gt => BinaryOp::Gt,
                TokenKind::LtEq => BinaryOp::LtEq,
                TokenKind::GtEq => BinaryOp::GtEq,
                _ => unreachable!(),
            };
            self.advance();
            let right = self.parse_term()?;
            let span = Span {
                start: left.meta.span.start,
                end: right.meta.span.end,
            };
            left = SpannedExpr::new(Expr::BinaryExpr {
                left: Box::new(left),
                op,
                right: Box::new(right),
            }, span);
        }
        Ok(left)
    }

    /// term = factor (('+' | '-') factor)*
    fn parse_term(&mut self) -> Result<SpannedExpr, FormulaError> {
        let mut left = self.parse_factor()?;
        while self.peek() == TokenKind::Plus || self.peek() == TokenKind::Minus {
            let op = match self.peek() {
                TokenKind::Plus => BinaryOp::Add,
                TokenKind::Minus => BinaryOp::Sub,
                _ => unreachable!(),
            };
            self.advance();
            let right = self.parse_factor()?;
            let span = Span {
                start: left.meta.span.start,
                end: right.meta.span.end,
            };
            left = SpannedExpr::new(Expr::BinaryExpr {
                left: Box::new(left),
                op,
                right: Box::new(right),
            }, span);
        }
        Ok(left)
    }

    /// factor = unary (('*' | '/') unary)*
    fn parse_factor(&mut self) -> Result<SpannedExpr, FormulaError> {
        let mut left = self.parse_unary()?;
        while self.peek() == TokenKind::Star || self.peek() == TokenKind::Slash {
            let op = match self.peek() {
                TokenKind::Star => BinaryOp::Mul,
                TokenKind::Slash => BinaryOp::Div,
                _ => unreachable!(),
            };
            self.advance();
            let right = self.parse_unary()?;
            let span = Span {
                start: left.meta.span.start,
                end: right.meta.span.end,
            };
            left = SpannedExpr::new(Expr::BinaryExpr {
                left: Box::new(left),
                op,
                right: Box::new(right),
            }, span);
        }
        Ok(left)
    }

    /// unary = ('-' | '!')? primary
    fn parse_unary(&mut self) -> Result<SpannedExpr, FormulaError> {
        if self.peek() == TokenKind::Minus || self.peek() == TokenKind::Bang {
            let op = match self.peek() {
                TokenKind::Minus => UnaryOp::Neg,
                TokenKind::Bang => UnaryOp::Not,
                _ => unreachable!(),
            };
            let op_span = token_span(self.tokens, self.pos);
            self.advance();
            let expr = self.parse_primary()?;
            let span = Span {
                start: op_span.start,
                end: expr.meta.span.end,
            };
            return Ok(SpannedExpr::new(Expr::UnaryExpr {
                op,
                expr: Box::new(expr),
            }, span));
        }
        self.parse_primary()
    }

    /// primary = NUMBER | STRING | TRUE | FALSE | IDENTIFIER | '(' expression ')' | function_call
    fn parse_primary(&mut self) -> Result<SpannedExpr, FormulaError> {
        let tok = self.advance();
        let span = tok.span;
        match &tok.kind {
            TokenKind::Number => {
                let n: f64 = tok.lexeme.parse().map_err(|_| {
                    FormulaError::new(ErrorKind::ParseError, "E003", "ไม่สามารถแปลงตัวเลขได้", Some(span))
                })?;
                Ok(SpannedExpr::new(Expr::Literal(crate::value::Value::Number(n)), span))
            }
            TokenKind::String => {
                Ok(SpannedExpr::new(Expr::Literal(crate::value::Value::String(tok.lexeme.clone())), span))
            }
            TokenKind::True => {
                Ok(SpannedExpr::new(Expr::Literal(crate::value::Value::Bool(true)), span))
            }
            TokenKind::False => {
                Ok(SpannedExpr::new(Expr::Literal(crate::value::Value::Bool(false)), span))
            }
            TokenKind::Identifier => {
                let name = tok.lexeme.clone();
                // ถ้าเจอ '(' ข้างหน้า คือ function call
                if self.peek() == TokenKind::LParen {
                    self.advance(); // กิน '('
                    let mut args = Vec::new();
                    if self.peek() != TokenKind::RParen {
                        loop {
                            args.push(self.parse_expression()?);
                            if self.peek() == TokenKind::Comma {
                                self.advance();
                            } else {
                                break;
                            }
                        }
                    }
                    self.expect(TokenKind::RParen, "ต้องการ ')'")?;
                    Ok(SpannedExpr::new(Expr::FunctionCall { name, args }, span))
                } else {
                    // ตัวแปร
                    Ok(SpannedExpr::new(Expr::Variable(name), span))
                }
            }
            TokenKind::LParen => {
                let inner = self.parse_expression()?;
                self.expect(TokenKind::RParen, "ต้องการ ')'")?;
                Ok(SpannedExpr::new(Expr::Grouping(Box::new(inner)), span))
            }
            _ => {
                Err(FormulaError::new(
                    ErrorKind::ParseError,
                    "E004",
                    &format!("ไม่คาดคิด token: {:?}", tok.kind),
                    Some(span),
                ))
            }
        }
    }
}

pub fn parse(tokens: &[Token]) -> Result<SpannedExpr, FormulaError> {
    let mut parser = Parser::new(tokens);
    parser.parse_expression()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::tokenize;

    #[test]
    fn test_arithmetic() {
        let tokens = tokenize("1 + 2 * 3").unwrap();
        let ast = parse(&tokens).unwrap();
        if let Expr::BinaryExpr { left, op, right } = &ast.expr {
            assert_eq!(*op, BinaryOp::Add);
            if let Expr::BinaryExpr { left: _, op: op2, right: _ } = &right.expr {
                assert_eq!(*op2, BinaryOp::Mul);
            } else {
                panic!("expected multiplication");
            }
        } else {
            panic!("expected binary add");
        }
    }

    #[test]
    fn test_function_call() {
        let tokens = tokenize("len(\"abc\")").unwrap();
        let ast = parse(&tokens).unwrap();
        match ast.expr {
            Expr::FunctionCall { name, args } => {
                assert_eq!(name, "len");
                assert_eq!(args.len(), 1);
            }
            _ => panic!("expected function call"),
        }
    }
}