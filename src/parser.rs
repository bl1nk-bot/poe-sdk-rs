use crate::ast::*;
use crate::error::{ErrorKind, FormulaError};
use crate::lexer::{Token, TokenKind};
use crate::span::Span;

/// สร้าง Span จาก token list โดยอ้างอิง index
fn token_span(tokens: &[Token], idx: usize) -> Span {
    tokens.get(idx).map(|t| t.span).expect("BUG: token index out of bounds")
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

    /// Helper สำหรับ parse binary operators แบบ left associative
    fn parse_left_associative_binary<F>(
        &mut self,
        mut next_parser: F,
        token_ops: &[(TokenKind, BinaryOp)],
    ) -> Result<SpannedExpr, FormulaError>
    where
        F: FnMut(&mut Self) -> Result<SpannedExpr, FormulaError>,
    {
        let mut left = next_parser(self)?;
        while let Some((_, op)) = token_ops.iter().find(|(token, _)| *token == self.peek()) {
            self.advance();
            let right = next_parser(self)?;
            let span = Span {
                start: left.meta.span.start,
                end: right.meta.span.end,
            };
            left = SpannedExpr::new(Expr::BinaryExpr {
                left: Box::new(left),
                op: op.clone(),
                right: Box::new(right),
            }, span);
        }
        Ok(left)
    }

    /// expression = logical_or
    pub fn parse_expression(&mut self) -> Result<SpannedExpr, FormulaError> {
        self.parse_logical_or()
    }

    /// logical_or = logical_and ('||' logical_and)*
    fn parse_logical_or(&mut self) -> Result<SpannedExpr, FormulaError> {
        self.parse_left_associative_binary(
            Self::parse_logical_and,
            &[(TokenKind::OrOr, BinaryOp::Or)],
        )
    }

    /// logical_and = equality ('&&' equality)*
    fn parse_logical_and(&mut self) -> Result<SpannedExpr, FormulaError> {
        self.parse_left_associative_binary(
            Self::parse_equality,
            &[(TokenKind::AndAnd, BinaryOp::And)],
        )
    }

    /// equality = comparison (('==' | '!=') comparison)*
    fn parse_equality(&mut self) -> Result<SpannedExpr, FormulaError> {
        self.parse_left_associative_binary(
            Self::parse_comparison,
            &[(TokenKind::EqEq, BinaryOp::Eq), (TokenKind::NotEq, BinaryOp::NotEq)],
        )
    }

    /// comparison = term (('<' | '>' | '<=' | '>=') term)*
    fn parse_comparison(&mut self) -> Result<SpannedExpr, FormulaError> {
        self.parse_left_associative_binary(
            Self::parse_term,
            &[
                (TokenKind::Lt, BinaryOp::Lt),
                (TokenKind::Gt, BinaryOp::Gt),
                (TokenKind::LtEq, BinaryOp::LtEq),
                (TokenKind::GtEq, BinaryOp::GtEq),
            ],
        )
    }

    /// term = factor (('+' | '-') factor)*
    fn parse_term(&mut self) -> Result<SpannedExpr, FormulaError> {
        self.parse_left_associative_binary(
            Self::parse_factor,
            &[(TokenKind::Plus, BinaryOp::Add), (TokenKind::Minus, BinaryOp::Sub)],
        )
    }

    /// factor = unary (('*' | '/') unary)*
    fn parse_factor(&mut self) -> Result<SpannedExpr, FormulaError> {
        self.parse_left_associative_binary(
            Self::parse_unary,
            &[(TokenKind::Star, BinaryOp::Mul), (TokenKind::Slash, BinaryOp::Div)],
        )
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
            TokenKind::Null => {
                Ok(SpannedExpr::new(Expr::Literal(crate::value::Value::Null), span))
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
            TokenKind::LBracket => {
                // parse array literal: '[' (expression (',' expression)*)? ']'
                let mut elements = Vec::new();
                if self.peek() != TokenKind::RBracket {
                    loop {
                        elements.push(self.parse_expression()?);
                        if self.peek() == TokenKind::Comma {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                }
                self.expect(TokenKind::RBracket, "ต้องการ ']' ปิดท้าย array")?;
                Ok(SpannedExpr::new(Expr::ArrayLiteral(elements), span))
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
    let expr = parser.parse_expression()?;
    if parser.pos < tokens.len() - 1 { // -1 เพราะมี EOF
        return Err(FormulaError::new(
            ErrorKind::ParseError,
            "E002",
            "มี tokens เหลือหลัง parse expression",
            Some(token_span(tokens, parser.pos)),
        ));
    }
    Ok(expr)
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

    #[test]
    fn test_unclosed_parenthesis() {
        let tokens = tokenize("1 + (2 * 3").unwrap();
        let result = parse(&tokens);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind, ErrorKind::ParseError);
        assert_eq!(err.code, "E002");
    }

    #[test]
    fn test_missing_comma_in_function() {
        let tokens = tokenize("if(true \"pass\" \"fail\")").unwrap();
        let result = parse(&tokens);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind, ErrorKind::ParseError);
    }

    #[test]
    fn test_invalid_number() {
        let tokens = tokenize("123abc").unwrap();
        let result = parse(&tokens);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind, ErrorKind::ParseError);
        assert_eq!(err.code, "E002"); // มี tokens เหลือหลัง parse expression
    }
}