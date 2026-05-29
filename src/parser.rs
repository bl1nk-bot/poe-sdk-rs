use crate::ast::*;
use crate::error::{ErrorKind, FormulaError};
use crate::lexer::{Token, TokenKind};
use crate::span::Span;
use crate::value::Value;

fn token_span(tokens: &[Token], idx: usize) -> Span {
    tokens
        .get(idx)
        .map(|t| t.span)
        .expect("BUG: token index out of bounds")
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
        self.tokens
            .get(self.pos)
            .map(|t| t.kind.clone())
            .unwrap_or(TokenKind::Eof)
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
                "E201",
                err_msg,
                Some(token_span(self.tokens, self.pos)),
            ));
        }
        self.advance();
        Ok(())
    }

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
            left = SpannedExpr::new(
                Expr::BinaryExpr {
                    left: Box::new(left),
                    op: op.clone(),
                    right: Box::new(right),
                },
                span,
            );
        }
        Ok(left)
    }

    pub fn parse_expression(&mut self) -> Result<SpannedExpr, FormulaError> {
        if self.peek() == TokenKind::Fn {
            return self.parse_function_def();
        }
        if self.peek() == TokenKind::LParen {
            let checkpoint = self.pos;
            if let Ok(lambda) = self.parse_lambda() {
                return Ok(lambda);
            }
            self.pos = checkpoint;
        }
        self.parse_logical_or()
    }

    fn parse_function_def(&mut self) -> Result<SpannedExpr, FormulaError> {
        let start_pos = self.pos;
        self.advance(); // fn
        let name_tok = self.advance();
        if name_tok.kind != TokenKind::Identifier {
            return Err(FormulaError::new(
                ErrorKind::ParseError,
                "E201",
                "คาดหวังชื่อฟังก์ชันหลัง 'fn'",
                Some(name_tok.span),
            ));
        }
        let name = name_tok.lexeme.clone();
        self.expect(TokenKind::LParen, "คาดหวัง '(' หลังชื่อฟังก์ชัน")?;
        let mut params = Vec::new();
        if self.peek() != TokenKind::RParen {
            loop {
                let p = self.advance();
                if p.kind != TokenKind::Identifier {
                    return Err(FormulaError::new(
                        ErrorKind::ParseError,
                        "E201",
                        "พารามิเตอร์ต้องเป็น identifier",
                        Some(p.span),
                    ));
                }
                params.push(p.lexeme.clone());
                if self.peek() == TokenKind::Comma {
                    self.advance();
                } else {
                    break;
                }
            }
        }
        self.expect(TokenKind::RParen, "คาดหวัง ')' หลังพารามิเตอร์")?;
        self.expect(TokenKind::Equal, "คาดหวัง '=' ก่อนนิพจน์ของฟังก์ชัน")?;
        let body = self.parse_expression()?;
        let span = Span {
            start: token_span(self.tokens, start_pos).start,
            end: body.meta.span.end,
        };
        Ok(SpannedExpr::new(
            Expr::FunctionDef {
                name,
                params,
                body: Box::new(body),
            },
            span,
        ))
    }

    fn parse_lambda(&mut self) -> Result<SpannedExpr, FormulaError> {
        let start_pos = self.pos;
        self.expect(TokenKind::LParen, "คาดหวัง '('")?;
        let mut params = Vec::new();
        if self.peek() != TokenKind::RParen {
            loop {
                let p = self.advance();
                if p.kind != TokenKind::Identifier {
                    return Err(FormulaError::new(
                        ErrorKind::ParseError,
                        "E201",
                        "พารามิเตอร์ต้องเป็น identifier",
                        Some(p.span),
                    ));
                }
                params.push(p.lexeme.clone());
                if self.peek() == TokenKind::Comma {
                    self.advance();
                } else {
                    break;
                }
            }
        }
        self.expect(TokenKind::RParen, "คาดหวัง ')'")?;
        self.expect(TokenKind::Arrow, "คาดหวัง '=>'")?;
        let body = self.parse_expression()?;
        let span = Span {
            start: token_span(self.tokens, start_pos).start,
            end: body.meta.span.end,
        };
        Ok(SpannedExpr::new(
            Expr::LambdaExpr {
                params,
                body: Box::new(body),
            },
            span,
        ))
    }

    fn parse_logical_or(&mut self) -> Result<SpannedExpr, FormulaError> {
        self.parse_left_associative_binary(
            Self::parse_logical_and,
            &[(TokenKind::OrOr, BinaryOp::Or)],
        )
    }

    fn parse_logical_and(&mut self) -> Result<SpannedExpr, FormulaError> {
        self.parse_left_associative_binary(
            Self::parse_equality,
            &[(TokenKind::AndAnd, BinaryOp::And)],
        )
    }

    fn parse_equality(&mut self) -> Result<SpannedExpr, FormulaError> {
        self.parse_left_associative_binary(
            Self::parse_comparison,
            &[
                (TokenKind::EqEq, BinaryOp::Eq),
                (TokenKind::NotEq, BinaryOp::NotEq),
            ],
        )
    }

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

    fn parse_term(&mut self) -> Result<SpannedExpr, FormulaError> {
        self.parse_left_associative_binary(
            Self::parse_factor,
            &[
                (TokenKind::Plus, BinaryOp::Add),
                (TokenKind::Minus, BinaryOp::Sub),
            ],
        )
    }

    fn parse_factor(&mut self) -> Result<SpannedExpr, FormulaError> {
        self.parse_left_associative_binary(
            Self::parse_unary,
            &[
                (TokenKind::Star, BinaryOp::Mul),
                (TokenKind::Slash, BinaryOp::Div),
            ],
        )
    }

    fn parse_unary(&mut self) -> Result<SpannedExpr, FormulaError> {
        if self.peek() == TokenKind::Minus || self.peek() == TokenKind::Bang {
            let op = match self.peek() {
                TokenKind::Minus => UnaryOp::Neg,
                TokenKind::Bang => UnaryOp::Not,
                _ => unreachable!(),
            };
            let start_pos = self.pos;
            self.advance();
            let expr = self.parse_primary()?;
            let span = Span {
                start: token_span(self.tokens, start_pos).start,
                end: expr.meta.span.end,
            };
            return Ok(SpannedExpr::new(
                Expr::UnaryExpr {
                    op,
                    expr: Box::new(expr),
                },
                span,
            ));
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Result<SpannedExpr, FormulaError> {
        let tok = self.advance();
        let span = tok.span;
        match &tok.kind {
            TokenKind::Number => {
                let n: f64 = tok.lexeme.parse().map_err(|_| {
                    FormulaError::new(
                        ErrorKind::ParseError,
                        "E202",
                        "ไม่สามารถแปลงตัวเลขได้",
                        Some(span),
                    )
                })?;
                Ok(SpannedExpr::new(Expr::Literal(Value::Number(n)), span))
            }
            TokenKind::String => Ok(SpannedExpr::new(
                Expr::Literal(Value::String(tok.lexeme.clone())),
                span,
            )),
            TokenKind::True => Ok(SpannedExpr::new(Expr::Literal(Value::Bool(true)), span)),
            TokenKind::False => Ok(SpannedExpr::new(Expr::Literal(Value::Bool(false)), span)),
            TokenKind::Null => Ok(SpannedExpr::new(Expr::Literal(Value::Null), span)),
            TokenKind::Identifier => {
                let mut name = tok.lexeme.clone();
                while self.peek() == TokenKind::Dot {
                    self.advance();
                    let field = self.advance();
                    if field.kind != TokenKind::Identifier {
                        return Err(FormulaError::new(
                            ErrorKind::ParseError,
                            "E201",
                            "Expected identifier after dot",
                            Some(field.span),
                        ));
                    }
                    name = format!("{}.{}", name, field.lexeme);
                }
                if self.peek() == TokenKind::LParen {
                    self.advance();
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
                    Ok(SpannedExpr::new(Expr::Variable(name), span))
                }
            }
            TokenKind::LParen => {
                let inner = self.parse_expression()?;
                self.expect(TokenKind::RParen, "ต้องการ ')'")?;
                Ok(SpannedExpr::new(Expr::Grouping(Box::new(inner)), span))
            }
            TokenKind::LBracket => {
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
            TokenKind::LBrace => {
                let mut pairs = Vec::new();
                if self.peek() != TokenKind::RBrace {
                    loop {
                        let key_tok = self.advance();
                        if key_tok.kind != TokenKind::Identifier {
                            return Err(FormulaError::new(
                                ErrorKind::ParseError,
                                "E201",
                                "key ใน map ต้องเป็น identifier",
                                Some(token_span(self.tokens, self.pos - 1)),
                            ));
                        }
                        let key = key_tok.lexeme.clone();
                        self.expect(TokenKind::Colon, "ต้องการ ':' หลัง key")?;
                        let value = self.parse_expression()?;
                        pairs.push((key, value));
                        if self.peek() == TokenKind::Comma {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                }
                self.expect(TokenKind::RBrace, "ต้องการ '}' ปิดท้าย map")?;
                Ok(SpannedExpr::new(Expr::MapLiteral(pairs), span))
            }
            _ => Err(FormulaError::new(
                ErrorKind::ParseError,
                "E203",
                &format!("ไม่คาดคิด token: {:?}", tok.kind),
                Some(span),
            )),
        }
    }
}

pub fn parse(tokens: &[Token]) -> Result<SpannedExpr, FormulaError> {
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expression()?;
    if parser.pos < tokens.len() - 1 {
        return Err(FormulaError::new(
            ErrorKind::ParseError,
            "E201",
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
    fn test_lambda_parsing() {
        let tokens = tokenize("(x) => x").unwrap();
        let ast = parse(&tokens).unwrap();
        match ast.expr {
            Expr::LambdaExpr { params, .. } => assert_eq!(params, vec!["x"]),
            _ => panic!("expected lambda"),
        }
    }
}
