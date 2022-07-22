use thiserror::Error;

use std::backtrace::Backtrace;

use crate::lexer::{AssignOp, BinaryOp, StateDesignator, Token, TokenKind, ValueKind};

#[derive(Debug, Clone)]
pub enum Value {
    String(String),
    Number(u64),
    Bool(bool),
    // TODO: The fields of a struct literal aren't really typed, right?
    StructLiteral { fields: Vec<StructField> },
}

impl Value {
    pub fn from_token(token: &Token) -> Self {
        match token.kind {
            TokenKind::Value(ValueKind::String) => Self::String(token.lexeme.clone()),
            TokenKind::Value(ValueKind::Number) => Self::Number(str::parse(&token.lexeme).unwrap()),
            TokenKind::Value(ValueKind::True) => Self::Bool(true),
            TokenKind::Value(ValueKind::False) => Self::Bool(false),
            _ => panic!("Value::from_token() with bad token '{:?}'", token),
        }
    }

    pub fn empty_struct() -> Value {
        Value::StructLiteral { fields: Vec::new() }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IdentifierAccessPath {
    identifier_name: String,
    field_names: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum SimpleExpression {
    Value(Value),
    IdentifierAccess(IdentifierAccessPath),
    ProcedureCall(String), // TODO: Arguments
    StateAccess {
        state: StateDesignator,
        path: IdentifierAccessPath,
    },
}

#[derive(Debug, Clone)]
pub enum Expression {
    Simple(SimpleExpression),
    Binary {
        lhs: SimpleExpression,
        op: BinaryOp,
        rhs: Box<Expression>,
    },
}

#[derive(Debug, Clone)]
pub enum Statement {
    AssertionCall {
        name: String,
    },
    IdentifierAssignment {
        lhs: IdentifierAccessPath,
        assign_op: AssignOp,
        rhs: Expression,
    },
    Expression(Expression),
}

#[derive(Debug, Clone)]
pub struct Procedure {
    name: String,
    body: Vec<Statement>,
}

#[derive(Debug, Clone)]
pub struct StructField {
    name: String,
    type_name: String,
    value: Value,
}

#[derive(Debug, Clone)]
pub struct Struct {
    name: String,
    fields: Vec<StructField>,
}

#[derive(Debug, Clone)]
pub struct Assertion {
    name: String,
    predicate: Expression,
}

#[derive(Debug, Clone)]
pub struct Program {
    state: Struct,
    procedures: Vec<Procedure>,
    assertions: Vec<Assertion>,
    main: Procedure,
}

#[derive(Debug, Clone, Error)]
pub enum ParseError {
    #[error("main procedure missing")]
    MissingMain,
    #[error("state struct missing")]
    MissingState,
    #[error("Procedure '{0}' is already defined")]
    ProcedureRedefinition(String),
    #[error("Assertion '{0}' is already defined")]
    AssertionRedefinition(String),
    #[error("Struct '{0}' is already defined")]
    StructRedefinition(String),
    #[error("Expected '{expected}' but got '{got:?}'")]
    Expected { expected: String, got: Token },
    #[error("Unexpected end of input")]
    End(String),
    #[error("Error: '{0}'")]
    Other(String),
}

use ParseError::*;

#[derive(Debug, Error)]
#[error("{error}")]
pub struct ParseErrorWrapper {
    pub error: ParseError,
    backtrace: std::backtrace::Backtrace,
}

macro_rules! err {
    ($error:expr) => {
        Err(ParseErrorWrapper {
            error: $error,
            backtrace: Backtrace::force_capture(),
        })
    };
}

macro_rules! wrap {
    ($error:expr) => {
        ParseErrorWrapper {
            error: $error,
            backtrace: Backtrace::force_capture(),
        }
    };
}

pub type ParseResult<T> = Result<T, ParseErrorWrapper>;

#[derive(Debug, Clone)]
pub struct Parser {
    tokens: Vec<Token>,
    index: usize,
    main: Option<Procedure>,
    state: Option<Struct>,
    assertions: Vec<Assertion>,
    procedures: Vec<Procedure>,
    structs: Vec<Struct>,
}

// (self, pattern) -> ParseResult<Option<&Token>>
macro_rules! advance_if_matches {
    ($self:ident, $pattern:pat) => {{
        // TODO: Not sure if EOF is acceptable here or not
        let token = $self
            .tokens
            .get($self.index)
            .ok_or_else(|| ParseErrorWrapper {
                error: End("Unexpected end on input".to_string()),
                backtrace: Backtrace::force_capture(),
            })?;

        Ok(matches!(token.kind, $pattern).then(|| {
            $self.index += 1;
            token
        }))
    }};
}

// (self, pattern) -> ParseResult<Option<PatternExtraction>>
macro_rules! extract_if_kind_matches {
    ($self:expr, $p:path) => {{
        let token = $self
            .tokens
            .get($self.index)
            .ok_or_else(|| ParseErrorWrapper {
                error: End("Unexpected end on input".to_string()),
                backtrace: Backtrace::force_capture(),
            })?;

        match token.kind {
            $p(value) => {
                $self.index += 1;
                Ok(Some(value))
            }
            _ => Ok(None),
        }
    }};
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            index: 0,
            main: None,
            state: None,
            assertions: Vec::new(),
            procedures: Vec::new(),
            structs: Vec::new(),
        }
    }

    fn at_end(&self) -> bool {
        self.index >= self.tokens.len()
    }

    fn go_back(&mut self) {
        // TODO: If I add more state than just index, make sure this works
        assert!(self.index > 0);
        self.index -= 1;
    }

    fn peek(&self) -> ParseResult<&Token> {
        let token = self
            .tokens
            .get(self.index)
            .ok_or_else(|| wrap!(ParseError::End("Unexpected end on input".to_string())))?;

        println!("Current token {:#?}", token);

        Ok(token)
    }

    fn advance(&mut self) -> ParseResult<&Token> {
        let token = self
            .tokens
            .get(self.index)
            .ok_or_else(|| wrap!(ParseError::End("Unexpected end on input".to_string())))?;

        println!("Next token: {:#?}", token);

        self.index += 1;

        Ok(token)
    }

    fn expect(&mut self, expected: TokenKind) -> ParseResult<&Token> {
        let token = self.advance()?;

        if token.kind != expected {
            unexpected(format!("{expected:?}"), token)?;
        }

        Ok(token)
    }

    fn next_is_not(&self, kind: TokenKind) -> bool {
        self.tokens
            .get(self.index)
            .map(|t| t.kind != kind)
            .unwrap_or(false)
    }

    fn define_procedure(&mut self, name: String, body: Vec<Statement>) -> ParseResult<()> {
        if self.procedures.iter().any(|p| p.name == name) {
            return err!(ProcedureRedefinition(name));
        }

        let proc = Procedure {
            name: name.clone(),
            body,
        };

        if name == "main" {
            self.main = Some(proc.clone());
        }

        // TODO: Not sure if allowing to refer to main like any other procedure is smart.
        self.procedures.push(proc);

        Ok(())
    }

    fn define_assertion(&mut self, name: String, predicate: Expression) -> ParseResult<()> {
        if self.assertions.iter().any(|p| p.name == name) {
            return err!(AssertionRedefinition(name));
        }

        self.assertions.push(Assertion { name, predicate });

        Ok(())
    }

    fn define_struct(&mut self, name: String, fields: Vec<StructField>) -> ParseResult<()> {
        if self.structs.iter().any(|p| p.name == name) {
            return err!(StructRedefinition(name));
        }

        let structure = Struct {
            name: name.clone(),
            fields,
        };

        if name == "state" {
            self.state = Some(structure.clone());
        }

        self.structs.push(structure);

        Ok(())
    }

    pub fn parse(mut self) -> ParseResult<Program> {
        while !self.at_end() {
            self.top_level_decl()?;
            println!("{self:?}\n\n\n\n\n");
        }

        if self.main.is_none() {
            return err!(MissingMain);
        }

        if self.state.is_none() {
            return err!(MissingState);
        }

        Ok(Program {
            main: self.main.unwrap(),
            state: self.state.unwrap(),
            assertions: self.assertions,
            procedures: self.procedures,
        })
    }

    fn top_level_decl(&mut self) -> ParseResult<()> {
        let token = self.advance()?;

        match token.kind {
            TokenKind::Procedure => self.procedure(),
            TokenKind::Assertion => self.assertion(),
            TokenKind::Struct => self.structure(),
            _ => err("Invalid top level item", token)?,
        }
    }

    fn structure(&mut self) -> ParseResult<()> {
        let name = self.expect(TokenKind::Identifier)?.lexeme.clone();

        self.expect(TokenKind::LeftBrace)?;

        let mut fields = Vec::new();

        while self.next_is_not(TokenKind::RightBrace) {
            let name = self.expect(TokenKind::Identifier)?.lexeme.clone();
            self.expect(TokenKind::Colon)?;

            let type_name = self.expect(TokenKind::Identifier)?.lexeme.clone();

            self.expect(TokenKind::AssignOp(AssignOp::Equal))?;

            let next = self.advance()?;

            let initializer = match next.kind {
                TokenKind::LeftBrace => {
                    self.expect(TokenKind::RightBrace)?;

                    Value::empty_struct()
                }
                TokenKind::Value(_) => Value::from_token(next),
                _ => unexpected("Value or struct literal", next)?,
            };

            self.expect(TokenKind::Comma)?;

            fields.push(StructField {
                name,
                type_name,
                value: initializer,
            })
        }

        self.expect(TokenKind::RightBrace)?;

        self.define_struct(name, fields)?;

        Ok(())
    }

    fn assertion(&mut self) -> ParseResult<()> {
        let name = self.expect(TokenKind::Identifier)?.lexeme.clone();

        self.expect(TokenKind::LeftBrace)?;

        let predicate = self.expression()?;

        self.expect(TokenKind::RightBrace)?;

        self.define_assertion(name, predicate)?;

        Ok(())
    }

    fn procedure(&mut self) -> ParseResult<()> {
        let name = self.expect(TokenKind::Identifier)?.lexeme.clone();

        self.expect(TokenKind::LeftBrace)?;

        let mut body = Vec::new();

        while self.next_is_not(TokenKind::RightBrace) {
            body.push(self.statement()?);
        }

        self.expect(TokenKind::RightBrace)?;

        self.define_procedure(name, body)?;

        Ok(())
    }

    fn statement(&mut self) -> ParseResult<Statement> {
        let next = self.advance()?.clone();

        Ok(match next.kind {
            TokenKind::Bang => self.assertion_statement()?,
            // TODO: This should be extracted into a function somehow
            TokenKind::Identifier => {
                // - 1 because the identifier itself was already consumed
                let old_index = self.index - 1;

                let lhs = self.identifier_access(next.lexeme)?;

                if let Some(assign_op) = extract_if_kind_matches!(self, TokenKind::AssignOp)? {
                    let rhs = self.expression()?;

                    self.expect(TokenKind::SemiColon)?;

                    Statement::IdentifierAssignment {
                        lhs,
                        assign_op,
                        rhs,
                    }
                } else {
                    // Was an expression after all, so put everything back
                    self.index = old_index;
                    let expr = self.expression()?;
                    self.expect(TokenKind::SemiColon)?;
                    Statement::Expression(expr)
                }
            }
            _ => {
                self.go_back();
                let expr = self.expression()?;
                self.expect(TokenKind::SemiColon)?;
                Statement::Expression(expr)
            }
        })
    }

    fn assertion_statement(&mut self) -> ParseResult<Statement> {
        let name = self.expect(TokenKind::Identifier)?.lexeme.clone();
        self.expect(TokenKind::SemiColon)?;

        Ok(Statement::AssertionCall { name })
    }

    fn expression(&mut self) -> ParseResult<Expression> {
        let lhs = self.simple_expression()?;

        if let Some(op) = extract_if_kind_matches!(self, TokenKind::BinaryOp)? {
            let rhs = Box::new(self.expression()?);

            Ok(Expression::Binary { lhs, op, rhs })
        } else {
            Ok(Expression::Simple(lhs))
        }
    }

    fn identifier_access(&mut self, identifier_name: String) -> ParseResult<IdentifierAccessPath> {
        let mut field_names = Vec::new();
        while advance_if_matches!(self, TokenKind::Dot)?.is_some() {
            let field = self.expect(TokenKind::Identifier)?;

            field_names.push(field.lexeme.clone());
        }

        Ok(IdentifierAccessPath {
            identifier_name,
            field_names,
        })
    }

    fn simple_expression(&mut self) -> ParseResult<SimpleExpression> {
        let next = self.advance()?.clone();

        match next.kind {
            TokenKind::Value(_) => Ok(SimpleExpression::Value(Value::from_token(&next))),
            // TODO: Allow struct literals here
            TokenKind::Identifier => {
                if advance_if_matches!(self, TokenKind::LeftParen)?.is_some() {
                    self.expect(TokenKind::RightParen)?;

                    Ok(SimpleExpression::ProcedureCall(next.lexeme))
                } else {
                    Ok(SimpleExpression::IdentifierAccess(
                        self.identifier_access(next.lexeme)?,
                    ))
                }
            }
            TokenKind::StateDesignator(state) => {
                self.expect(TokenKind::Colon)?;
                let name = self.expect(TokenKind::Identifier)?.lexeme.clone();
                let path = self.identifier_access(name)?;

                Ok(SimpleExpression::StateAccess { state, path })
            }

            _ => unexpected("Value or identifer access", &next)?,
        }
    }
}

fn unexpected(expected: impl AsRef<str>, got: &Token) -> ParseResult<!> {
    err!(ParseError::Expected {
        expected: expected.as_ref().to_string(),
        got: got.clone(),
    })
}

fn err(reason: impl AsRef<str>, token: &Token) -> ParseResult<!> {
    let reason = format!("{}: {}", token.error_string(), reason.as_ref());
    err!(ParseError::Other(reason))
}
