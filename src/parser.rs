use thiserror::Error;

use std::{backtrace::Backtrace, collections::HashMap};

use crate::{
    interpreter::{InterpretationError, InterpretationResult},
    lexer::{AssignOp, BinaryOp, ExecutionDesignator, Token, TokenKind, ValueKind},
};

#[derive(Debug, Clone, PartialEq)]
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

    pub fn type_matches(&self, other: &Value) -> bool {
        // TODO: This doesn't work for structs that are different
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }

    pub fn empty_struct() -> Value {
        Value::StructLiteral { fields: Vec::new() }
    }

    pub fn is_truthy(&self) -> InterpretationResult<bool> {
        // TODO: Do I even want to have that? Better to typecheck, and then we simply don't have
        // the 'Value' enum at all

        match self {
            Value::Bool(boolean) => Ok(*boolean),
            _ => Err(InterpretationError::InvalidTypeConversion {
                from: self.type_name().to_string(),
                to: Value::Bool(true).type_name().to_string(),
            }),
        }
    }

    pub fn type_name(&self) -> &str {
        match self {
            Value::String(_) => "string",
            Value::Number(_) => "u64",
            Value::Bool(_) => "bool",
            Value::StructLiteral { .. } => "struct", // TODO: Print fields
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccessPath {
    pub name: String,
    pub fields: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SimpleExpression {
    Value(Value),
    StateAccess(AccessPath),
    ExecutionAccess {
        execution: ExecutionDesignator,
        path: AccessPath,
    },
    BindingAccess(AccessPath),
    FunctionCall {
        name: String,
        arguments: Vec<Expression>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    Simple(SimpleExpression),
    Binary {
        lhs: SimpleExpression,
        op: BinaryOp,
        rhs: Box<Expression>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    AssertionCall {
        name: String,
    },
    ProcedureCall {
        name: String,
    },
    StateAssignment {
        lhs: AccessPath,
        assign_op: AssignOp,
        rhs: Expression,
    },
    Expression(Expression),
}

#[derive(Debug, Clone)]
pub struct Procedure {
    pub name: String,
    pub body: Vec<Statement>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructField {
    pub name: String,
    pub type_name: String,
    pub initial_value: Value, //  TODO: Make clearer that this is just initial, nothing you can actually change at runtime
}

#[derive(Debug, Clone)]
pub struct Struct {
    pub name: String,
    pub fields: Vec<StructField>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionParameter {
    pub name: String,
    pub type_name: String,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub params: Vec<FunctionParameter>,
    pub return_type: String,
    pub expression: Expression,
}

#[derive(Debug, Clone)]
pub struct Assertion {
    pub name: String,
    pub predicate: Expression,
}

#[derive(Debug, Clone)]
pub struct Program {
    // TODO: This duplicates the name. But I think I really want to be able to pass a Procedure
    // around without having to also pass it's name.
    pub procedures: HashMap<String, Procedure>,
    pub assertions: HashMap<String, Assertion>,
    pub functions: HashMap<String, Function>,
    pub structs: HashMap<String, Struct>,
    pub main: Procedure,
}

impl Program {
    // This really shouldn't be here, but instead done at compile-time
    pub fn ensure_uses_defined_components(&self, stmt: &Statement) -> InterpretationResult<()> {
        match stmt {
            Statement::AssertionCall { name } => {
                self.assertions
                    .contains_key(name)
                    .then_some(())
                    .ok_or_else(|| InterpretationError::UnknownAssertion(name.to_string()))?
            }
            Statement::StateAssignment { .. } => (), // TODO: actually check this
            Statement::Expression(expression) => match expression {
                Expression::Simple(simple) => match simple {
                    SimpleExpression::Value(_) => (),
                    SimpleExpression::StateAccess(_) => todo!(),
                    SimpleExpression::ExecutionAccess { execution, path } => todo!(),
                    SimpleExpression::FunctionCall { name, arguments } => todo!(),
                    SimpleExpression::BindingAccess(_) => todo!(),
                },
                Expression::Binary { .. } => todo!(),
            },
            Statement::ProcedureCall { name } => {
                self.procedures
                    .contains_key(name)
                    .then_some(())
                    .ok_or_else(|| InterpretationError::UnknownProcedure(name.to_string()))?
            }
        };

        Ok(())
    }
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
    #[error("Function '{0}' is already defined")]
    FunctionRedefinition(String),
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
    assertions: HashMap<String, Assertion>,
    functions: HashMap<String, Function>,
    procedures: HashMap<String, Procedure>,
    structs: HashMap<String, Struct>,
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
            assertions: HashMap::new(),
            functions: HashMap::new(),
            procedures: HashMap::new(),
            structs: HashMap::new(),
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

    fn define_procedure(&mut self, name: String, body: Vec<Statement>) -> ParseResult<()> {
        if self.procedures.contains_key(&name) {
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
        self.procedures.insert(name, proc);

        Ok(())
    }

    fn define_function(
        &mut self,
        name: String,
        params: Vec<FunctionParameter>,
        return_type: String,
        expression: Expression,
    ) -> ParseResult<()> {
        if self.functions.contains_key(&name) {
            return err!(FunctionRedefinition(name));
        }

        let func = Function {
            name: name.clone(),
            params,
            return_type,
            expression,
        };

        self.functions.insert(name, func);

        Ok(())
    }

    fn define_assertion(&mut self, name: String, predicate: Expression) -> ParseResult<()> {
        if self.assertions.contains_key(&name) {
            return err!(AssertionRedefinition(name));
        }

        self.assertions
            .insert(name.clone(), Assertion { name, predicate });

        Ok(())
    }

    fn define_struct(&mut self, name: String, fields: Vec<StructField>) -> ParseResult<()> {
        if self.structs.contains_key(&name) {
            return err!(StructRedefinition(name));
        }

        let structure = Struct {
            name: name.clone(),
            fields,
        };

        if name == "state" {
            self.state = Some(structure.clone());
        }

        self.structs.insert(name, structure);

        Ok(())
    }

    pub fn parse(mut self) -> ParseResult<(Struct, Program)> {
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

        Ok((
            self.state.unwrap(),
            Program {
                main: self.main.unwrap(),
                assertions: self.assertions,
                functions: self.functions,
                procedures: self.procedures,
                structs: self.structs,
            },
        ))
    }

    fn top_level_decl(&mut self) -> ParseResult<()> {
        let token = self.advance()?;

        match token.kind {
            TokenKind::Procedure => self.procedure(),
            TokenKind::Assertion => self.assertion(),
            TokenKind::Function => self.function(),
            TokenKind::Struct => self.structure(),
            _ => err("Invalid top level item", token)?,
        }
    }

    fn structure(&mut self) -> ParseResult<()> {
        let name = self.expect(TokenKind::Identifier)?.lexeme.clone();

        self.expect(TokenKind::LeftBrace)?;

        let mut fields = Vec::new();

        while advance_if_matches!(self, TokenKind::RightBrace)?.is_none() {
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
                initial_value: initializer,
            })
        }

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

    fn function(&mut self) -> ParseResult<()> {
        let name = self.expect(TokenKind::Identifier)?.lexeme.clone();

        self.expect(TokenKind::LeftParen)?;

        let mut params = Vec::new();

        while advance_if_matches!(self, TokenKind::RightParen)?.is_none() {
            let name = self.expect(TokenKind::Identifier)?.lexeme.clone();

            self.expect(TokenKind::Colon)?;

            let type_name = self.expect(TokenKind::Identifier)?.lexeme.clone();

            params.push(FunctionParameter { name, type_name });
        }

        self.expect(TokenKind::SmallArrow)?;

        let return_type = self.expect(TokenKind::Identifier)?.lexeme.clone();

        self.expect(TokenKind::LeftBrace)?;

        let expression = self.expression()?;

        self.expect(TokenKind::RightBrace)?;

        self.define_function(name, params, return_type, expression)?;

        Ok(())
    }

    fn procedure(&mut self) -> ParseResult<()> {
        let name = self.expect(TokenKind::Identifier)?.lexeme.clone();

        self.expect(TokenKind::LeftBrace)?;

        let mut body = Vec::new();

        while advance_if_matches!(self, TokenKind::RightBrace)?.is_none() {
            body.push(self.statement()?);
        }

        self.define_procedure(name, body)?;

        Ok(())
    }

    fn statement(&mut self) -> ParseResult<Statement> {
        let next = self.advance()?.clone();

        Ok(match next.kind {
            TokenKind::Bang => self.assertion_statement()?,
            // TODO: This should be extracted into a function somehow
            TokenKind::DotDot => self.procedure_call_statement()?,
            TokenKind::Dot => {
                // - 1 because the dot itself was already consumed
                let old_index = self.index - 1;

                let lhs = self.state_access()?;

                if let Some(assign_op) = extract_if_kind_matches!(self, TokenKind::AssignOp)? {
                    let rhs = self.expression()?;

                    self.expect(TokenKind::SemiColon)?;

                    Statement::StateAssignment {
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

    fn procedure_call_statement(&mut self) -> ParseResult<Statement> {
        let name = self.expect(TokenKind::Identifier)?.lexeme.clone();
        self.expect(TokenKind::SemiColon)?;

        Ok(Statement::ProcedureCall { name })
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

    fn state_access(&mut self) -> ParseResult<AccessPath> {
        let identifier_name = self.expect(TokenKind::Identifier)?.lexeme.clone();

        let mut field_names = Vec::new();
        while advance_if_matches!(self, TokenKind::Dot)?.is_some() {
            let field = self.expect(TokenKind::Identifier)?;

            field_names.push(field.lexeme.clone());
        }

        Ok(AccessPath {
            name: identifier_name,
            fields: field_names,
        })
    }

    fn binding_access(&mut self, name: String) -> ParseResult<AccessPath> {
        let mut field_names = Vec::new();
        while advance_if_matches!(self, TokenKind::Dot)?.is_some() {
            let field = self.expect(TokenKind::Identifier)?;

            field_names.push(field.lexeme.clone());
        }

        Ok(AccessPath {
            name,
            fields: field_names,
        })
    }

    fn simple_expression(&mut self) -> ParseResult<SimpleExpression> {
        let next = self.advance()?.clone();

        match next.kind {
            TokenKind::Value(_) => Ok(SimpleExpression::Value(Value::from_token(&next))),
            // TODO: Allow struct literals here
            TokenKind::Dot => Ok(SimpleExpression::StateAccess(self.state_access()?)),
            TokenKind::Identifier => {
                if advance_if_matches!(self, TokenKind::LeftParen)?.is_some() {
                    // Function call
                    let mut arguments = Vec::new();

                    while advance_if_matches!(self, TokenKind::RightParen)?.is_none() {
                        arguments.push(self.expression()?);
                        advance_if_matches!(self, TokenKind::Comma)?;
                    }

                    Ok(SimpleExpression::FunctionCall {
                        name: next.lexeme,
                        arguments,
                    })
                } else {
                    Ok(SimpleExpression::BindingAccess(
                        self.binding_access(next.lexeme)?,
                    ))
                }
            }
            TokenKind::ExecutionDesignator(execution) => {
                self.expect(TokenKind::Colon)?;
                self.expect(TokenKind::Dot)?;
                let path = self.state_access()?;

                Ok(SimpleExpression::ExecutionAccess { execution, path })
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
