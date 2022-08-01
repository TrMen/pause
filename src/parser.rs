use thiserror::Error;

use std::{backtrace::Backtrace, collections::HashMap};

use crate::{
    interpreter::{InterpretationError, InterpretationResult},
    lexer::{AssignOp, BinaryOp, ExecutionDesignator, Token, TokenKind},
    typechecker::BuildinType,
};

#[derive(Debug, Clone, PartialEq)]
pub enum Indirection {
    Field(String),
    Subscript(Box<Expression>),
    // TODO: Add function calls
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvaluatedIndirection {
    Field(String),
    Subscript(usize),
    // TODO: Add function calls
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvaluatedAccessPath {
    pub name: String,
    pub indirections: Vec<EvaluatedIndirection>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AccessPath {
    pub name: String,
    pub indirections: Vec<Indirection>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SimpleExpression {
    Boolean(bool),
    String(String),
    NumberLiteral(u64),
    StructLiteral(Struct),
    ArrayLiteral(Vec<Expression>),
    ArrayAccess {
        target: Box<Expression>,
        index: Box<Expression>,
    },
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
pub enum ParsedType {
    // TODO: Add things like reference, tuples, generics, functions
    Simple { name: String },
    Array { inner: Box<ParsedType> },
    Void,
    Unknown, // For things like struct literals that are assigned expressions
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructField {
    pub name: String,
    pub parsed_type: ParsedType,
    // TODO: Make sure this expression only uses literals
    pub initializer: Expression,
    // TODO: Add modifiers like is_mutable
}

#[derive(Debug, Clone, PartialEq)]
pub struct Struct {
    pub name: String,
    pub fields: Vec<StructField>,
}
// TODO: This looks a lot like StructField. But prolly makes sense to keep them separate
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionParameter {
    pub name: String,
    pub parsed_type: ParsedType,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub params: Vec<FunctionParameter>,
    pub return_type: ParsedType,
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
                    SimpleExpression::StateAccess(_) => todo!(),
                    SimpleExpression::ExecutionAccess { execution, path } => todo!(),
                    SimpleExpression::FunctionCall { name, arguments } => todo!(),
                    SimpleExpression::BindingAccess(_) => todo!(),
                    SimpleExpression::ArrayLiteral(_) => todo!(),
                    SimpleExpression::ArrayAccess { target, index } => todo!(),
                    SimpleExpression::Boolean(_) => todo!(),
                    SimpleExpression::String(_) => todo!(),
                    SimpleExpression::NumberLiteral(_) => todo!(),
                    SimpleExpression::StructLiteral(_) => todo!(),
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

// TODO: Why was that a macro again? I think because of borrowing rules
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

// TODO: This is one macro too deep. This is nasty

// (self, Fn(&mut Self) -> ParseResult<T>, separator: Pattern, terminator: Pattern)
// -> ParseResult<Vec<T>>
macro_rules! collect_list {
    // This is needed to turn the pattern into a string for the inner macro invocation
    ($self:ident, $parser:expr, $separator:pat, $terminator:pat) => {
        collect_list!(
            $self,
            $parser,
            $separator,
            $terminator,
            stringify!($terminator)
        )
    };

    ($self:ident, $parser:expr, $separator:pat, $terminator:pat, $string_terminator:expr) => {{
        let mut list = Vec::new();

        if advance_if_matches!($self, $terminator)?.is_none() {
            list.push($parser($self)?);

            while advance_if_matches!($self, $separator)?.is_some() {
                if matches!($self.peek()?.kind, $terminator) {
                    // A trailing separator is fine in all lists
                    break;
                }
                list.push($parser($self)?);
            }

            advance_if_matches!($self, $terminator)?.ok_or_else(|| ParseErrorWrapper {
                error: Expected {
                    expected: $string_terminator.to_string(),
                    got: $self.peek().unwrap().clone(),
                },
                backtrace: Backtrace::force_capture(),
            })?;
        }

        Ok(list)
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
        } else {
            self.procedures.insert(name, proc);
        }

        Ok(())
    }

    fn define_function(
        &mut self,
        name: String,
        params: Vec<FunctionParameter>,
        return_type: ParsedType,
        expression: Expression,
    ) -> ParseResult<()> {
        let func = Function {
            name: name.clone(),
            params,
            return_type,
            expression,
        };

        if self.functions.contains_key(&name) {
            return err!(FunctionRedefinition(name));
        }

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

        let fields = collect_list!(
            self,
            |this: &mut Self| {
                let name = this.expect(TokenKind::Identifier)?.lexeme.clone();
                this.expect(TokenKind::Colon)?;

                let parsed_type = this.parsed_type()?;

                this.expect(TokenKind::AssignOp(AssignOp::Equal))?;

                let next = this.advance()?;

                let initializer = self.expression()?;

                println!("{}: {:?} = {:?}", name, parsed_type, initializer);

                println!("NEXT: {:?}", this.peek()?);

                Ok(StructField {
                    name,
                    parsed_type,
                    initializer,
                })
            },
            TokenKind::Comma,
            TokenKind::RightBrace
        )?;

        self.define_struct(name, fields)?;

        Ok(())
    }

    fn parsed_type(&mut self) -> ParseResult<ParsedType> {
        let next = self.advance()?;
        match next.kind {
            TokenKind::LeftBracket => {
                let inner = Box::new(self.parsed_type()?);
                self.expect(TokenKind::RightBracket)?;
                Ok(ParsedType::Array { inner })
            }
            TokenKind::Identifier => Ok(ParsedType::Simple {
                name: next.lexeme.to_string(),
            }),
            // TODO: How to handle void? I don't think anything is ever explicitly 'void' type, so
            // doesn't need to be parsed
            _ => unexpected("type name", next)?,
        }
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

        let params = collect_list!(
            self,
            |this: &mut Self| {
                let name = this.expect(TokenKind::Identifier)?.lexeme.clone();

                this.expect(TokenKind::Colon)?;

                let parsed_type = this.parsed_type()?;

                Ok(FunctionParameter { name, parsed_type })
            },
            TokenKind::Comma,
            TokenKind::RightParen
        )?;

        self.expect(TokenKind::SmallArrow)?;

        let return_type = self.parsed_type()?;

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

                let lhs = self.access_path()?;

                if let Some(assign_op) = extract_if_kind_matches!(self, TokenKind::AssignOp)? {
                    let rhs = self.expression()?;

                    self.expect(TokenKind::SemiColon)?;

                    // TODO: Check that lhs is state assignment (prolly needs to happen at runtime
                    // fro now)
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

    fn access_path(&mut self) -> ParseResult<AccessPath> {
        let name = self.expect(TokenKind::Identifier)?.lexeme.clone();

        let mut indirections = Vec::new();
        loop {
            if advance_if_matches!(self, TokenKind::Dot)?.is_some() {
                let field = self.expect(TokenKind::Identifier)?;
                indirections.push(Indirection::Field(field.lexeme.clone()));
            } else if advance_if_matches!(self, TokenKind::LeftBracket)?.is_some() {
                let index = Box::new(self.expression()?);
                self.expect(TokenKind::RightBracket)?;

                indirections.push(Indirection::Subscript(index));
            } else {
                break;
            }
        }

        Ok(AccessPath { name, indirections })
    }

    fn simple_expression(&mut self) -> ParseResult<SimpleExpression> {
        let next = self.advance()?.clone();

        let simple_expression = match next.kind {
            TokenKind::LeftBracket => {
                let array = collect_list!(
                    self,
                    |this: &mut Self| { this.expression() },
                    TokenKind::Comma,
                    TokenKind::RightBracket
                )?;

                // TODO: Arrays should hold the same type

                Ok(SimpleExpression::ArrayLiteral(array))
            }
            TokenKind::Number => Ok(SimpleExpression::NumberLiteral(
                str::parse(&next.lexeme).unwrap(),
            )),
            TokenKind::String => Ok(SimpleExpression::String(next.lexeme.to_string())),
            TokenKind::True => Ok(SimpleExpression::Boolean(true)),
            TokenKind::False => Ok(SimpleExpression::Boolean(false)),
            // TODO: Allow struct literals here
            TokenKind::Dot => Ok(SimpleExpression::StateAccess(self.access_path()?)),
            TokenKind::Identifier => {
                if advance_if_matches!(self, TokenKind::LeftParen)?.is_some() {
                    // Function call
                    let arguments = collect_list!(
                        self,
                        |this: &mut Self| this.expression(),
                        TokenKind::Comma,
                        TokenKind::RightParen
                    )?;

                    Ok(SimpleExpression::FunctionCall {
                        name: next.lexeme,
                        arguments,
                    })
                } else {
                    // Pub back the identifier
                    self.go_back();
                    Ok(SimpleExpression::BindingAccess(self.access_path()?))
                }
            }
            TokenKind::ExecutionDesignator(execution) => {
                self.expect(TokenKind::Colon)?;
                self.expect(TokenKind::Dot)?;
                let path = self.access_path()?;

                Ok(SimpleExpression::ExecutionAccess { execution, path })
            }

            _ => unexpected("Value or identifer access", &next)?,
        }?;

        // TODO: There's no way this is enough. Array access needs proper precedence parsing :<
        if advance_if_matches!(self, TokenKind::LeftBracket)?.is_some() {
            let index = Box::new(self.expression()?);

            self.expect(TokenKind::RightBracket)?;

            Ok(SimpleExpression::ArrayAccess {
                target: Box::new(Expression::Simple(simple_expression)),
                index,
            })
        } else {
            Ok(simple_expression)
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
