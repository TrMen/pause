use serde::{Deserialize, Serialize};
use thiserror::Error;

use std::{backtrace::Backtrace, collections::HashMap};

use crate::lexer::{AssignOp, BinaryOp, ExecutionDesignator, Token, TokenKind, UnaryOp};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Pattern {
    Value(u64), // TODO: Allow more than just numbers
    Else,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub rhs: Expression,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SimpleExpression {
    Boolean(bool),
    String(String),
    NumberLiteral(u64),
    StructLiteral {
        name: Option<String>,
        fields: Vec<StructField>,
    },
    Match {
        expr: Box<Expression>,
        arms: Vec<MatchArm>,
    },
    ArrayLiteral(Vec<Expression>),
    EnumVariant {
        enum_name: String, // I don't think a full parsed_type goes here, since it can only be an enum name
        variant_name: String,
        initializer: Option<Box<Expression>>,
    },
    ExecutionAccess {
        execution: ExecutionDesignator,
        access_expression: Box<AccessExpression>,
    },
    BindingAccess {
        name: String,
    },
    StateAccess {
        name: String,
    },
    FunctionCall {
        name: String,
        arguments: Vec<Expression>,
    },
    Parentheses {
        inner: Box<Expression>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    Unary(UnaryExpression),
    Binary {
        lhs: UnaryExpression,
        op: BinaryOp,
        rhs: Box<Expression>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryExpression {
    Access(AccessExpression),
    WithUnary {
        op: UnaryOp,
        expr: Box<UnaryExpression>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Indirection {
    Subscript { index_expr: Expression },
    Field { field_name: String },
}

#[derive(Debug, Clone, PartialEq)]
pub struct AccessExpression {
    pub lhs: SimpleExpression,
    pub indirections: Vec<Indirection>,
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
        lhs: AccessExpression,
        assign_op: AssignOp,
        rhs: Expression,
    },
    Expression(Expression),
    For {
        iterator: Expression,
        body: Vec<Statement>,
    },
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
pub struct EnumVariant {
    pub name: String,
    pub parsed_type: ParsedType,
}

#[derive(Debug, Clone)]
pub struct Enum {
    pub name: String,
    pub variants: Vec<EnumVariant>,
}

#[derive(Debug, Clone)]
pub struct Program {
    // TODO: This duplicates the name. But I think I really want to be able to pass a Procedure
    // around without having to also pass it's name.
    pub procedures: HashMap<String, Procedure>,
    pub assertions: HashMap<String, Assertion>,
    pub functions: HashMap<String, Function>,
    pub structs: HashMap<String, Struct>,
    pub enums: HashMap<String, Enum>,
    pub main: Procedure,
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
    #[error("Enum '{0}' is already defined")]
    EnumRedefinition(String),
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
    pub backtrace: std::backtrace::Backtrace,
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
    enums: HashMap<String, Enum>,
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

    ($self:ident, $parser:expr, $terminator:pat) => {{
        let mut list = Vec::new();

        while advance_if_matches!($self, $terminator)?.is_none() {
            list.push($parser($self)?);
        }

        Ok(list)
    }};

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
        let token = $self.peek()?;

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
            enums: HashMap::new(),
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
            self.main = Some(proc);
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
                enums: self.enums,
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
            TokenKind::Enum => self.enumeration(),
            _ => err("Invalid top level item", token)?,
        }
    }

    fn enumeration(&mut self) -> ParseResult<()> {
        let name = self.expect(TokenKind::Identifier)?.lexeme.clone();

        self.expect(TokenKind::LeftBrace)?;

        let variants = collect_list!(
            self,
            |this: &mut Self| {
                let name = this.expect(TokenKind::Identifier)?.lexeme.clone();

                this.expect(TokenKind::Colon)?;

                let parsed_type = this.parsed_type()?;

                Ok(EnumVariant { parsed_type, name })
            },
            TokenKind::Comma,
            TokenKind::RightBrace
        )?;

        if self.enums.contains_key(&name) {
            return err!(EnumRedefinition(name));
        }

        self.enums.insert(name.clone(), Enum { name, variants });

        Ok(())
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

                let initializer = this.expression()?;

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
                // TODO: Allow nested array types.
                if advance_if_matches!(self, TokenKind::RightBracket)?.is_some() {
                    Ok(ParsedType::Array {
                        inner: Box::new(ParsedType::Unknown),
                    })
                } else {
                    let inner = Box::new(self.parsed_type()?);
                    self.expect(TokenKind::RightBracket)?;
                    Ok(ParsedType::Array { inner })
                }
            }
            TokenKind::Identifier => {
                if next.lexeme == "void" {
                    Ok(ParsedType::Void)
                } else {
                    Ok(ParsedType::Simple {
                        name: next.lexeme.to_string(),
                    })
                }
            }
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

        let body = collect_list!(
            self,
            |this: &mut Self| this.statement(),
            TokenKind::RightBrace
        )?;

        self.define_procedure(name, body)?;

        Ok(())
    }

    fn statement(&mut self) -> ParseResult<Statement> {
        let next = self.advance()?.clone();

        Ok(match next.kind {
            TokenKind::For => {
                let iterator = self.expression()?;

                self.expect(TokenKind::LeftBrace)?;

                let body = collect_list!(
                    self,
                    |this: &mut Self| this.statement(),
                    TokenKind::RightBrace
                )?;

                Statement::For { iterator, body }
            }
            TokenKind::Bang => self.assertion_statement()?,
            // TODO: This should be extracted into a function somehow
            TokenKind::DotDot => self.procedure_call_statement()?,
            TokenKind::Dot => {
                // - 1 because the dot itself was already consumed
                let old_index = self.index - 1;

                // let the expr parse the dot
                self.go_back();

                let lhs = self.access_expression()?;

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
        let lhs = self.unary_expression()?;

        if let Some(op) = extract_if_kind_matches!(self, TokenKind::BinaryOp)? {
            let rhs = Box::new(self.expression()?);

            Ok(Expression::Binary { lhs, op, rhs })
        } else {
            Ok(Expression::Unary(lhs))
        }
    }

    fn access_expression(&mut self) -> ParseResult<AccessExpression> {
        let lhs = self.simple_expression()?;

        let mut indirections = Vec::new();

        while matches!(self.peek()?.kind, TokenKind::Dot | TokenKind::LeftBracket) {
            if advance_if_matches!(self, TokenKind::Dot)?.is_some() {
                indirections.push(Indirection::Field {
                    field_name: self.expect(TokenKind::Identifier)?.lexeme.to_string(),
                })
            } else if advance_if_matches!(self, TokenKind::LeftBracket)?.is_some() {
                let index_expr = self.expression()?;
                self.expect(TokenKind::RightBracket)?;
                indirections.push(Indirection::Subscript { index_expr });
            } else {
                panic!("Should be caught by matches! above");
            }
        }

        Ok(AccessExpression { lhs, indirections })
    }

    fn unary_expression(&mut self) -> ParseResult<UnaryExpression> {
        if let Some(op) = extract_if_kind_matches!(self, TokenKind::UnaryOp)? {
            let expr = Box::new(self.unary_expression()?);

            Ok(UnaryExpression::WithUnary { op, expr })
        } else {
            Ok(UnaryExpression::Access(self.access_expression()?))
        }
    }

    fn pattern(&mut self) -> ParseResult<Pattern> {
        if let Some(num) = advance_if_matches!(self, TokenKind::Number)? {
            Ok(Pattern::Value(str::parse(&num.lexeme).unwrap()))
        } else if advance_if_matches!(self, TokenKind::Else)?.is_some() {
            Ok(Pattern::Else)
        } else {
            unexpected("Pattern (value or 'else')", self.peek()?)?;
        }
    }

    fn simple_expression(&mut self) -> ParseResult<SimpleExpression> {
        let next = self.advance()?.clone();

        match next.kind {
            TokenKind::Match => {
                let expr = Box::new(self.expression()?);

                self.expect(TokenKind::LeftBrace)?;

                let arms = collect_list!(
                    self,
                    |this: &mut Self| {
                        let pattern = this.pattern()?;

                        this.expect(TokenKind::FatArrow)?;

                        let rhs = this.expression()?;

                        Ok(MatchArm { pattern, rhs })
                    },
                    TokenKind::Comma,
                    TokenKind::RightBrace
                )?;

                Ok(SimpleExpression::Match { expr, arms })
            }
            TokenKind::LeftParen => {
                let inner = Box::new(self.expression()?);
                self.expect(TokenKind::RightParen)?;

                Ok(SimpleExpression::Parentheses { inner })
            }
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
            TokenKind::String => Ok(SimpleExpression::String(next.lexeme)),
            TokenKind::True => Ok(SimpleExpression::Boolean(true)),
            TokenKind::False => Ok(SimpleExpression::Boolean(false)),
            // TODO: Allow struct literals here
            TokenKind::Dot => Ok(SimpleExpression::StateAccess {
                name: self.expect(TokenKind::Identifier)?.lexeme.to_string(),
            }),
            TokenKind::LeftBrace => {
                // TODO: Add more than just empty struct literals.
                self.expect(TokenKind::RightBrace)?;

                Ok(SimpleExpression::StructLiteral {
                    name: None,
                    fields: Vec::new(),
                })
            }
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
                } else if advance_if_matches!(self, TokenKind::Colon)?.is_some() {
                    self.expect(TokenKind::Colon)?;

                    let variant_name = self.expect(TokenKind::Identifier)?.lexeme.clone();

                    let initializer = advance_if_matches!(self, TokenKind::LeftParen)?
                        .cloned()
                        .map(|_| {
                            let initializer = Box::new(self.expression()?);

                            self.expect(TokenKind::RightParen)?;

                            Ok(initializer)
                        })
                        .transpose()?;

                    Ok(SimpleExpression::EnumVariant {
                        enum_name: next.lexeme,
                        variant_name,
                        initializer,
                    })
                } else {
                    Ok(SimpleExpression::BindingAccess { name: next.lexeme })
                }
            }
            TokenKind::ExecutionDesignator(execution) => {
                self.expect(TokenKind::Colon)?;

                let access_expression = Box::new(self.access_expression()?);

                Ok(SimpleExpression::ExecutionAccess {
                    execution,
                    access_expression,
                })
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
