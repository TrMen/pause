use std::fmt::Display;

use thiserror::Error;

use crate::parser::{Expression, SimpleExpression};

#[derive(Debug, Clone, PartialEq)]
pub enum CheckedValue {
    String(String),
    Number(u64),
    Bool(bool),
    // TODO: The fields of a struct literal aren't really typed, right?
    Struct(Struct),
    Array(Vec<ValueLiteral>),
}

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
    Value(ValueLiteral),
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
pub struct StructField {
    pub name: String,
    pub type_name: String,
    pub initial_value: ValueLiteral, //  TODO: Make clearer that this is just initial, nothing you can actually change at runtime
    pub value: ValueLiteral, //  TODO: Make clearer that this is just initial, nothing you can actually change at runtime
}

#[derive(Debug, Clone, PartialEq)]
pub struct Struct {
    pub name: String,
    pub fields: Vec<StructField>,
}

impl Struct {
    pub fn get_field(&self, name: &str) -> InterpretationResult<&StructField> {
        self.fields
            .iter()
            .find(|field| field.name == name)
            .ok_or_else(|| InterpretationError::UndefinedStructField {
                structure: self.name.clone(),
                field_name: name.to_string(),
            })
    }

    pub fn get_field_mut(&mut self, name: &str) -> InterpretationResult<&mut StructField> {
        self.fields
            .iter_mut()
            .find(|field| field.name == name)
            .ok_or_else(|| InterpretationError::UndefinedStructField {
                structure: self.name.clone(),
                field_name: name.to_string(),
            })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
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

#[derive(Debug, Clone, PartialEq)]
pub enum BuildinType {
    U64,
    String,
    Bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Array {
        size: usize,
        element_type: Box<Type>,
    },
    Struct(String),
    Buildin(BuildinType),
    Unknown, // TODO: Not sure if useful. But I think yes, because [] is Array<Unknown>
}

impl Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedExpression {
    expression: Expression,
    typ: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedSimpleExpression {
    simple: SimpleExpression,
    typ: Type,
}

#[derive(Error, Debug)]
pub enum TypeCheckError {
    #[error("Expression '{expr:#?}' in array has type '{expr_type}'. Type of array was previously inferred to be '{arr_type}'")]
    ArrayMissmatch {
        expr: Expression,
        expr_type: Type,
        arr_type: Type,
    },
}

use TypeCheckError::*;

pub type TypeCheckResult<T> = Result<T, TypeCheckError>;

pub fn typecheck_expression(expression: Expression) -> TypeCheckResult<TypedExpression> {
    let typ = match expression {
        Expression::Simple(simple) => typecheck_simple_expression(simple),
        Expression::Binary { lhs, op, rhs } => todo!(),
    };
    todo!()
}

pub fn typecheck_simple_expression(
    simple: SimpleExpression,
) -> TypeCheckResult<TypedSimpleExpression> {
    Ok(match &simple {
        SimpleExpression::Value(value) => TypedSimpleExpression {
            typ: value.typ(),
            simple,
        },
        SimpleExpression::ArrayLiteral(array) => {
            let arr_type = Type::Unknown;

            let mut checked_arr = Vec::new();
            for expr in array {
                let checked = typecheck_expression(expr.clone())?;
                if checked.typ != arr_type {
                    return Err(ArrayMissmatch {
                        expr: expr.clone(),
                        expr_type: checked.typ,
                        arr_type,
                    });
                }
                checked_arr.push(checked);
            }

            TypedSimpleExpression {
                simple,
                typ: arr_type,
            }
        }
        SimpleExpression::ArrayAccess { target, index } => todo!(),
        SimpleExpression::StateAccess(_) => todo!(),
        SimpleExpression::ExecutionAccess { execution, path } => todo!(),
        SimpleExpression::BindingAccess(_) => todo!(),
        SimpleExpression::FunctionCall { name, arguments } => todo!(),
    })
}
