use std::{collections::HashMap, fmt::Display};

use thiserror::Error;

use crate::{
    lexer::{AssignOp, BinaryOp, ExecutionDesignator},
    parser::{
        AccessPath, Assertion, Expression, Function, ParsedType, Procedure, Program,
        SimpleExpression, Struct,
    },
};

#[derive(Debug, Clone, Copy)]
pub struct TypeId {
    id: usize,
}

impl PartialEq for TypeId {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl TypeId {
    fn from_parsed(
        defined_types: &Vec<(TypeId, String)>,
        parsed_type: &ParsedType,
    ) -> TypeCheckResult<Self> {
        match &parsed_type {
            // Unwrap: ParsedTypes occur in the source code, so they must be build-in or
            // user-d&efined
            ParsedType::Simple { name } => Self::from_string(
                &defined_types.iter().find(|p| &p.1 == name).unwrap().1,
                defined_types,
            ),
            ParsedType::Array { inner } => todo!(),
            ParsedType::Void => Self::from_string("void", defined_types),
            ParsedType::Unknown => todo!(),
        }
    }

    fn type_id(&self) -> Self {
        *self
    }

    fn new(id: usize) -> Self {
        Self { id }
    }

    fn from_string(name: &str, defined_types: &Vec<(TypeId, String)>) -> TypeCheckResult<Self> {
        todo!()
    }

    fn to_string(self, defined_types: &Vec<(TypeId, String)>) -> String {
        todo!()
    }

    fn unknown() -> Self {
        todo!()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum CheckedSimpleExpression {
    Boolean(bool),
    String(String),
    NumberLiteral(u64), // TODO: Add type when more than one number type exists.
    StructLiteral {
        fields: Vec<CheckedStructField>,
    },
    ArrayLiteral {
        elements: Vec<CheckedExpression>,
        type_id: TypeId,
    },
    ArrayAccess {
        target: Box<CheckedExpression>,
        index: Box<CheckedExpression>,
        type_id: TypeId,
    },
    StateAccess {
        path: AccessPath,
        type_id: TypeId,
    },
    ExecutionAccess {
        execution: ExecutionDesignator,
        path: AccessPath,
        type_id: TypeId,
    },
    BindingAccess {
        path: AccessPath,
        type_id: TypeId,
    },
    FunctionCall {
        name: String,
        arguments: Vec<CheckedExpression>,
        type_id: TypeId,
    },
}

impl CheckedSimpleExpression {
    pub fn type_id(&self) -> TypeId {
        match self {
            CheckedSimpleExpression::Boolean(_) => todo!(),
            CheckedSimpleExpression::String(_) => todo!(),
            CheckedSimpleExpression::NumberLiteral(_) => todo!(),
            CheckedSimpleExpression::StructLiteral { fields } => todo!(),
            CheckedSimpleExpression::ArrayLiteral { elements, type_id } => todo!(),
            CheckedSimpleExpression::ArrayAccess {
                target,
                index,
                type_id,
            } => todo!(),
            CheckedSimpleExpression::StateAccess { path, type_id } => todo!(),
            CheckedSimpleExpression::ExecutionAccess {
                execution,
                path,
                type_id,
            } => todo!(),
            CheckedSimpleExpression::BindingAccess { path, type_id } => todo!(),
            CheckedSimpleExpression::FunctionCall {
                name,
                arguments,
                type_id,
            } => todo!(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum CheckedExpression {
    Simple(CheckedSimpleExpression),
    Binary {
        lhs: CheckedSimpleExpression,
        op: BinaryOp,
        rhs: Box<CheckedSimpleExpression>,
    },
}

impl CheckedExpression {
    pub fn type_id(&self) -> TypeId {
        match self {
            CheckedExpression::Simple(simple) => simple.type_id(),
            CheckedExpression::Binary { lhs, .. } => lhs.type_id(), // TODO: Might fail if op produces a diff type
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CheckedStructField {
    pub name: String,
    pub type_id: TypeId,
    // TODO: Make sure this expression only uses literals
    pub initializer: CheckedExpression,
    // TODO: Add modifiers like is_mutable
}

#[derive(Debug, Clone, PartialEq)]
pub struct CheckedStruct {
    pub name: String,
    pub fields: Vec<CheckedStructField>,
}
// TODO: This looks a lot like StructField. But prolly makes sense to keep them separate
#[derive(Debug, Clone, PartialEq)]
pub struct CheckedFunctionParameter {
    pub name: String,
    pub type_id: TypeId,
}

#[derive(Debug, Clone)]
pub struct CheckedFunction {
    pub name: String,
    pub params: Vec<CheckedFunctionParameter>,
    pub return_type: TypeId,
    pub expression: CheckedExpression,
}

#[derive(Debug, Clone)]
pub struct CheckedAssertion {
    pub name: String,
    pub predicate: Expression,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CheckedStatement {
    AssertionCall {
        name: String,
    },
    ProcedureCall {
        name: String,
    },
    StateAssignment {
        lhs: AccessPath,
        assign_op: AssignOp,
        rhs: CheckedExpression,
    },
    Expression(CheckedExpression),
}

#[derive(Debug, Clone)]
pub struct CheckedProcedure {
    pub name: String,
    pub body: Vec<CheckedStatement>,
}

#[derive(Debug, Clone)]
pub struct CheckedProgram {
    // TODO: This duplicates the name. But I think I really want to be able to pass a Procedure
    // around without having to also pass it's name.
    pub procedures: HashMap<String, CheckedProcedure>,
    pub assertions: HashMap<String, CheckedAssertion>,
    pub functions: HashMap<String, CheckedFunction>,
    pub structs: HashMap<String, CheckedStruct>,
    pub main: CheckedProcedure,
    pub defined_types: Vec<(TypeId, String)>,
}

#[derive(Error, Debug)]
pub enum TypeCheckError {
    #[error("Expression in array has wrong type. '{common}'")]
    ArrayMissmatch { common: TypeCheckErrorCommon },
    #[error("Function expression does not match return type. '{common}'")]
    FunctionReturn { common: TypeCheckErrorCommon },
}

#[derive(Debug)]
pub struct TypeCheckErrorCommon {
    expr: CheckedExpression,
    actual: String,
    expected: String,
}

impl Display for TypeCheckErrorCommon {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Expression '{:#?}' evaluates to type '{}'. Expected '{}'.",
            self.expr, self.actual, self.expected
        )
    }
}

macro_rules! check_types {
    ($defined_types:expr, $lhs:expr, $rhs:expr, $error:ident) => {
        if $lhs.type_id() != $rhs.type_id() {
            return Err(TypeCheckError::$error {
                common: TypeCheckErrorCommon {
                    expr: $lhs.clone(),
                    actual: $lhs.type_id().to_string($defined_types),
                    expected: $rhs.type_id().to_string($defined_types),
                },
            });
        }
    };
}

pub type TypeCheckResult<T> = Result<T, TypeCheckError>;

pub fn typecheck_program(program: Program) -> TypeCheckResult<CheckedProgram> {
    // Predecl all structs so they can be used in function definitions
    let predecl_structs = program
        .structs
        .iter()
        .map(|s| {
            (
                s.0.clone(),
                CheckedStruct {
                    name: s.0.clone(),
                    fields: Vec::new(),
                },
            )
        })
        .collect();

    let predecl_functions = program
        .functions
        .iter()
        .map(|f| {
            (
                f.0.clone(),
                CheckedFunction {
                    name: f.0.clone(),
                    params: Vec::new(),
                    return_type: TypeId::unknown(),
                    // TODO: Maybe avoid having to put random stuff here.
                    expression: CheckedExpression::Simple(CheckedSimpleExpression::Boolean(false)),
                },
            )
        })
        .collect();

    let builtin_types = vec![
        ("void", 0),
        ("bool", 1),
        ("u64", 2),
        ("string", 3),
        ("<unknown>", 4),
    ]
    .into_iter()
    .map(|p| (TypeId::new(p.1), p.0.to_string()))
    .collect();

    let mut checked_program = CheckedProgram {
        structs: predecl_structs,
        functions: predecl_functions,
        assertions: HashMap::new(),
        procedures: HashMap::new(),
        main: CheckedProcedure {
            name: "main".to_string(),
            body: Vec::new(),
        },
        defined_types: builtin_types,
    };

    // Then typecheck functions, since they can only use structs and buildin types, or other
    // functions
    checked_program.functions = program
        .functions
        .into_iter()
        .map(|f| Ok((f.0.clone(), typecheck_function(&checked_program, f.1)?)))
        .collect::<Result<HashMap<_, _>, _>>()?;

    // Then typecheck structs since they can only use expressions (including other structs and
    // funcs) in their field initializers
    checked_program.structs = program
        .structs
        .into_iter()
        .map(|s| Ok((s.0.clone(), typecheck_struct(&checked_program, s.1)?)))
        .collect::<Result<HashMap<_, _>, _>>()?;

    // Then check assertions since they can be used by procedures.
    checked_program.assertions = program
        .assertions
        .into_iter()
        .map(|a| Ok((a.0.clone(), typecheck_assertion(&checked_program, a.1)?)))
        .collect::<Result<HashMap<_, _>, _>>()?;

    // Then predecl procedures since they can refer to other procedures
    checked_program.procedures = program
        .procedures
        .iter()
        .map(|p| {
            (
                p.0.clone(),
                CheckedProcedure {
                    name: p.0.clone(),
                    body: Vec::new(),
                },
            )
        })
        .collect();

    // Then typecheck procedures
    checked_program.procedures = program
        .procedures
        .into_iter()
        .map(|p| Ok((p.0.clone(), typecheck_procedure(&checked_program, p.1)?)))
        .collect::<Result<HashMap<_, _>, _>>()?;

    // And finally main (is just a procedure, but may be different in the future)
    checked_program.main = typecheck_procedure(&checked_program, program.main)?;

    Ok(checked_program)
}

fn typecheck_struct(program: &CheckedProgram, structure: Struct) -> TypeCheckResult<CheckedStruct> {
    todo!()
}

fn typecheck_function(
    program: &CheckedProgram,
    function: Function,
) -> TypeCheckResult<CheckedFunction> {
    let expression = typecheck_expression(program, function.expression)?;
    let return_type = TypeId::from_parsed(&program.defined_types, &function.return_type)?;
    let params = function
        .params
        .iter()
        .map(|param| {
            Ok(CheckedFunctionParameter {
                name: param.name,
                type_id: TypeId::from_parsed(&program.defined_types, &param.parsed_type)?,
            })
        })
        .collect::<Result<_, _>>()?;

    check_types!(
        &program.defined_types,
        expression,
        return_type,
        FunctionReturn
    );

    Ok(CheckedFunction {
        name: function.name,
        params,
        return_type,
        expression,
    })
}

fn typecheck_procedure(
    program: &CheckedProgram,
    procedure: Procedure,
) -> TypeCheckResult<CheckedProcedure> {
    todo!()
}

fn typecheck_assertion(
    program: &CheckedProgram,
    procedure: Assertion,
) -> TypeCheckResult<CheckedAssertion> {
    todo!()
}

pub fn typecheck_expression(
    program: &CheckedProgram,
    expression: Expression,
) -> TypeCheckResult<CheckedExpression> {
    let typ = match expression {
        Expression::Simple(simple) => typecheck_simple_expression(program, simple),
        Expression::Binary { lhs, op, rhs } => todo!(),
    };
    todo!()
}

pub fn typecheck_simple_expression(
    program: &CheckedProgram,
    simple: SimpleExpression,
) -> TypeCheckResult<CheckedSimpleExpression> {
    Ok(match &simple {
        SimpleExpression::Boolean(value) => CheckedSimpleExpression::Boolean(*value),
        SimpleExpression::NumberLiteral(value) => CheckedSimpleExpression::NumberLiteral(*value),
        SimpleExpression::String(value) => CheckedSimpleExpression::String(value.clone()),
        SimpleExpression::ArrayLiteral(array) => {
            let arr_type = TypeId::unknown();

            let mut elements = Vec::new();
            for expr in array {
                let checked = typecheck_expression(program, expr.clone())?;

                check_types!(&program.defined_types, checked, arr_type, ArrayMissmatch);

                elements.push(checked);
            }

            CheckedSimpleExpression::ArrayLiteral {
                elements,
                type_id: arr_type,
            }
        }
        SimpleExpression::ArrayAccess { target, index } => todo!(),
        SimpleExpression::StateAccess(_) => todo!(),
        SimpleExpression::ExecutionAccess { execution, path } => todo!(),
        SimpleExpression::BindingAccess(_) => todo!(),
        SimpleExpression::FunctionCall { name, arguments } => todo!(),
        SimpleExpression::StructLiteral(_) => todo!(),
    })
}
