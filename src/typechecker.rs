use std::{
    collections::{HashMap, HashSet},
    fmt::Display,
    hash::Hash,
};

use thiserror::Error;

use crate::{
    lexer::{AssignOp, BinaryOp, ExecutionDesignator},
    parser::{
        AccessExpression, Assertion, Expression, Function, Indirection, ParsedType, Procedure,
        Program, SimpleExpression, Statement, Struct,
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
    fn new(id: usize) -> Self {
        Self { id }
    }

    fn unknown() -> Self {
        Self { id: usize::MAX }
    }

    fn is_builtin(&self) -> bool {
        self.id < BUILTIN_TYPES.len()
    }
}

trait Typed {
    fn type_id(&self) -> TypeId;
}

impl Typed for TypeId {
    fn type_id(&self) -> TypeId {
        *self
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
    ExecutionAccess {
        execution: ExecutionDesignator,
        access_expression: Box<CheckedAccessExpression>,
        type_id: TypeId,
    },
    BindingAccess {
        name: String,
        type_id: TypeId,
    },
    StateAccess {
        name: String,
        type_id: TypeId,
    },
    FunctionCall {
        name: String,
        arguments: Vec<CheckedExpression>,
        type_id: TypeId,
    },
    Parentheses {
        inner: Box<Expression>,
    },
}

impl Typed for CheckedSimpleExpression {
    fn type_id(&self) -> TypeId {
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
            CheckedSimpleExpression::ExecutionAccess {
                execution,
                access_expression,
                type_id,
            } => todo!(),
            CheckedSimpleExpression::BindingAccess { name, type_id } => todo!(),
            CheckedSimpleExpression::StateAccess { name, type_id } => todo!(),
            CheckedSimpleExpression::FunctionCall {
                name,
                arguments,
                type_id,
            } => todo!(),
            CheckedSimpleExpression::Parentheses { inner } => todo!(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum CheckedExpression {
    Access(CheckedAccessExpression),
    Binary {
        lhs: CheckedAccessExpression,
        op: BinaryOp,
        rhs: Box<CheckedExpression>,
        type_id: TypeId,
    },
}

impl Typed for CheckedExpression {
    fn type_id(&self) -> TypeId {
        match self {
            CheckedExpression::Access(expr) => expr.type_id(),
            CheckedExpression::Binary { type_id, .. } => *type_id,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum CheckedAccessExpression {
    Subscript {
        lhs: CheckedSimpleExpression,
        index_expr: Box<CheckedExpression>,
        rest: Option<Box<CheckedAccessExpression>>,
        type_id: TypeId,
    },
    FieldAccess {
        lhs: CheckedSimpleExpression,
        rest: Box<CheckedAccessExpression>,
        type_id: TypeId,
    },
    Simple(CheckedSimpleExpression),
}

impl Typed for CheckedAccessExpression {
    fn type_id(&self) -> TypeId {
        match self {
            CheckedAccessExpression::Subscript { type_id, .. } => *type_id,
            CheckedAccessExpression::FieldAccess { type_id, .. } => *type_id,
            CheckedAccessExpression::Simple(expr) => expr.type_id(),
        }
    }
}

#[derive(Debug, Clone)]
struct DefinedTypes {
    types: HashMap<String, TypeId>,
    next_id: usize,
}

const BUILTIN_TYPES: [&str; 5] = ["void", "bool", "u64", "string", "<unknown>"];

impl DefinedTypes {
    pub fn from_builtin() -> Self {
        let mut next_id = 0;

        let builtin_types = BUILTIN_TYPES
            .iter()
            .map(|p| {
                next_id += 1;
                (p.to_string(), TypeId::new(next_id - 1))
            })
            .collect();

        Self {
            types: builtin_types,
            next_id,
        }
    }

    pub fn define_type(&mut self, name: String) -> TypeId {
        let id = TypeId { id: self.next_id };
        self.next_id += 1;
        self.types.insert(name, id);
        id
    }

    pub fn get_typename(&self, id: TypeId) -> &str {
        // I only give out type ids from this struct, so a name must exist
        self.types
            .iter()
            .find(|p| p.1 == &id)
            .map(|p| &p.0)
            .unwrap()
    }

    pub fn get_id(&self, name: &str) -> TypeCheckResult<TypeId> {
        self.types
            .get(name)
            .ok_or_else(|| UndefinedType {
                name: name.to_string(),
            })
            .copied()
    }

    fn check_parsed(&self, parsed_type: &ParsedType) -> TypeCheckResult<TypeId> {
        match &parsed_type {
            // Unwrap: ParsedTypes occur in the source code, so they must be build-in or
            // user-d&efined
            ParsedType::Simple { name } => self.get_id(name),
            ParsedType::Array { inner } => self.check_parsed(inner),
            ParsedType::Void => Ok(self.types["void"]),
            ParsedType::Unknown => todo!(),
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

impl Typed for CheckedStructField {
    fn type_id(&self) -> TypeId {
        self.type_id
    }
}


#[derive(Debug, Clone, PartialEq)]
pub struct CheckedStruct {
    pub name: String,
    pub fields: Vec<CheckedStructField>,
}

impl CheckedStruct {
    pub fn type_of_field(&self, field_name: &str) -> TypeCheckResult<TypeId> {
        self.fields.iter().find(|field| field.name == field_name).ok_or_else(|| UndefinedStructField { struct_name: self.name.to_string(), field_name: field_name.to_string()}).map(|field| field.type_id())
    }
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
    pub predicate: CheckedExpression,
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
        lhs: CheckedAccessExpression,
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
    pub structs: HashMap<String, CheckedStruct>,
    pub functions: HashMap<String, CheckedFunction>,
    pub assertions: HashMap<String, CheckedAssertion>,
    pub procedures: HashMap<String, CheckedProcedure>,
    pub main: CheckedProcedure,
    pub defined_types: DefinedTypes,
}

impl CheckedProgram {
    pub fn new(
        structs: HashMap<String, CheckedStruct>,
        functions: HashMap<String, CheckedFunction>,
        assertions: HashMap<String, CheckedAssertion>,
        procedures: HashMap<String, CheckedProcedure>,
        main: CheckedProcedure,
        defined_types: DefinedTypes,
    ) -> Self {
        let mut program = Self {
            procedures,
            assertions,
            functions,
            structs,
            main,
            defined_types,
        };

        for structure in program.structs {
            program.defined_types.define_type(structure.0);
        }

        program
    }

    pub fn get_struct_for_type(&self, id: TypeId) -> TypeCheckResult<&CheckedStruct> {
        let type_name = self.defined_types.get_typename(id);

        if id.is_builtin() {
            return Err(BuiltinStructUse {
                type_name: type_name.to_string(),
            });
        }

        self.structs.get(type_name).ok_or_else(|| UndefinedType {
            name: type_name.to_string(),
        })
    }
}

#[derive(Error, Debug)]
pub enum TypeCheckError {
    #[error("Expression in array has wrong type. '{common}'")]
    ArrayMissmatch { common: TypeCheckErrorCommon },
    #[error("Function expression does not match return type. '{common}'")]
    FunctionReturn { common: TypeCheckErrorCommon },
    #[error("Name '{name}' is already defined in this scope. Must be unique.")]
    BindingAlreadyDefined { name: String },
    #[error("Declared type of structure field does not match type of initializer expression. '{common}'")]
    FieldInitializerMissmatch { common: TypeCheckErrorCommon },
    #[error("Trying to assign value of incorrect type to state. '{common}'")]
    AssignmentMissmatch { common: TypeCheckErrorCommon },
    #[error("Incompatible types in binary expression. '{common}'")]
    BinaryExpressionMissmatch { common: TypeCheckErrorCommon },
    #[error("Assertions must contain exactly one expression that evalutes to a boolean value. '{common}'")]
    AssertionExpressionType { common: TypeCheckErrorCommon },
    #[error("Call to undefined assertion '{name}'")]
    UndefinedAssertion { name: String },
    #[error("Call to undefined function '{name}'")]
    UndefinedFunction { name: String },
    #[error("Call to undefined function '{name}'")]
    UndefinedProcedure { name: String },
    #[error("Undefined type '{name}'")]
    UndefinedType { name: String },
    #[error("Trying to use types '{lhs_type}' and '{rhs_type}' in binary operation '{op:#?}'. Types must be '{expected_type}'.")]
    BinaryOpType {
        lhs_type: String,
        rhs_type: String,
        op: BinaryOp,
        expected_type: String,
    },
    #[error("Trying to use builtin type '{type_name}' as if it were a struct.")]
    BuiltinStructUse { type_name: String },
    #[error("Struct '{struct_name}' has no field '{field_name}'")]
    UndefinedStructField {
        struct_name: String,
        field_name: String,
    }
}

use TypeCheckError::*;

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

fn check_unique<T>(name_iter: impl Iterator<Item = T>) -> TypeCheckResult<()>
where
    T: Hash + Display + Eq,
{
    let mut names = HashSet::new();

    name_iter
        .find(|name| !names.insert(name))
        .map(|name| {
            Err(BindingAlreadyDefined {
                name: name.to_string(),
            })
        })
        .unwrap_or(Ok(()))
}

macro_rules! check_types {
    ($program:expr, $lhs:expr, $rhs:expr, $error:ident) => {{
        let lhs_checked = typecheck_expression($program, $lhs)?;

        if lhs_checked.type_id() != $rhs.type_id() {
            Err(TypeCheckError::$error {
                common: TypeCheckErrorCommon {
                    expr: lhs_checked,
                    actual: type_name($program, lhs_checked.type_id()),
                    expected: type_name($program, $rhs.type_id()),
                },
            })
        } else {
            Ok(lhs_checked)
        }
    }};
}

pub fn type_name(program: &CheckedProgram, type_id: TypeId) -> String {
    program.defined_types.get_typename(type_id).to_string()
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
                    expression: CheckedExpression::Access(CheckedAccessExpression::Simple(
                        CheckedSimpleExpression::Boolean(false),
                    )),
                },
            )
        })
        .collect();

    let mut checked_program = CheckedProgram::new(
        predecl_structs,
        predecl_functions,
        HashMap::new(),
        HashMap::new(),
        CheckedProcedure {
            name: "main".to_string(),
            body: Vec::new(),
        },
        DefinedTypes::from_builtin(),
    );

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
    check_unique(structure.fields.iter().map(|f| f.name))?;

    let fields = structure
        .fields
        .into_iter()
        .map(|field| {
            let initializer = check_types!(
                program,
                field.initializer,
                program.defined_types.check_parsed(&field.parsed_type)?,
                FieldInitializerMissmatch
            )?;

            Ok(CheckedStructField {
                name: field.name,
                type_id: initializer.type_id(),
                initializer,
            })
        })
        .collect::<Result<_, _>>()?;

    Ok(CheckedStruct {
        name: structure.name,
        fields,
    })
}

fn typecheck_function(
    program: &CheckedProgram,
    function: Function,
) -> TypeCheckResult<CheckedFunction> {
    check_unique(function.params.iter().map(|param| param.name))?;

    let return_type = program.defined_types.check_parsed(&function.return_type)?;

    let params = function
        .params
        .iter()
        .map(|param| {
            Ok(CheckedFunctionParameter {
                name: param.name,
                type_id: program.defined_types.check_parsed(&param.parsed_type)?,
            })
        })
        .collect::<Result<_, _>>()?;

    let expression = check_types!(program, function.expression, return_type, FunctionReturn)?;

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
    let body = procedure
        .body
        .into_iter()
        .map(|p| typecheck_statement(program, p))
        .collect::<Result<_, _>>()?;

    Ok(CheckedProcedure {
        name: procedure.name,
        body,
    })
}

fn typecheck_assertion(
    program: &CheckedProgram,
    assertion: Assertion,
) -> TypeCheckResult<CheckedAssertion> {
    let predicate = check_types!(
        program,
        assertion.predicate,
        program
            .defined_types
            .get_id("bool")
            .expect("Builtin type bool not defined"),
        AssertionExpressionType
    )?;

    Ok(CheckedAssertion {
        name: assertion.name,
        predicate,
    })
}

pub fn typecheck_statement(
    program: &CheckedProgram,
    statement: Statement,
) -> TypeCheckResult<CheckedStatement> {
    Ok(match statement {
        Statement::AssertionCall { name } => {
            // TODO: Do I need to check the called assertion here? I don't think so,
            // since I check every top-level thing anyway
            program
                .assertions
                .get(&name)
                .ok_or_else(|| UndefinedAssertion { name })?;

            CheckedStatement::AssertionCall { name }
        }
        Statement::ProcedureCall { name } => {
            // TODO: Do I need to check the called procedure here? I don't think so,
            // since I check every top-level thing anyway
            program
                .procedures
                .get(&name)
                .ok_or_else(|| UndefinedProcedure { name })?;

            CheckedStatement::ProcedureCall { name }
        }
        Statement::StateAssignment {
            lhs,
            assign_op,
            rhs,
        } => {
            let lhs = typecheck_access_expression(program, lhs)?;

            let rhs = check_types!(program, rhs, lhs.type_id(), AssignmentMissmatch)?;

            // Just here so the compile fails if I add more assing ops that might change the type.
            match assign_op {
                AssignOp::Equal => (),
                AssignOp::PlusEqual => (),
            };

            CheckedStatement::StateAssignment {
                lhs,
                assign_op,
                rhs,
            }
        }
        Statement::Expression(expression) => {
            CheckedStatement::Expression(typecheck_expression(program, expression)?)
        }
    })
}

pub fn typecheck_expression(
    program: &CheckedProgram,
    expression: Expression,
) -> TypeCheckResult<CheckedExpression> {
    Ok(match expression {
        Expression::Access(access) => {
            CheckedExpression::Access(typecheck_access_expression(program, access)?)
        }
        Expression::Binary { lhs, op, rhs } => {
            let lhs = typecheck_access_expression(program, lhs)?;

            let rhs = Box::new(check_types!(
                program,
                *rhs,
                lhs.type_id(),
                BinaryExpressionMissmatch
            )?);

            let type_id = match op {
                // TODO: Allow more than just 'u64' to pass here
                BinaryOp::Plus | BinaryOp::Minus => {
                    check_binary_op(program, lhs.type_id(), rhs.type_id(), op, "u64")?;
                    program.defined_types.get_id("u64").expect("u64 undefined")
                }
                BinaryOp::EqualEqual => {
                    program // TODO: Restrict what types can be equality-compared
                        .defined_types
                        .get_id("bool")
                        .expect("bool undefined")
                }
            };

            CheckedExpression::Binary {
                lhs,
                op,
                rhs,
                type_id,
            }
        }
    })
}

fn check_binary_op(
    program: &CheckedProgram,
    lhs_type: TypeId,
    rhs_type: TypeId,
    op: BinaryOp,
    expected_type: &str,
) -> TypeCheckResult<()> {
    let expected_type_id = program
        .defined_types
        .get_id(expected_type)
        .expect("Explicitly requested type name is undefined");

    if lhs_type != expected_type_id || rhs_type != expected_type_id {
        Err(BinaryOpType {
            lhs_type: type_name(program, lhs_type),
            rhs_type: type_name(program, lhs_type),
            op,
            expected_type: expected_type.to_string(),
        })
    } else {
        Ok(())
    }
}

pub fn typecheck_access_expression(
    program: &CheckedProgram,
    access: AccessExpression,
) -> TypeCheckResult<CheckedAccessExpression> {
    match access {
        AccessExpression::Subscript {
            lhs,
            index_expr,
            rest,
        } => todo!(),
        AccessExpression::FieldAccess { lhs, rest } => {
            let lhs = typecheck_simple_expression(program, lhs)?;

            let rest = Box::new(typecheck_access_expression(program, *rest)?);

            let lhs_struct = program.get_struct_for_type(lhs.type_id())?;

            lhs_struct.type_of_field()?

            Ok(CheckedAccessExpression::FieldAccess { lhs, rest, type_id })
        }
        AccessExpression::Simple(simple) => Ok(CheckedAccessExpression::Simple(
            typecheck_simple_expression(program, simple)?,
        )),
    }
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
            let arr_type = if let Some(first) = array.first() {
                typecheck_expression(program, first.clone())?.type_id()
            } else {
                TypeId::unknown()
            };

            let elements = array
                .into_iter()
                .map(|expr| check_types!(program, *expr, arr_type, ArrayMissmatch))
                .collect::<Result<_, _>>()?;

            CheckedSimpleExpression::ArrayLiteral {
                elements,
                type_id: arr_type,
            }
        }
        SimpleExpression::FunctionCall { name, arguments } => todo!(),
        SimpleExpression::StructLiteral(_) => todo!(),
        SimpleExpression::Parentheses { inner } => todo!(),
        SimpleExpression::StateAccess { name } => todo!(),
        SimpleExpression::ExecutionAccess {
            execution,
            access_expression,
        } => todo!(),
        SimpleExpression::BindingAccess { name } => todo!(),
    })
}
