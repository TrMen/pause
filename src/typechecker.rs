use std::{
    collections::{HashMap, HashSet},
    fmt::Display,
    hash::Hash,
};

use std::backtrace::Backtrace;

use thiserror::Error;

use crate::{
    lexer::{AssignOp, BinaryOp, ExecutionDesignator},
    parser::{
        AccessExpression, Assertion, Expression, Function, Indirection, ParsedType, Procedure,
        Program, SimpleExpression, Statement, Struct,
    },
};

macro_rules! err {
    ($error:expr) => {
        Err(TypeCheckErrorWrapper {
            error: $error,
            backtrace: Backtrace::force_capture(),
        })
    };
}

macro_rules! wrap {
    ($error:expr) => {
        TypeCheckErrorWrapper {
            error: $error,
            backtrace: Backtrace::force_capture(),
        }
    };
}

#[derive(Debug, Clone, Copy)]
pub struct TypeId {
    id: usize,
}

impl TypeId {
    fn is_compatible_with(&self, other: TypeId) -> bool {
        if self.id == TypeId::unchecked().id || other.id == TypeId::unchecked().id {
            true
        } else {
            self.id == other.id
        }
    }

    fn builtin(name: &str) -> Self {
        Self {
            id: BUILTIN_TYPES
                .iter()
                .position(|builtin| builtin == &name)
                .unwrap(),
        }
    }

    fn unknown() -> Self {
        Self { id: usize::MAX }
    }

    fn unchecked() -> Self {
        Self { id: usize::MAX - 1 }
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

#[derive(Clone, Debug)]
pub struct Binding {
    name: String,
    type_id: TypeId,
}

impl Typed for Binding {
    fn type_id(&self) -> TypeId {
        self.type_id
    }
}

#[derive(Clone, Debug)]
pub struct Scope {
    bindings: Vec<Binding>,
    outer: Option<Box<Scope>>,
}

// TODO: Add buildins to global scope
static GLOBAL_SCOPE: Scope = Scope {
    outer: None,
    bindings: Vec::new(),
};

// TODO: Make scope a lifetime thing to avoid having deepcopy outer
impl Scope {
    pub fn inner_with_bindings(&self, bindings: Vec<Binding>) -> Self {
        Self {
            outer: Some(Box::new(self.clone())),
            bindings,
        }
    }

    pub fn find_binding(&self, name: &str) -> TypeCheckResult<&Binding> {
        let outer_find = || self.outer.as_ref().map(|outer| outer.find_binding(name));

        self.bindings
            .iter()
            .find(|binding| binding.name == name)
            .map_or_else(outer_find, |b| Some(Ok(b)))
            .ok_or_else(|| {
                wrap!(UndefinedBinding {
                    name: name.to_string(),
                })
            })?
    }
}

#[derive(Debug, Clone)]
pub enum CheckedSimpleExpression {
    Boolean(bool),
    String(String),
    NumberLiteral(u64), // TODO: Add type when more than one number type exists.
    StructLiteral {
        type_name: Option<String>,
        fields: Vec<CheckedStructField>,
        type_id: TypeId,
    },
    ArrayLiteral {
        elements: Vec<CheckedExpression>,
        type_id: TypeId,
    },
    ExecutionAccess {
        execution: ExecutionDesignator,
        access_expression: Box<CheckedAccessExpression>,
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
        inner: Box<CheckedExpression>,
    },
}

impl Typed for CheckedSimpleExpression {
    fn type_id(&self) -> TypeId {
        match self {
            CheckedSimpleExpression::Boolean(_) => TypeId::builtin("bool"),
            CheckedSimpleExpression::String(_) => TypeId::builtin("string"),
            CheckedSimpleExpression::NumberLiteral(_) => TypeId::builtin("u64"),
            CheckedSimpleExpression::StructLiteral {
                type_id,
                type_name,
                fields: _,
            } => {
                if type_name.is_some() {
                    todo!("type of struct nonempty struct literals")
                } else {
                    *type_id
                }
            }
            CheckedSimpleExpression::ArrayLiteral { type_id, .. } => *type_id,
            CheckedSimpleExpression::ExecutionAccess {
                access_expression, ..
            } => access_expression.type_id(),
            CheckedSimpleExpression::BindingAccess { type_id, .. } => *type_id,
            CheckedSimpleExpression::StateAccess { type_id, .. } => *type_id,
            CheckedSimpleExpression::FunctionCall { type_id, .. } => *type_id,
            CheckedSimpleExpression::Parentheses { inner } => inner.type_id(),
        }
    }
}

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
pub enum CheckedIndirection {
    Subscript { index_expr: CheckedExpression },
    Field { field_name: String },
}

#[derive(Debug, Clone)]
pub struct CheckedAccessExpression {
    pub lhs: CheckedSimpleExpression,
    pub indirections: Vec<CheckedIndirection>,
    pub type_id: TypeId,
}

impl Typed for CheckedAccessExpression {
    fn type_id(&self) -> TypeId {
        self.type_id
    }
}

#[derive(Debug, Clone)]
pub struct DefinedTypes {
    types: HashMap<String, TypeId>,
    next_id: usize,
}

const BUILTIN_TYPES: [&str; 4] = ["void", "bool", "u64", "string"];

impl DefinedTypes {
    pub fn from_builtin() -> Self {
        let mut next_id = 0;

        let builtin_types = BUILTIN_TYPES
            .iter()
            .map(|p| {
                next_id += 1;
                (p.to_string(), TypeId { id: next_id - 1 })
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
        if id.is_compatible_with(TypeId::unknown()) {
            return "<unknown>";
        }

        // I only give out type ids from this struct, so a name must exist
        self.types
            .iter()
            .find(|p| p.1.is_compatible_with(id))
            .map(|p| p.0)
            .unwrap()
    }

    pub fn get_id(&self, name: &str) -> TypeCheckResult<TypeId> {
        self.types
            .get(name)
            .ok_or_else(|| {
                wrap!(UndefinedType {
                    name: name.to_string(),
                })
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
            ParsedType::Unknown => Ok(TypeId::unknown()),
        }
    }
}

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
pub struct CheckedStruct {
    pub name: String,
    pub fields: Vec<CheckedStructField>,
}

impl CheckedStruct {
    pub fn type_of_field(&self, field_name: &str) -> TypeCheckResult<TypeId> {
        self.fields
            .iter()
            .find(|field| field.name == field_name)
            .ok_or_else(|| {
                wrap!(UndefinedStructField {
                    struct_name: self.name.to_string(),
                    field_name: field_name.to_string(),
                })
            })
            .map(|field| field.type_id())
    }
}

// TODO: This looks a lot like StructField. But prolly makes sense to keep them separate
#[derive(Debug, Clone)]
pub struct CheckedFunctionParameter {
    pub name: String,
    pub type_id: TypeId,
}

impl Typed for CheckedFunctionParameter {
    fn type_id(&self) -> TypeId {
        self.type_id
    }
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

#[derive(Debug, Clone)]
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

        for structure in &program.structs {
            program.defined_types.define_type(structure.0.clone());
        }

        program
    }

    pub fn get_struct_for_type(&self, id: TypeId) -> TypeCheckResult<&CheckedStruct> {
        let type_name = self.defined_types.get_typename(id);

        if id.is_builtin() {
            return err!(BuiltinStructUse {
                type_name: type_name.to_string(),
            });
        }

        self.structs.get(type_name).ok_or_else(|| {
            wrap!(UndefinedType {
                name: type_name.to_string(),
            })
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
    #[error("Binding '{name}' is not defined")]
    UndefinedBinding { name: String },
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
    },
    #[error("Trying to use expression '{:#?}' of type '{}' to index into array. Array indexing must be done with '{}'", .common.expr, .common.actual, .common.expected)]
    ArraySubscript { common: TypeCheckErrorCommon },
    #[error("Trying to call function '{func_name}' with '{arg_count}' arguments. '{param_count}' arguments are required")]
    FunctionArgumentCount {
        func_name: String,
        arg_count: usize,
        param_count: usize,
    },
    #[error("Incorrect argument for call to function '{}'. Expression '{:#?}' evaluates to '{}'. Expected '{}'", .common.context["function"], .common.expr, .common.actual, .common.expected)]
    ParamArgMissmatch { common: TypeCheckErrorCommon },
}

use TypeCheckError::*;

#[derive(Debug)]
pub struct TypeCheckErrorCommon {
    expr: CheckedExpression,
    actual: String,
    expected: String,
    context: HashMap<String, String>,
}

#[derive(Debug, Error)]
#[error("{error}")]
pub struct TypeCheckErrorWrapper {
    pub error: TypeCheckError,
    pub backtrace: std::backtrace::Backtrace,
}

pub type TypeCheckResult<T> = Result<T, TypeCheckErrorWrapper>;

impl Display for TypeCheckErrorCommon {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Expression '{:#?}' evaluates to type '{}'. Expected '{}'.",
            self.expr, self.actual, self.expected
        )
    }
}

fn check_unique<'a, T>(name_iter: impl Iterator<Item = &'a T>) -> TypeCheckResult<()>
where
    T: Hash + Display + Eq + 'a,
{
    let mut names = HashSet::new();

    for name in name_iter {
        if !names.insert(name) {
            return err!(BindingAlreadyDefined {
                name: name.to_string(),
            });
        }
    }

    Ok(())
}

macro_rules! eval_lhs_and_check {
    ($program:expr, $scope:expr ,$lhs:expr, $rhs:expr, $error:ident) => {{
        let lhs_checked = typecheck_expression($program, $scope, $lhs)?;

        if !lhs_checked.type_id().is_compatible_with($rhs.type_id()) {
            err!(TypeCheckError::$error {
                common: TypeCheckErrorCommon {
                    actual: type_name($program, lhs_checked.type_id()),
                    expr: lhs_checked,
                    expected: type_name($program, $rhs.type_id()),
                    context: HashMap::new(),
                },
            })
        } else {
            Ok(lhs_checked)
        }
    }};

    ($program:expr, $scope:expr, $lhs:expr, $rhs:expr, $error:ident, $context:expr) => {{
        let context = $context
            .into_iter()
            .map(|p| (p.0.to_string(), p.1.to_string()))
            .collect();

        let lhs_checked = typecheck_expression($program, $scope, $lhs)?;

        if !lhs_checked.type_id().is_compatible_with($rhs.type_id()) {
            err!(TypeCheckError::$error {
                common: TypeCheckErrorCommon {
                    actual: type_name($program, lhs_checked.type_id()),
                    expr: lhs_checked,
                    expected: type_name($program, $rhs.type_id()),
                    context,
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
                    expression: CheckedExpression::Access(CheckedAccessExpression {
                        lhs: CheckedSimpleExpression::Boolean(false),
                        indirections: Vec::new(),
                        type_id: TypeId::unknown(),
                    }),
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
        .map(|f| {
            Ok((
                f.0.clone(),
                typecheck_function(&checked_program, &GLOBAL_SCOPE, f.1)?,
            ))
        })
        .collect::<Result<HashMap<_, _>, _>>()?;

    // Then typecheck structs since they can only use expressions (including other structs and
    // funcs) in their field initializers
    checked_program.structs = program
        .structs
        .into_iter()
        .map(|s| {
            Ok((
                s.0.clone(),
                typecheck_struct(&checked_program, &GLOBAL_SCOPE, s.1)?,
            ))
        })
        .collect::<Result<HashMap<_, _>, _>>()?;

    // Then check assertions since they can be used by procedures.
    checked_program.assertions = program
        .assertions
        .into_iter()
        .map(|a| {
            Ok((
                a.0.clone(),
                typecheck_assertion(&checked_program, &GLOBAL_SCOPE, a.1)?,
            ))
        })
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
        .map(|p| {
            Ok((
                p.0.clone(),
                typecheck_procedure(&checked_program, &GLOBAL_SCOPE, p.1)?,
            ))
        })
        .collect::<Result<HashMap<_, _>, _>>()?;

    // And finally main (is just a procedure, but may be different in the future)
    checked_program.main = typecheck_procedure(&checked_program, &GLOBAL_SCOPE, program.main)?;

    Ok(checked_program)
}

fn typecheck_struct(
    program: &CheckedProgram,
    scope: &Scope,
    structure: Struct,
) -> TypeCheckResult<CheckedStruct> {
    check_unique(structure.fields.iter().map(|f| &f.name))?;

    let fields = structure
        .fields
        .into_iter()
        .map(|field| {
            let initializer = eval_lhs_and_check!(
                program,
                scope,
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
    scope: &Scope,
    function: Function,
) -> TypeCheckResult<CheckedFunction> {
    check_unique(function.params.iter().map(|param| &param.name))?;

    let return_type = program.defined_types.check_parsed(&function.return_type)?;

    let params: Vec<_> = function
        .params
        .iter()
        .map(|param| {
            Ok(CheckedFunctionParameter {
                name: param.name.clone(),
                type_id: program.defined_types.check_parsed(&param.parsed_type)?,
            })
        })
        .collect::<Result<_, _>>()?;

    let bindings = params
        .iter()
        .map(|CheckedFunctionParameter { name, type_id }| Binding {
            name: name.clone(),
            type_id: *type_id,
        })
        .collect();

    let function_scope = scope.inner_with_bindings(bindings);

    let expression = eval_lhs_and_check!(
        program,
        &function_scope,
        function.expression,
        return_type,
        FunctionReturn
    )?;

    Ok(CheckedFunction {
        name: function.name,
        params,
        return_type,
        expression,
    })
}

fn typecheck_procedure(
    program: &CheckedProgram,
    scope: &Scope,
    procedure: Procedure,
) -> TypeCheckResult<CheckedProcedure> {
    let body = procedure
        .body
        .into_iter()
        .map(|p| typecheck_statement(program, scope, p))
        .collect::<Result<_, _>>()?;

    Ok(CheckedProcedure {
        name: procedure.name,
        body,
    })
}

fn typecheck_assertion(
    program: &CheckedProgram,
    scope: &Scope,
    assertion: Assertion,
) -> TypeCheckResult<CheckedAssertion> {
    let predicate = eval_lhs_and_check!(
        program,
        scope,
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
    scope: &Scope,
    statement: Statement,
) -> TypeCheckResult<CheckedStatement> {
    Ok(match statement {
        Statement::AssertionCall { name } => {
            // TODO: Do I need to check the called assertion here? I don't think so,
            // since I check every top-level thing anyway
            program.assertions.get(&name).ok_or_else(|| {
                wrap!(UndefinedAssertion {
                    name: name.to_string(),
                })
            })?;

            CheckedStatement::AssertionCall { name }
        }
        Statement::ProcedureCall { name } => {
            // TODO: Do I need to check the called procedure here? I don't think so,
            // since I check every top-level thing anyway
            program
                .procedures
                .get(&name)
                .ok_or_else(|| wrap!(UndefinedProcedure { name: name.clone() }))?;

            CheckedStatement::ProcedureCall { name }
        }
        Statement::StateAssignment {
            lhs,
            assign_op,
            rhs,
        } => {
            let lhs = typecheck_access_expression(program, scope, lhs)?;

            let rhs = eval_lhs_and_check!(program, scope, rhs, lhs.type_id(), AssignmentMissmatch)?;

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
            CheckedStatement::Expression(typecheck_expression(program, scope, expression)?)
        }
    })
}

pub fn typecheck_expression(
    program: &CheckedProgram,
    scope: &Scope,
    expression: Expression,
) -> TypeCheckResult<CheckedExpression> {
    Ok(match expression {
        Expression::Access(access) => {
            CheckedExpression::Access(typecheck_access_expression(program, scope, access)?)
        }
        Expression::Binary { lhs, op, rhs } => {
            let lhs = typecheck_access_expression(program, scope, lhs)?;

            let rhs = Box::new(eval_lhs_and_check!(
                program,
                scope,
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

    if !lhs_type.is_compatible_with(expected_type_id)
        || !rhs_type.is_compatible_with(expected_type_id)
    {
        err!(BinaryOpType {
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
    scope: &Scope,
    access: AccessExpression,
) -> TypeCheckResult<CheckedAccessExpression> {
    let lhs = typecheck_simple_expression(program, scope, access.lhs)?;

    let mut type_id = lhs.type_id();

    let indirections = access
        .indirections
        .into_iter()
        .map(|indirection| match indirection {
            Indirection::Subscript { index_expr } => {
                let index_expr = eval_lhs_and_check!(
                    program,
                    scope,
                    index_expr,
                    program.defined_types.get_id("u64").expect("u64 undefined"),
                    ArraySubscript
                )?;

                // TODO: If array subscript can change the type then we have to adjust type_id
                // here. But I think so far, it can't change the type

                Ok(CheckedIndirection::Subscript { index_expr })
            }
            Indirection::Field { field_name } => {
                let accessed_struct = program.get_struct_for_type(type_id)?;
                type_id = accessed_struct.type_of_field(&field_name)?;

                Ok(CheckedIndirection::Field { field_name })
            }
        })
        .collect::<Result<_, _>>()?;

    Ok(CheckedAccessExpression {
        lhs,
        indirections,
        type_id,
    })
}

pub fn typecheck_simple_expression(
    program: &CheckedProgram,
    scope: &Scope,
    simple: SimpleExpression,
) -> TypeCheckResult<CheckedSimpleExpression> {
    Ok(match simple {
        SimpleExpression::Boolean(value) => CheckedSimpleExpression::Boolean(value),
        SimpleExpression::NumberLiteral(value) => CheckedSimpleExpression::NumberLiteral(value),
        SimpleExpression::String(value) => CheckedSimpleExpression::String(value),
        SimpleExpression::ArrayLiteral(array) => {
            let arr_type = if let Some(first) = array.first() {
                typecheck_expression(program, scope, first.clone())?.type_id()
            } else {
                // TODO: Is just ignoring type-check on an empty array just always fine? I don't
                // think so.
                TypeId::unchecked()
            };

            let elements = array
                .into_iter()
                .map(|expr| eval_lhs_and_check!(program, scope, expr, arr_type, ArrayMissmatch))
                .collect::<Result<_, _>>()?;

            CheckedSimpleExpression::ArrayLiteral {
                elements,
                type_id: arr_type,
            }
        }
        SimpleExpression::FunctionCall { name, arguments } => {
            let function = program
                .functions
                .get(&name)
                .ok_or_else(|| wrap!(UndefinedFunction { name }))?;

            if arguments.len() != function.params.len() {
                return err!(FunctionArgumentCount {
                    func_name: function.name.to_string(),
                    arg_count: arguments.len(),
                    param_count: function.params.len(),
                });
            }

            let arguments = function
                .params
                .iter()
                .zip(arguments.into_iter())
                .map(|(param, arg)| {
                    eval_lhs_and_check!(
                        program,
                        scope,
                        arg,
                        param,
                        ParamArgMissmatch,
                        [("function", &function.name)]
                    )
                })
                .collect::<Result<_, _>>()?;

            CheckedSimpleExpression::FunctionCall {
                name: function.name.clone(),
                arguments,
                type_id: function.return_type,
            }
        }
        SimpleExpression::StructLiteral { name, fields } => {
            if name.is_none() && fields.is_empty() {
                CheckedSimpleExpression::StructLiteral {
                    type_name: None,
                    fields: Vec::new(),
                    type_id: TypeId::unchecked(),
                }
            } else {
                todo!("I'm not sure if I want to allow {{a: 2}} or {{a: int = 2}} literals without a type name, or just c-style literals.")
            }
        }
        SimpleExpression::Parentheses { inner } => CheckedSimpleExpression::Parentheses {
            inner: Box::new(typecheck_expression(program, scope, *inner)?),
        },
        SimpleExpression::StateAccess { name } => {
            let type_id = program
                .structs
                .get("state")
                .expect("Parser should ensure state struct is defined.")
                .type_of_field(&name)?;

            CheckedSimpleExpression::StateAccess { name, type_id }
        }
        SimpleExpression::ExecutionAccess {
            execution,
            access_expression,
        } => CheckedSimpleExpression::ExecutionAccess {
            execution,
            access_expression: Box::new(typecheck_access_expression(
                program,
                scope,
                *access_expression,
            )?),
        },
        SimpleExpression::BindingAccess { name } => {
            let Binding { type_id, .. } = scope.find_binding(&name)?;

            CheckedSimpleExpression::BindingAccess {
                name,
                type_id: *type_id,
            }
        }
    })
}
