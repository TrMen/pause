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

macro_rules! err {
    ($error:expr) => {
        Err(TypeCheckErrorWrapper {
            error: $error,
            backtrace: std::backtrace::Backtrace::force_capture(),
        })
    };
}

macro_rules! wrap {
    ($error:expr) => {
        TypeCheckErrorWrapper {
            error: $error,
            backtrace: std::backtrace::Backtrace::force_capture(),
        }
    };
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TypeId {
    id: usize,
}

impl TypeId {
    fn is_exactly(&self, other: TypeId) -> bool {
        self.id == other.id
    }

    fn unknown() -> Self {
        Self { id: usize::MAX }
    }

    fn is_builtin(&self) -> bool {
        self.id < std::mem::variant_count::<BuiltinType>()
    }

    fn builtin(name: &str) -> TypeId {
        let id = match name {
            "void" => BuiltinType::Void,
            "bool" => BuiltinType::Bool,
            "u64" => BuiltinType::U64,
            "string" => BuiltinType::String,
            _ => panic!("Trying to get non-builtin type '{name}'"),
        };

        TypeId { id: id as usize }
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

#[derive(Debug, Clone, PartialEq)]
#[repr(u8)]
pub enum BuiltinType {
    Void,
    Bool,
    U64,
    String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeDescription {
    Builtin(BuiltinType),
    Unknown,
    Struct(TypeId),
    GenericInstance {
        generic_id: GenericId,
        type_arguments: Vec<TypeId>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Type {
    description: TypeDescription,
    type_id: TypeId,
}

impl Typed for Type {
    fn type_id(&self) -> TypeId {
        self.type_id
    }
}

#[derive(Debug, Clone)]
pub enum TypeParameterCount {
    Fixed(usize),
    Variadic,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenericId {
    id: usize,
}

#[derive(Debug, Clone)]
pub struct GenericType {
    name: String,
    generic_id: GenericId,
    parameter_count: TypeParameterCount,
}

#[derive(Debug, Clone)]
pub struct DefinedTypes {
    // Includes builtins, all generic type instances used in the program,
    // and structs. Structs are also kept in the struct field for lookup by name.
    concrete_types: Vec<Type>,

    structs: HashMap<String, TypeId>,
    // Includes all generic types like array that can't be directly instanciated.
    generic_types: Vec<GenericType>,
    next_id: usize,
    next_generic_id: usize,
}

impl DefinedTypes {
    pub fn from_builtin() -> Self {
        use BuiltinType::*;

        let builtin_types = vec![Void, Bool, U64, String];

        let mut this = Self {
            structs: HashMap::new(),
            generic_types: Vec::new(),
            next_id: builtin_types.len(),
            concrete_types: builtin_types
                .into_iter()
                .enumerate()
                .map(|(id, builtin)| Type {
                    type_id: TypeId { id },
                    description: TypeDescription::Builtin(builtin),
                })
                .collect(),
            next_generic_id: 0,
        };

        this.define_generic("array".to_string(), TypeParameterCount::Fixed(1));

        this.define_struct("<empty_struct>".to_string()).unwrap();

        this
    }

    fn define_generic(
        &mut self,
        name: String,
        parameter_count: TypeParameterCount,
    ) -> &GenericType {
        self.generic_types.push(GenericType {
            name,
            generic_id: GenericId {
                id: self.next_generic_id,
            },
            parameter_count,
        });

        self.next_generic_id += 1;

        self.generic_types.last().unwrap()
    }

    fn get_generic(&self, name: &str) -> TypeCheckResult<GenericId> {
        self.generic_types
            .iter()
            .find(|g| g.name == name)
            .map(|g| g.generic_id)
            .ok_or_else(|| {
                wrap!(UndefinedGenericType {
                    name: name.to_string()
                })
            })
    }

    fn are_assignment_compatible(&self, lhs: TypeId, rhs: TypeId) -> bool {
        if let TypeDescription::Struct(_) = self.get_type_description(lhs)
            && rhs == self.get_struct("<empty_struct>").unwrap() {
            true
        }
        else if let TypeDescription::GenericInstance {generic_id: generic_id_lhs, ..} = self.get_type_description(lhs)
            && let TypeDescription::GenericInstance { generic_id: generic_id_rhs, type_arguments: type_arguments_rhs } = self.get_type_description(rhs)
            && generic_id_lhs == self.get_generic("array").unwrap()
            && generic_id_rhs == self.get_generic("array").unwrap()
            && *type_arguments_rhs.first().unwrap() == TypeId::unknown(){
                // rhs is an empty array literal, and lhs is any array type
                true
            }
        else {lhs.is_exactly(rhs)}
    }

    pub fn define_struct(&mut self, name: String) -> TypeCheckResult<TypeId> {
        if self.structs.contains_key(&name) {
            return err!(StructRedefinition { name });
        }

        let type_id = TypeId { id: self.next_id };
        self.next_id += 1;

        self.concrete_types.push(Type {
            type_id,
            description: TypeDescription::Struct(type_id),
        });

        self.structs.insert(name, type_id);

        Ok(type_id)
    }

    fn get_type_description(&self, id: TypeId) -> TypeDescription {
        if id == TypeId::unknown() {
            TypeDescription::Unknown
        } else {
            // Because we insert them in order and give out the type id we just inserted
            self.concrete_types[id.id].description.clone()
        }
    }

    pub fn type_name(&self, id: TypeId) -> String {
        let description = self.get_type_description(id);

        match description {
            TypeDescription::Builtin(builtin) => match builtin {
                BuiltinType::Void => "void".to_string(),
                BuiltinType::Bool => "bool".to_string(),
                BuiltinType::U64 => "u64".to_string(),
                BuiltinType::String => "string".to_string(),
            },
            TypeDescription::Unknown => "<unknown>".to_string(),
            TypeDescription::Struct(type_id) => self
                .structs
                .iter()
                .find(|p| p.1.is_exactly(type_id))
                .map(|p| p.0.to_string())
                .unwrap_or_else(|| {
                    panic!("Asked for struct by undefined type id '{:#?}'", type_id)
                }),
            TypeDescription::GenericInstance {
                generic_id,
                type_arguments,
            } => {
                let generic_type = self
                    .generic_types
                    .iter()
                    .find(|g| g.generic_id == generic_id)
                    .unwrap();

                if generic_type.name == "array" {
                    assert_eq!(type_arguments.len(), 1);

                    format!("[{}]", self.type_name(type_arguments[0]))
                } else {
                    todo!("Non-array generics");
                }
            }
        }
    }

    fn get_struct(&self, name: &str) -> TypeCheckResult<TypeId> {
        self.structs
            .get(name)
            .ok_or_else(|| {
                wrap!(UndefinedType {
                    name: name.to_string()
                })
            })
            .copied()
    }

    fn typecheck_parsed_type(&mut self, parsed_type: &ParsedType) -> TypeCheckResult<TypeId> {
        Ok(match &parsed_type {
            // Unwrap: ParsedTypes occur in the source code, so they must be build-in or
            // user-d&efined
            ParsedType::Simple { name } => match name.as_str() {
                name @ ("u64" | "string" | "bool") => TypeId::builtin(name),
                _ => self.get_struct(name)?,
            },
            ParsedType::Array { inner } => {
                let inner_id = self.typecheck_parsed_type(inner)?;

                let array_type_id = self
                    .generic_types
                    .iter()
                    .find(|generic| generic.name == "array")
                    .expect("array type undefined")
                    .generic_id;

                self.get_or_define_type(TypeDescription::GenericInstance {
                    generic_id: array_type_id,
                    type_arguments: vec![inner_id],
                })
            }

            ParsedType::Void => TypeId::builtin("void"),
            ParsedType::Unknown => TypeId::unknown(),
        })
    }

    fn get_or_define_type(&mut self, description: TypeDescription) -> TypeId {
        if let Some(existing) = self
            .concrete_types
            .iter()
            .enumerate()
            .find(|(_, typ)| typ.description == description)
        {
            TypeId { id: existing.0 }
        } else {
            self.next_id += 1;
            let type_id = TypeId {
                id: self.next_id - 1,
            };

            self.concrete_types.push(Type {
                type_id,
                description,
            });

            type_id
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
}

impl CheckedProgram {
    pub fn new(
        structs: HashMap<String, CheckedStruct>,
        functions: HashMap<String, CheckedFunction>,
        assertions: HashMap<String, CheckedAssertion>,
        procedures: HashMap<String, CheckedProcedure>,
        main: CheckedProcedure,
    ) -> Self {
        Self {
            procedures,
            assertions,
            functions,
            structs,
            main,
        }
    }
}

pub fn get_struct_for_type<'a>(
    program: &'a CheckedProgram,
    types: &DefinedTypes,
    id: TypeId,
) -> TypeCheckResult<&'a CheckedStruct> {
    let type_name = types.type_name(id);

    Ok(program
        .structs
        .get(&type_name)
        .expect("Struct is not defined in program despite being found in defined types"))
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
    #[error("Undefined generic type '{name}'")]
    UndefinedGenericType { name: String },
    #[error("Trying to use types '{lhs_type}' and '{rhs_type}' in binary operation '{op:#?}'. Types must be '{expected_type}'.")]
    BinaryOpType {
        lhs_type: String,
        rhs_type: String,
        op: BinaryOp,
        expected_type: String,
    },
    #[error("Type '{struct_name}' has no field '{field_name}'")]
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
    #[error("Incorrect argument type for call to function '{}'. Expression '{:#?}' evaluates to '{}'. Expected '{}'", .common.context["function"], .common.expr, .common.actual, .common.expected)]
    ParamArgMissmatch { common: TypeCheckErrorCommon },
    #[error("Struct '{name}' is already defined.")]
    StructRedefinition { name: String },
    #[error(
        "Trying index into expression '{expr:#?}' of type '{expr_type}' as if it were an array."
    )]
    NonArraySubscript {
        expr: AccessExpression,
        expr_type: String,
    },
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

macro_rules! eval_rhs_and_check {
    ($program:expr, $types:expr, $scope:expr ,$assignee:expr, $rhs:expr, $error:ident) => {{
        let rhs_checked = typecheck_expression($program, $types, $scope, $rhs)?;

        if !$types.are_assignment_compatible($assignee.type_id(), rhs_checked.type_id()) {
            err!(TypeCheckError::$error {
                common: TypeCheckErrorCommon {
                    actual: $types.type_name(rhs_checked.type_id()),
                    expr: rhs_checked,
                    expected: $types.type_name($assignee.type_id()),
                    context: HashMap::new(),
                },
            })
        } else {
            Ok(rhs_checked)
        }
    }};

    ($program:expr, $types:expr, $scope:expr, $assignee:expr, $rhs:expr, $error:ident, $context:expr) => {{
        let context = $context
            .into_iter()
            .map(|p| (p.0.to_string(), p.1.to_string()))
            .collect();

        let rhs_checked = typecheck_expression($program, $types, $scope, $rhs)?;

        if !$types.are_assignment_compatible($assignee.type_id(), rhs_checked.type_id()) {
            err!(TypeCheckError::$error {
                common: TypeCheckErrorCommon {
                    actual: $types.type_name(rhs_checked.type_id()),
                    expr: rhs_checked,
                    expected: $types.type_name($assignee.type_id()),
                    context,
                },
            })
        } else {
            Ok(rhs_checked)
        }
    }};
}

pub fn typecheck_program(program: Program) -> TypeCheckResult<CheckedProgram> {
    // Predecl all structs so they can be used in function definitions
    let predecl_structs: HashMap<_, _> = program
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

    let mut types = DefinedTypes::from_builtin();

    for name in predecl_structs.keys() {
        types.define_struct(name.clone())?;
    }

    let mut checked_program = CheckedProgram::new(
        predecl_structs,
        predecl_functions,
        HashMap::new(),
        HashMap::new(),
        CheckedProcedure {
            name: "main".to_string(),
            body: Vec::new(),
        },
    );

    // Then typecheck functions, since they can only use structs and buildin types, or other
    // functions
    checked_program.functions = program
        .functions
        .into_iter()
        .map(|f| {
            Ok((
                f.0.clone(),
                typecheck_function(&checked_program, &mut types, &GLOBAL_SCOPE, f.1)?,
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
                typecheck_struct(&checked_program, &mut types, &GLOBAL_SCOPE, s.1)?,
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
                typecheck_assertion(&checked_program, &mut types, &GLOBAL_SCOPE, a.1)?,
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
                typecheck_procedure(&checked_program, &mut types, &GLOBAL_SCOPE, p.1)?,
            ))
        })
        .collect::<Result<HashMap<_, _>, _>>()?;

    // And finally main (is just a procedure, but may be different in the future)
    checked_program.main =
        typecheck_procedure(&checked_program, &mut types, &GLOBAL_SCOPE, program.main)?;

    Ok(checked_program)
}

fn typecheck_struct(
    program: &CheckedProgram,
    types: &mut DefinedTypes,
    scope: &Scope,
    structure: Struct,
) -> TypeCheckResult<CheckedStruct> {
    check_unique(structure.fields.iter().map(|f| &f.name))?;

    let fields = structure
        .fields
        .into_iter()
        .map(|field| {
            let field_type = types.typecheck_parsed_type(&field.parsed_type)?;
            let initializer = typecheck_expression(program, types, scope, field.initializer)?;

            if !types.are_assignment_compatible(field_type.type_id(), initializer.type_id()) {
                return err!(FieldInitializerMissmatch {
                    common: TypeCheckErrorCommon {
                        actual: types.type_name(initializer.type_id()),
                        expr: initializer,
                        expected: types.type_name(field_type.type_id()),
                        context: HashMap::new(),
                    },
                });
            }

            Ok(CheckedStructField {
                name: field.name,
                type_id: field_type,
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
    types: &mut DefinedTypes,
    scope: &Scope,
    function: Function,
) -> TypeCheckResult<CheckedFunction> {
    check_unique(function.params.iter().map(|param| &param.name))?;

    let return_type = types.typecheck_parsed_type(&function.return_type)?;

    let params: Vec<_> = function
        .params
        .iter()
        .map(|param| {
            Ok(CheckedFunctionParameter {
                name: param.name.clone(),
                type_id: types.typecheck_parsed_type(&param.parsed_type)?,
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

    let expression = eval_rhs_and_check!(
        program,
        types,
        &function_scope,
        return_type,
        function.expression,
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
    types: &mut DefinedTypes,
    scope: &Scope,
    procedure: Procedure,
) -> TypeCheckResult<CheckedProcedure> {
    let body = procedure
        .body
        .into_iter()
        .map(|p| typecheck_statement(program, types, scope, p))
        .collect::<Result<_, _>>()?;

    Ok(CheckedProcedure {
        name: procedure.name,
        body,
    })
}

fn typecheck_assertion(
    program: &CheckedProgram,
    types: &mut DefinedTypes,
    scope: &Scope,
    assertion: Assertion,
) -> TypeCheckResult<CheckedAssertion> {
    let predicate = eval_rhs_and_check!(
        program,
        types,
        scope,
        TypeId::builtin("bool"),
        assertion.predicate,
        AssertionExpressionType
    )?;

    Ok(CheckedAssertion {
        name: assertion.name,
        predicate,
    })
}

pub fn typecheck_statement(
    program: &CheckedProgram,
    types: &mut DefinedTypes,
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
            let lhs = typecheck_access_expression(program, types, scope, lhs)?;

            let rhs = eval_rhs_and_check!(program, types, scope, lhs, rhs, AssignmentMissmatch)?;

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
            CheckedStatement::Expression(typecheck_expression(program, types, scope, expression)?)
        }
    })
}

pub fn typecheck_expression(
    program: &CheckedProgram,
    types: &mut DefinedTypes,
    scope: &Scope,
    expression: Expression,
) -> TypeCheckResult<CheckedExpression> {
    Ok(match expression {
        Expression::Access(access) => {
            CheckedExpression::Access(typecheck_access_expression(program, types, scope, access)?)
        }
        Expression::Binary { lhs, op, rhs } => {
            let lhs = typecheck_access_expression(program, types, scope, lhs)?;

            let rhs = Box::new(eval_rhs_and_check!(
                program,
                types,
                scope,
                lhs,
                *rhs,
                BinaryExpressionMissmatch
            )?);

            let type_id = match op {
                // TODO: Allow more than just 'u64' to pass here
                BinaryOp::Plus | BinaryOp::Minus => {
                    check_binary_op(
                        types,
                        lhs.type_id(),
                        rhs.type_id(),
                        op,
                        TypeId::builtin("u64"),
                    )?;
                    TypeId::builtin("u64")
                }
                BinaryOp::EqualEqual => TypeId::builtin("bool"),
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
    types: &mut DefinedTypes,
    lhs_type: TypeId,
    rhs_type: TypeId,
    op: BinaryOp,
    expected_type_id: TypeId,
) -> TypeCheckResult<()> {
    if !types.are_assignment_compatible(lhs_type, expected_type_id)
        || !types.are_assignment_compatible(rhs_type, expected_type_id)
    {
        err!(BinaryOpType {
            lhs_type: types.type_name(lhs_type),
            rhs_type: types.type_name(rhs_type),
            op,
            expected_type: types.type_name(expected_type_id),
        })
    } else {
        Ok(())
    }
}

pub fn typecheck_access_expression(
    program: &CheckedProgram,
    types: &mut DefinedTypes,
    scope: &Scope,
    access: AccessExpression,
) -> TypeCheckResult<CheckedAccessExpression> {
    let access_clone = access.clone(); // TODO: Remove
    let lhs = typecheck_simple_expression(program, types, scope, access.lhs)?;

    let mut type_id = lhs.type_id();

    let indirections = access
        .indirections
        .into_iter()
        .map(|indirection| match indirection {
            Indirection::Subscript { index_expr } => {
                let type_description = types.get_type_description(type_id);

                // Check that the current type_id is an array
                if let TypeDescription::GenericInstance {
                    generic_id,
                    type_arguments,
                } = &type_description
                {
                    let array_type = types
                        .get_generic("array")
                        .expect("array generic type undefined");

                    if *generic_id != array_type {
                        return err!(NonArraySubscript {
                            expr_type: types.type_name(type_id),
                            expr: access_clone.clone(), // TODO: This is a little imprecise
                        });
                    }

                    assert_eq!(type_arguments.len(), 1);

                    // Expr type becomes the inner array type (first and only type argument)
                    type_id = *type_arguments.first().unwrap();
                } else {
                    return err!(NonArraySubscript {
                        expr_type: types.type_name(type_id),
                        expr: access_clone.clone(), // TODO: This is a little imprecise
                    });
                }

                // types.get_or_define_type(type_description);

                let index_expr = eval_rhs_and_check!(
                    program,
                    types,
                    scope,
                    TypeId::builtin("u64"),
                    index_expr,
                    ArraySubscript
                )?;

                // TODO: If array subscript can change the type then we have to adjust type_id
                // here. But I think so far, it can't change the type

                Ok(CheckedIndirection::Subscript { index_expr })
            }
            Indirection::Field { field_name } => {
                if type_id.is_builtin() {
                    return err!(UndefinedStructField {
                        struct_name: types.type_name(type_id),
                        field_name,
                    });
                }

                let accessed_struct = get_struct_for_type(program, types, type_id)?;
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
    types: &mut DefinedTypes,
    scope: &Scope,
    simple: SimpleExpression,
) -> TypeCheckResult<CheckedSimpleExpression> {
    Ok(match simple {
        SimpleExpression::Boolean(value) => CheckedSimpleExpression::Boolean(value),
        SimpleExpression::NumberLiteral(value) => CheckedSimpleExpression::NumberLiteral(value),
        SimpleExpression::String(value) => CheckedSimpleExpression::String(value),
        SimpleExpression::ArrayLiteral(array) => {
            let element_type = if let Some(first) = array.first() {
                typecheck_expression(program, types, scope, first.clone())?.type_id()
            } else {
                TypeId::unknown()
            };

            let elements = array
                .into_iter()
                .map(|expr| {
                    eval_rhs_and_check!(program, types, scope, element_type, expr, ArrayMissmatch)
                })
                .collect::<Result<_, _>>()?;

            let arr_type = types.get_or_define_type(TypeDescription::GenericInstance {
                generic_id: types.get_generic("array").expect("array undefined"),
                type_arguments: vec![element_type],
            });

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
                    eval_rhs_and_check!(
                        program,
                        types,
                        scope,
                        param,
                        arg,
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
                    type_id: types
                        .get_struct("<empty_struct>")
                        .expect("Empty struct undefined"),
                }
            } else {
                todo!("I'm not sure if I want to allow {{a: 2}} or {{a: int = 2}} literals without a type name, or just c-style literals.")
            }
        }
        SimpleExpression::Parentheses { inner } => CheckedSimpleExpression::Parentheses {
            inner: Box::new(typecheck_expression(program, types, scope, *inner)?),
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
                types,
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
