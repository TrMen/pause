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
        Program, SimpleExpression, Statement, Struct, Enum,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    EnumVariant {
        type_id: TypeId,
        variant_name: String,
        initializer: Option<Box<CheckedExpression>>,
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
            CheckedSimpleExpression::EnumVariant { type_id, .. } => *type_id,
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

#[derive(Debug, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum BuiltinType {
    Void,
    Bool,
    U64,
    String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeDescription {
    Builtin(BuiltinType),
    Unknown,
    Struct(TypeId),
    Enum(TypeId),
    GenericInstance {
        generic_id: GenericId,
        type_arguments: Vec<TypeId>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    // and structs.
    concrete_types: Vec<Type>,

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

    fn is_instance_of(&self, instance: TypeId, generic: GenericId) -> bool {
        if let TypeDescription::GenericInstance {generic_id, ..} = self.get_type_description(instance) 
            && generic_id == generic {
            true
        }
        else {
            false
        }
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

    fn get_type_description(&self, id: TypeId) -> TypeDescription {
        if id == TypeId::unknown() {
            TypeDescription::Unknown
        } else {
            // Because we insert them in order and give out the type id we just inserted
            self.concrete_types[id.id].description.clone()
        }
    }

    fn get_or_define_type(&mut self, description: TypeDescription) -> TypeId {
        if let Some(existing) = self
            .concrete_types
            .iter()
            .enumerate()
            .find(|(_, typ)| {
                match description {
                    TypeDescription::Unknown => panic!("Trying to define unknown type"),
                    TypeDescription::Struct(type_id) => if type_id == TypeId::unknown() { false } else { description == typ.description },
                    TypeDescription::Enum(type_id) => if type_id == TypeId::unknown() { false } else { description == typ.description },
                    _ => description == typ.description,
                }
            })
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

fn print_defined_types(program: &CheckedProgram, types: &DefinedTypes) {
    println!("Concrete types:");
    for concrete_type in &types.concrete_types {
        println!("{}: '{}'", concrete_type.type_id().id, type_name(program, types, concrete_type.type_id()))
    }
    println!("\nGeneric Types:");
    for generic_type in &types.generic_types {
        println!("{}: '{}'", generic_type.generic_id.id, generic_type.name);
    }
}

fn are_assignment_compatible(program: &CheckedProgram, types: &DefinedTypes, lhs: TypeId, rhs: TypeId) -> bool {
    if let TypeDescription::Struct(_)= types.get_type_description(lhs)
        && rhs == program.structs.get("<empty_struct>").unwrap().type_id {
        true
    }
    else if types.is_instance_of(lhs, types.get_generic("array").unwrap()) && 
        let TypeDescription::GenericInstance { generic_id: generic_id_rhs, type_arguments: type_arguments_rhs } = types.get_type_description(rhs)
        && generic_id_rhs == types.get_generic("array").unwrap()
        && *type_arguments_rhs.first().unwrap() == TypeId::unknown(){
            // rhs is an empty array literal, and lhs is any array type
            true
        }
    else {lhs.is_exactly(rhs)}
}

pub fn type_name(program: &CheckedProgram, types: &DefinedTypes, id: TypeId) -> String {
    match types.get_type_description(id) {
        TypeDescription::Builtin(builtin) => match builtin {
            BuiltinType::Void => "void".to_string(),
            BuiltinType::Bool => "bool".to_string(),
            BuiltinType::U64 => "u64".to_string(),
            BuiltinType::String => "string".to_string(),
        },
        TypeDescription::Unknown => "<unknown>".to_string(),
        TypeDescription::Struct(_)=> {
                program
                .structs
                .iter()
                .find(|p| p.1.type_id().is_exactly(id))
                .map(|p| p.0.to_string())
                .unwrap_or_else(|| {
                    panic!("Asked for struct by undefined type id '{:#?}'", id)
                })
            }
        TypeDescription::GenericInstance {
            generic_id,
            type_arguments,
        } => {
            let generic_type = types
                .generic_types
                .iter()
                .find(|g| g.generic_id == generic_id)
                .unwrap();

            if generic_type.name == "array" {
                assert_eq!(type_arguments.len(), 1);

                format!("[{}]", type_name(program, types, type_arguments[0]))
            } else {
                todo!("Non-array generics");
            }
        }
        TypeDescription::Enum(_)=>{
            program
            .enums
            .iter()
            .find(|e| e.1.type_id().is_exactly(id))
            .map(|e| e.0.to_string())
            .unwrap_or_else(|| {
                panic!("Asked for struct by undefined type id '{:#?}'", id)
            })
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
    pub type_id: TypeId,
}

impl Typed for CheckedStruct {
    fn type_id(&self) -> TypeId {
        self.type_id
    }
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
    For {
        iterator: CheckedExpression,
        body: Vec<CheckedStatement>,
    }
}

#[derive(Debug, Clone)]
pub struct CheckedProcedure {
    pub name: String,
    pub body: Vec<CheckedStatement>,
}


#[derive(Debug, Clone)]
pub struct CheckedEnumVariant {
    pub name: String,
    pub variant_type: TypeId,
}

impl Typed for CheckedEnumVariant {
    fn type_id(&self) -> TypeId {
        self.variant_type
    }
}

#[derive(Debug, Clone)]
pub struct CheckedEnum {
    pub name: String,
    pub variants: Vec<CheckedEnumVariant>,
    pub type_id: TypeId,
}

impl Typed for CheckedEnum {
    fn type_id(&self) -> TypeId {
        self.type_id
    }
}


#[derive(Debug, Clone)]
pub struct CheckedProgram {
    // TODO: This duplicates the name. But I think I really want to be able to pass a Procedure
    // around without having to also pass it's name.
    pub structs: HashMap<String, CheckedStruct>,
    pub functions: HashMap<String, CheckedFunction>,
    pub assertions: HashMap<String, CheckedAssertion>,
    pub procedures: HashMap<String, CheckedProcedure>,
    pub enums: HashMap<String, CheckedEnum>,
    pub main: CheckedProcedure,
}

impl CheckedProgram {
    pub fn new(
        structs: HashMap<String, CheckedStruct>,
        functions: HashMap<String, CheckedFunction>,
        assertions: HashMap<String, CheckedAssertion>,
        procedures: HashMap<String, CheckedProcedure>,
        enums: HashMap<String, CheckedEnum>,
        main: CheckedProcedure,
    ) -> Self {
        Self {
            procedures,
            assertions,
            functions,
            structs,
            enums,
            main,
        }
    }
}

pub fn get_struct_for_type<'a>(
    program: &'a CheckedProgram,
    types: &DefinedTypes,
    id: TypeId,
) -> TypeCheckResult<&'a CheckedStruct> {
    let type_name = type_name(program, types, id);

    Ok(program
        .structs
        .get(&type_name)
        .unwrap_or_else(|| panic!("Struct '{}' is not defined in program despite being found in defined types", type_name)))
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
    #[error("Undefined enum '{name}'")]
    UndefinedEnum { name: String },
    #[error("Variant '{variant_name}' of enum '{enum_name}' is undefined")]
    UndefinedEnumVariant{ variant_name: String , enum_name: String },
    #[error("Variant '{variant_name}' of enum '{enum_name}' has non-void type '{variant_type_name}' but no initializer")]
    MissingEnumVariantInitializer {
        enum_name: String,
        variant_type_name: String,
        variant_name: String,
    },
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
    #[error("Trying to initialize variant '{}' of enum '{}' with expression '{:#?} that evaluates to type '{}'. Variant has type '{}'", .common.context["variant_name"], .common.context["enum_name"], .common.expr, .common.actual, .common.expected)]
    EnumVariantInitMissmatch { common: TypeCheckErrorCommon },
    #[error("Struct '{name}' is already defined.")]
    StructRedefinition { name: String },
    #[error(
        "Trying index into expression '{expr:#?}' of type '{expr_type}' as if it were an array."
    )]
    NonArraySubscript {
        expr: AccessExpression,
        expr_type: String,
    },
    #[error(
        "Trying to iterate over expression '{:#?}' of type '{type_name}'. Only arrays can be iterated over"
    )]
    NonArrayIteration {
        type_name: String,
        expr: CheckedExpression,
    }
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

        if !are_assignment_compatible($program, $types, $assignee.type_id(), rhs_checked.type_id()) {
            err!(TypeCheckError::$error {
                common: TypeCheckErrorCommon {
                    actual: type_name($program, $types, rhs_checked.type_id()),
                    expr: rhs_checked,
                    expected: type_name($program, $types, $assignee.type_id()),
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

        if !are_assignment_compatible($program, $types,$assignee.type_id(), rhs_checked.type_id()) {
            err!(TypeCheckError::$error {
                common: TypeCheckErrorCommon {
                    actual: type_name($program, $types, rhs_checked.type_id()),
                    expr: rhs_checked,
                    expected: type_name($program, $types, $assignee.type_id()),
                    context,
                },
            })
        } else {
            Ok(rhs_checked)
        }
    }};
}

pub fn typecheck_program(mut program: Program) -> TypeCheckResult<CheckedProgram> {
    let mut types = DefinedTypes::from_builtin();

    // TODO: remove this hack
    program.structs.insert("<empty_struct>".to_string(), Struct {name: "<empty_struct>".to_string(), fields: Vec::new()});

    // Predecl all structs so they can be used in function type signatures
    let predecl_structs: HashMap<_, _> = program
        .structs
        .iter()
        .map(|s| {
            let type_id = types.get_or_define_type(TypeDescription::Struct(TypeId::unknown()));
            (
                s.0.clone(),
                CheckedStruct {
                    name: s.0.clone(),
                    fields: Vec::new(),
                    type_id,
                },
            )
        })
        .collect();


    // Predecl all enums so they can be used in function type signatures
    let predecl_enums: HashMap<_, _> = program
        .enums
        .iter()
        .map(|s| {
                let type_id = types.get_or_define_type(TypeDescription::Enum(TypeId::unknown()));
            (
                s.0.clone(),
                CheckedEnum {
                    name: s.0.clone(),
                    variants: Vec::new(),
                    type_id,
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
        predecl_enums,
        CheckedProcedure {
            name: "main".to_string(),
            body: Vec::new(),
        },
    );

    print_defined_types(&checked_program, &types);

    checked_program.enums = program.enums.into_iter().map(|e| {
        Ok((e.0.clone(), typecheck_enum(&checked_program, &mut types, &GLOBAL_SCOPE, e.1)?))
    }).collect::<Result<_, _>>()?;


    // TODO: I need another step here because at this point struct types are not evaluated, so
    // using structs inside a function expression won't work. In realtity, I need to typecheck
    // function signatures, struct signatures, and enum signatures first, then function expressions
    // and struct field initializers.

    // Then typecheck functions, since they can only use structs and buildin types, or other
    // functions
    checked_program.functions = program
        .functions
        .into_iter()
        .map(|f| {
            Ok((
                f.0.clone(),
                typecheck_function(&checked_program, &mut types, &GLOBAL_SCOPE, f.1)?
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

fn typecheck_enum(
    program: &CheckedProgram,
    types: &mut DefinedTypes,
    _scope: &Scope,
    enumeration: Enum,
) -> TypeCheckResult<CheckedEnum> {
    check_unique(enumeration.variants.iter().map(|e| &e.name))?;

    let variants = enumeration
        .variants
        .into_iter()
        .map(|variant| {
            let variant_type = typecheck_parsed_type(program, types, &variant.parsed_type)?;

            Ok(CheckedEnumVariant {
                name: variant.name,
                variant_type,
            })
        })
        .collect::<Result<_, _>>()?;

    Ok(CheckedEnum {
        type_id: program.enums.get(&enumeration.name).expect("Enum not predeclared").type_id,
        name: enumeration.name,
        variants,
    })
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
            let field_type = typecheck_parsed_type(program, types, &field.parsed_type)?;
            let initializer = typecheck_expression(program, types, scope, field.initializer)?;

            if !are_assignment_compatible(program, types, field_type.type_id(), initializer.type_id()) {
                return err!(FieldInitializerMissmatch {
                    common: TypeCheckErrorCommon {
                        actual: type_name(program, types, initializer.type_id()),
                        expr: initializer,
                        expected: type_name(program, types,field_type.type_id()),
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
        type_id: program.structs.get(&structure.name).expect("Struct not predeclared").type_id,
        name: structure.name,
        fields,
    })
}

    fn typecheck_parsed_type(program: &CheckedProgram, types: &mut DefinedTypes, parsed_type: &ParsedType) -> TypeCheckResult<TypeId> {
        Ok(match &parsed_type {
            // Unwrap: ParsedTypes occur in the source code, so they must be build-in or
            // user-d&efined
            ParsedType::Simple { name } => match name.as_str() {
                name @ ("u64" | "string" | "bool") => TypeId::builtin(name),
                _ => program.structs.get(name).map(|s| s.type_id())
                    .or_else(|| program.enums.get(name).map(|e| e.type_id()))
                    .ok_or_else(|| wrap!(UndefinedType{ name: name.clone() }))?,
            },
            ParsedType::Array { inner } => {
                let inner_id = typecheck_parsed_type(program, types, inner)?;

                let array_type_id = types
                    .generic_types
                    .iter()
                    .find(|generic| generic.name == "array")
                    .expect("array type undefined")
                    .generic_id;

                types.get_or_define_type(TypeDescription::GenericInstance {
                    generic_id: array_type_id,
                    type_arguments: vec![inner_id],
                })
            }

            ParsedType::Void => TypeId::builtin("void"),
            ParsedType::Unknown => TypeId::unknown(),
        })
    }



fn typecheck_function(
    program: &CheckedProgram,
    types: &mut DefinedTypes,
    scope: &Scope,
    function: Function,
) -> TypeCheckResult<CheckedFunction> {
    check_unique(function.params.iter().map(|param| &param.name))?;

    let return_type = typecheck_parsed_type(program, types, &function.return_type)?;

    let params: Vec<_> = function
        .params
        .iter()
        .map(|param| {
            Ok(CheckedFunctionParameter {
                name: param.name.clone(),
                type_id: typecheck_parsed_type(program, types, &param.parsed_type)?,
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
        Statement::For { iterator, body } => {
            let iterator = typecheck_expression(program, types, scope, iterator)?;

            if let TypeDescription::GenericInstance { generic_id, type_arguments } = types.get_type_description(iterator.type_id())
                && generic_id == types.get_generic("array").unwrap() {

                let bindings = vec![Binding {name: "it".to_string(), type_id: *type_arguments.first().unwrap()},
                                    Binding {name: "idx".to_string(), type_id: TypeId::builtin("u64") }];

                let for_scope = scope.inner_with_bindings(bindings);

                let body = body.into_iter().map(|statement| {
                    typecheck_statement(program, types, &for_scope, statement)
                }).collect::<Result<_,_>>()?;

                CheckedStatement::For { iterator, body }
            }
            else {
                return err!(NonArrayIteration {
                    type_name: type_name(program, types, iterator.type_id()),
                    expr: iterator,
                });
            }
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
                    typecheck_binary_op(
                        program,
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

fn typecheck_binary_op(
    program: &CheckedProgram,
    types: &mut DefinedTypes,
    lhs_type: TypeId,
    rhs_type: TypeId,
    op: BinaryOp,
    expected_type: TypeId,
) -> TypeCheckResult<()> {
    if !lhs_type.is_exactly(expected_type)
        || !rhs_type.is_exactly(expected_type)
    {
        err!(BinaryOpType {
            lhs_type: type_name(program, types, lhs_type),
            rhs_type: type_name(program, types, rhs_type),
            op,
            expected_type: type_name(program, types, expected_type),
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
                            expr_type: type_name(program, types,type_id),
                            expr: access_clone.clone(), // TODO: This is a little imprecise
                        });
                    }

                    assert_eq!(type_arguments.len(), 1);

                    // Expr type becomes the inner array type (first and only type argument)
                    type_id = *type_arguments.first().unwrap();
                } else {
                    return err!(NonArraySubscript {
                        expr_type: type_name(program, types,type_id),
                        expr: access_clone.clone(), // TODO: This is a little imprecise
                    });
                }

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
                        struct_name: type_name(program, types,type_id),
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
                    type_id: program.structs.get("<empty_struct>")
                        .expect("Empty struct undefined").type_id,
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
        SimpleExpression::EnumVariant { enum_name, variant_name, initializer } => {
            let enumeration = program.enums.get(&enum_name).ok_or_else(|| wrap!(UndefinedEnum{name: enum_name.to_string()}))?;

            let variant = enumeration.variants.iter().find(|variant| variant.name == variant_name).ok_or_else(||wrap!(UndefinedEnumVariant {enum_name: enum_name.clone(), variant_name: variant_name.clone()}))?;


            if type_name(program, types, variant.type_id()) != "void" && initializer.is_none() {
                return err!(MissingEnumVariantInitializer { variant_name, enum_name, variant_type_name: type_name(program, types, variant.type_id())});
            }

            let initializer = initializer.map(|init| {
                eval_rhs_and_check!(program, types, scope, variant.type_id(), *init, EnumVariantInitMissmatch, [("enum_name", enum_name), ("variant_name", variant_name.clone())]).map(Box::new)
            }).transpose()?;


            CheckedSimpleExpression::EnumVariant { type_id: enumeration.type_id(), variant_name, initializer }

        },
    })
}
