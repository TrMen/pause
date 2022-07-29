use thiserror::Error;

use crate::{
    lexer::{AssignOp, ExecutionDesignator},
    parser::{
        AccessPath, EvaluatedAccessPath, EvaluatedIndirection, Expression, Indirection, Program,
        SimpleExpression, Statement, Struct, Value,
    },
};

#[derive(Error, Debug)]
pub enum InterpretationError {
    #[error("Procedure call failed '{0}'")]
    CallFailed(String),
    #[error("Assertion failed '{0}'")]
    AssertionFailed(String),
    #[error("Unkown procedure '{0}'")]
    UnknownProcedure(String),
    #[error("Unkown function '{0}'")]
    UnknownFunction(String),
    #[error("Unkown assertion '{0}'")]
    UnknownAssertion(String),
    #[error("Access of undefined binding '{0}'")]
    UndefinedBinding(String),
    #[error("Trying to access undefined field '{field_name}' of struct '{structure}'")]
    UndefinedStructField {
        structure: String,
        field_name: String,
    },
    #[error("Binding '{0}' is already defined")]
    BindingRedefinition(String),
    #[error("Index '{0}' is out of program bounds")]
    OutOfProgramBounds(usize),
    #[error("Unexpected end of program for execution '{0:?}'")]
    UnexpectedDone(ExecutionDesignator),
    #[error("Invalid conversion from type '{from}' to '{to}'")]
    InvalidTypeConversion { from: String, to: String }, // TODO: Should be some enum or int identifying types, rather than a string
    #[error("Value returned from function has incorrect type '{actual}'. Expected '{expected}'")]
    ReturnTypeMissmatch { expected: String, actual: String }, // TODO: Should be some enum or int identifying types, rather than a string
    #[error("Trying to assign value '{value:#?}' of type '{actual}' to field or binding of type '{expected}'")]
    AssignmentTypeMissmatch {
        value: Value,
        expected: String,
        actual: String,
    }, // TODO: Should be some enum or int identifying types, rather than a string
    #[error("Incorrect argument type for '{param_name}: {expected_type}' in call to function '{func_name}'. Argument is '{actual_val:#?}' of type '{actual_type}'")]
    FunctionArgumentTypeMissmatch {
        func_name: String,
        param_name: String,
        expected_type: String,
        actual_val: Value,
        actual_type: String,
    }, //
    #[error("Incorrect number of arguments for call to function '{name}'. Expected '{expected}' arguments, got '{actual}'.")]
    MissmatchedArity {
        name: String,
        expected: usize,
        actual: usize,
    },
    #[error("Trying to access array with index value '{value:#?}' of type '{type_name}'. Only integers are allowed.")]
    ArrayIndexTypeError { value: Value, type_name: String },
    #[error("Trying to access value '{value:#?}' as if it were an array.")]
    ArrayTypeError { value: Value },
    #[error("Out of bounds array access on '{:#?}'. Index: '{index}'")]
    OutOfBoundsArrayAccess { array: Value, index: usize },
}

use InterpretationError::*;

pub(crate) type InterpretationResult<T> = Result<T, InterpretationError>;

#[derive(Clone, Debug)]
pub struct Binding {
    name: String,
    type_name: String,
    value: Value,
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

    pub fn find_binding(&self, name: &str) -> InterpretationResult<&Binding> {
        let outer_find = || self.outer.as_ref().map(|outer| outer.find_binding(name));

        self.bindings
            .iter()
            .find(|binding| binding.name == name)
            .map_or_else(outer_find, |b| Some(Ok(b)))
            .ok_or_else(|| InterpretationError::UndefinedBinding(name.to_string()))?
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Interpreter {
    pub state: Struct,

    main_index: usize,
}

impl Interpreter {
    pub fn new(state: Struct) -> Self {
        Self {
            state,
            main_index: 0,
        }
    }

    pub fn step(&mut self, program: &Program) -> InterpretationResult<Option<Statement>> {
        // TODO: Doesn't need to return by value.
        //
        // TODO: Make this more sophisticated than just "steps in main procedure". Prolly the whole
        // idea of stepping doesn't really make sense, and I'll just tell the interpreter to run
        // for a bit, then it comes back with a list of statements it ran, and optionally an error.
        if self.main_index >= program.main.body.len() {
            return Ok(None);
        }

        let executed_statement = &program.main.body[self.main_index];

        self.main_index += 1;

        self.statement(&GLOBAL_SCOPE, program, executed_statement)?;

        Ok(Some(executed_statement.clone()))
    }

    pub fn execute_statements<'a>(
        &mut self,
        program: &Program,
        stmts: impl Iterator<Item = &'a Statement>,
    ) -> InterpretationResult<()> {
        // TODO: Allow more than just global scope
        for stmt in stmts {
            self.statement(&GLOBAL_SCOPE, program, stmt)?;
        }

        Ok(())

        // TODO: Need to track early termination somehow, but that's not important without exit()
        // or return so far
    }

    fn statement(
        &mut self,
        scope: &Scope,
        program: &Program,
        stmt: &Statement,
    ) -> InterpretationResult<()> {
        match stmt {
            Statement::AssertionCall { name } => self.call_assertion(scope, program, name),
            Statement::StateAssignment {
                lhs,
                assign_op,
                rhs,
            } => self.state_assignment(scope, program, lhs, assign_op, rhs),
            Statement::ProcedureCall { name } => self.procedure_call(scope, program, name),
            Statement::Expression(expression) => {
                let _expr_result = evaluate_expression(scope, &self.state, program, expression)?;
                // TODO: Check if result was marked must-use (or is that even a thing?)
                Ok(())
            }
        }
    }

    fn call_assertion(
        &mut self,
        scope: &Scope,
        program: &Program,
        name: &str,
    ) -> InterpretationResult<()> {
        let assertion = program
            .assertions
            .get(name)
            .ok_or_else(|| UnknownAssertion(name.to_string()))?;

        let passed =
            evaluate_expression(scope, &self.state, program, &assertion.predicate)?.is_truthy()?;

        if !passed {
            return Err(AssertionFailed(name.to_string()));
        }

        Ok(())
    }

    fn state_assignment(
        &mut self,
        scope: &Scope,
        program: &Program,
        lhs: &AccessPath,
        assign_op: &AssignOp,
        rhs: &Expression,
    ) -> InterpretationResult<()> {
        let rhs_value = evaluate_expression(scope, &self.state, program, rhs)?;

        let path = evaluate_path(scope, lhs, &self.state, program)?;

        let field = self
            .state
            .get_field_mut(&path.name)?
            .value
            .access_path_mut(&path)?;

        // TODO: This isn't quite right. It checks against whatever is currently in the field,
        // which is fine since it always checks. But this doesn't work right for arrays. Should
        // check against the declared type of the struct field
        if rhs_value.type_name() != field.type_name() {
            return Err(AssignmentTypeMissmatch {
                value: rhs_value.clone(),
                expected: field.type_name().to_string(),
                actual: rhs_value.type_name().to_string(),
            });
        }

        match assign_op {
            // TODO: This should not use the Struct strucutre, since that's for source code
            // definitions.
            AssignOp::Equal => *field = rhs_value,
            AssignOp::PlusEqual => {
                if let (Value::Number(field), Value::Number(rhs)) = (field, rhs_value) {
                    *field += rhs;
                } else {
                    todo!("Invalid assignment");
                }
            }
        };

        Ok(())
    }

    fn procedure_call(
        &mut self,
        scope: &Scope,
        program: &Program,
        name: &str,
    ) -> InterpretationResult<()> {
        let procedure = program
            .procedures
            .get(name)
            .ok_or_else(|| UnknownProcedure(name.to_string()))?;

        for stmt in &procedure.body {
            // TODO: Do procedures ever return something? I'm leaning towards no. But if yes, I
            // need a return stmt or implicit last-expr-returns and I need to check that here.
            self.statement(scope, program, stmt)?;
        }

        Ok(())
    }
}

pub fn evaluate_path(
    scope: &Scope,
    path: &AccessPath,
    state: &Struct,
    program: &Program,
) -> InterpretationResult<EvaluatedAccessPath> {
    let mut eval_path = EvaluatedAccessPath {
        name: path.name.clone(),
        indirections: Vec::new(),
    };

    for indirection in &path.indirections {
        match indirection {
            Indirection::Field(_) => todo!("Do I need to do anything here? "),
            Indirection::Subscript(index_expr) => {
                let evaluated = evaluate_expression(scope, state, program, index_expr)?;
                if let Value::Number(index) = evaluated {
                    eval_path
                        .indirections
                        .push(EvaluatedIndirection::Subscript(index.try_into().unwrap()));
                } else {
                    return Err(ArrayIndexTypeError {
                        value: evaluated.clone(),
                        type_name: evaluated.type_name().to_string(),
                    });
                }
            }
        }
    }

    Ok(eval_path)
}

// TODO: This can be pub if expressions have no side effects. They can right now through
// procedure calls
pub fn evaluate_expression(
    scope: &Scope,
    state: &Struct,
    program: &Program,
    expression: &Expression,
) -> InterpretationResult<Value> {
    match expression {
        Expression::Binary { lhs, op, rhs } => {
            let lhs = evaluate_simple_expression(scope, state, program, lhs)?;

            let rhs = evaluate_expression(scope, state, program, rhs)?;

            match op {
                crate::lexer::BinaryOp::Plus => {
                    Ok(Value::Number(lhs.as_number() + rhs.as_number()))
                }
                crate::lexer::BinaryOp::Minus => {
                    Ok(Value::Number(lhs.as_number() - rhs.as_number()))
                }
                crate::lexer::BinaryOp::EqualEqual => Ok(Value::Bool(lhs == rhs)),
            }
        }
        Expression::Simple(simple) => evaluate_simple_expression(scope, state, program, simple),
    }
}

pub fn evaluate_simple_expression(
    scope: &Scope,
    state: &Struct,
    program: &Program,
    simple: &SimpleExpression,
) -> InterpretationResult<Value> {
    match simple {
        SimpleExpression::Value(value) => Ok(value.clone()),
        SimpleExpression::StateAccess(access_path) => {
            struct_field_access(scope, state, program, access_path).cloned()
        }
        SimpleExpression::FunctionCall { name, arguments } => {
            evaluate_function(scope, state, program, name, arguments)
        }
        SimpleExpression::ExecutionAccess { execution, path } => todo!(),
        SimpleExpression::BindingAccess(path) => {
            let path = evaluate_path(scope, path, state, program)?;

            scope
                .find_binding(&path.name)?
                .value
                .access_path(&path)
                .cloned()
        }
        SimpleExpression::ArrayLiteral(array) => {
            let mut evaluated_array = Vec::new();
            for expression in array {
                evaluated_array.push(evaluate_expression(scope, state, program, expression)?);
            }

            Ok(Value::Array(evaluated_array))
        }
        // TODO: This should just be a path access. But have to parse that correctly
        SimpleExpression::ArrayAccess { target, index } => {
            let target = evaluate_expression(scope, state, program, target)?;
            let index_val = evaluate_expression(scope, state, program, index)?;

            if let Value::Number(index) = index_val {
                Ok(target.access_array(index.try_into().unwrap())?.clone())
            } else {
                Err(ArrayIndexTypeError {
                    value: index_val.clone(),
                    type_name: index_val.type_name().to_string(),
                })
            }
        }
    }
}

pub fn struct_field_access<'a>(
    scope: &Scope,
    structure: &'a Struct,
    program: &Program,
    path: &AccessPath,
) -> InterpretationResult<&'a Value> {
    structure
        .get_field(&path.name)?
        .value
        .access_path(&evaluate_path(scope, path, structure, program)?)
}

pub fn struct_field_access_mut<'a>(
    scope: &Scope,
    structure: &'a mut Struct,
    program: &Program,
    path: &AccessPath,
) -> InterpretationResult<&'a Value> {
    let path = evaluate_path(scope, path, structure, program)?;

    structure
        .get_field_mut(&path.name)?
        .value
        .access_path(&path)
}

pub fn evaluate_function(
    scope: &Scope,
    state: &Struct,
    program: &Program,
    name: &str,
    arguments: &Vec<Expression>,
) -> InterpretationResult<Value> {
    let function = program
        .functions
        .get(name)
        .ok_or_else(|| UnknownFunction(name.to_string()))?;

    if function.params.len() != arguments.len() {
        return Err(MissmatchedArity {
            name: name.to_string(),
            expected: function.params.len(),
            actual: arguments.len(),
        });
    }

    let bindings = function
        .params
        .iter()
        .zip(arguments.iter())
        .map(|(param, arg)| -> InterpretationResult<Binding> {
            let arg = evaluate_expression(scope, state, program, arg)?;
            if param.type_name != arg.type_name() {
                return Err(FunctionArgumentTypeMissmatch {
                    func_name: function.name.to_string(),
                    param_name: param.name.clone(),
                    expected_type: param.type_name.clone(),
                    actual_type: arg.type_name().to_string(),
                    actual_val: arg,
                });
            }
            Ok(Binding {
                name: param.name.clone(),
                type_name: param.type_name.clone(),
                value: arg,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let inner_scope = scope.inner_with_bindings(bindings);

    let function_value = evaluate_expression(&inner_scope, state, program, &function.expression)?;

    // TODO: remove this runtime type checking from here
    if function_value.type_name() != function.return_type {
        return Err(ReturnTypeMissmatch {
            expected: function.return_type.clone(),
            actual: function_value.type_name().to_string(),
        });
    }

    Ok(function_value)
}
