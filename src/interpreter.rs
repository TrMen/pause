use std::collections::HashMap;

use thiserror::Error;

use crate::{
    lexer::{AssignOp, ExecutionDesignator},
    parser::{
        AccessPath, Expression, Indirection, Program, SimpleExpression, Statement, Struct, Value,
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
    OutOfBoundsArrayAccess { array: Value, index: u64 },
}

use InterpretationError::*;

pub(crate) type InterpretationResult<T> = Result<T, InterpretationError>;

#[derive(Clone, Debug)]
struct Binding {
    name: String,
    type_name: String,
    value: Value,
}

#[derive(Clone, Debug)]
pub(crate) struct Interpreter {
    pub state: Struct,

    main_index: usize,

    bindings: HashMap<String, Binding>,
}

impl Interpreter {
    pub fn new(state: Struct) -> Self {
        Self {
            state,
            bindings: HashMap::new(),
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

        self.statement(program, executed_statement)?;

        Ok(Some(executed_statement.clone()))
    }

    pub fn execute_statements<'a>(
        &mut self,
        program: &Program,
        stmts: impl Iterator<Item = &'a Statement>,
    ) -> InterpretationResult<()> {
        for stmt in stmts {
            self.statement(program, stmt)?;
        }

        Ok(())

        // TODO: Need to track early termination somehow, but that's not important without exit()
        // or return so far
    }

    fn statement(&mut self, program: &Program, stmt: &Statement) -> InterpretationResult<()> {
        match stmt {
            Statement::AssertionCall { name } => self.call_assertion(program, name),
            Statement::StateAssignment {
                lhs,
                assign_op,
                rhs,
            } => self.state_assignment(program, lhs, assign_op, rhs),
            Statement::ProcedureCall { name } => self.procedure_call(program, name),
            Statement::Expression(expression) => {
                let _expr_result = self.evaluate_expression(program, expression)?;
                // TODO: Check if result was marked must-use (or is that even a thing?)
                Ok(())
            }
        }
    }

    fn call_assertion(&mut self, program: &Program, name: &str) -> InterpretationResult<()> {
        let assertion = program
            .assertions
            .get(name)
            .ok_or_else(|| UnknownAssertion(name.to_string()))?;

        let passed = self
            .evaluate_expression(program, &assertion.predicate)?
            .is_truthy()?;

        if !passed {
            return Err(AssertionFailed(name.to_string()));
        }

        Ok(())
    }

    fn state_assignment(
        &mut self,
        program: &Program,
        lhs: &AccessPath,
        assign_op: &AssignOp,
        rhs: &Expression,
    ) -> InterpretationResult<()> {
        let rhs_value = self.evaluate_expression(program, rhs)?;

        let state_field = self
            .state
            .fields
            .iter_mut()
            .find(|f| f.name == lhs.name)
            .ok_or_else(|| UndefinedStructField {
                structure: "state".to_string(),
                field_name: lhs.name.clone(),
            })?;

        let mut lhs_value = &mut state_field.initial_value;

        for indirection in &lhs.indirections {
            match indirection {
                crate::parser::Indirection::Field(_) => todo!("Struct field access"),
                crate::parser::Indirection::Subscript(index) => {
                    let index = self.evaluate_expression(program, &index)?;
                    lhs_value = self.access_array_mut(lhs_value, index)?
                }
            }
        }

        //        let lhs_value =
        //           self.access_indirections(program, &mut state_field.initial_value, &lhs.indirections)?;

        if rhs_value.type_name() != state_field.type_name {
            return Err(AssignmentTypeMissmatch {
                value: rhs_value.clone(),
                expected: state_field.type_name.clone(),
                actual: rhs_value.type_name().to_string(),
            });
        }

        match assign_op {
            // TODO: This should not use the Struct strucutre, since that's for source code
            // definitions.
            AssignOp::Equal => *lhs_value = rhs_value,
            AssignOp::PlusEqual => {
                if let (Value::Number(lhs), Value::Number(rhs)) = (lhs_value, rhs_value) {
                    *lhs += rhs;
                } else {
                    todo!("Invalid assignment");
                }
            }
        };

        Ok(())
    }

    // TODO: This can be pub if expressions have no side effects. They can right now through
    // procedure calls
    fn evaluate_expression(
        &self,
        program: &Program,
        expression: &Expression,
    ) -> InterpretationResult<Value> {
        match expression {
            Expression::Binary { lhs, op, rhs } => {
                let lhs = self.evaluate_simple_expression(program, lhs)?;

                let rhs = self.evaluate_expression(program, rhs)?;

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
            Expression::Simple(simple) => self.evaluate_simple_expression(program, simple),
        }
    }

    fn evaluate_simple_expression(
        &self,
        program: &Program,
        simple: &SimpleExpression,
    ) -> InterpretationResult<Value> {
        match simple {
            SimpleExpression::Value(value) => Ok(value.clone()),
            SimpleExpression::StateAccess(access_path) => self.state_access(access_path),
            SimpleExpression::FunctionCall { name, arguments } => {
                self.evaluate_function(program, name, arguments)
            }
            SimpleExpression::ExecutionAccess { execution, path } => todo!(),
            SimpleExpression::BindingAccess(path) => self.binding_access(program, path),
            SimpleExpression::ArrayLiteral(array) => {
                let mut evaluated_array = Vec::new();
                for expression in array {
                    evaluated_array.push(self.evaluate_expression(program, expression)?);
                }

                Ok(Value::Array(evaluated_array))
            }
            SimpleExpression::ArrayAccess { target, index } => {
                let target = self.evaluate_expression(program, target)?;
                let index = self.evaluate_expression(program, index)?;

                self.access_array(&target, index).cloned()
            }
        }
    }

    fn access_array_mut(
        &self,
        array: &mut Value,
        index: Value,
    ) -> InterpretationResult<&mut Value> {
        if let Value::Number(index) = &index {
            if let Value::Array(array) = array {
                Ok(array
                    .get_mut(*index as usize)
                    .ok_or_else(|| OutOfBoundsArrayAccess {
                        array: Value::Array(array.clone()),
                        index: *index,
                    })?)
            } else {
                Err(ArrayTypeError {
                    value: array.clone(),
                })
            }
        } else {
            Err(ArrayIndexTypeError {
                value: index.clone(),
                type_name: index.type_name().to_string(),
            })
        }
    }

    // TODO: Don't replicate these so much
    fn access_array(&self, array: &Value, index: Value) -> InterpretationResult<&Value> {
        if let Value::Number(index) = &index {
            if let Value::Array(array) = &array {
                Ok(array
                    .get_mut(*index as usize)
                    .ok_or_else(|| OutOfBoundsArrayAccess {
                        array: Value::Array(array.clone()),
                        index: *index,
                    })?)
            } else {
                Err(ArrayTypeError {
                    value: array.clone(),
                })
            }
        } else {
            Err(ArrayIndexTypeError {
                value: index.clone(),
                type_name: index.type_name().to_string(),
            })
        }
    }

    fn state_access(&self, path: &AccessPath) -> InterpretationResult<Value> {
        let field = self
            .state
            .fields
            .iter()
            .find(|field| field.name == path.name)
            .ok_or_else(|| UndefinedStructField {
                structure: "state".to_string(),
                field_name: path.name.clone(),
            })?;

        // TODO: Initial value shouldn't be here
        let value = field.initial_value.clone();

        if !path.indirections.is_empty() {
            todo!("Struct access");
        }

        Ok(value)
    }

    fn binding_access(&self, program: &Program, path: &AccessPath) -> InterpretationResult<Value> {
        let binding = self
            .bindings
            .get(&path.name)
            .ok_or_else(|| UndefinedBinding(path.name.clone()))?;

        let value = binding.value.clone();

        let value = self.access_indirections(program, &mut value, &path.indirections)?;

        Ok(value.clone())
    }

    fn access_indirections(
        &self,
        program: &Program,
        mut value: &Value,
        indirections: &Vec<Indirection>,
    ) -> InterpretationResult<&Value> {
        for indirection in indirections {
            match indirection {
                crate::parser::Indirection::Field(_) => todo!("Struct field access"),
                crate::parser::Indirection::Subscript(index) => {
                    let index = self.evaluate_expression(program, &index)?;
                    value = self.access_array(&value, index)?
                }
            }
        }

        Ok(value)
    }

    fn access_indirections_mut(
        &mut self,
        program: &Program,
        mut value: &mut Value,
        indirections: &Vec<Indirection>,
    ) -> InterpretationResult<&mut Value> {
        for indirection in indirections {
            match indirection {
                crate::parser::Indirection::Field(_) => todo!("Struct field access"),
                crate::parser::Indirection::Subscript(index) => {
                    let index = self.evaluate_expression(program, &index)?;
                    value = self.access_array_mut(&mut value, index)?
                }
            }
        }

        Ok(value)
    }

    fn evaluate_function(
        &self,
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

        for (parameter, argument) in function.params.iter().zip(arguments.iter()) {
            let value = self.evaluate_expression(program, argument)?;
            self.define_binding(parameter.name.clone(), parameter.type_name.clone(), value)?;
        }

        let value = self.evaluate_expression(program, &function.expression)?;

        // TODO: remove this runtime type checking from here
        if value.type_name() != function.return_type {
            return Err(ReturnTypeMissmatch {
                expected: function.return_type.clone(),
                actual: value.type_name().to_string(),
            });
        }

        for parameter in function.params.iter() {
            self.delete_binding(&parameter.name);
        }

        Ok(value)
    }

    fn define_binding(
        &mut self,
        name: String,
        type_name: String,
        value: Value,
    ) -> InterpretationResult<()> {
        if self.bindings.contains_key(&name) {
            return Err(BindingRedefinition(name));
        }

        self.bindings.insert(
            name.clone(),
            Binding {
                name,
                type_name,
                value,
            },
        );

        Ok(())
    }

    fn delete_binding(&mut self, name: &str) {
        self.bindings.remove(name);
    }

    fn procedure_call(&mut self, program: &Program, name: &str) -> InterpretationResult<()> {
        let procedure = program
            .procedures
            .get(name)
            .ok_or_else(|| UnknownProcedure(name.to_string()))?;

        for stmt in &procedure.body {
            // TODO: Do procedures ever return something? I'm leaning towards no. But if yes, I
            // need a return stmt or implicit last-expr-returns and I need to check that here.
            self.statement(program, stmt)?;
        }

        Ok(())
    }
}

//    pub fn set(&mut self, idx: usize, new: ProgramEntry) -> FunctionResult<()> {
//        let existing = self
//            .entries
//            .get_mut(idx)
//            .ok_or(FunctionFailure::OutOfProgramBounds(idx))?;
//
//        *existing = new;
//
//        Ok(())
//    }
//
//    pub fn push_fn(&mut self, fn_name: &str) {
//        self.entries
//            .push(ProgramEntry::Function(fn_name.to_string()));
//    }
//
//    pub fn push_assertion(&mut self, assertion_name: &str) {
//        self.entries
//            .push(ProgramEntry::Assertion(assertion_name.to_string()));
//    }
//
//    // TODO: This returning FunctionResult is an indicator that might not be named well.
//    pub fn get(&self, idx: usize) -> &ProgramEntry {
//        // Should only be used from TrackedState, so the function idx must exist.
//        &self.entries[idx]
//    }
