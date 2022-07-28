use std::collections::HashMap;

use thiserror::Error;

use crate::{
    lexer::{AssignOp, ExecutionDesignator},
    parser::{AccessPath, Expression, Program, Statement, Struct, Value},
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
        program.ensure_uses_defined_components(stmt)?;

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
        let value = self.evaluate_expression(program, rhs)?;

        let state_field = self
            .state
            .fields
            .iter_mut()
            .find(|f| f.name == lhs.name)
            .ok_or_else(|| UndefinedStructField {
                structure: "state".to_string(),
                field_name: lhs.name.clone(),
            })?;

        if !lhs.fields.is_empty() {
            todo!("Struct access for state")
        }

        if value.type_name() != state_field.type_name {
            return Err(AssignmentTypeMissmatch {
                value: value.clone(),
                expected: state_field.type_name.clone(),
                actual: value.type_name().to_string(),
            });
        }

        match assign_op {
            // TODO: This should not use the Struct strucutre, since that's for source code
            // definitions.
            AssignOp::Equal => state_field.initial_value = value,
            AssignOp::PlusEqual => {
                if let (Value::Number(lhs), Value::Number(rhs)) =
                    (&mut state_field.initial_value, value)
                {
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
        &mut self,
        program: &Program,
        expression: &Expression,
    ) -> InterpretationResult<Value> {
        match expression {
            Expression::Simple(simple) => match simple {
                crate::parser::SimpleExpression::Value(value) => Ok(value.clone()),
                crate::parser::SimpleExpression::StateAccess(_) => todo!(),
                crate::parser::SimpleExpression::FunctionCall { name, arguments } => {
                    self.evaluate_function(program, name, arguments)
                }
                crate::parser::SimpleExpression::ExecutionAccess { execution, path } => todo!(),
                crate::parser::SimpleExpression::BindingAccess(path) => self.binding_access(path),
            },
            Expression::Binary { lhs, op, rhs } => todo!(),
        }
    }

    fn binding_access(&self, path: &AccessPath) -> InterpretationResult<Value> {
        let binding = self
            .bindings
            .get(&path.name)
            .ok_or_else(|| UndefinedBinding(path.name.clone()))?;

        let value = binding.value.clone();

        if !path.fields.is_empty() {
            todo!("Struct access")
        }

        Ok(value)
    }

    fn evaluate_function(
        &mut self,
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
