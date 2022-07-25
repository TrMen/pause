use std::{fmt::Display, io::stdin};

use crate::{
    interpreter::{InterpretationError, InterpretationResult, Interpreter},
    lexer::ExecutionDesignator,
    parser::{Program, Statement, Struct},
};

use ExecutionDesignator::*;

// TODO: This becomes a real, graph structure that can give you relations between all possible
// TemporalDesignator
#[derive(Debug, Clone)]
struct ProgramExecutionLog {
    past_to_current: Vec<Statement>,
}

// TODO: Better name that signifies this is between two specific points
#[derive(Debug)]
struct ExecutionLogDiff<'a> {
    start: ExecutionDesignator,
    end: ExecutionDesignator,
    log: &'a mut Vec<Statement>,
}

impl ExecutionLogDiff<'_> {
    fn push(&mut self, statement: Statement) {
        self.log.push(statement);
    }

    fn len(&self) -> usize {
        self.log.len()
    }

    fn get(&self, index: usize) -> Option<&Statement> {
        self.log.get(index)
    }

    fn statements(&self) -> impl Iterator<Item = &Statement> {
        self.log.iter()
    }

    fn remove(&mut self, index: usize) {
        self.log.remove(index);
    }

    fn change(&mut self, index: usize, new_entry: Statement) {
        *self.log.get_mut(index).unwrap() = new_entry;
    }
}

impl ProgramExecutionLog {
    fn new() -> Self {
        Self {
            past_to_current: Vec::new(),
        }
    }

    fn fmt_between(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        start: ExecutionDesignator,
        end: ExecutionDesignator,
    ) -> std::fmt::Result {
        {
            writeln!(f, "Exection log between {:?} and {:?}", start, end)?;

            // TODO: Work with more than past and present

            for (i, entry) in self.past_to_current.iter().enumerate() {
                writeln!(f, "{i}: {entry:#?}")?;
            }

            Ok(())
        }
    }

    fn between(
        &mut self,
        start: ExecutionDesignator,
        end: ExecutionDesignator,
    ) -> ExecutionLogDiff {
        // TODO: work with more than past and present
        assert!([Current, Past].contains(&start));
        assert!([Current, Past].contains(&end));

        ExecutionLogDiff {
            start,
            end,
            log: &mut self.past_to_current,
        }
    }

    // Returns a tree structure that encompasses all possible executions from the start
    // node
    fn from(&mut self, start: ExecutionDesignator) -> ExecutionTree {
        // TODO: more than just one
        assert!(start == Past);
        ExecutionTree {
            logs: vec![&mut self.past_to_current],
        }
    }
}

#[derive(Debug)]
struct ExecutionTree<'a> {
    // TODO: The logs should be VecDeq I guess, since I remove from both ends pretty often
    logs: Vec<&'a mut Vec<Statement>>,
}

impl ExecutionTree<'_> {
    // For each possible execution, remove the first entry
    fn remove_one(&mut self) {
        for log in &mut self.logs {
            log.remove(0);
        }
    }
}

#[derive(Debug)]
pub struct MetaInterpreter {
    current: Interpreter,
    // These are stored here to avoid double borrow issues in Interpreter. They are really
    // immutable views for Interpreter
    // TODO: I feel like there must be some way to tell the compiler: I'm never mutating this
    // field, ignore it in mut xor shared self analysis
    current_program: Program,
    past: Interpreter,
    past_program: Program,
    max_past_offset: usize,
    execution_log: ProgramExecutionLog,
}

impl MetaInterpreter {
    pub fn new(state: Struct, program: Program) -> Self {
        let interpreter = Interpreter::new(state);
        Self {
            current: interpreter.clone(),
            current_program: program.clone(),
            past: interpreter,
            past_program: program,
            max_past_offset: 50, // TODO: Magic number
            execution_log: ProgramExecutionLog::new(),
        }
    }

    fn step_one(
        &mut self,
        execution: ExecutionDesignator,
    ) -> InterpretationResult<Option<Statement>> {
        // TODO: Add alternative executions here
        let result = match execution {
            Past => self.past.step(&self.past_program),
            Current => self.current.step(&self.current_program),
        };

        match result {
            Ok(Some(good)) => {
                self.execution_log.to(execution).push(good);
                self.execution_log.from(execution).remove_one();

                Ok(Some(good))
            }
            Ok(None) => {
                todo!("Handle end of execution")
            }
            Err(err) => {
                self.handle_error(execution, err);

                todo!("Gotta make sure that previous step didn't change any state when failing, then can 'just' step again");
            }
        }
    }

    pub fn step(&mut self) -> InterpretationResult<Option<Statement>> {
        // TODO: Should this return anything?
        // TODO: Execution log

        if self.execution_log.between(Past, Current).len() < self.max_past_offset {
            let past_stmt = self.step_one(Past)?;
            assert_eq!(
                self.execution_log.between(Past, Current).get(0),
                past_stmt.as_ref()
            );
            self.execution_log.between(Past, Current).remove(0);
        }

        let current_stmt = self.step_one(Current)?;

        if let Some(current_stmt) = &current_stmt {
            self.execution_log
                .between(Past, Current)
                .push(current_stmt.clone())
        }
        // TODO: Otherwise, make sure we don't just go on

        Ok(current_stmt)
    }

    // TODO: can this really just return no error. That's suspicious
    pub fn handle_error(&mut self, execution: ExecutionDesignator, error: InterpretationError) {
        match self.user_interaction(execution, error) {
            Ok(()) => (),
            Err(err) => self.handle_error(execution, err),
        }
    }

    fn user_interaction(
        &mut self,
        execution: ExecutionDesignator,
        error: InterpretationError,
    ) -> InterpretationResult<()> {
        println!("{}", self);
        println!("{}", error);
        println!("{:?} failed", execution);

        loop {
            match get_user_choice() {
                // TODO: This shouldn't just be a permanent change
                UserChoice::RemoveLogEntry(idx) => {
                    self.execution_log.between(Past, Current).remove(idx);
                }
                UserChoice::ExecutePastTillCurrent => {
                    // TODO: Allow (staged but not committed) changes to execution log here
                    let mut alternative = self.past.clone();
                    let alternative_program = self.past_program.clone();

                    alternative.execute_statements(
                        &alternative_program,
                        self.execution_log.between(Past, Current).statements(),
                    )?;

                    // TODO: Most common case is prolly to then assert equality (e.g. replace something in
                    // past, then check if that was equivalent). But not always. Sometimes you just wanna
                    // change the past.

                    println!(
                        "{:#?}\nExecuted past(previous output is alternative current)...",
                        alternative.state,
                    );
                }
                UserChoice::Print => println!("{}", self),
                UserChoice::ChangeProgramEntry {
                    offset_from_past,
                    new_entry,
                } => match &new_entry {
                    Statement::AssertionCall { .. } | Statement::ProcedureCall { .. } => {
                        // TODO: I think this might get confusing. User defines a new fn
                        // interactively, but is that defined in past, current, or alt?
                        self.past_program
                            .ensure_uses_defined_components(&new_entry)?;

                        self.execution_log
                            .between(Past, Current)
                            .change(offset_from_past, new_entry);
                    }
                    Statement::StateAssignment { .. } => {
                        todo!("Didn't give the user the chance to do that yet")
                    }
                    _ => todo!("Arbitrary insertions not yet implemented"),
                },
                UserChoice::ShowDefinedAssertions => {
                    for (i, assertion) in self.current_program.assertions.iter().enumerate() {
                        println!("{i}: {}", assertion.0);
                    }
                }
                UserChoice::ShowDefinedProcedures => {
                    // TODO: This is too restrictive. Only works for past->current
                    println!("Defined Procedures: ");
                    for (i, procedure) in self.current_program.procedures.iter().enumerate() {
                        println!("{i}: {}", procedure.0);
                    }
                }

                UserChoice::ShowDefinedStructs => {
                    for (i, structure) in self.current_program.structs.iter().enumerate() {
                        println!("{i}: {:#?}", structure.1);
                    }
                }
                UserChoice::BadInput => println!("Bad input..."),
                UserChoice::Step(execution) => {
                    let _ = self.step_one(execution)?;
                }
                UserChoice::Continue => todo!(
                    "Make sure the user can actually do something useful in repl, then continue"
                ),
            }
        }
    }
}

impl Display for MetaInterpreter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Display will not print the whole program. A seperate method will do that.
        self.execution_log.fmt_between(f, Past, Current)?;
        writeln!(f)?;
        //writeln!( TODO: Insert a failing entry again
        //    f,
        //    "Failing Entry: '{}\n",
        //)?;
        writeln!(f, "Current state: {:#?}", self.current.state)?;
        writeln!(f, "Past state: {:#?}", self.past.state)
    }
}

enum UserChoice {
    Continue,
    BadInput,
    RemoveLogEntry(usize),
    ExecutePastTillCurrent,
    // TODO: This is actually hard, since in general this would be allowing
    // the user to change the AST. The real question then is: What do I wanna let them change?
    // Is it execution log or the actual ast of the past program?
    ChangeProgramEntry {
        offset_from_past: usize,
        new_entry: Statement,
    },
    Print, // TODO: Allow choosing what to print
    Step(ExecutionDesignator),
    ShowDefinedAssertions,
    ShowDefinedProcedures,
    ShowDefinedStructs,
}

fn get_user_input(prompt: &str) -> String {
    let mut inp = String::new();

    println!(">{prompt}");

    stdin().read_line(&mut inp).unwrap();

    inp.strip_suffix('\n').unwrap().to_string()
}

fn get_user_choice() -> UserChoice {
    let input = get_user_input(
        "step|r(emove)|e(xecute) past|p(rint) state|c(hange) past|procedures|assertions|continue",
    );

    // TODO: Make this not just die with bad input. This must be robust

    match input.as_str() {
        "r" | "remove" => UserChoice::RemoveLogEntry(
            str::parse(&get_user_input("Enter an exection log index to remove")).unwrap(),
        ),
        "e" | "execute" => UserChoice::ExecutePastTillCurrent,
        "p" | "print" => UserChoice::Print,
        "c" | "change" => {
            let offset_from_past: usize = str::parse(&get_user_input(
                "Enter the index of a program entry to change",
            ))
            .unwrap();

            // TODO: Should not let the user distinguish between entries. Just let them put a name,
            // good enough
            let sel = get_user_input(
                "Enter a(ssertion)|p(rocedure) and the name of the respective thing to insert.",
            );
            let sel = sel.split_whitespace().take(2).collect::<Vec<&str>>();

            let new_entry = match sel[0] {
                "a" | "assertion" => Statement::AssertionCall {
                    name: sel[1].to_string(),
                },
                "p" | "procedure" => Statement::ProcedureCall {
                    name: sel[1].to_string(),
                },
                _ => {
                    return UserChoice::BadInput;
                }
            };

            UserChoice::ChangeProgramEntry {
                offset_from_past,
                new_entry,
            }
        }
        "assertions" => UserChoice::ShowDefinedAssertions,
        "continue" => UserChoice::Continue,
        "procedures" => UserChoice::ShowDefinedProcedures,
        "structs" => UserChoice::ShowDefinedStructs,
        "step" => match get_user_input("past|current").as_str() {
            "past" => UserChoice::Step(Past),
            "current" => UserChoice::Step(Current),
            _ => UserChoice::BadInput,
        },
        "step current" => UserChoice::Step(Current),
        "step past" => UserChoice::Step(Past),
        _ => UserChoice::BadInput,
    }
}
