#![feature(never_type)]
#![feature(backtrace)]
mod lexer;
mod parser;
mod util;

use once_cell::sync::Lazy;
use std::{collections::HashMap, fmt::Display, io::stdin, process::exit, sync::Mutex};

use crate::parser::Parser;

#[derive(Clone, Debug)]
struct State {
    pub num: usize,
    variables: HashMap<String, Value>,
}

impl State {
    pub fn new() -> Self {
        Self {
            num: 0,
            variables: HashMap::new(),
        }
    }
    // TODO: Make generic over return type, or return Value sumtype
    fn get_variable<'a>(&'a self, name: &str) -> &'a Value {
        self.variables.get(name).unwrap()
    }

    fn assign_variable(&mut self, name: &str, value: Value) {
        self.variables.insert(name.to_string(), value);
    }
}

impl Display for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

struct Functions {
    pub functions: HashMap<String, Box<Function>>,
    pub assertions: HashMap<String, Box<Assertion>>,
}

impl Functions {
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
            assertions: HashMap::new(),
        }
    }
    pub fn get_fn(&self, name: &str) -> Option<&Function> {
        // TODO: The double refer seems a bit weird
        self.functions.get(name).map(|v| &**v)
    }

    pub fn add_fn(&mut self, name: &str, func: &'static Function) {
        self.functions.insert(name.to_string(), Box::new(func));
    }

    pub fn get_assertion(&self, name: &str) -> Option<&Assertion> {
        // TODO: The double refer seems a bit weird
        self.assertions.get(name).map(|v| &**v)
    }

    pub fn add_assertion(&mut self, name: &str, func: &'static Assertion) {
        self.assertions.insert(name.to_string(), Box::new(func));
    }
}

static FUNCTIONS: Lazy<Mutex<Functions>> = Lazy::new(|| Mutex::new(Functions::new()));

macro_rules! get_fn {
    ($name:ident) => {
        FUNCTIONS
            .lock()
            .unwrap()
            .get_fn($name)
            .ok_or_else(|| FunctionFailure::UnknownFunction($name.clone()))
    };
}

macro_rules! get_assertion {
    ($name:ident) => {
        FUNCTIONS
            .lock()
            .unwrap()
            .get_assertion($name)
            .ok_or_else(|| FunctionFailure::UnknownFunction($name.clone()))
    };
}

#[derive(Clone)]
enum ProgramEntry {
    Function(String),
    Assertion(String),
}

impl ProgramEntry {
    pub fn exists(&self) -> bool {
        match self {
            ProgramEntry::Function(name) => get_fn!(name).is_ok(),
            ProgramEntry::Assertion(name) => get_assertion!(name).is_ok(),
        }
    }
}

impl Display for ProgramEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProgramEntry::Function(name) => write!(f, "Function '{name}'"),
            ProgramEntry::Assertion(name) => write!(f, "Assertion '{name}'"),
        }
    }
}

#[derive(Clone)]
struct Program {
    entries: Vec<ProgramEntry>,
}

impl Program {
    // TODO: Replace with a proper init method
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn set(&mut self, idx: usize, new: ProgramEntry) -> FunctionResult<()> {
        let existing = self
            .entries
            .get_mut(idx)
            .ok_or(FunctionFailure::OutOfProgramBounds(idx))?;

        *existing = new;

        Ok(())
    }

    pub fn push_fn(&mut self, fn_name: &str) {
        self.entries
            .push(ProgramEntry::Function(fn_name.to_string()));
    }

    pub fn push_assertion(&mut self, assertion_name: &str) {
        self.entries
            .push(ProgramEntry::Assertion(assertion_name.to_string()));
    }

    // End is exclusive
    pub fn iter(&self, start: usize, end: usize) -> impl Iterator<Item = &ProgramEntry> + '_ {
        self.entries.iter().skip(start).take(end - start)
    }

    // TODO: This returning FunctionResult is an indicator that might not be named well.
    pub fn get(&self, idx: usize) -> &ProgramEntry {
        // Should only be used from TrackedState, so the function idx must exist.
        &self.entries[idx]
    }
}

// TODO: Sumtype
type Value = usize;

#[derive(Clone)]
struct TrackedState {
    pub current: State,
    pub past: State,
    // Program keeps the full program text, from beginning to end.
    // Past and current are states that point to a function in program. States in past and current
    // change as they move along program. You can change program, but that does not cause current
    // to be recomputed. TODO: Maybe it should though.
    program: Program,
    current_program_index: usize,
    past_program_index: usize,
    past_offset: usize, // How far apart current and past should be (how far they really are apart starts at 0)
                        // Future is not special. It's just a branch where current becomes past, then the new current goes
                        // ahead.
}

impl Display for TrackedState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Display will not print the whole program. A seperate method will do that.
        writeln!(f, "Entries between past and current: ")?;
        for (i, entry) in self
            .program
            .iter(self.past_program_index, self.current_program_index)
            .enumerate()
        {
            let func_name = match entry {
                ProgramEntry::Function(name) => name,
                ProgramEntry::Assertion(name) => name,
            };
            writeln!(f, "{i}: {func_name}")?;
        }
        writeln!(f)?;
        writeln!(
            f,
            "Failing Entry: '{}\n",
            self.program.get(self.current_program_index)
        )?;
        writeln!(f, "Current: {}", self.current)?;
        writeln!(f, "Past: {}", self.past)
    }
}

#[derive(Debug)]
enum StateDesignator {
    Past,
    Current,
}

enum ProgramProgress {
    Done,
    NotDone,
}

impl TrackedState {
    fn new(program: Program, past_offset: usize) -> Self {
        Self {
            current: State::new(),
            past: State::new(),
            program,
            past_program_index: 0,
            current_program_index: 0,
            past_offset,
        }
    }

    fn step_state(&mut self, state_designator: StateDesignator) -> FunctionResult<()> {
        let (idx, state) = match state_designator {
            StateDesignator::Past => (self.past_program_index, &mut self.past),
            StateDesignator::Current => (self.current_program_index, &mut self.current),
        };

        match self.program.get(idx) {
            ProgramEntry::Function(fn_name) => get_fn!(fn_name)?(state)?,
            ProgramEntry::Assertion(assertion_name) => get_assertion!(assertion_name)?(self)
                .then(|| ())
                .ok_or_else(|| FunctionFailure::AssertionFailed(assertion_name.to_string()))?,
        };

        match state_designator {
            StateDesignator::Past => self.past_program_index += 1,
            StateDesignator::Current => self.current_program_index += 1,
        };

        Ok(())
    }

    fn step(&mut self) -> FunctionResult<ProgramProgress> {
        self.step_state(StateDesignator::Current)?;

        if (self.current_program_index - self.past_program_index) > self.past_offset {
            self.step_state(StateDesignator::Past)?;
        }

        if self.current_program_index == self.program.len() {
            Ok(ProgramProgress::Done)
        } else {
            Ok(ProgramProgress::NotDone)
        }
    }

    // TODO: Here we can insert changable elements.
    // and which operate on alternative executions
    fn sync_past_to_current(&mut self) -> FunctionResult<()> {
        while self.past_program_index < self.current_program_index {
            self.step_state(StateDesignator::Past)?;
        }

        Ok(())
        // TODO: Most common case is prolly to then assert equality (e.g. replace something in
        // past, then check if that was equivalent). But not always. Sometimes you just wanna
        // change the past.
    }

    fn change_program(
        &mut self,
        offset_from_past: usize,
        new_entry: ProgramEntry,
    ) -> FunctionResult<()> {
        assert!(new_entry.exists(), "{new_entry} does not exist");
        self.program
            .set(self.past_program_index + offset_from_past, new_entry)
    }

    fn skip_entry_for(&mut self, state_designator: StateDesignator) {
        match state_designator {
            StateDesignator::Past => self.past_program_index += 1,
            StateDesignator::Current => self.current_program_index += 1,
        };
        // TODO: This likely isn't enough
    }
}

#[derive(Debug)]
enum FunctionFailure {
    CallFailed(String),
    AssertionFailed(String),
    UnknownFunction(String),
    OutOfProgramBounds(usize),
}

impl Display for FunctionFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FunctionFailure::CallFailed(tracked_state) => {
                write!(f, "Application Failed: {tracked_state}")
            }
            FunctionFailure::AssertionFailed(tracked_state) => {
                write!(f, "Assertion failed: {tracked_state}")
            }
            FunctionFailure::UnknownFunction(fn_name) => write!(f, "Unknown function: '{fn_name}'"),
            FunctionFailure::OutOfProgramBounds(idx) => {
                write!(f, "Index '{idx}' is out of program bounds")
            }
        }
    }
}

// TODO: Generic
type FunctionResult<T> = Result<T, FunctionFailure>;

// TODO: Macro, and with variadic parameters
// TODO: Make Err generic
// TODO: Closures
// TODO: Make this be able to print its inner computations somehow. Prolly needs recursive
// definition in terms of other functions all the way down
type Function = (dyn Fn(&mut State) -> FunctionResult<()> + Sync + Send);
// Assertions have (immuatable) access to all the tracked state. TODO: Maybe even to more than one
// branch in the future
type Assertion = (dyn Fn(&TrackedState) -> bool + Sync + Send);

fn build_program() -> Program {
    let mut program = Program::new();

    {
        // Define functions we'll use in this program
        let mut funcs = FUNCTIONS.lock().unwrap();
        funcs.add_fn("increment", &|state| {
            state.num += 1;
            Ok(())
        });

        funcs.add_fn("multiply", &|state| {
            state.num *= 2;
            Ok(())
        });

        funcs.add_fn("fallible", &|_state| {
            Err(FunctionFailure::CallFailed("Unlucky".to_string()))
        });

        funcs.add_fn("assign_num", &|state| {
            state.assign_variable("num", state.num);
            Ok(())
        });

        funcs.add_assertion("increment_good", &|tracked| {
            tracked.current.num == *tracked.current.get_variable("num") + 1
        });

        funcs.add_assertion("multiply_good", &|tracked| {
            tracked.current.num == *tracked.current.get_variable("num") * 2
        });
    }

    // Build the program
    for _ in 0..200 {
        program.push_fn("assign_num");
        program.push_fn("increment");
        program.push_assertion("increment_good");
    }

    program.push_fn("fallible");

    for _ in 0..10 {
        program.push_fn("assign_num");
        program.push_fn("multiply");
        program.push_assertion("multiply_good");
    }

    program
}

enum UserChoice {
    BadInput,
    Skip,
    ExecutePastTillCurrent,
    ChangeProgramEntry {
        offset_from_past: usize,
        new_entry: ProgramEntry,
    },
    PrintState, // TODO: Allow choosing what to print
    Step(StateDesignator),
    ShowDefinedAssertions,
    ShowDefinedFunctions,
}

fn get_user_input(prompt: &str) -> String {
    let mut inp = String::new();

    println!(">{prompt}");

    stdin().read_line(&mut inp).unwrap();

    inp.strip_suffix('\n').unwrap().to_string()
}

fn get_user_choice() -> UserChoice {
    println!("step|skip current|e(xecute) past|p(rint) state|c(hange) past|functions|assertions");

    // TODO: Make this not just die with bad input. This must be robust
    let mut input = String::new();
    stdin().read_line(&mut input).unwrap();

    match input.as_str().strip_suffix('\n').unwrap() {
        "skip" => UserChoice::Skip,
        "e" | "execute" => UserChoice::ExecutePastTillCurrent,
        "p" | "print" => UserChoice::PrintState,
        "c" | "change" => {
            let offset_from_past: usize = str::parse(&get_user_input(
                "Enter the index of a program entry to change",
            ))
            .unwrap();

            // TODO: Should not let the user distinguish between entries. Just let them put a name,
            // good enough
            let sel = get_user_input(
                "Enter a(ssertion)|f(unction) and the name of the respective thing to insert.",
            );
            let sel = sel.split_whitespace().take(2).collect::<Vec<&str>>();

            let new_entry = match sel[0] {
                "a" | "assertion" => ProgramEntry::Assertion(sel[1].to_string()),
                "f" | "function" => ProgramEntry::Function(sel[1].to_string()),
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
        "functions" => UserChoice::ShowDefinedFunctions,
        "step" => match get_user_input("past|current").as_str() {
            "past" => UserChoice::Step(StateDesignator::Past),
            "current" => UserChoice::Step(StateDesignator::Current),
            _ => UserChoice::BadInput,
        },
        "step current" => UserChoice::Step(StateDesignator::Current),
        "step past" => UserChoice::Step(StateDesignator::Past),
        _ => UserChoice::BadInput,
    }
}

fn user_interaction(error: FunctionFailure, tracked: &mut TrackedState) -> FunctionResult<()> {
    println!("{}", tracked);
    println!("{}", error);

    let mut alternative_executions = Vec::new(); // TODO: Actually use this

    loop {
        match get_user_choice() {
            UserChoice::Skip => {
                tracked.skip_entry_for(StateDesignator::Current);
                break;
            }
            UserChoice::ExecutePastTillCurrent => {
                let mut alternative = tracked.clone();

                alternative.sync_past_to_current()?;
                println!(
                    "{}\nExecuted past(previous output is alternative current)...",
                    alternative
                );
                alternative_executions.push(alternative);
            }
            UserChoice::PrintState => println!("{}", tracked),
            UserChoice::ChangeProgramEntry {
                offset_from_past,
                new_entry,
            } => {
                if !new_entry.exists() {
                    println!(
                        "Entry '{}' does not exist. Cannot replace program with that.",
                        new_entry
                    );
                    continue;
                }
                // TODO: I'm just gonna change that in place for now, but it should really branch
                tracked.change_program(offset_from_past, new_entry).unwrap();

                println!("Changed past");
            }
            UserChoice::ShowDefinedAssertions => {
                println!("Defined Assertions: ");
                FUNCTIONS
                    .lock()
                    .unwrap()
                    .assertions
                    .iter()
                    .for_each(|assertion| {
                        println!("{}", assertion.0);
                    });
            }
            UserChoice::ShowDefinedFunctions => {
                println!("Defined Functions: ");
                FUNCTIONS
                    .lock()
                    .unwrap()
                    .functions
                    .iter()
                    .for_each(|function| {
                        println!("{}", function.0);
                    });
            }
            UserChoice::BadInput => println!("Bad input..."),
            UserChoice::Step(state_designator) => tracked.step_state(state_designator)?,
        }
    }
    Ok(())
}

fn handle_error(error: FunctionFailure, tracked: &mut TrackedState) {
    match user_interaction(error, tracked) {
        Ok(()) => (),
        Err(err) => handle_error(err, tracked),
    }
}

fn main() {
    let file_content = std::fs::read_to_string("main.pause").unwrap();

    let tokens = lexer::Lexer::lex(file_content.as_bytes());

    println!("{tokens:#?}");

    let parser = Parser::new(tokens);

    let program = parser.parse().unwrap();

    println!("{:#?}", program);

    exit(0);

    let program = build_program();

    let mut tracked = TrackedState::new(program, 50);

    loop {
        match tracked.step() {
            Ok(ProgramProgress::Done) => {
                println!("Done:\n{}", tracked);
                break;
            }
            Ok(ProgramProgress::NotDone) => continue,
            Err(error) => handle_error(error, &mut tracked),
        }
    }

    // TODO: Next steps:
    // - Make errors that happen in past and current distinguishable
    // - Make a simple parser so you can dynamically load source files to define functions.
    // - Try to implement just enough so you can solve the simplest leetcode
}
