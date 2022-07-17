use once_cell::sync::Lazy;
use std::{collections::HashMap, fmt::Display, io::stdin, sync::Mutex};

// TODO: Do I even need cicular buffer anywhere?
#[derive(Clone)]
pub struct CircularBuffer<T> {
    buffer: Vec<T>,
    pub max_size: usize,
    // Cursor always points at the next elem to be inserted
    // e.g. the oldest element in the buffer
    cursor: usize,
    stage: Option<T>,
}

impl<T: Display> Display for CircularBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stage_str = self
            .stage
            .as_ref()
            .map(|s| format!("{}", s))
            .unwrap_or_else(|| "None".to_string());

        writeln!(f, "{{ ({}/{})", self.cursor, self.max_size)?;

        write!(f, "[")?;
        let mut offset = 0;
        // Print from oldest to newest computation
        // TODO: When arguments, get them printed in here somehow
        for i in self.cursor..self.buffer.len() {
            writeln!(f, "{offset}:\t{},", self.buffer[i])?;
            offset += 1;
        }
        for i in 0..self.cursor {
            writeln!(f, "{offset}:\t{},", self.buffer[i])?;
            offset += 1;
        }
        write!(f, "]")?;

        writeln!(f, "Stage: '{}' }}", stage_str)
    }
}

impl<T> CircularBuffer<T> {
    fn new(size: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(size),
            max_size: size,
            cursor: 0,
            stage: None,
        }
    }

    pub fn newest(&self) -> Option<&T> {
        if self.cursor == 0 {
            self.buffer.get(self.buffer.len())
        } else {
            self.buffer.get(self.cursor - 1)
        }
    }

    pub fn oldest(&self) -> Option<&T> {
        if self.is_full() {
            Some(&self.buffer[self.cursor])
        } else {
            self.buffer.get(0)
        }
    }

    pub fn insert(&mut self, value: T) {
        // TODO: I'm sure there's a better way to do this
        if !self.is_full() {
            self.buffer.push(value);
            self.cursor = (self.cursor + 1) % self.max_size;
        } else {
            self.buffer[self.cursor] = value;
            self.cursor = (self.cursor + 1) % self.max_size;
        }
    }

    pub fn stage(&mut self, value: T) {
        self.stage = Some(value);
    }

    pub fn insert_stage(&mut self) {
        assert!(self.stage.is_some());
        let staged = self.stage.take().unwrap();
        self.insert(staged);
    }

    pub fn change_max_size(&mut self, new_size: usize) {
        self.buffer.reserve(new_size);
        self.max_size = new_size;
    }

    pub fn replace(&mut self, offset_from_oldest: usize, new: T) {
        assert!(offset_from_oldest <= self.buffer.len()); // Circular behavior is probably a bug in the caller
        let idx = (self.cursor + offset_from_oldest) % self.buffer.len();
        self.buffer[idx] = new;
    }

    pub fn is_full(&self) -> bool {
        self.buffer.len() == self.max_size
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> + '_ {
        self.buffer
            .iter()
            .skip(self.cursor)
            .chain(self.buffer.iter().take(self.cursor))
    }

    // TODO: Can make an endless circular buffer method (esp a mut one)
}

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

    pub fn add_fn(&mut self, name: &str, func: &Function) {
        self.functions.insert(name.to_string(), Box::new(func));
    }

    pub fn get_assertion(&self, name: &str) -> Option<&Assertion> {
        // TODO: The double refer seems a bit weird
        self.assertions.get(name).map(|v| &**v)
    }

    pub fn add_assertion(&mut self, name: &str, func: &Assertion) {
        self.assertions.insert(name.to_string(), Box::new(func));
    }
}

static FUNCTIONS: Lazy<Mutex<Functions>> = Lazy::new(|| Mutex::new(Functions::new()));

fn get_fn(name: &String) -> FunctionResult<&Function> {
    FUNCTIONS
        .lock()
        .unwrap()
        .get_fn(name)
        .ok_or_else(|| FunctionFailure::UnknownFunction(name.clone()))
}

fn get_assertion(name: &String) -> FunctionResult<&Assertion> {
    FUNCTIONS
        .lock()
        .unwrap()
        .get_assertion(name)
        .ok_or_else(|| FunctionFailure::UnknownFunction(name.clone()))
}

fn add_fn(name: &str, func: &Function) {
    FUNCTIONS.lock().unwrap().add_fn(name, func);
}

#[derive(Clone)]
struct Program {
    entries: Vec<ProgramEntry>,
    idx: usize,
}

#[derive(Clone)]
enum ProgramEntry {
    Function(String),
    Assertion(String),
}

impl Program {
    // TODO: Replace with a proper init method
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            idx: 0,
        }
    }

    pub fn set(&mut self, idx: usize, new: ProgramEntry) -> FunctionResult<()> {
        let mut existing = self
            .entries
            .get_mut(idx)
            .ok_or_else(|| FunctionFailure::OutOfProgramBounds(idx))?;

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

    // End is inclusive
    pub fn iter(&self, start: usize, end: usize) -> impl Iterator<Item = &ProgramEntry> + '_ {
        self.entries.iter().skip(start).take(end - start + 1)
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
    current_function_index: usize,
    past_function_index: usize,
    past_offset: usize, // How far apart current and past should be (how far they really are apart starts at 0)
                        // Future is not special. It's just a branch where current becomes past, then the new current goes
                        // ahead.
}

impl Display for TrackedState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Display will not print the whole program. A seperate method will do that.
        writeln!(f, "Current: {}", self.current)?;
        writeln!(f, "Past: {}", self.past)?;
        write!(f, "Functions between past and current: ")?;
        for entry in self
            .program
            .iter(self.current_function_index, self.past_function_index)
        {
            let func_name = match entry {
                ProgramEntry::Function(name) => name,
                ProgramEntry::Assertion(name) => name,
            };
            write!(f, "{}", func_name)?;
        }
        writeln!(f)
    }
}

enum StateDesignator {
    Past,
    Current,
}

impl TrackedState {
    fn new(program: Program, past_offset: usize) -> Self {
        Self {
            current: State::new(),
            past: State::new(),
            program,
            past_function_index: 0,
            current_function_index: 0,
            past_offset,
        }
    }

    fn step_state(&mut self, state: StateDesignator) -> FunctionResult<()> {
        let (idx, state) = match state {
            StateDesignator::Past => (&mut self.past_function_index, &self.past),
            StateDesignator::Current => (&mut self.current_function_index, &self.current),
        };

        match self.program.get(*idx) {
            ProgramEntry::Function(fn_name) => get_fn(fn_name)?(&mut self.current)?,
            ProgramEntry::Assertion(assertion_name) => get_assertion(assertion_name)?(&self)?,
        };

        *idx += 1;

        Ok(())
    }

    fn step(&mut self) -> FunctionResult<()> {
        self.step_state(StateDesignator::Current)?;

        if self.current_function_index - self.past_function_index < self.past_offset {
            self.step_state(StateDesignator::Past)?;
        }

        Ok(())
    }

    fn assert(&self, assertion: impl FnOnce(&TrackedState) -> bool) -> FunctionResult<()> {
        if !assertion(self) {
            // TODO: Prolly too many copies in the long run
            Err(FunctionFailure::AssertionFailed(self.clone()))
        } else {
            Ok(())
        }
    }

    fn past_to_current(&self) -> impl Iterator<Item = &ProgramEntry> + '_ {
        self.program
            .iter(self.past_function_index, self.current_function_index)
    }

    // TODO: Here we can insert changable elements.
    // and which operate on alternative executions
    fn sync_past_to_current(&mut self) -> FunctionResult<()> {
        while self.past_function_index < self.current_function_index {
            self.step_state(StateDesignator::Past)?;
        }

        Ok(())
        // TODO: Most common case is prolly to then assert equality (e.g. replace something in
        // past, then check if that was equivalent). But not always. Sometimes you just wanna
        // change the past.
    }

    fn change_program(&mut self, index: usize, new_entry: ProgramEntry) {
        self.program.set(index, new_entry);
    }
}

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
        let funcs = FUNCTIONS.lock().unwrap();
        funcs.add_fn("increment", &|state| {
            state.num += 1;
            Ok(())
        });

        funcs.add_fn("multiply", &|state| {
            state.num *= 2;
            Ok(())
        });

        funcs.add_fn("fallible", &|state| {
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
    Ignore,
    ExecutePastTillCurrent,
    ChangeProgramEntry { index: usize, new_fn_name: String },
    PrintState, // TODO: Allow choosing what to print
    ShowDefinedAssertions,
    ShowDefinedFunctions,
}

fn get_user_choice() -> UserChoice {
    // TODO: Make this not just die with bad input. This must be robust
    let mut input = String::new();
    stdin().read_line(&mut input).unwrap();

    match input.as_str().strip_suffix('\n').unwrap() {
        "i" | "ignore" => UserChoice::Ignore,
        "e" | "execute" => UserChoice::ExecutePastTillCurrent,
        "p" | "print" => UserChoice::PrintState,
        "c" | "change" => {
            println!("Enter index of func to change and new function name");
            let mut selection = String::new();
            stdin().read_line(&mut selection).unwrap();

            let sel = selection.split_whitespace().take(2).collect::<Vec<&str>>();

            let index = str::parse::<usize>(sel[0]).unwrap();
            let new_fn_name = sel[1].to_string();

            UserChoice::ChangeProgramEntry { index, new_fn_name }
        }
        _ => UserChoice::PrintState,
    }
}

fn user_interaction(mut error: FunctionFailure, tracked: &mut TrackedState) {
    println!("{}", error);

    let mut alternative_executions = Vec::new(); // TODO: Actually use this

    loop {
        println!("i(gnore)|e(xecute) past|p(rint) state|c(hange) past");

        match get_user_choice() {
            UserChoice::Ignore => break,
            UserChoice::ExecutePastTillCurrent => {
                let mut alternative = tracked.clone();

                alternative.sync_past_to_current().unwrap_or_else(|state| {
                    // TODO: Don't unwrap, allow another user interaction loop
                    println!("Failed interactively: {}", state);
                    todo!("Failing execute past end")
                });
                println!(
                    "{}Executed past(previous out is alternative current)",
                    alternative
                );
                alternative_executions.push(alternative);
            }
            UserChoice::PrintState => println!("{}", tracked),
            UserChoice::ChangeProgramEntry { index, new_fn_name } => {
                let new_func = if let Some(func) = get_fn(new_fn_name.as_str()) {
                    func
                } else {
                    continue;
                };
                // TODO: I'm just gonna change that in place for now, but it should really branch
                tracked.change_program(index, new_func);

                println!("Changed past");
            }
            UserChoice::ShowDefinedAssertions => {
                println!("Defined Functions: ");
                for function in FUNCTIONS.lock().unwrap().functions {
                    println!("{}", function.0);
                }
            }
            UserChoice::ShowDefinedFunctions => {
                println!("Defined Assertions: ");
                for assertion in FUNCTIONS.lock().unwrap().assertions {
                    println!("{}", assertion.0);
                }
            }
        }
    }
}

fn main() {
    let program = build_program();

    let mut tracked = TrackedState::new(program, 50);

    loop {
        match tracked.step() {
            Ok(()) => {
                println!("Done:\n{}", tracked);
                break;
            }
            Err(error) => user_interaction(error, &mut tracked),
        }
    }
}
