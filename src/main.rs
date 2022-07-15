use std::{collections::HashMap, fmt::Display, io::stdin};

use rand::Rng;

#[macro_use]
extern crate lazy_static;

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
}

impl State {
    pub fn new() -> Self {
        Self { num: 0 }
    }
}

impl Display for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Clone)]
struct TrackedState {
    pub current: State,
    pub past: State,
    past_to_current: CircularBuffer<&'static PrintableFunction>,
    // pub future: Option<State>, TODO:
}

impl Display for TrackedState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Current: {}", self.current)?;
        writeln!(f, "Past: {}", self.past)?;
        writeln!(f, "Applied functions: {}", self.past_to_current)
    }
}

impl TrackedState {
    fn new(past_offset: usize) -> Self {
        Self {
            current: State::new(),
            past: State::new(),
            past_to_current: CircularBuffer::new(past_offset),
        }
    }

    fn as_computation_result(&mut self, result: Result<(), String>) -> ComputationResult<()> {
        result.map_err(|reason| {
            println!("Breaking with reason: '{reason}'");
            ComputationError::ApplicationFailed(std::mem::replace(
                self,
                TrackedState::new(self.past_to_current.max_size),
            ))
        })
    }

    fn apply(&mut self, func: &'static PrintableFunction) -> ComputationResult<()> {
        // Stage computation so it shows up if any call here fails
        self.past_to_current.stage(func);

        if self.past_to_current.is_full() {
            let res = (func.f)(&mut self.current);
            self.as_computation_result(res)?; // TODO: There has to be a better way to do this
        } else {
            let res = (func.f)(&mut self.current);
            self.as_computation_result(res)?;
            let res = (func.f)(&mut self.past);
            self.as_computation_result(res)?;
        }

        self.past_to_current.insert_stage();

        Ok(())
    }

    fn assert(&mut self, assertion: impl FnOnce(&State) -> bool) -> ComputationResult<()> {
        if !assertion(&self.current) {
            // TODO: Prolly too many copies in the long run
            Err(ComputationError::AssertionFailed(std::mem::replace(
                self,
                TrackedState::new(self.past_to_current.max_size),
            )))
        } else {
            Ok(())
        }
    }

    // TODO: Here we can insert changable elements.
    // TODO: Need to cleanly separate which functions destroy the inner state of this object
    // and which operate on alternative executions
    fn sync_past_to_current(&mut self) -> ComputationResult<()> {
        let mut result = Ok(());
        // Weird construt to avoid borrow checking rules
        for past_fn in self.past_to_current.iter() {
            match (past_fn.f)(&mut self.past) {
                Ok(_) => continue,
                Err(fail) => {
                    result = Err(fail);
                    break;
                }
            }
        }

        self.as_computation_result(result)
        // TODO: Most common case is prolly to then assert equality (e.g. replace something in
        // past, then check if that was equivalent). But not always. Sometimes you just wanna
        // change the past.
    }

    // TODO: Unify terminology around computation/function/what's recorded and what's applicable
    fn change_past_computation(&mut self, index: usize, new: &'static PrintableFunction) {
        self.past_to_current.replace(index, new);
    }
}

enum ComputationError {
    ApplicationFailed(TrackedState),
    AssertionFailed(TrackedState),
}

impl ComputationError {
    fn take_tracked_state(self) -> TrackedState {
        match self {
            ComputationError::ApplicationFailed(tracked_state) => tracked_state,
            ComputationError::AssertionFailed(tracked_state) => tracked_state,
        }
    }
    fn get_tracked_state_mut(&mut self) -> &mut TrackedState {
        match self {
            ComputationError::ApplicationFailed(tracked_state) => tracked_state,
            ComputationError::AssertionFailed(tracked_state) => tracked_state,
        }
    }
    fn get_tracked_state(&self) -> &TrackedState {
        match self {
            ComputationError::ApplicationFailed(tracked_state) => tracked_state,
            ComputationError::AssertionFailed(tracked_state) => tracked_state,
        }
    }
}

impl Display for ComputationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComputationError::ApplicationFailed(tracked_state) => {
                write!(f, "Application Failed: {}", tracked_state)
            }
            ComputationError::AssertionFailed(tracked_state) => {
                write!(f, "Assertion failed: {}", tracked_state)
            }
        }
    }
}

// TODO: Generic
type ComputationResult<T> = Result<T, ComputationError>;

// TODO: Macro, and with variadic parameters
// TODO: Make Err generic
struct PrintableFunction {
    name: &'static str,
    f: &'static (dyn Fn(&mut State) -> Result<(), String> + Sync),
}

impl PrintableFunction {
    pub fn new(
        name: &'static str,
        f: &'static (dyn Fn(&mut State) -> Result<(), String> + Sync),
    ) -> Self {
        Self { f, name }
    }
}

impl Display for PrintableFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}()", self.name)
    }
}

lazy_static! {
    static ref INCREMENT: PrintableFunction = PrintableFunction::new("increment", &|state| {
        state.num += 1;

        if rand::thread_rng().gen_range(0..100) == 1 {
            state.num += 1;
        }

        Ok(())
    });
    static ref MULTIPLY: PrintableFunction = PrintableFunction::new("multiply", &|state| {
        state.num *= 2;
        Ok(())
    });
    static ref FALLIBLE_SUB: PrintableFunction = PrintableFunction::new("fallible", &|state| {
        if rand::thread_rng().gen_range(0..5) == 1 {
            Err("Unlucky".to_string())
        } else {
            state.num -= 1;
            Ok(())
        }
    });
}

lazy_static! {
    static ref PRINTABLE_FUNCTIONS_ARRAY: [&'static PrintableFunction; 3] =
        [&INCREMENT, &MULTIPLY, &FALLIBLE_SUB];
    static ref PRINTABLE_FUNCTIONS: HashMap<&'static str, &'static PrintableFunction> =
        HashMap::from(PRINTABLE_FUNCTIONS_ARRAY.map(|f| (f.name, f)));
}

fn computation(tracked: &mut TrackedState) -> ComputationResult<()> {
    for _ in 0..200 {
        let num = tracked.current.num;

        tracked.apply(&INCREMENT)?;

        tracked.assert(|current| current.num == num + 1)?
    }

    for _ in 0..10 {
        let num = tracked.current.num;
        tracked.apply(&MULTIPLY)?;
        tracked.assert(|current| current.num == num * 2)?;
    }

    for _ in 0..2 {
        let num = tracked.current.num;
        tracked.apply(&FALLIBLE_SUB)?;
        tracked.assert(|current| current.num == num - 1)?;
    }

    Ok(())
}

enum UserChoice {
    Ignore,
    ExecutePastTillCurrent,
    ChangePastComputation { index: usize, new_fn_name: String },
    PrintState, // TODO: Allow choosing what to print
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

            UserChoice::ChangePastComputation { index, new_fn_name }
        }
        _ => UserChoice::PrintState,
    }
}

fn user_interaction(mut error: ComputationError) -> TrackedState {
    println!("{}", error);

    let mut alternative_executions = Vec::new(); // TODO: Actually use this
    let tracked_state = error.get_tracked_state_mut();

    loop {
        println!("i(gnore)|e(xecute) past|p(rint) state|c(hange) past");

        match get_user_choice() {
            UserChoice::Ignore => break,
            UserChoice::ExecutePastTillCurrent => {
                let mut alternative = tracked_state.clone();

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
            UserChoice::PrintState => println!("{}", tracked_state),
            UserChoice::ChangePastComputation { index, new_fn_name } => {
                let new_func = if let Some(func) = PRINTABLE_FUNCTIONS.get(new_fn_name.as_str()) {
                    func
                } else {
                    continue;
                };
                // TODO: I'm just gonna change that in place for now, but it should really branch
                tracked_state.change_past_computation(index, new_func);

                println!("Changed past");
            }
        }
    }

    error.take_tracked_state()
}

fn main() {
    let mut current = TrackedState::new(50);

    loop {
        match computation(&mut current) {
            Ok(()) => {
                println!("Done:\n{}", current);
                break;
            }
            Err(paused) => current = user_interaction(paused),
        }
    }
}
