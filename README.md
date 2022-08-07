# pause

Pause is an early-stage, experimental, programming language. Its main idea is to allow interactive bug-solving while running the program. 
For this, when an assertion failure is hit, the program will not abort, but instead pause execution and drop the user into a repl where they can inspect the program
state and do various operations to recover correct state and fix the underlying issue. 

## Current state of development
It's still very early stages. Currently, there is no ability to execute Pause programs. Only parsing and typechecking is implemented so far. 
The specifics of how the runtime will work are not decided yet. 
Likely, I will ahead-of-time typecheck and compile the code to LLVM IR, but I'm only just starting with codegen.


## Planned capabilities
- To allow human inspection in case of error, store debug information in the program (potentially externally later).
- While the program runs, it always stores a copy of the current application state as it was x (state-changing) instructions ago.
    - When an error is encountered, the user can compare this past state and the current state, and inspect instructions between them.
    - The specifics of how much past state is stored is not decided yet. Possible ideas include:
        - Storing a fixed number of instructions, configurable at compile-time/run-time.
        - Letting the programmer annotate when to store the current state and drop past states.
        - Storing as much as fits into a given memory limit, and using deduplication and cache-eviction strategies to clean up as needed.

When the program pauses, I plan to give the user the following capabilities (non-exhaustive):
- Inspecting all program state. To make this easy, all state the application as must be stored in a `state` struct.
- Run expressions against the current state. For this, all `function`s in Pause are purely functional. State changes are done in `procedure`s.
- Change state. Either explicitly or by running a `procedure`.
- Change the instructions between the past and current state.
    - And then run an alternative execution from past to present, to see what the resulting state is.
- Allow changing the source code at runtime.
    - This is obviously tricky, because LLVM might inline anything. So it will probably require some additional debug information.
    - The idea is that you change the source code at runtime, rerun past-to-current and then check if something sensible came out.
    - If yes, you can then store those changes. After the interactive session ends, they are stored as diffs on disk, and you can choose to include them in the real source code.
- Storing a snapshot of state for later comparisson.
- Continuing execution of the current timeline.




The language has many ideas, many of which I will probably throw out later. For now, a sample of the syntax can be found in `main.pause`.
