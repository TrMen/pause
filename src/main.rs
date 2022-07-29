#![feature(never_type)]
#![feature(backtrace)]

use std::process::exit;

use crate::{meta_interpreter::MetaInterpreter, parser::Parser};
mod interpreter;
mod lexer;
mod meta_interpreter;
mod parser;
mod typechecker;
mod util;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file_content = std::fs::read_to_string("main.pause").unwrap();

    let tokens = lexer::Lexer::lex(file_content.as_bytes());

    println!("{tokens:#?}");

    let parser = Parser::new(tokens);

    let (state, program) = parser.parse().unwrap_or_else(|err| {
        println!("{:#?}", err);

        exit(1);
    });

    println!("{:#?}", program);

    let mut meta = MetaInterpreter::new(state, program);

    loop {
        match meta.step()? {
            Some(_) => continue,
            None => {
                println!("{}", meta);
                println!("\n\nDone!");
                break;
            }
        }
    }

    Ok(())

    // TODO: Next steps:
    // - Make errors that happen in past and current distinguishable
    // - Make a simple parser so you can dynamically load source files to define functions.
    // - Try to implement just enough so you can solve the simplest leetcode
}
