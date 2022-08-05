#![feature(never_type)]
#![feature(backtrace)]
#![feature(variant_count)]
#![feature(let_chains)]

use std::process::exit;

// use crate::{meta_interpreter::MetaInterpreter, parser::Parser};
use crate::{parser::Parser, typechecker::typecheck_program};
// mod interpreter;
mod lexer;
// mod meta_interpreter;
mod codegen;
mod parser;
mod typechecker;
mod util;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file_content = std::fs::read_to_string("main.pause").unwrap();

    let tokens = lexer::Lexer::lex(file_content.as_bytes());

    let parser = Parser::new(tokens);

    let (state, program) = parser.parse().unwrap_or_else(|err| {
        println!("{}\n{err}", err.backtrace);

        exit(1);
    });

    let checked_program = typecheck_program(program);

    if let Err(err) = checked_program {
        println!("{}\n{err}", err.backtrace);
    } else {
        println!("SUCCESS!!!");
        let ast = serde_json::to_string(&checked_program?)?;

        std::fs::write("ast", ast)?;
    }

    //let codegen = typecheck_program(checked_program.unwrap());

    //    match codegen {
    //        Ok(_) => {
    //            println!("SUCCESS!!!")
    //        }
    //        Err(err) => {
    //            println!("{}\n{err}", err.backtrace);
    //        }
    //    }
    //
    /*

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

    */

    Ok(())

    // TODO: Next steps:
    // - Make errors that happen in past and current distinguishable
    // - Make a simple parser so you can dynamically load source files to define functions.
    // - Try to implement just enough so you can solve the simplest leetcode
}
