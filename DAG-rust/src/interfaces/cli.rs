//! Command-line interface shell.

use crate::foundations::error::{DagError, Result};

pub fn run() -> Result<i32> {
    run_from(std::env::args())
}

pub fn run_from<I, S>(args: I) -> Result<i32>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let args = args.into_iter().map(Into::into).collect::<Vec<_>>();
    match args.get(1).map(String::as_str) {
        None | Some("--help") | Some("-h") => {
            print_help();
            Ok(0)
        }
        Some("build" | "merge" | "validate" | "stats" | "export" | "export-adphmm") => Err(
            DagError::UnsupportedOperation("CLI command is scaffolded but not implemented yet"),
        ),
        Some(_) => Err(DagError::UnsupportedOperation("unknown CLI command")),
    }
}

fn print_help() {
    println!("dag-rust\n\nCommands: build, merge, validate, stats, export, export-adphmm\n");
}
