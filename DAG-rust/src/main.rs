fn main() {
    match dag_rust::interfaces::cli::run() {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            eprintln!("dag-rust: {error}");
            std::process::exit(1);
        }
    }
}
