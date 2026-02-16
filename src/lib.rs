use pyo3::prelude::*;

/// Greets a user by name.
#[pyfunction]
fn greet(name: &str) -> String {
    format!("Hello, {name}! Welcome to Rusty Agent SDK.")
}

/// A Python module implemented in Rust.
#[pymodule]
mod rusty_agent_sdk {
    #[pymodule_export]
    use super::greet;
}
