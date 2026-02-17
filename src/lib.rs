//! Rusty Agent SDK — a Rust-powered Python SDK for LLM text generation.
//!
//! This crate exposes a Python extension module via PyO3.

use pyo3::prelude::*;

mod errors;
mod generate;
mod http;
mod models;
mod provider;
mod stream;

pub use provider::Provider;
pub use stream::TextStream;

#[doc(hidden)]
pub mod internal {
    pub use crate::models::{
        ChatMessage, ChatRequest, GenerationParams, StreamEvent, api_error_message,
        parse_chat_response, parse_sse_event, parse_sse_line,
    };
    pub use crate::provider::{
        build_chat_completions_url, resolve_provider_values, resolve_runtime_config,
    };
}

#[pymodule]
mod rusty_agent_sdk {
    #[pymodule_export]
    use super::Provider;

    #[pymodule_export]
    use super::TextStream;
}
