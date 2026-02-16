use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// Configuration for an OpenAI-compatible LLM API provider.
///
/// Holds the API key and base URL needed to authenticate and route requests
/// to any OpenAI-compatible chat completions endpoint. By default, requests
/// are sent to OpenRouter (https://openrouter.ai/api/v1).
///
/// The API key can be supplied explicitly or read from the
/// ``OPENROUTER_API_KEY`` environment variable. If neither is available,
/// a ``ValueError`` is raised at construction time.
///
/// Examples:
///     Using the environment variable (recommended)::
///
///         from dotenv import load_dotenv
///         from rusty_agent_sdk import Provider
///
///         load_dotenv()  # loads OPENROUTER_API_KEY from .env
///         provider = Provider()
///
///     Passing an explicit key and custom endpoint::
///
///         provider = Provider(
///             api_key="sk-...",
///             base_url="https://api.openai.com/v1",
///         )
///
/// Notes:
///     - The provider is immutable after creation; create a new instance
///       to change the key or URL.
///     - The API key is never printed in ``repr()`` output for safety.
#[pyclass(from_py_object)]
#[derive(Clone)]
struct Provider {
    api_key: String,
    base_url: String,
}

#[pymethods]
impl Provider {
    /// Create a new Provider.
    ///
    /// Args:
    ///     api_key (str | None): API key for the LLM service. If ``None``,
    ///         the ``OPENROUTER_API_KEY`` environment variable is used.
    ///     base_url (str | None): Base URL of the OpenAI-compatible API.
    ///         Defaults to ``"https://openrouter.ai/api/v1"``.
    ///
    /// Returns:
    ///     Provider: A configured provider instance.
    ///
    /// Raises:
    ///     ValueError: If no ``api_key`` is provided and the
    ///         ``OPENROUTER_API_KEY`` environment variable is not set.
    ///
    /// Examples:
    ///     Default (reads env var)::
    ///
    ///         provider = Provider()
    ///
    ///     Explicit key::
    ///
    ///         provider = Provider(api_key="sk-...")
    ///
    ///     Custom endpoint (e.g. OpenAI direct)::
    ///
    ///         provider = Provider(
    ///             api_key="sk-...",
    ///             base_url="https://api.openai.com/v1",
    ///         )
    #[new]
    #[pyo3(signature = (*, api_key=None, base_url=None))]
    #[pyo3(text_signature = "(*, api_key=None, base_url=None)")]
    fn new(api_key: Option<String>, base_url: Option<String>) -> PyResult<Self> {
        let base_url = base_url.unwrap_or_else(|| "https://openrouter.ai/api/v1".to_string());

        let api_key = match api_key {
            Some(key) => key,
            None => std::env::var("OPENROUTER_API_KEY").map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(
                    "No api_key provided and OPENROUTER_API_KEY environment variable is not set.",
                )
            })?,
        };

        Ok(Provider { api_key, base_url })
    }

    fn __repr__(&self) -> String {
        format!("Provider(base_url='{}')", self.base_url)
    }
}

// --- OpenAI-compatible request/response types ---

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatResponseMessage,
}

#[derive(Deserialize)]
struct ChatResponseMessage {
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ErrorDetail {
    message: String,
}

#[derive(Deserialize)]
struct ErrorResponse {
    error: ErrorDetail,
}

/// Generate a complete text response from an LLM (blocking).
///
/// Sends a single user message to the chat completions endpoint and blocks
/// until the full response is available. This is the simplest way to get a
/// response from a model when you don't need streaming.
///
/// Under the hood, a temporary Tokio runtime is created for the async HTTP
/// call so the function can be used from synchronous Python code without
/// ``asyncio``.
///
/// Args:
///     provider (Provider): The provider configuration (API key + base URL).
///     model (str): Model identifier, e.g. ``"openai/gpt-4o-mini"`` or
///         ``"anthropic/claude-sonnet-4-5-20250514"``.
///     prompt (str): The user message to send to the model.
///
/// Returns:
///     str: The model's complete text response.
///
/// Raises:
///     ConnectionError: If the HTTP request to the API fails (network error,
///         DNS resolution failure, timeout, etc.).
///     RuntimeError: If the API returns a non-2xx status code. The error
///         message includes the HTTP status and the API's error description
///         when available.
///     ValueError: If the API returns a 2xx response but the body cannot be
///         parsed, or if the response contains no choices.
///
/// Examples:
///     Basic usage::
///
///         from rusty_agent_sdk import Provider, generate_text
///
///         provider = Provider(api_key="sk-...")
///         response = generate_text(provider, "openai/gpt-4o-mini", "Hello!")
///         print(response)
///
///     With a custom endpoint::
///
///         provider = Provider(
///             api_key="sk-...",
///             base_url="https://api.openai.com/v1",
///         )
///         response = generate_text(provider, "gpt-4o-mini", "Hello!")
///
/// Notes:
///     - Only the ``user`` role is sent. System/assistant messages are not
///       yet supported.
///     - A new HTTP client and Tokio runtime are created per call. For
///       high-throughput use cases, prefer ``stream_text`` or batch at the
///       application level.
#[pyfunction]
#[pyo3(text_signature = "(provider, model, prompt)")]
fn generate_text(provider: &Provider, model: &str, prompt: &str) -> PyResult<String> {
    let url = format!("{}/chat/completions", provider.base_url);

    let body = ChatRequest {
        model: model.to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        }],
    };

    let api_key = provider.api_key.clone();

    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    rt.block_on(async {
        let client = reqwest::Client::new();

        let response = client
            .post(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| pyo3::exceptions::PyConnectionError::new_err(e.to_string()))?;

        let status = response.status();
        let response_text = response
            .text()
            .await
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        if !status.is_success() {
            if let Ok(err) = serde_json::from_str::<ErrorResponse>(&response_text) {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "API error ({}): {}",
                    status, err.error.message
                )));
            }
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "API error ({}): {}",
                status, response_text
            )));
        }

        let chat_response: ChatResponse = serde_json::from_str(&response_text)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!(
                "Failed to parse response: {}",
                e
            )))?;

        chat_response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("No choices returned in API response")
            })
    })
}

// --- SSE streaming response types ---

#[derive(Serialize)]
struct StreamChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
}

#[derive(Deserialize)]
struct DeltaMessage {
    content: Option<String>,
}

#[derive(Deserialize)]
struct StreamChoice {
    delta: DeltaMessage,
}

#[derive(Deserialize)]
struct StreamChunk {
    choices: Vec<StreamChoice>,
}

/// An iterator that yields text chunks from a streaming LLM response.
///
/// ``TextStream`` implements Python's iterator protocol, so you can use it
/// in a ``for`` loop or call ``next()`` on it manually. Each iteration
/// yields the next content fragment as a ``str``.
///
/// The stream is backed by a background thread that reads Server-Sent
/// Events (SSE) from the API and forwards parsed content deltas through an
/// internal channel. Iteration blocks until the next chunk arrives or the
/// stream ends.
///
/// You do not construct ``TextStream`` directly — it is returned by
/// ``stream_text()``.
///
/// Examples:
///     Print chunks as they arrive::
///
///         for chunk in stream_text(provider, model, prompt):
///             print(chunk, end="", flush=True)
///         print()  # newline after stream ends
///
///     Collect the full response::
///
///         full = "".join(stream_text(provider, model, prompt))
///
/// Raises:
///     RuntimeError: If the background streaming thread encounters an error
///         (e.g. a malformed SSE event or a dropped connection mid-stream),
///         the error is raised on the **next** call to ``__next__``.
///
/// Notes:
///     - The iterator is single-use. Once exhausted, calling ``next()``
///       will raise ``StopIteration``.
///     - The background thread is automatically cleaned up when the stream
///       ends or when the ``TextStream`` object is garbage-collected.
#[pyclass]
struct TextStream {
    receiver: std::sync::Mutex<std::sync::mpsc::Receiver<PyResult<String>>>,
    _handle: Option<std::thread::JoinHandle<()>>,
}

#[pymethods]
impl TextStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self) -> Option<PyResult<String>> {
        let rx = self.receiver.lock().unwrap();
        match rx.recv() {
            Ok(chunk) => Some(chunk),
            Err(_) => None,
        }
    }
}

/// Stream text from an LLM, returning an iterator that yields chunks in
/// real time.
///
/// Sends a chat completions request with ``stream: true`` and returns a
/// ``TextStream`` iterator. Each call to ``next()`` on the iterator blocks
/// until the next content chunk arrives from the API's Server-Sent Events
/// (SSE) stream.
///
/// A dedicated background thread is spawned to handle the async HTTP
/// streaming, keeping the Python thread free to process chunks as they
/// arrive. Communication between the background thread and the iterator
/// uses a bounded channel.
///
/// Args:
///     provider (Provider): The provider configuration (API key + base URL).
///     model (str): Model identifier, e.g. ``"openai/gpt-4o-mini"`` or
///         ``"anthropic/claude-sonnet-4-5-20250514"``.
///     prompt (str): The user message to send to the model.
///
/// Returns:
///     TextStream: An iterator yielding ``str`` chunks. Use in a ``for``
///         loop or call ``next()`` manually.
///
/// Raises:
///     ConnectionError: If the initial HTTP connection to the API fails.
///     RuntimeError: If the API returns a non-2xx status code before
///         streaming begins. The error is raised on the **first** call to
///         ``next()`` on the returned ``TextStream``.
///
/// Examples:
///     Print a streaming response::
///
///         from rusty_agent_sdk import Provider, stream_text
///
///         provider = Provider(api_key="sk-...")
///         for chunk in stream_text(provider, "openai/gpt-4o-mini", "Hello!"):
///             print(chunk, end="", flush=True)
///         print()
///
///     Collect full text from a stream::
///
///         text = "".join(stream_text(provider, "openai/gpt-4o-mini", "Hello!"))
///
/// Notes:
///     - Like ``generate_text``, only the ``user`` role is sent.
///     - Empty content deltas from the SSE stream are silently skipped.
///     - The ``[DONE]`` sentinel in the SSE stream is handled internally
///       and never yielded to the caller.
#[pyfunction]
#[pyo3(text_signature = "(provider, model, prompt)")]
fn stream_text(provider: &Provider, model: &str, prompt: &str) -> PyResult<TextStream> {
    let (sender, receiver) = std::sync::mpsc::channel::<PyResult<String>>();

    let url = format!("{}/chat/completions", provider.base_url);
    let api_key = provider.api_key.clone();
    let body = StreamChatRequest {
        model: model.to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        }],
        stream: true,
    };

    let handle = std::thread::spawn(move || {
        let rt = match tokio::runtime::Runtime::new() {
            Ok(rt) => rt,
            Err(e) => {
                let _ = sender.send(Err(pyo3::exceptions::PyRuntimeError::new_err(
                    e.to_string(),
                )));
                return;
            }
        };

        rt.block_on(async {
            let client = reqwest::Client::new();

            let response = match client
                .post(&url)
                .header("Authorization", format!("Bearer {}", api_key))
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    let _ = sender.send(Err(pyo3::exceptions::PyConnectionError::new_err(
                        e.to_string(),
                    )));
                    return;
                }
            };

            if !response.status().is_success() {
                let status = response.status();
                let text = response.text().await.unwrap_or_default();
                let msg = if let Ok(err) = serde_json::from_str::<ErrorResponse>(&text) {
                    format!("API error ({}): {}", status, err.error.message)
                } else {
                    format!("API error ({}): {}", status, text)
                };
                let _ = sender.send(Err(pyo3::exceptions::PyRuntimeError::new_err(msg)));
                return;
            }

            // Read SSE stream chunk by chunk
            let mut buffer = String::new();
            let mut stream = response.bytes_stream();

            use futures_util::StreamExt;
            while let Some(chunk_result) = stream.next().await {
                let bytes = match chunk_result {
                    Ok(b) => b,
                    Err(e) => {
                        let _ = sender.send(Err(pyo3::exceptions::PyRuntimeError::new_err(
                            e.to_string(),
                        )));
                        return;
                    }
                };

                buffer.push_str(&String::from_utf8_lossy(&bytes));

                // Process complete lines from the buffer
                while let Some(newline_pos) = buffer.find('\n') {
                    let line = buffer[..newline_pos].trim().to_string();
                    buffer = buffer[newline_pos + 1..].to_string();

                    if line.is_empty() {
                        continue;
                    }

                    if let Some(data) = line.strip_prefix("data: ") {
                        if data == "[DONE]" {
                            return;
                        }

                        if let Ok(chunk) = serde_json::from_str::<StreamChunk>(data) {
                            if let Some(choice) = chunk.choices.first() {
                                if let Some(content) = &choice.delta.content {
                                    if !content.is_empty() && sender.send(Ok(content.clone())).is_err() {
                                        return;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });
    });

    Ok(TextStream {
        receiver: std::sync::Mutex::new(receiver),
        _handle: Some(handle),
    })
}

/// Rusty Agent SDK — a Rust-powered Python SDK for LLM text generation.
///
/// This module provides high-performance functions for interacting with
/// OpenAI-compatible chat completions APIs. The networking and SSE parsing
/// are implemented in Rust for speed and reliability, while the public API
/// is designed to feel natural in Python.
///
/// Exported symbols:
///     - ``Provider``: API configuration (key + base URL).
///     - ``generate_text``: Blocking single-shot text generation.
///     - ``stream_text``: Streaming text generation returning a ``TextStream``.
///     - ``TextStream``: Iterator over streamed text chunks.
#[pymodule]
mod rusty_agent_sdk {
    #[pymodule_export]
    use super::generate_text;

    #[pymodule_export]
    use super::stream_text;

    #[pymodule_export]
    use super::Provider;

    #[pymodule_export]
    use super::TextStream;
}
