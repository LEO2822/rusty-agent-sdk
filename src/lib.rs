use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// An LLM provider configuration (API key + base URL).
#[pyclass(from_py_object)]
#[derive(Clone)]
struct Provider {
    api_key: String,
    base_url: String,
}

#[pymethods]
impl Provider {
    #[new]
    #[pyo3(signature = (*, api_key=None, base_url=None))]
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

/// Generate text from an LLM using the given provider, model, and prompt.
#[pyfunction]
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

/// A streaming iterator that yields text chunks from an LLM response.
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

/// Stream text from an LLM, returning an iterator that yields chunks.
#[pyfunction]
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

/// A Python module implemented in Rust.
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
