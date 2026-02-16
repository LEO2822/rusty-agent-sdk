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

/// A Python module implemented in Rust.
#[pymodule]
mod rusty_agent_sdk {
    #[pymodule_export]
    use super::generate_text;

    #[pymodule_export]
    use super::Provider;
}
