use crate::errors::SdkError;
use crate::generate;
use crate::stream::{self, TextStream};
use pyo3::prelude::*;

pub const DEFAULT_BASE_URL: &str = "https://openrouter.ai/api/v1";

/// Build a normalized chat completions URL from the configured provider base URL.
pub fn build_chat_completions_url(base_url: &str) -> String {
    format!("{}/chat/completions", base_url.trim_end_matches('/'))
}

pub fn resolve_provider_values(
    api_key: Option<String>,
    base_url: Option<String>,
    env_api_key: Option<String>,
) -> Result<(String, String), SdkError> {
    let base_url = base_url
        .unwrap_or_else(|| DEFAULT_BASE_URL.to_string())
        .trim_end_matches('/')
        .to_string();

    let api_key = match api_key {
        Some(key) => key,
        None => env_api_key.ok_or_else(|| {
            SdkError::value(
                "No api_key provided and OPENROUTER_API_KEY environment variable is not set.",
            )
        })?,
    };

    Ok((api_key, base_url))
}

/// Configuration for an OpenAI-compatible LLM API provider.
///
/// Holds the API key, base URL, and default model needed to authenticate
/// and route requests to any OpenAI-compatible chat completions endpoint.
/// By default, requests are sent to OpenRouter (https://openrouter.ai/api/v1).
///
/// The API key can be supplied explicitly or read from the
/// ``OPENROUTER_API_KEY`` environment variable. If neither is available,
/// a ``ValueError`` is raised at construction time.
///
/// Examples (Python):
///
/// ```text
/// provider = Provider("openai/gpt-4o-mini")
/// for chunk in provider.stream_text("Hello!"):
///     print(chunk, end="", flush=True)
/// ```
///
/// ```text
/// provider = Provider(
///     "gpt-4o-mini",
///     api_key="sk-...",
///     base_url="https://api.openai.com/v1",
/// )
/// response = provider.generate_text("Hello!")
/// ```
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct Provider {
    pub(crate) api_key: String,
    pub(crate) base_url: String,
    pub(crate) model: String,
}

#[pymethods]
impl Provider {
    /// Create a new Provider.
    ///
    /// Args:
    ///     model (str): Model identifier, e.g. ``"openai/gpt-4o-mini"``
    ///         or ``"anthropic/claude-sonnet-4-5-20250514"``.
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
    #[new]
    #[pyo3(signature = (model, *, api_key=None, base_url=None))]
    #[pyo3(text_signature = "(model, *, api_key=None, base_url=None)")]
    fn new(model: String, api_key: Option<String>, base_url: Option<String>) -> PyResult<Self> {
        let env_api_key = std::env::var("OPENROUTER_API_KEY").ok();
        let (api_key, base_url) = resolve_provider_values(api_key, base_url, env_api_key)
            .map_err(SdkError::into_pyerr)?;

        Ok(Self {
            api_key,
            base_url,
            model,
        })
    }

    /// Generate a complete text response from the LLM (blocking).
    ///
    /// Args:
    ///     prompt (str): The user message to send to the model.
    ///
    /// Returns:
    ///     str: The model's complete text response.
    ///
    /// Raises:
    ///     ConnectionError: If the HTTP request fails.
    ///     RuntimeError: If the API returns a non-2xx status code.
    ///     ValueError: If the response cannot be parsed.
    #[pyo3(text_signature = "(self, prompt)")]
    fn generate_text(&self, prompt: &str) -> PyResult<String> {
        generate::run(self, prompt)
    }

    /// Stream text from the LLM, returning an iterator of chunks.
    ///
    /// Args:
    ///     prompt (str): The user message to send to the model.
    ///
    /// Returns:
    ///     TextStream: An iterator yielding ``str`` chunks.
    ///
    /// Raises:
    ///     ConnectionError: If the initial HTTP connection fails.
    ///     RuntimeError: If the API returns a non-2xx status code.
    #[pyo3(text_signature = "(self, prompt)")]
    fn stream_text(&self, prompt: &str) -> PyResult<TextStream> {
        stream::run(self, prompt)
    }

    fn __repr__(&self) -> String {
        format!(
            "Provider(model='{}', base_url='{}')",
            self.model, self.base_url
        )
    }
}
