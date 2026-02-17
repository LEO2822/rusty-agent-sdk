use crate::errors::SdkError;
use crate::generate;
use crate::stream::{self, TextStream};
use pyo3::prelude::*;
use std::time::Duration;

pub const DEFAULT_BASE_URL: &str = "https://openrouter.ai/api/v1";
pub const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 60;
pub const DEFAULT_CONNECT_TIMEOUT_SECS: u64 = 10;
pub const DEFAULT_MAX_RETRIES: u32 = 2;
pub const DEFAULT_RETRY_BACKOFF_MS: u64 = 250;

const REQUEST_TIMEOUT_ENV: &str = "RUSTY_AGENT_REQUEST_TIMEOUT_SECS";
const CONNECT_TIMEOUT_ENV: &str = "RUSTY_AGENT_CONNECT_TIMEOUT_SECS";
const MAX_RETRIES_ENV: &str = "RUSTY_AGENT_MAX_RETRIES";
const RETRY_BACKOFF_ENV: &str = "RUSTY_AGENT_RETRY_BACKOFF_MS";

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

#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    pub request_timeout: Duration,
    pub connect_timeout: Duration,
    pub max_retries: u32,
    pub retry_backoff: Duration,
}

pub fn resolve_runtime_config(
    request_timeout_env: Option<String>,
    connect_timeout_env: Option<String>,
    max_retries_env: Option<String>,
    retry_backoff_env: Option<String>,
) -> Result<RuntimeConfig, SdkError> {
    let request_timeout_secs = parse_positive_u64_env(
        request_timeout_env,
        REQUEST_TIMEOUT_ENV,
        DEFAULT_REQUEST_TIMEOUT_SECS,
    )?;
    let connect_timeout_secs = parse_positive_u64_env(
        connect_timeout_env,
        CONNECT_TIMEOUT_ENV,
        DEFAULT_CONNECT_TIMEOUT_SECS,
    )?;
    let retry_backoff_ms = parse_positive_u64_env(
        retry_backoff_env,
        RETRY_BACKOFF_ENV,
        DEFAULT_RETRY_BACKOFF_MS,
    )?;
    let max_retries = parse_u32_env(max_retries_env, MAX_RETRIES_ENV, DEFAULT_MAX_RETRIES)?;

    Ok(RuntimeConfig {
        request_timeout: Duration::from_secs(request_timeout_secs),
        connect_timeout: Duration::from_secs(connect_timeout_secs),
        max_retries,
        retry_backoff: Duration::from_millis(retry_backoff_ms),
    })
}

fn parse_positive_u64_env(
    value: Option<String>,
    name: &str,
    default: u64,
) -> Result<u64, SdkError> {
    let Some(raw) = value else {
        return Ok(default);
    };

    let parsed = raw.parse::<u64>().map_err(|_| {
        SdkError::value(format!(
            "{} must be a positive integer, got '{}'.",
            name, raw
        ))
    })?;

    if parsed == 0 {
        return Err(SdkError::value(format!(
            "{} must be greater than zero.",
            name
        )));
    }

    Ok(parsed)
}

fn parse_u32_env(value: Option<String>, name: &str, default: u32) -> Result<u32, SdkError> {
    let Some(raw) = value else {
        return Ok(default);
    };

    raw.parse::<u32>().map_err(|_| {
        SdkError::value(format!(
            "{} must be a non-negative integer, got '{}'.",
            name, raw
        ))
    })
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
    pub(crate) request_timeout: Duration,
    pub(crate) connect_timeout: Duration,
    pub(crate) max_retries: u32,
    pub(crate) retry_backoff: Duration,
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
        let runtime_config = resolve_runtime_config(
            std::env::var(REQUEST_TIMEOUT_ENV).ok(),
            std::env::var(CONNECT_TIMEOUT_ENV).ok(),
            std::env::var(MAX_RETRIES_ENV).ok(),
            std::env::var(RETRY_BACKOFF_ENV).ok(),
        )
        .map_err(SdkError::into_pyerr)?;

        Ok(Self {
            api_key,
            base_url,
            model,
            request_timeout: runtime_config.request_timeout,
            connect_timeout: runtime_config.connect_timeout,
            max_retries: runtime_config.max_retries,
            retry_backoff: runtime_config.retry_backoff,
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
