use crate::embed;
use crate::errors::SdkError;
use crate::generate;
use crate::models::{
    ChatMessage, EmbeddingInput, EmbeddingResultData, EmbeddingUsage, GenerationParams,
    ParsedChatResult, Usage,
};
use crate::stream::{self, TextStream};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyList, PyString};
use serde_json::Value;
use std::time::Duration;

// ---------------------------------------------------------------------------
// GenerateResult pyclass
// ---------------------------------------------------------------------------

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct GenerateResult {
    text: String,
    usage: Option<Usage>,
    finish_reason: Option<String>,
    model: Option<String>,
}

#[pymethods]
impl GenerateResult {
    #[getter]
    fn text(&self) -> &str {
        &self.text
    }

    #[getter]
    fn prompt_tokens(&self) -> Option<u64> {
        self.usage.as_ref().map(|u| u.prompt_tokens)
    }

    #[getter]
    fn completion_tokens(&self) -> Option<u64> {
        self.usage.as_ref().map(|u| u.completion_tokens)
    }

    #[getter]
    fn total_tokens(&self) -> Option<u64> {
        self.usage.as_ref().map(|u| u.total_tokens)
    }

    #[getter]
    fn finish_reason(&self) -> Option<&str> {
        self.finish_reason.as_deref()
    }

    #[getter]
    fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    fn __str__(&self) -> &str {
        &self.text
    }

    fn __repr__(&self) -> String {
        format!(
            "GenerateResult(text='{}...', finish_reason={:?}, prompt_tokens={:?}, completion_tokens={:?})",
            &self.text.chars().take(50).collect::<String>(),
            self.finish_reason,
            self.usage.as_ref().map(|u| u.prompt_tokens),
            self.usage.as_ref().map(|u| u.completion_tokens),
        )
    }
}

impl GenerateResult {
    pub fn from_parsed(result: ParsedChatResult) -> Self {
        Self {
            text: result.text,
            usage: result.usage,
            finish_reason: result.finish_reason,
            model: result.model,
        }
    }
}

// ---------------------------------------------------------------------------
// EmbeddingResult pyclass
// ---------------------------------------------------------------------------

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct EmbeddingResult {
    embeddings: Vec<Vec<f64>>,
    usage: Option<EmbeddingUsage>,
    model: Option<String>,
}

#[pymethods]
impl EmbeddingResult {
    #[getter]
    fn embeddings(&self) -> Vec<Vec<f64>> {
        self.embeddings.clone()
    }

    #[getter]
    fn prompt_tokens(&self) -> Option<u64> {
        self.usage.as_ref().map(|u| u.prompt_tokens)
    }

    #[getter]
    fn total_tokens(&self) -> Option<u64> {
        self.usage.as_ref().map(|u| u.total_tokens)
    }

    #[getter]
    fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    fn __repr__(&self) -> String {
        format!(
            "EmbeddingResult(count={}, prompt_tokens={:?})",
            self.embeddings.len(),
            self.usage.as_ref().map(|u| u.prompt_tokens),
        )
    }
}

impl EmbeddingResult {
    fn from_data(data: EmbeddingResultData) -> Self {
        Self {
            embeddings: data.embeddings,
            usage: data.usage,
            model: data.model,
        }
    }
}

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

/// Build a normalized embeddings URL from the configured provider base URL.
pub fn build_embeddings_url(base_url: &str) -> String {
    format!("{}/embeddings", base_url.trim_end_matches('/'))
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

// ---------------------------------------------------------------------------
// Python → Rust conversion helpers
// ---------------------------------------------------------------------------

/// Recursively convert a Python object to `serde_json::Value`.
///
/// PyBool is checked before integer extraction because in Python
/// `bool` is a subclass of `int`.
fn py_to_json(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        Ok(Value::Null)
    } else if let Ok(b) = obj.cast::<PyBool>() {
        Ok(Value::Bool(b.is_true()))
    } else if let Ok(i) = obj.extract::<i64>() {
        Ok(Value::from(i))
    } else if let Ok(f) = obj.cast::<PyFloat>() {
        let v = f.value();
        Ok(Value::from(v))
    } else if let Ok(s) = obj.cast::<PyString>() {
        Ok(Value::String(s.to_string()))
    } else if let Ok(list) = obj.cast::<PyList>() {
        let items: PyResult<Vec<Value>> = list.iter().map(|item| py_to_json(&item)).collect();
        Ok(Value::Array(items?))
    } else if let Ok(dict) = obj.cast::<PyDict>() {
        let mut map = serde_json::Map::new();
        for (k, v) in dict.iter() {
            let key: String = k.extract()?;
            map.insert(key, py_to_json(&v)?);
        }
        Ok(Value::Object(map))
    } else {
        Err(SdkError::value(format!(
            "Cannot convert Python type '{}' to JSON.",
            obj.get_type().name()?
        ))
        .into_pyerr())
    }
}

/// Extract a Python list of `{"role": ..., "content": ...}` dicts into `Vec<ChatMessage>`.
fn extract_messages(py_messages: &Bound<'_, PyList>) -> PyResult<Vec<ChatMessage>> {
    let mut messages = Vec::with_capacity(py_messages.len());
    for item in py_messages.iter() {
        let role: String = item.get_item("role")?.extract()?;
        let content: String = item.get_item("content")?.extract()?;
        messages.push(ChatMessage { role, content });
    }
    Ok(messages)
}

/// Convert a Python `str | list[str]` to `serde_json::Value`.
fn extract_stop(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if let Ok(s) = obj.extract::<String>() {
        return Ok(Value::String(s));
    }
    if let Ok(list) = obj.cast::<PyList>() {
        let strings: Vec<String> = list.extract()?;
        return Ok(serde_json::json!(strings));
    }
    Err(SdkError::value("'stop' must be a string or list of strings.").into_pyerr())
}

/// Build `GenerationParams` from Python keyword arguments.
#[expect(clippy::too_many_arguments)] // mirrors the Python-facing API surface
fn build_generation_params(
    prompt: Option<&str>,
    system_prompt: Option<&str>,
    messages: Option<&Bound<'_, PyList>>,
    temperature: Option<f64>,
    max_tokens: Option<u64>,
    top_p: Option<f64>,
    stop: Option<&Bound<'_, PyAny>>,
    frequency_penalty: Option<f64>,
    presence_penalty: Option<f64>,
    seed: Option<i64>,
    response_format: Option<&Bound<'_, PyAny>>,
) -> PyResult<GenerationParams> {
    let raw_messages = messages.map(extract_messages).transpose()?;
    let stop_val = stop.map(extract_stop).transpose()?;
    let rf_val = response_format.map(py_to_json).transpose()?;

    let msgs = GenerationParams::build_messages(prompt, system_prompt, raw_messages)
        .map_err(SdkError::into_pyerr)?;

    Ok(GenerationParams {
        messages: msgs,
        temperature,
        max_tokens,
        top_p,
        stop: stop_val,
        frequency_penalty,
        presence_penalty,
        seed,
        response_format: rf_val,
    })
}

// ---------------------------------------------------------------------------
// Provider pyclass
// ---------------------------------------------------------------------------

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
    ///     prompt (str | None): The user message to send (shorthand for a
    ///         single user message).
    ///     system_prompt (str | None): System prompt, prepended to messages.
    ///     messages (list[dict] | None): Full conversation history as a
    ///         list of ``{"role": ..., "content": ...}`` dicts.
    ///     temperature (float | None): Sampling temperature (0-2).
    ///     max_tokens (int | None): Maximum tokens to generate.
    ///     top_p (float | None): Nucleus sampling threshold (0-1).
    ///     stop (str | list[str] | None): Up to 4 stop sequences.
    ///     frequency_penalty (float | None): Frequency penalty (-2 to 2).
    ///     presence_penalty (float | None): Presence penalty (-2 to 2).
    ///     seed (int | None): Random seed for deterministic generation.
    ///     response_format (dict | None): Response format configuration.
    ///
    /// Returns:
    ///     str: The model's complete text response.
    ///
    /// Raises:
    ///     ConnectionError: If the HTTP request fails.
    ///     RuntimeError: If the API returns a non-2xx status code.
    ///     ValueError: If the response cannot be parsed, or if neither
    ///         prompt nor messages is provided.
    #[expect(clippy::too_many_arguments)] // PyO3 requires flat params for Python kwargs
    #[pyo3(signature = (
        prompt = None,
        *,
        system_prompt = None,
        messages = None,
        temperature = None,
        max_tokens = None,
        top_p = None,
        stop = None,
        frequency_penalty = None,
        presence_penalty = None,
        seed = None,
        response_format = None,
        include_usage = false,
    ))]
    #[pyo3(
        text_signature = "(self, prompt=None, *, system_prompt=None, messages=None, temperature=None, max_tokens=None, top_p=None, stop=None, frequency_penalty=None, presence_penalty=None, seed=None, response_format=None, include_usage=False)"
    )]
    fn generate_text(
        &self,
        py: Python<'_>,
        prompt: Option<&str>,
        system_prompt: Option<&str>,
        messages: Option<&Bound<'_, PyList>>,
        temperature: Option<f64>,
        max_tokens: Option<u64>,
        top_p: Option<f64>,
        stop: Option<&Bound<'_, PyAny>>,
        frequency_penalty: Option<f64>,
        presence_penalty: Option<f64>,
        seed: Option<i64>,
        response_format: Option<&Bound<'_, PyAny>>,
        include_usage: bool,
    ) -> PyResult<Py<PyAny>> {
        let params = build_generation_params(
            prompt,
            system_prompt,
            messages,
            temperature,
            max_tokens,
            top_p,
            stop,
            frequency_penalty,
            presence_penalty,
            seed,
            response_format,
        )?;

        if include_usage {
            let result = generate::run_full(self, params)?;
            Ok(GenerateResult::from_parsed(result)
                .into_pyobject(py)?
                .into_any()
                .unbind())
        } else {
            let text = generate::run(self, params)?;
            Ok(text.into_pyobject(py)?.into_any().unbind())
        }
    }

    /// Stream text from the LLM, returning an iterator of chunks.
    ///
    /// Accepts the same parameters as ``generate_text``.
    ///
    /// Returns:
    ///     TextStream: An iterator yielding ``str`` chunks.
    ///
    /// Raises:
    ///     ConnectionError: If the initial HTTP connection fails.
    ///     RuntimeError: If the API returns a non-2xx status code.
    ///     ValueError: If neither prompt nor messages is provided.
    #[expect(clippy::too_many_arguments)] // PyO3 requires flat params for Python kwargs
    #[pyo3(signature = (
        prompt = None,
        *,
        system_prompt = None,
        messages = None,
        temperature = None,
        max_tokens = None,
        top_p = None,
        stop = None,
        frequency_penalty = None,
        presence_penalty = None,
        seed = None,
        response_format = None,
        include_usage = false,
    ))]
    #[pyo3(
        text_signature = "(self, prompt=None, *, system_prompt=None, messages=None, temperature=None, max_tokens=None, top_p=None, stop=None, frequency_penalty=None, presence_penalty=None, seed=None, response_format=None, include_usage=False)"
    )]
    fn stream_text(
        &self,
        prompt: Option<&str>,
        system_prompt: Option<&str>,
        messages: Option<&Bound<'_, PyList>>,
        temperature: Option<f64>,
        max_tokens: Option<u64>,
        top_p: Option<f64>,
        stop: Option<&Bound<'_, PyAny>>,
        frequency_penalty: Option<f64>,
        presence_penalty: Option<f64>,
        seed: Option<i64>,
        response_format: Option<&Bound<'_, PyAny>>,
        include_usage: bool,
    ) -> PyResult<TextStream> {
        let params = build_generation_params(
            prompt,
            system_prompt,
            messages,
            temperature,
            max_tokens,
            top_p,
            stop,
            frequency_penalty,
            presence_penalty,
            seed,
            response_format,
        )?;

        if include_usage {
            stream::run_with_metadata(self, params)
        } else {
            stream::run(self, params)
        }
    }

    /// Generate embeddings for a single text input.
    ///
    /// Args:
    ///     text (str): The text to embed.
    ///
    /// Returns:
    ///     EmbeddingResult: Contains the embedding vector and usage metadata.
    #[pyo3(signature = (text))]
    #[pyo3(text_signature = "(self, text)")]
    fn embed(&self, text: String) -> PyResult<EmbeddingResult> {
        let data = embed::run(self, EmbeddingInput::Single(text))?;
        Ok(EmbeddingResult::from_data(data))
    }

    /// Generate embeddings for multiple text inputs in a single request.
    ///
    /// Args:
    ///     texts (list[str]): The texts to embed.
    ///
    /// Returns:
    ///     EmbeddingResult: Contains the embedding vectors (one per input) and usage metadata.
    #[pyo3(signature = (texts))]
    #[pyo3(text_signature = "(self, texts)")]
    fn embed_many(&self, texts: Vec<String>) -> PyResult<EmbeddingResult> {
        let data = embed::run(self, EmbeddingInput::Multiple(texts))?;
        Ok(EmbeddingResult::from_data(data))
    }

    /// Create a Provider pre-configured for OpenAI's API.
    ///
    /// Args:
    ///     model (str): Model identifier, e.g. ``"gpt-4o-mini"``.
    ///     api_key (str | None): API key. Defaults to ``OPENAI_API_KEY`` env var.
    #[classmethod]
    #[pyo3(signature = (model, *, api_key=None))]
    #[pyo3(text_signature = "(model, *, api_key=None)")]
    fn openai(
        _cls: &Bound<'_, pyo3::types::PyType>,
        model: String,
        api_key: Option<String>,
    ) -> PyResult<Self> {
        Self::from_preset(
            model,
            api_key,
            "https://api.openai.com/v1",
            "OPENAI_API_KEY",
        )
    }

    /// Create a Provider pre-configured for Anthropic's API.
    ///
    /// Args:
    ///     model (str): Model identifier, e.g. ``"claude-sonnet-4-5-20250514"``.
    ///     api_key (str | None): API key. Defaults to ``ANTHROPIC_API_KEY`` env var.
    #[classmethod]
    #[pyo3(signature = (model, *, api_key=None))]
    #[pyo3(text_signature = "(model, *, api_key=None)")]
    fn anthropic(
        _cls: &Bound<'_, pyo3::types::PyType>,
        model: String,
        api_key: Option<String>,
    ) -> PyResult<Self> {
        Self::from_preset(
            model,
            api_key,
            "https://api.anthropic.com/v1",
            "ANTHROPIC_API_KEY",
        )
    }

    /// Create a Provider pre-configured for OpenRouter's API.
    ///
    /// Args:
    ///     model (str): Model identifier, e.g. ``"openai/gpt-4o-mini"``.
    ///     api_key (str | None): API key. Defaults to ``OPENROUTER_API_KEY`` env var.
    #[classmethod]
    #[pyo3(signature = (model, *, api_key=None))]
    #[pyo3(text_signature = "(model, *, api_key=None)")]
    fn openrouter(
        _cls: &Bound<'_, pyo3::types::PyType>,
        model: String,
        api_key: Option<String>,
    ) -> PyResult<Self> {
        Self::from_preset(
            model,
            api_key,
            "https://openrouter.ai/api/v1",
            "OPENROUTER_API_KEY",
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "Provider(model='{}', base_url='{}')",
            self.model, self.base_url
        )
    }
}

impl Provider {
    fn from_preset(
        model: String,
        api_key: Option<String>,
        base_url: &str,
        env_var: &str,
    ) -> PyResult<Self> {
        let env_api_key = std::env::var(env_var).ok();
        let (api_key, base_url) =
            resolve_provider_values(api_key, Some(base_url.to_string()), env_api_key).map_err(
                |_| {
                    SdkError::value(format!(
                        "No api_key provided and {} environment variable is not set.",
                        env_var
                    ))
                    .into_pyerr()
                },
            )?;
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
}
