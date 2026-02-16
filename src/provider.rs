use crate::errors::SdkError;
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
/// Holds the API key and base URL needed to authenticate and route requests
/// to any OpenAI-compatible chat completions endpoint. By default, requests
/// are sent to OpenRouter (https://openrouter.ai/api/v1).
///
/// The API key can be supplied explicitly or read from the
/// ``OPENROUTER_API_KEY`` environment variable. If neither is available,
/// a ``ValueError`` is raised at construction time.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct Provider {
    pub(crate) api_key: String,
    pub(crate) base_url: String,
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
    #[new]
    #[pyo3(signature = (*, api_key=None, base_url=None))]
    #[pyo3(text_signature = "(*, api_key=None, base_url=None)")]
    fn new(api_key: Option<String>, base_url: Option<String>) -> PyResult<Self> {
        let env_api_key = std::env::var("OPENROUTER_API_KEY").ok();
        let (api_key, base_url) = resolve_provider_values(api_key, base_url, env_api_key)
            .map_err(SdkError::into_pyerr)?;

        Ok(Self { api_key, base_url })
    }

    fn __repr__(&self) -> String {
        format!("Provider(base_url='{}')", self.base_url)
    }
}
