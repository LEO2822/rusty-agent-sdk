use crate::errors::SdkError;
use crate::http::{is_retryable_error, is_retryable_status, retry_delay};
use crate::models::{
    GenerationParams, ParsedChatResult, api_error_message, parse_chat_response,
    parse_chat_response_full,
};
use crate::provider::{Provider, build_chat_completions_url};
use pyo3::prelude::*;
use tokio::time::sleep;

/// Core generation logic, called by `Provider.generate_text()`.
pub fn run(provider: &Provider, params: GenerationParams) -> PyResult<String> {
    let body = params.into_chat_request(provider.model.clone(), None, None);
    run_request(provider, &body, parse_chat_response)
}

/// Generation with full metadata, called by `Provider.generate_text(include_usage=True)`.
pub fn run_full(provider: &Provider, params: GenerationParams) -> PyResult<ParsedChatResult> {
    let body = params.into_chat_request(provider.model.clone(), None, None);
    run_request(provider, &body, parse_chat_response_full)
}

fn run_request<T>(
    provider: &Provider,
    body: &crate::models::ChatRequest,
    parse: impl FnOnce(&str) -> Result<T, SdkError>,
) -> PyResult<T> {
    let url = build_chat_completions_url(&provider.base_url);
    let api_key = provider.api_key.clone();
    let request_timeout = provider.request_timeout;
    let connect_timeout = provider.connect_timeout;
    let max_retries = provider.max_retries;
    let retry_backoff = provider.retry_backoff;
    let body_json =
        serde_json::to_value(body).map_err(|e| SdkError::runtime(e.to_string()).into_pyerr())?;

    let runtime = tokio::runtime::Runtime::new()
        .map_err(|e| SdkError::runtime(e.to_string()).into_pyerr())?;

    runtime
        .block_on(async move {
            let client = reqwest::Client::builder()
                .connect_timeout(connect_timeout)
                .build()
                .map_err(|e| SdkError::runtime(e.to_string()))?;

            for attempt in 0..=max_retries {
                let response_result = client
                    .post(&url)
                    .header("Authorization", format!("Bearer {}", api_key))
                    .header("Content-Type", "application/json")
                    .timeout(request_timeout)
                    .json(&body_json)
                    .send()
                    .await;

                match response_result {
                    Ok(response) => {
                        let status = response.status();
                        let response_text = response
                            .text()
                            .await
                            .map_err(|e| SdkError::runtime(e.to_string()))?;

                        if status.is_success() {
                            return parse(&response_text);
                        }

                        if is_retryable_status(status) && attempt < max_retries {
                            sleep(retry_delay(retry_backoff, attempt)).await;
                            continue;
                        }

                        return Err(SdkError::runtime(api_error_message(status, &response_text)));
                    }
                    Err(error) => {
                        if is_retryable_error(&error) && attempt < max_retries {
                            sleep(retry_delay(retry_backoff, attempt)).await;
                            continue;
                        }

                        return Err(SdkError::connection(error.to_string()));
                    }
                }
            }

            Err(SdkError::runtime(
                "Request failed after retries were exhausted.",
            ))
        })
        .map_err(SdkError::into_pyerr)
}
