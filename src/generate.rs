use crate::errors::SdkError;
use crate::http::{is_retryable_error, is_retryable_status, retry_delay};
use crate::models::{GenerationParams, api_error_message, parse_chat_response};
use crate::provider::{Provider, build_chat_completions_url};
use pyo3::prelude::*;
use tokio::time::sleep;

/// Core generation logic, called by `Provider.generate_text()`.
pub fn run(provider: &Provider, params: GenerationParams) -> PyResult<String> {
    let url = build_chat_completions_url(&provider.base_url);
    let api_key = provider.api_key.clone();
    let model = provider.model.clone();
    let request_timeout = provider.request_timeout;
    let connect_timeout = provider.connect_timeout;
    let max_retries = provider.max_retries;
    let retry_backoff = provider.retry_backoff;

    let body = params.into_chat_request(model, None);

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
                    .json(&body)
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
                            return parse_chat_response(&response_text);
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
