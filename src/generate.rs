use crate::errors::SdkError;
use crate::models::{ChatMessage, ChatRequest, api_error_message, parse_chat_response};
use crate::provider::{Provider, build_chat_completions_url};
use pyo3::prelude::*;

/// Core generation logic, called by `Provider.generate_text()`.
pub fn run(provider: &Provider, prompt: &str) -> PyResult<String> {
    let url = build_chat_completions_url(&provider.base_url);
    let api_key = provider.api_key.clone();
    let model = provider.model.clone();

    let body = ChatRequest {
        model,
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        }],
    };

    let runtime = tokio::runtime::Runtime::new()
        .map_err(|e| SdkError::runtime(e.to_string()).into_pyerr())?;

    runtime
        .block_on(async move {
            let client = reqwest::Client::new();
            let response = client
                .post(&url)
                .header("Authorization", format!("Bearer {}", api_key))
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await
                .map_err(|e| SdkError::connection(e.to_string()))?;

            let status = response.status();
            let response_text = response
                .text()
                .await
                .map_err(|e| SdkError::runtime(e.to_string()))?;

            if !status.is_success() {
                return Err(SdkError::runtime(api_error_message(status, &response_text)));
            }

            parse_chat_response(&response_text)
        })
        .map_err(SdkError::into_pyerr)
}
