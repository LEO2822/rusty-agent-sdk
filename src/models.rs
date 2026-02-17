use crate::errors::SdkError;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
pub(crate) struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Serialize)]
pub(crate) struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
}

#[derive(Serialize)]
pub(crate) struct StreamChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub stream: bool,
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

pub fn parse_chat_response(response_text: &str) -> Result<String, SdkError> {
    let chat_response: ChatResponse = serde_json::from_str(response_text)
        .map_err(|e| SdkError::value(format!("Failed to parse response: {}", e)))?;

    chat_response
        .choices
        .first()
        .map(|choice| choice.message.content.clone())
        .ok_or_else(|| SdkError::value("No choices returned in API response"))
}

pub fn api_error_message(status: StatusCode, response_text: &str) -> String {
    if let Ok(err) = serde_json::from_str::<ErrorResponse>(response_text) {
        return format!("API error ({}): {}", status, err.error.message);
    }

    format!("API error ({}): {}", status, response_text)
}

#[derive(Debug, PartialEq, Eq)]
pub enum StreamEvent {
    Done,
    Content(String),
    Ignore,
}

pub fn parse_sse_line(line: &str) -> Result<StreamEvent, SdkError> {
    let trimmed = line.trim_end_matches('\r');
    if trimmed.trim().is_empty() {
        return Ok(StreamEvent::Ignore);
    }

    parse_sse_event(trimmed)
}

pub fn parse_sse_event(event: &str) -> Result<StreamEvent, SdkError> {
    let mut data_lines = Vec::new();
    for line in event.lines() {
        let trimmed = line.trim_end_matches('\r');
        if let Some(data) = trimmed.strip_prefix("data:") {
            data_lines.push(data.trim_start());
        }
    }

    if data_lines.is_empty() {
        return Ok(StreamEvent::Ignore);
    }

    parse_sse_data(&data_lines.join("\n"))
}

fn parse_sse_data(data: &str) -> Result<StreamEvent, SdkError> {
    if data == "[DONE]" {
        return Ok(StreamEvent::Done);
    }

    let chunk: StreamChunk = serde_json::from_str(data).map_err(|e| {
        SdkError::runtime(format!("Failed to parse streaming response chunk: {}", e))
    })?;

    let content = chunk
        .choices
        .first()
        .and_then(|choice| choice.delta.content.as_ref());

    match content {
        Some(content) if !content.is_empty() => Ok(StreamEvent::Content(content.clone())),
        _ => Ok(StreamEvent::Ignore),
    }
}
