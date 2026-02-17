use crate::errors::SdkError;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Serialize, Clone, Debug)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<Value>,
}

/// Internal parameters extracted from Python keyword arguments.
///
/// This is not a pyclass — it exists to pass generation options from
/// `Provider` methods to `generate::run` and `stream::run`.
pub struct GenerationParams {
    pub messages: Vec<ChatMessage>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u64>,
    pub top_p: Option<f64>,
    pub stop: Option<Value>,
    pub frequency_penalty: Option<f64>,
    pub presence_penalty: Option<f64>,
    pub seed: Option<i64>,
    pub response_format: Option<Value>,
}

impl GenerationParams {
    /// Build the messages list from Python-side inputs.
    ///
    /// Priority:
    /// 1. If `messages` is non-empty, use it. If `system_prompt` is also
    ///    provided, prepend it as a system message.
    /// 2. If only `prompt` is provided, create a single user message.
    ///    If `system_prompt` is also provided, prepend it.
    /// 3. If neither is provided, return an error.
    pub fn build_messages(
        prompt: Option<&str>,
        system_prompt: Option<&str>,
        raw_messages: Option<Vec<ChatMessage>>,
    ) -> Result<Vec<ChatMessage>, SdkError> {
        let mut messages = Vec::new();

        if let Some(sys) = system_prompt {
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: sys.to_string(),
            });
        }

        match (raw_messages, prompt) {
            (Some(msgs), _) if !msgs.is_empty() => {
                messages.extend(msgs);
            }
            (_, Some(p)) => {
                messages.push(ChatMessage {
                    role: "user".to_string(),
                    content: p.to_string(),
                });
            }
            _ => {
                return Err(SdkError::value(
                    "Either 'prompt' or 'messages' must be provided.",
                ));
            }
        }

        Ok(messages)
    }

    /// Convert into a serialisable `ChatRequest`.
    pub fn into_chat_request(self, model: String, stream: Option<bool>) -> ChatRequest {
        ChatRequest {
            model,
            messages: self.messages,
            stream,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            top_p: self.top_p,
            stop: self.stop,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            seed: self.seed,
            response_format: self.response_format,
        }
    }
}

// ---------------------------------------------------------------------------
// Response parsing (unchanged)
// ---------------------------------------------------------------------------

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
