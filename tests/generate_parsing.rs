use reqwest::StatusCode;
use rusty_agent_sdk::internal::{
    Usage, api_error_message, parse_chat_response, parse_chat_response_full,
};

#[test]
fn parse_chat_response_returns_first_choice_content() {
    let body = r#"{"choices":[{"message":{"content":"Hello"}}]}"#;

    let content = parse_chat_response(body).expect("response should parse");

    assert_eq!(content, "Hello");
}

#[test]
fn parse_chat_response_fails_when_choices_are_missing() {
    let body = r#"{"choices":[]}"#;

    let err = parse_chat_response(body).expect_err("missing choices should fail");
    let message = format!("{:?}", err);

    assert!(message.contains("No choices returned"));
}

#[test]
fn parse_chat_response_fails_on_invalid_json() {
    let err = parse_chat_response("not-json").expect_err("invalid json should fail");
    let message = format!("{:?}", err);

    assert!(message.contains("Failed to parse response"));
}

#[test]
fn api_error_message_uses_structured_error_when_available() {
    let body = r#"{"error":{"message":"Invalid key"}}"#;

    let message = api_error_message(StatusCode::UNAUTHORIZED, body);

    assert_eq!(message, "API error (401 Unauthorized): Invalid key");
}

#[test]
fn api_error_message_falls_back_to_raw_body() {
    let body = "upstream unavailable";

    let message = api_error_message(StatusCode::BAD_GATEWAY, body);

    assert_eq!(message, "API error (502 Bad Gateway): upstream unavailable");
}

// ---------------------------------------------------------------------------
// parse_chat_response_full tests
// ---------------------------------------------------------------------------

#[test]
fn parse_chat_response_full_extracts_all_fields() {
    let body = r#"{
        "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "model": "gpt-4"
    }"#;

    let result = parse_chat_response_full(body).expect("should parse full response");

    assert_eq!(result.text, "Hello!");
    assert_eq!(result.finish_reason, Some("stop".to_string()));
    assert_eq!(result.model, Some("gpt-4".to_string()));

    let usage = result.usage.expect("usage should be present");
    assert_eq!(
        usage,
        Usage {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
        }
    );
}

#[test]
fn parse_chat_response_full_with_missing_optional_fields() {
    let body = r#"{"choices": [{"message": {"content": "Hi"}}]}"#;

    let result = parse_chat_response_full(body).expect("should parse without optionals");

    assert_eq!(result.text, "Hi");
    assert!(result.usage.is_none());
    assert!(result.finish_reason.is_none());
    assert!(result.model.is_none());
}

#[test]
fn parse_chat_response_full_fails_on_empty_choices() {
    let body = r#"{"choices": []}"#;

    let err = parse_chat_response_full(body).expect_err("empty choices should fail");
    let msg = format!("{:?}", err);

    assert!(msg.contains("No choices returned"));
}

#[test]
fn parse_chat_response_full_fails_on_invalid_json() {
    let err = parse_chat_response_full("not-json").expect_err("invalid json should fail");
    let msg = format!("{:?}", err);

    assert!(msg.contains("Failed to parse response"));
}
