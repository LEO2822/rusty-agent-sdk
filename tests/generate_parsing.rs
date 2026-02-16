use reqwest::StatusCode;
use rusty_agent_sdk::internal::{api_error_message, parse_chat_response};

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
