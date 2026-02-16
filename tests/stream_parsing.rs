use rusty_agent_sdk::internal::{StreamEvent, parse_sse_line};

#[test]
fn parse_sse_line_extracts_content_chunk() {
    let line = r#"data: {"choices":[{"delta":{"content":"Hel"}}]}"#;

    let event = parse_sse_line(line).expect("line should parse");

    assert_eq!(event, StreamEvent::Content("Hel".to_string()));
}

#[test]
fn parse_sse_line_marks_done_sentinel() {
    let event = parse_sse_line("data: [DONE]").expect("done sentinel should parse");

    assert_eq!(event, StreamEvent::Done);
}

#[test]
fn parse_sse_line_ignores_non_data_lines() {
    let event = parse_sse_line("event: completion").expect("non-data line should be ignored");

    assert_eq!(event, StreamEvent::Ignore);
}

#[test]
fn parse_sse_line_ignores_empty_content() {
    let line = r#"data: {"choices":[{"delta":{"content":""}}]}"#;

    let event = parse_sse_line(line).expect("line should parse");

    assert_eq!(event, StreamEvent::Ignore);
}

#[test]
fn parse_sse_line_returns_error_for_malformed_data_payload() {
    let err = parse_sse_line("data: {not-json}").expect_err("malformed payload should fail");
    let message = format!("{:?}", err);

    assert!(message.contains("Failed to parse streaming response chunk"));
}
