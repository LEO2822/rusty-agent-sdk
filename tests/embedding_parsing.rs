use rusty_agent_sdk::internal::{EmbeddingUsage, parse_embedding_response};

#[test]
fn parse_single_embedding_response() {
    let body = r#"{
        "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
        "model": "text-embedding-ada-002"
    }"#;

    let result = parse_embedding_response(body).expect("should parse single embedding");

    assert_eq!(result.embeddings.len(), 1);
    assert_eq!(result.embeddings[0], vec![0.1, 0.2, 0.3]);
    assert_eq!(result.model, Some("text-embedding-ada-002".to_string()));
    assert!(result.usage.is_none());
}

#[test]
fn parse_batch_embedding_response() {
    let body = r#"{
        "data": [
            {"embedding": [0.4, 0.5], "index": 1},
            {"embedding": [0.1, 0.2], "index": 0},
            {"embedding": [0.7, 0.8], "index": 2}
        ],
        "model": "text-embedding-ada-002"
    }"#;

    let result = parse_embedding_response(body).expect("should parse batch embeddings");

    assert_eq!(result.embeddings.len(), 3);
    // Embeddings should be sorted by index
    assert_eq!(result.embeddings[0], vec![0.1, 0.2]);
    assert_eq!(result.embeddings[1], vec![0.4, 0.5]);
    assert_eq!(result.embeddings[2], vec![0.7, 0.8]);
}

#[test]
fn parse_embedding_response_with_usage() {
    let body = r#"{
        "data": [{"embedding": [0.1, 0.2], "index": 0}],
        "model": "text-embedding-ada-002",
        "usage": {"prompt_tokens": 8, "total_tokens": 8}
    }"#;

    let result = parse_embedding_response(body).expect("should parse with usage");

    let usage = result.usage.expect("usage should be present");
    assert_eq!(
        usage,
        EmbeddingUsage {
            prompt_tokens: 8,
            total_tokens: 8,
        }
    );
}

#[test]
fn parse_embedding_response_fails_on_empty_data() {
    let body = r#"{"data": [], "model": "text-embedding-ada-002"}"#;

    let err = parse_embedding_response(body).expect_err("empty data should fail");
    let msg = format!("{:?}", err);

    assert!(msg.contains("No embeddings returned"));
}

#[test]
fn parse_embedding_response_fails_on_invalid_json() {
    let err = parse_embedding_response("not-json").expect_err("invalid json should fail");
    let msg = format!("{:?}", err);

    assert!(msg.contains("Failed to parse embedding response"));
}
