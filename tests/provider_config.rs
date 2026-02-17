use rusty_agent_sdk::internal::{
    build_chat_completions_url, resolve_provider_values, resolve_runtime_config,
};
use std::time::Duration;

#[test]
fn provider_uses_env_key_when_api_key_not_provided() {
    let (api_key, base_url) = resolve_provider_values(None, None, Some("env-key".to_string()))
        .expect("config should be valid");

    assert_eq!(api_key, "env-key");
    assert_eq!(base_url, "https://openrouter.ai/api/v1");
}

#[test]
fn provider_prefers_explicit_api_key_over_env() {
    let (api_key, base_url) = resolve_provider_values(
        Some("explicit-key".to_string()),
        Some("https://api.openai.com/v1/".to_string()),
        Some("env-key".to_string()),
    )
    .expect("config should be valid");

    assert_eq!(api_key, "explicit-key");
    assert_eq!(base_url, "https://api.openai.com/v1");
}

#[test]
fn provider_returns_error_when_no_api_key_is_available() {
    let err = resolve_provider_values(None, None, None).expect_err("missing api key should fail");
    let message = format!("{:?}", err);
    assert!(message.contains("OPENROUTER_API_KEY"));
}

#[test]
fn chat_url_builder_normalizes_trailing_slash() {
    let url = build_chat_completions_url("https://openrouter.ai/api/v1/");

    assert_eq!(url, "https://openrouter.ai/api/v1/chat/completions");
}

#[test]
fn runtime_config_uses_defaults_when_env_is_missing() {
    let config = resolve_runtime_config(None, None, None, None).expect("config should be valid");

    assert_eq!(config.request_timeout, Duration::from_secs(60));
    assert_eq!(config.connect_timeout, Duration::from_secs(10));
    assert_eq!(config.max_retries, 2);
    assert_eq!(config.retry_backoff, Duration::from_millis(250));
}

#[test]
fn runtime_config_reads_env_values() {
    let config = resolve_runtime_config(
        Some("90".to_string()),
        Some("5".to_string()),
        Some("4".to_string()),
        Some("500".to_string()),
    )
    .expect("config should parse");

    assert_eq!(config.request_timeout, Duration::from_secs(90));
    assert_eq!(config.connect_timeout, Duration::from_secs(5));
    assert_eq!(config.max_retries, 4);
    assert_eq!(config.retry_backoff, Duration::from_millis(500));
}

#[test]
fn runtime_config_rejects_invalid_values() {
    let err = resolve_runtime_config(Some("0".to_string()), None, None, None)
        .expect_err("request timeout of 0 should fail");
    assert!(format!("{:?}", err).contains("RUSTY_AGENT_REQUEST_TIMEOUT_SECS"));

    let err = resolve_runtime_config(None, None, Some("bad".to_string()), None)
        .expect_err("invalid retry count should fail");
    assert!(format!("{:?}", err).contains("RUSTY_AGENT_MAX_RETRIES"));
}
