use rusty_agent_sdk::internal::{build_chat_completions_url, resolve_provider_values};

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
