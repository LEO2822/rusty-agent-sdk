use rusty_agent_sdk::internal::{ChatMessage, GenerationParams};

#[test]
fn build_messages_from_prompt_only() {
    let msgs = GenerationParams::build_messages(Some("Hello"), None, None)
        .expect("should build from prompt");
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0].role, "user");
    assert_eq!(msgs[0].content, "Hello");
}

#[test]
fn build_messages_with_system_prompt_and_prompt() {
    let msgs = GenerationParams::build_messages(Some("Hello"), Some("You are helpful"), None)
        .expect("should build with system_prompt");
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[0].role, "system");
    assert_eq!(msgs[0].content, "You are helpful");
    assert_eq!(msgs[1].role, "user");
    assert_eq!(msgs[1].content, "Hello");
}

#[test]
fn build_messages_from_messages_list() {
    let input = vec![
        ChatMessage {
            role: "user".into(),
            content: "Hi".into(),
        },
        ChatMessage {
            role: "assistant".into(),
            content: "Hello".into(),
        },
        ChatMessage {
            role: "user".into(),
            content: "How are you?".into(),
        },
    ];
    let msgs =
        GenerationParams::build_messages(None, None, Some(input)).expect("should use messages");
    assert_eq!(msgs.len(), 3);
    assert_eq!(msgs[0].role, "user");
    assert_eq!(msgs[2].content, "How are you?");
}

#[test]
fn build_messages_with_system_prompt_and_messages_list() {
    let input = vec![ChatMessage {
        role: "user".into(),
        content: "Hi".into(),
    }];
    let msgs = GenerationParams::build_messages(None, Some("Be concise"), Some(input))
        .expect("should prepend system_prompt");
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[0].role, "system");
    assert_eq!(msgs[0].content, "Be concise");
    assert_eq!(msgs[1].role, "user");
}

#[test]
fn build_messages_prefers_messages_over_prompt() {
    let input = vec![ChatMessage {
        role: "user".into(),
        content: "From messages".into(),
    }];
    let msgs = GenerationParams::build_messages(Some("From prompt"), None, Some(input))
        .expect("should prefer messages");
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0].content, "From messages");
}

#[test]
fn build_messages_fails_when_neither_prompt_nor_messages() {
    let err = GenerationParams::build_messages(None, None, None).unwrap_err();
    let msg = format!("{:?}", err);
    assert!(msg.contains("Either 'prompt' or 'messages'"));
}

#[test]
fn chat_request_serialization_omits_none_fields() {
    let params = GenerationParams {
        messages: vec![ChatMessage {
            role: "user".into(),
            content: "Hi".into(),
        }],
        temperature: None,
        max_tokens: None,
        top_p: None,
        stop: None,
        frequency_penalty: None,
        presence_penalty: None,
        seed: None,
        response_format: None,
    };
    let req = params.into_chat_request("gpt-4".into(), None, None);
    let json = serde_json::to_string(&req).expect("should serialise");

    assert!(!json.contains("temperature"));
    assert!(!json.contains("max_tokens"));
    assert!(!json.contains("stream"));
    assert!(!json.contains("response_format"));
    assert!(!json.contains("stop"));
    assert!(!json.contains("seed"));
    assert!(!json.contains("stream_options"));

    assert!(json.contains("model"));
    assert!(json.contains("messages"));
}

#[test]
fn chat_request_serialization_includes_set_fields() {
    let params = GenerationParams {
        messages: vec![ChatMessage {
            role: "user".into(),
            content: "Hi".into(),
        }],
        temperature: Some(0.7),
        max_tokens: Some(100),
        top_p: None,
        stop: Some(serde_json::json!(["END", "STOP"])),
        frequency_penalty: None,
        presence_penalty: None,
        seed: Some(42),
        response_format: Some(serde_json::json!({"type": "json_object"})),
    };
    let req = params.into_chat_request("gpt-4".into(), Some(true), None);
    let json: serde_json::Value = serde_json::to_value(&req).expect("should serialise");

    assert_eq!(json["temperature"], 0.7);
    assert_eq!(json["max_tokens"], 100);
    assert_eq!(json["stream"], true);
    assert_eq!(json["seed"], 42);
    assert_eq!(json["response_format"]["type"], "json_object");
    assert!(json.get("top_p").is_none());
    assert!(json.get("frequency_penalty").is_none());
    assert!(json.get("stream_options").is_none());
}

#[test]
fn chat_request_includes_stream_options_when_set() {
    let params = GenerationParams {
        messages: vec![ChatMessage {
            role: "user".into(),
            content: "Hi".into(),
        }],
        temperature: None,
        max_tokens: None,
        top_p: None,
        stop: None,
        frequency_penalty: None,
        presence_penalty: None,
        seed: None,
        response_format: None,
    };
    let stream_opts = serde_json::json!({"include_usage": true});
    let req = params.into_chat_request("gpt-4".into(), Some(true), Some(stream_opts));
    let json: serde_json::Value = serde_json::to_value(&req).expect("should serialise");

    assert_eq!(json["stream_options"]["include_usage"], true);
    assert_eq!(json["stream"], true);
}

#[test]
fn chat_request_omits_stream_options_when_none() {
    let params = GenerationParams {
        messages: vec![ChatMessage {
            role: "user".into(),
            content: "Hi".into(),
        }],
        temperature: None,
        max_tokens: None,
        top_p: None,
        stop: None,
        frequency_penalty: None,
        presence_penalty: None,
        seed: None,
        response_format: None,
    };
    let req = params.into_chat_request("gpt-4".into(), Some(true), None);
    let json = serde_json::to_string(&req).expect("should serialise");

    assert!(!json.contains("stream_options"));
}
