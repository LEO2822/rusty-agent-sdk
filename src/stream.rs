use crate::errors::SdkError;
use crate::models::{
    ChatMessage, StreamChatRequest, StreamEvent, api_error_message, parse_sse_line,
};
use crate::provider::{Provider, build_chat_completions_url};
use futures_util::StreamExt;
use pyo3::prelude::*;
use std::sync::Mutex;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::thread::JoinHandle;

const STREAM_CHANNEL_CAPACITY: usize = 128;

/// An iterator that yields text chunks from a streaming LLM response.
#[pyclass]
pub struct TextStream {
    receiver: Mutex<Receiver<Result<String, SdkError>>>,
    _handle: Option<JoinHandle<()>>,
}

#[pymethods]
impl TextStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self) -> Option<PyResult<String>> {
        let receiver = match self.receiver.lock() {
            Ok(receiver) => receiver,
            Err(_) => {
                return Some(Err(SdkError::runtime(
                    "Internal stream state is unavailable.",
                )
                .into_pyerr()));
            }
        };

        match receiver.recv() {
            Ok(Ok(chunk)) => Some(Ok(chunk)),
            Ok(Err(err)) => Some(Err(err.into_pyerr())),
            Err(_) => None,
        }
    }
}

/// Core streaming logic, called by `Provider.stream_text()`.
pub fn run(provider: &Provider, prompt: &str) -> PyResult<TextStream> {
    let (sender, receiver) = sync_channel::<Result<String, SdkError>>(STREAM_CHANNEL_CAPACITY);

    let url = build_chat_completions_url(&provider.base_url);
    let api_key = provider.api_key.clone();
    let body = StreamChatRequest {
        model: provider.model.clone(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        }],
        stream: true,
    };

    let handle = std::thread::spawn(move || {
        run_stream_thread(sender, url, api_key, body);
    });

    Ok(TextStream {
        receiver: Mutex::new(receiver),
        _handle: Some(handle),
    })
}

fn run_stream_thread(
    sender: SyncSender<Result<String, SdkError>>,
    url: String,
    api_key: String,
    body: StreamChatRequest,
) {
    let runtime = match tokio::runtime::Runtime::new() {
        Ok(runtime) => runtime,
        Err(e) => {
            let _ = sender.send(Err(SdkError::runtime(e.to_string())));
            return;
        }
    };

    runtime.block_on(async move {
        let client = reqwest::Client::new();

        let response = match client
            .post(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
        {
            Ok(response) => response,
            Err(e) => {
                let _ = sender.send(Err(SdkError::connection(e.to_string())));
                return;
            }
        };

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            let _ = sender.send(Err(SdkError::runtime(api_error_message(status, &text))));
            return;
        }

        let mut stream = response.bytes_stream();
        let mut buffer = String::new();

        while let Some(chunk_result) = stream.next().await {
            let bytes = match chunk_result {
                Ok(bytes) => bytes,
                Err(e) => {
                    let _ = sender.send(Err(SdkError::runtime(e.to_string())));
                    return;
                }
            };

            buffer.push_str(&String::from_utf8_lossy(&bytes));

            while let Some(newline_pos) = buffer.find('\n') {
                let line = buffer[..newline_pos].to_string();
                buffer = buffer[newline_pos + 1..].to_string();

                if handle_sse_line(&sender, &line) {
                    return;
                }
            }
        }

        if !buffer.trim().is_empty() {
            let _ = handle_sse_line(&sender, &buffer);
        }
    });
}

fn handle_sse_line(sender: &SyncSender<Result<String, SdkError>>, line: &str) -> bool {
    match parse_sse_line(line) {
        Ok(StreamEvent::Done) => true,
        Ok(StreamEvent::Content(content)) => sender.send(Ok(content)).is_err(),
        Ok(StreamEvent::Ignore) => false,
        Err(err) => {
            let _ = sender.send(Err(err));
            true
        }
    }
}
