use crate::errors::SdkError;
use crate::http::{is_retryable_error, is_retryable_status, retry_delay};
use crate::models::{
    ChatMessage, StreamChatRequest, StreamEvent, api_error_message, parse_sse_event,
};
use crate::provider::{Provider, build_chat_completions_url};
use futures_util::StreamExt;
use pyo3::prelude::*;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::thread::JoinHandle;
use std::time::Duration;
use tokio::time::{Instant, sleep, timeout};

const STREAM_CHANNEL_CAPACITY: usize = 128;
const STREAM_CANCEL_POLL_INTERVAL: Duration = Duration::from_millis(100);

struct StreamWorkerConfig {
    url: String,
    api_key: String,
    body: StreamChatRequest,
    request_timeout: Duration,
    connect_timeout: Duration,
    max_retries: u32,
    retry_backoff: Duration,
    cancel_flag: Arc<AtomicBool>,
}

/// An iterator that yields text chunks from a streaming LLM response.
#[pyclass]
pub struct TextStream {
    receiver: Mutex<Receiver<Result<String, SdkError>>>,
    cancel_flag: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl Drop for TextStream {
    fn drop(&mut self) {
        self.cancel_flag.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
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
    let cancel_flag = Arc::new(AtomicBool::new(false));

    let url = build_chat_completions_url(&provider.base_url);
    let api_key = provider.api_key.clone();
    let request_timeout = provider.request_timeout;
    let connect_timeout = provider.connect_timeout;
    let max_retries = provider.max_retries;
    let retry_backoff = provider.retry_backoff;
    let body = StreamChatRequest {
        model: provider.model.clone(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        }],
        stream: true,
    };
    let thread_cancel_flag = Arc::clone(&cancel_flag);
    let config = StreamWorkerConfig {
        url,
        api_key,
        body,
        request_timeout,
        connect_timeout,
        max_retries,
        retry_backoff,
        cancel_flag: thread_cancel_flag,
    };

    let handle = std::thread::spawn(move || {
        run_stream_thread(sender, config);
    });

    Ok(TextStream {
        receiver: Mutex::new(receiver),
        cancel_flag,
        handle: Some(handle),
    })
}

fn run_stream_thread(sender: SyncSender<Result<String, SdkError>>, config: StreamWorkerConfig) {
    let runtime = match tokio::runtime::Runtime::new() {
        Ok(runtime) => runtime,
        Err(e) => {
            let _ = sender.send(Err(SdkError::runtime(e.to_string())));
            return;
        }
    };

    runtime.block_on(async move {
        let StreamWorkerConfig {
            url,
            api_key,
            body,
            request_timeout,
            connect_timeout,
            max_retries,
            retry_backoff,
            cancel_flag,
        } = config;

        let client = match reqwest::Client::builder()
            .connect_timeout(connect_timeout)
            .build()
        {
            Ok(client) => client,
            Err(e) => {
                let _ = sender.send(Err(SdkError::runtime(e.to_string())));
                return;
            }
        };

        let mut response = None;
        for attempt in 0..=max_retries {
            if cancel_flag.load(Ordering::Relaxed) {
                return;
            }

            let response_result = client
                .post(&url)
                .header("Authorization", format!("Bearer {}", api_key))
                .header("Content-Type", "application/json")
                .timeout(request_timeout)
                .json(&body)
                .send()
                .await;

            match response_result {
                Ok(resp) => {
                    if resp.status().is_success() {
                        response = Some(resp);
                        break;
                    }

                    let status = resp.status();
                    let text = resp.text().await.unwrap_or_default();
                    if is_retryable_status(status) && attempt < max_retries {
                        if sleep_with_cancellation(
                            &cancel_flag,
                            retry_delay(retry_backoff, attempt),
                        )
                        .await
                        {
                            return;
                        }
                        continue;
                    }

                    let _ = sender.send(Err(SdkError::runtime(api_error_message(status, &text))));
                    return;
                }
                Err(error) => {
                    if is_retryable_error(&error) && attempt < max_retries {
                        if sleep_with_cancellation(
                            &cancel_flag,
                            retry_delay(retry_backoff, attempt),
                        )
                        .await
                        {
                            return;
                        }
                        continue;
                    }

                    let _ = sender.send(Err(SdkError::connection(error.to_string())));
                    return;
                }
            }
        }

        let Some(response) = response else {
            let _ = sender.send(Err(SdkError::runtime(
                "Stream request failed after retries were exhausted.",
            )));
            return;
        };

        let mut stream = response.bytes_stream();
        let mut line_buffer = String::new();
        let mut event_buffer = String::new();
        let mut last_activity = Instant::now();

        loop {
            if cancel_flag.load(Ordering::Relaxed) {
                return;
            }

            let chunk_result = match timeout(STREAM_CANCEL_POLL_INTERVAL, stream.next()).await {
                Ok(chunk) => chunk,
                Err(_) => {
                    if last_activity.elapsed() >= request_timeout {
                        let _ = sender.send(Err(SdkError::runtime(format!(
                            "Streaming response timed out after {}s of inactivity.",
                            request_timeout.as_secs()
                        ))));
                        return;
                    }
                    continue;
                }
            };

            let Some(chunk_result) = chunk_result else {
                break;
            };

            let bytes = match chunk_result {
                Ok(bytes) => bytes,
                Err(e) => {
                    let _ = sender.send(Err(SdkError::runtime(e.to_string())));
                    return;
                }
            };
            last_activity = Instant::now();

            line_buffer.push_str(&String::from_utf8_lossy(&bytes));

            while let Some(newline_pos) = line_buffer.find('\n') {
                let mut line = line_buffer[..newline_pos].to_string();
                line_buffer = line_buffer[newline_pos + 1..].to_string();
                if line.ends_with('\r') {
                    line.pop();
                }

                if line.is_empty() {
                    if !event_buffer.is_empty() {
                        if handle_sse_event(&sender, &event_buffer) {
                            return;
                        }
                        event_buffer.clear();
                    }
                    continue;
                }

                if !event_buffer.is_empty() {
                    event_buffer.push('\n');
                }
                event_buffer.push_str(&line);
            }
        }

        let trailing_line = line_buffer.trim_end_matches('\r');
        if !trailing_line.is_empty() {
            if !event_buffer.is_empty() {
                event_buffer.push('\n');
            }
            event_buffer.push_str(trailing_line);
        }

        if !event_buffer.trim().is_empty() {
            let _ = handle_sse_event(&sender, &event_buffer);
        }
    });
}

async fn sleep_with_cancellation(cancel_flag: &AtomicBool, delay: Duration) -> bool {
    let start = Instant::now();
    while start.elapsed() < delay {
        if cancel_flag.load(Ordering::Relaxed) {
            return true;
        }
        sleep(STREAM_CANCEL_POLL_INTERVAL).await;
    }
    false
}

fn handle_sse_event(sender: &SyncSender<Result<String, SdkError>>, event: &str) -> bool {
    match parse_sse_event(event) {
        Ok(StreamEvent::Done) => true,
        Ok(StreamEvent::Content(content)) => sender.send(Ok(content)).is_err(),
        Ok(StreamEvent::Ignore) => false,
        Err(err) => {
            let _ = sender.send(Err(err));
            true
        }
    }
}
