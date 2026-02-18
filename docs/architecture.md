# Architecture

This document describes the internal Rust architecture of rusty-agent-sdk for contributors who want to understand, modify, or extend the codebase.

## Module Map

| File | Lines | Purpose |
|------|-------|---------|
| `lib.rs` | ~40 | PyO3 module root. Declares submodules, re-exports `Provider`, `TextStream`, and `GenerateResult`. Exposes a `pub mod internal` for integration test access to internal types. |
| `provider.rs` | ~640 | `Provider` pyclass: model, API key, base URL, timeouts, retry config. `GenerateResult` pyclass with token usage getters. Preset constructors (`openai`, `anthropic`, `openrouter`). Python-to-Rust conversion helpers (`py_to_json`, `extract_messages`, `extract_stop`, `build_generation_params`). |
| `generate.rs` | ~93 | `generate_text()` implementation: blocking LLM call via `tokio::runtime::Runtime::block_on`. Generic `run_request` function parameterized over a parser function (`parse_chat_response` or `parse_chat_response_full`). |
| `stream.rs` | ~381 | `stream_text()` + `TextStream` iterator. Background thread with its own tokio runtime. Uses `mpsc::sync_channel(128)` for backpressure. `AtomicBool` cancellation with 100ms poll interval. |
| `models.rs` | ~311 | Serde types for OpenAI-compatible chat completions (request/response). SSE parsing (`parse_sse_line`, `parse_sse_event`, `parse_sse_data`). `GenerationParams`, `ChatRequest`, `Usage`, `StreamMetadata`, `StreamEvent` types. |
| `http.rs` | ~23 | Retry helpers: retryable status codes (429, 500, 502, 503, 504), retryable errors (timeout, connect, request), exponential backoff delay (`base * 2^attempt`, capped at `2^8`). |
| `errors.rs` | ~32 | `SdkError` enum with three variants: `Connection` maps to `PyConnectionError`, `Runtime` maps to `PyRuntimeError`, `Value` maps to `PyValueError`. |

## Data Flow

### generate_text() Flow

```
Python call
  |
  v
build_generation_params()          # Extract Python kwargs into GenerationParams
  |
  v
GenerationParams::into_chat_request()  # Convert to serializable ChatRequest
  |
  v
run_request()                      # Generic over parser function
  |
  v
tokio::runtime::Runtime::new()     # Fresh runtime per call
  |
  v
Runtime::block_on(async { ... })
  |
  v
HTTP POST with retry loop         # reqwest POST to /chat/completions
  |                                # Retry on 429/5xx with exponential backoff
  v
parse response                     # parse_chat_response -> String
  |                                # parse_chat_response_full -> ParsedChatResult
  v
Return str or GenerateResult       # Depends on include_usage flag
```

### stream_text() Flow

```
Python call
  |
  v
build_generation_params()          # Extract Python kwargs into GenerationParams
  |
  v
GenerationParams::into_chat_request()  # Sets stream=true, optionally stream_options
  |
  v
spawn background thread           # std::thread::spawn
  |
  v
Thread creates Runtime::new()      # Own tokio runtime in background thread
  |
  v
HTTP POST with retry loop         # Same retry logic as generate
  |
  v
response.bytes_stream()           # Streaming byte chunks via futures_util::StreamExt
  |
  v
line_buffer -> event_buffer        # Accumulate bytes into lines, lines into SSE events
  |
  v
parse_sse_event()                  # Extract StreamEvent variants from SSE data
  |
  v
sync_channel(128) sender          # Send chunks through bounded channel
  |
  v
Python iterator recv()            # TextStream.__next__ calls recv(), blocks naturally
```

## Key Design Decisions

### Why block_on for generate_text

PyO3 requires synchronous return values from Python-callable methods. The `#[pymethods]` functions cannot be async. Each `generate_text` call creates a fresh `tokio::runtime::Runtime` and calls `block_on` to execute the async HTTP request. This is simple and avoids the complexity of managing a long-lived runtime across the Python-Rust boundary.

### Why a background thread for streaming

Streaming cannot hold the GIL while waiting for chunks from the network. The solution is to spawn a `std::thread` that owns its own async runtime. The thread reads from the HTTP stream and pushes chunks through a `sync_channel`. The Python-side `TextStream.__next__` calls `recv()`, which naturally blocks and releases the GIL while waiting.

### Why sync_channel(128)

A bounded channel provides backpressure. If the Python consumer is slow (e.g., doing expensive processing per chunk), the producer thread blocks once 128 chunks are buffered. This prevents unbounded memory growth without requiring explicit flow control.

### Why AtomicBool for cancellation

The `cancel_flag` is an `Arc<AtomicBool>` shared between the `TextStream` Python object and the background thread. It is checked:

- At every 100ms poll interval during `timeout(STREAM_CANCEL_POLL_INTERVAL, stream.next()).await`
- Before each retry attempt
- During retry backoff sleep (polled every 100ms via `sleep_with_cancellation`)

When `TextStream` is dropped (Python garbage collection or explicit `del`), its `Drop` implementation sets the cancel flag and joins the background thread.

### Why cdylib + rlib crate types

- `cdylib`: Required for building the Python extension module (`.so`/`.pyd`/`.dylib` binary)
- `rlib`: Enables Rust integration tests in `tests/` to import internal types through the `pub mod internal` re-export in `lib.rs`

## Dependency Choices

| Crate | Features | Rationale |
|-------|----------|-----------|
| `pyo3` | `abi3-py39` | Single wheel binary works across Python 3.9 through 3.13+. Uses the Python Stable ABI. |
| `reqwest` | `json`, `rustls`, `stream` | `rustls` avoids system OpenSSL dependency, producing portable wheels. `stream` enables `bytes_stream()` for streaming responses. `json` provides `.json()` request builder. |
| `tokio` | `rt-multi-thread`, `time` | `rt-multi-thread` required for `Runtime::new()` in both per-call (generate) and per-stream (streaming) threads. `time` for `sleep` and `timeout`. |
| `futures-util` | `sink`, `std` | `StreamExt` trait for iterating over `bytes_stream()` chunks. |
| `serde` / `serde_json` | `derive` | Serialization/deserialization of chat completion request and response JSON. |

## Error Handling

All internal errors use the `SdkError` enum, which maps cleanly to Python exceptions:

```rust
enum SdkError {
    Connection(String),  // -> Python ConnectionError  (network failures)
    Runtime(String),     // -> Python RuntimeError     (API errors, parse failures)
    Value(String),       // -> Python ValueError       (invalid arguments)
}
```

The `into_pyerr()` method converts an `SdkError` into the appropriate `PyErr`. This keeps error creation and conversion separate, allowing the core logic to work with `Result<T, SdkError>` without PyO3 imports.

## Test Structure

All integration tests live in the `tests/` directory and access internal types through `rusty_agent_sdk::internal::*`.

### tests/generate_parsing.rs

Tests for non-streaming response parsing:

- `parse_chat_response` extracts the first choice content from a valid response
- `parse_chat_response` fails on empty choices array
- `parse_chat_response` fails on invalid JSON
- `parse_chat_response_full` extracts text, usage, finish_reason, and model
- `parse_chat_response_full` handles missing optional fields gracefully
- `api_error_message` extracts structured error messages when available
- `api_error_message` falls back to raw response body

### tests/stream_parsing.rs

Tests for SSE (Server-Sent Events) parsing:

- `parse_sse_line` extracts content chunks from `data:` lines
- `parse_sse_line` recognizes the `[DONE]` sentinel
- `parse_sse_line` ignores non-data lines (e.g., `event:` lines)
- `parse_sse_line` ignores empty content deltas
- `parse_sse_line` returns errors for malformed JSON payloads
- `parse_sse_event` joins multi-line data payloads correctly
- `parse_sse_event` ignores events without data lines

### tests/request_building.rs

Tests for message building and request serialization:

- `build_messages` from prompt only, with system prompt, from message list, with both
- `build_messages` prefers messages list over prompt when both are provided
- `build_messages` fails when neither prompt nor messages is provided
- `ChatRequest` serialization omits `None` fields (uses `skip_serializing_if`)
- `ChatRequest` serialization includes set fields with correct values
- `stream_options` included/omitted based on presence

### tests/provider_config.rs

Tests for provider configuration and URL building:

- `resolve_provider_values` uses environment key when no explicit key is provided
- `resolve_provider_values` prefers explicit API key over environment variable
- `resolve_provider_values` errors when no API key is available
- `build_chat_completions_url` normalizes trailing slashes
- `resolve_runtime_config` uses default values when no environment variables are set
- `resolve_runtime_config` reads custom environment values correctly
- `resolve_runtime_config` rejects invalid values (zero timeouts, non-numeric strings)

## Cross-Compilation Notes

The CI pipeline builds wheels for 8 platform targets:

| Target | Notes |
|--------|-------|
| Linux x86_64 (manylinux) | Standard glibc build |
| Linux aarch64 (manylinux) | Cross-compiled with `--zig` flag (required because `aws-lc-sys` fails without it) |
| Linux x86_64 (musllinux) | Static musl libc for Alpine and similar |
| Linux aarch64 (musllinux) | Cross-compiled with `--zig` flag |
| Windows x64 | MSVC target |
| macOS x86_64 | Intel Macs |
| macOS aarch64 | Apple Silicon |
| sdist | Source distribution for fallback pip builds |

The `--zig` flag must be passed inside the `args` field of `maturin-action@v1`, not as a separate input parameter. This is a known gotcha with the maturin GitHub Action.

## Related Documentation

- [API Reference](api-reference.md)
- [Contributing](contributing.md)
- [Configuration](configuration.md)
