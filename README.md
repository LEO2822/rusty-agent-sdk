# rusty-agent-sdk

[![PyPI](https://img.shields.io/pypi/v/rusty-agent-sdk)](https://pypi.org/project/rusty-agent-sdk/)
[![License](https://img.shields.io/pypi/l/rusty-agent-sdk)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/rusty-agent-sdk)](https://pypi.org/project/rusty-agent-sdk/)
[![Platforms](https://img.shields.io/badge/platform-linux%20%7C%20macos%20%7C%20windows-blue)](https://pypi.org/project/rusty-agent-sdk/)

A Rust-powered Python SDK for OpenAI-compatible text generation and streaming.

- Native Rust performance with zero Python runtime dependencies
- Blocking and streaming text generation
- Multi-turn conversations with full message history
- System prompts and generation parameters (temperature, top_p, max_tokens, etc.)
- JSON mode via `response_format`
- Token usage tracking and response metadata
- Provider presets for OpenAI, Anthropic, and OpenRouter
- Automatic retries with exponential backoff on transient errors
- Configurable timeouts via environment variables
- Pre-built wheels for Linux (x86/ARM), macOS (x86/ARM), and Windows (x64)

## Installation

```bash
uv add rusty-agent-sdk
```

Or with pip:

```bash
pip install rusty-agent-sdk
```

Requires Python 3.9 or later. No additional dependencies are needed -- everything is compiled into the wheel.

## Quick Start

Set your API key (defaults to OpenRouter):

```bash
export OPENROUTER_API_KEY="your-key-here"
```

### Basic Generation

```python
from rusty_agent_sdk import Provider

provider = Provider("openai/gpt-4o-mini")
response = provider.generate_text("Explain Rust in one sentence.")
print(response)
```

### Streaming

Streaming is true real-time SSE -- each token is delivered to the Python iterator the moment the model generates it, with no buffering.

```python
from rusty_agent_sdk import Provider

provider = Provider("openai/gpt-4o-mini")
for chunk in provider.stream_text("Count from 1 to 5."):
    print(chunk, end="", flush=True)
print()
```

### Generation Parameters

```python
from rusty_agent_sdk import Provider

provider = Provider("openai/gpt-4o-mini")
response = provider.generate_text(
    "Write a haiku about systems programming.",
    system_prompt="You are a poet who writes concise verse.",
    temperature=0.7,
    max_tokens=100,
)
print(response)
```

## Provider Presets

Convenience constructors configure the correct base URL and API key environment variable for each provider.

```python
from rusty_agent_sdk import Provider

# OpenAI (reads OPENAI_API_KEY)
provider = Provider.openai("gpt-4o-mini")

# Anthropic (reads ANTHROPIC_API_KEY)
provider = Provider.anthropic("claude-sonnet-4-20250514")

# OpenRouter (reads OPENROUTER_API_KEY) -- same as the default constructor
provider = Provider.openrouter("openai/gpt-4o-mini")
```

Each preset also accepts an explicit `api_key` keyword argument to override the environment variable.

## Token Usage Tracking

Pass `include_usage=True` to get token counts and metadata alongside the response.

### With generate_text

```python
from rusty_agent_sdk import Provider

provider = Provider("openai/gpt-4o-mini")
result = provider.generate_text("Hello!", include_usage=True)

print(result.text)
print(f"Prompt tokens: {result.prompt_tokens}")
print(f"Completion tokens: {result.completion_tokens}")
print(f"Total tokens: {result.total_tokens}")
print(f"Model: {result.model}")
print(f"Finish reason: {result.finish_reason}")
```

### With stream_text

```python
from rusty_agent_sdk import Provider

provider = Provider("openai/gpt-4o-mini")
stream = provider.stream_text("Hello!", include_usage=True)

for chunk in stream:
    print(chunk, end="", flush=True)
print()

# Metadata is available after the stream is fully consumed
print(f"Prompt tokens: {stream.prompt_tokens}")
print(f"Completion tokens: {stream.completion_tokens}")
print(f"Total tokens: {stream.total_tokens}")
```

## Multi-Turn Conversations

Pass a list of message dicts for full conversation history:

```python
from rusty_agent_sdk import Provider

provider = Provider("openai/gpt-4o-mini")
response = provider.generate_text(messages=[
    {"role": "user", "content": "My name is Alice."},
    {"role": "assistant", "content": "Nice to meet you, Alice!"},
    {"role": "user", "content": "What is my name?"},
])
print(response)
```

## JSON Mode

```python
from rusty_agent_sdk import Provider

provider = Provider("openai/gpt-4o-mini")
response = provider.generate_text(
    "List 3 colors as JSON.",
    response_format={"type": "json_object"},
)
print(response)
```

## Runtime Configuration

Networking behavior can be tuned via environment variables without changing code:

| Variable | Default | Description |
|---|---|---|
| `RUSTY_AGENT_REQUEST_TIMEOUT_SECS` | 60 | Total request timeout in seconds |
| `RUSTY_AGENT_CONNECT_TIMEOUT_SECS` | 10 | TCP connection timeout in seconds |
| `RUSTY_AGENT_MAX_RETRIES` | 2 | Max retries on 429/5xx errors |
| `RUSTY_AGENT_RETRY_BACKOFF_MS` | 250 | Base backoff between retries in milliseconds |

## Documentation

For detailed documentation, see the `docs/` directory:

- [Getting Started](docs/getting-started.md) -- onboarding guide with step-by-step setup
- [API Reference](docs/api-reference.md) -- complete reference for all classes and methods
- [Examples](docs/examples.md) -- copy-paste recipes for streaming, multi-turn, JSON mode, and more
- [Configuration](docs/configuration.md) -- environment variables, timeouts, and retry behavior
- [Architecture](docs/architecture.md) -- Rust internals for contributors
- [Contributing](docs/contributing.md) -- dev setup, testing, and release process

## Development

```bash
# Build and install locally
uv run maturin develop

# Run checks
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

## License

Apache 2.0
