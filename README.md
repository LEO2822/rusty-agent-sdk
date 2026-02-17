# rusty-agent-sdk

[![PyPI](https://img.shields.io/pypi/v/rusty-agent-sdk)](https://pypi.org/project/rusty-agent-sdk/)
[![License](https://img.shields.io/pypi/l/rusty-agent-sdk)](LICENSE)

Rust-powered Python SDK for OpenAI-compatible text generation and streaming.

## Installation

```bash
pip install rusty-agent-sdk
```

## Quick Start

Set your API key:

```bash
export OPENROUTER_API_KEY="your-key"
```

Then use the SDK:

```python
from rusty_agent_sdk import Provider

provider = Provider("openai/gpt-4o-mini")

# Blocking generation
text = provider.generate_text("Say hello in one sentence.")
print(text)

# Streaming
for chunk in provider.stream_text("Count to 3."):
    print(chunk, end="", flush=True)
print()
```

## Custom Endpoint

```python
provider = Provider(
    "gpt-4o-mini",
    api_key="sk-...",
    base_url="https://api.openai.com/v1",
)
```

## Runtime Tuning (Optional)

Without changing code, you can tune networking behavior through environment variables:

```bash
export RUSTY_AGENT_REQUEST_TIMEOUT_SECS=60
export RUSTY_AGENT_CONNECT_TIMEOUT_SECS=10
export RUSTY_AGENT_MAX_RETRIES=2
export RUSTY_AGENT_RETRY_BACKOFF_MS=250
```

## Example

See [examples/basic_usage.py](examples/basic_usage.py) for a runnable demo.

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
