# rusty-agent-sdk

`rusty-agent-sdk` is a Rust-powered Python SDK for OpenAI-compatible text generation APIs.
It provides:

- `generate_text(...)` for blocking single-response generation
- `stream_text(...)` for Server-Sent Events (SSE) token streaming
- `Provider(...)` for API key and endpoint configuration

## Installation

Build and install locally with maturin:

```bash
uv run maturin develop
```

## Quick Start

Set your API key:

```bash
export OPENROUTER_API_KEY="your-key"
```

Then use the SDK:

```python
from rusty_agent_sdk import Provider, generate_text, stream_text

provider = Provider()

text = generate_text(provider, "openai/gpt-4o-mini", "Say hello in one sentence.")
print(text)

for chunk in stream_text(provider, "openai/gpt-4o-mini", "Count to 3."):
    print(chunk, end="", flush=True)
print()
```

## Example Script

See `/Users/a2954/Library/CloudStorage/OneDrive-JMRInfotechIndia(P)Ltd/Code/weekend-projects/rusty-agent-sdk/examples/basic_usage.py` for a runnable end-to-end example.

## Development Checks

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```
