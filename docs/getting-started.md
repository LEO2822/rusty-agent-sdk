# Getting Started

Get up and running with `rusty-agent-sdk` in minutes.

## Prerequisites

- **Python 3.9 or later**
- **An API key** from at least one LLM provider:
  - [OpenRouter](https://openrouter.ai/) (default)
  - [OpenAI](https://platform.openai.com/)
  - [Anthropic](https://console.anthropic.com/)

## Installation

```bash
uv add rusty-agent-sdk
```

Or with pip:

```bash
pip install rusty-agent-sdk
```

No additional dependencies are required -- the SDK is a self-contained Rust binary extension.

## API Key Setup

The SDK needs an API key to authenticate with your LLM provider. There are three ways to provide one, in order of priority:

### 1. Explicit parameter (highest priority)

```python
from rusty_agent_sdk import Provider

provider = Provider("openai/gpt-4o-mini", api_key="sk-...")
```

### 2. Environment variable

Set the appropriate environment variable for your provider:

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

Then create a provider without specifying `api_key`:

```python
provider = Provider("openai/gpt-4o-mini")
```

### 3. `.env` file with python-dotenv

Install `python-dotenv` and create a `.env` file in your project root:

```bash
uv add python-dotenv
```

```
# .env
OPENROUTER_API_KEY=sk-or-...
```

```python
from dotenv import load_dotenv
from rusty_agent_sdk import Provider

load_dotenv()
provider = Provider("openai/gpt-4o-mini")
```

If no API key is found through any of these methods, a `ValueError` is raised at construction time.

## Your First Generation

```python
from dotenv import load_dotenv
from rusty_agent_sdk import Provider

# Load API key from .env file
load_dotenv()

# Create a provider targeting OpenRouter (the default)
provider = Provider("openai/gpt-4o-mini")

# Generate a complete response (blocking call)
response = provider.generate_text("What is the capital of France?")

# response is a plain string
print(response)
```

Line-by-line:

1. `load_dotenv()` reads your `.env` file and sets environment variables.
2. `Provider("openai/gpt-4o-mini")` creates a provider instance. It reads `OPENROUTER_API_KEY` from the environment and targets the OpenRouter API.
3. `generate_text("What is the capital of France?")` sends a blocking HTTP request to the chat completions endpoint and returns the model's response as a string.

## Your First Stream

```python
from dotenv import load_dotenv
from rusty_agent_sdk import Provider

load_dotenv()
provider = Provider("openai/gpt-4o-mini")

# Stream the response token by token
for chunk in provider.stream_text("Write a haiku about programming."):
    print(chunk, end="", flush=True)

print()  # newline after streaming completes
```

Line-by-line:

1. `stream_text(...)` returns a `TextStream` iterator that yields string chunks as they arrive.
2. The `for` loop prints each chunk immediately with `end=""` to avoid extra newlines and `flush=True` to ensure output appears in real time.
3. A background thread handles the HTTP streaming connection, sending chunks through a channel to the Python iterator.

## Provider Presets

Instead of manually specifying `base_url`, use the built-in class methods for common providers:

| Preset                       | Base URL                          | Env Var              | Example Model                    |
|------------------------------|-----------------------------------|----------------------|----------------------------------|
| `Provider(model)`           | `https://openrouter.ai/api/v1`   | `OPENROUTER_API_KEY` | `"openai/gpt-4o-mini"`          |
| `Provider.openai(model)`    | `https://api.openai.com/v1`      | `OPENAI_API_KEY`     | `"gpt-4o-mini"`                 |
| `Provider.anthropic(model)` | `https://api.anthropic.com/v1`   | `ANTHROPIC_API_KEY`  | `"claude-sonnet-4-20250514"` |
| `Provider.openrouter(model)`| `https://openrouter.ai/api/v1`   | `OPENROUTER_API_KEY` | `"openai/gpt-4o-mini"`          |

```python
# OpenAI direct
openai = Provider.openai("gpt-4o-mini")

# Anthropic direct
anthropic = Provider.anthropic("claude-sonnet-4-20250514")

# OpenRouter (explicit preset, same as default constructor)
openrouter = Provider.openrouter("openai/gpt-4o-mini")
```

## Next Steps

- [API Reference](api-reference.md) -- full method signatures, parameters, and return types.
- [Examples](examples.md) -- copy-paste recipes for common tasks.
- [Configuration](configuration.md) -- timeouts, retries, and environment variables.
