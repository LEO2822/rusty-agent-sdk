# API Reference

Complete reference for the `rusty-agent-sdk` Python API.

## Provider

The `Provider` class is the single entry point for all LLM interactions. It holds the model identifier, API key, and base URL needed to route requests to any OpenAI-compatible chat completions endpoint.

### Constructor

```python
Provider(model: str, *, api_key: str | None = None, base_url: str | None = None)
```

| Parameter  | Type           | Default                              | Description                                       |
|------------|----------------|--------------------------------------|---------------------------------------------------|
| `model`    | `str`          | *(required)*                         | Model identifier, e.g. `"openai/gpt-4o-mini"`     |
| `api_key`  | `str \| None`  | `None`                               | API key. Falls back to `OPENROUTER_API_KEY` env var |
| `base_url` | `str \| None`  | `"https://openrouter.ai/api/v1"`     | Base URL of the OpenAI-compatible API              |

**Raises:** `ValueError` if no `api_key` is provided and the `OPENROUTER_API_KEY` environment variable is not set.

```python
from rusty_agent_sdk import Provider

# Uses OPENROUTER_API_KEY env var and default base URL
provider = Provider("openai/gpt-4o-mini")

# Explicit key and custom endpoint
provider = Provider(
    "gpt-4o-mini",
    api_key="sk-...",
    base_url="https://api.openai.com/v1",
)
```

### Class Methods (Provider Presets)

Pre-configured constructors for common providers. Each sets the appropriate base URL and reads the API key from the provider-specific environment variable.

#### `Provider.openai(model, *, api_key=None)`

```python
Provider.openai(model: str, *, api_key: str | None = None) -> Provider
```

- **base_url:** `https://api.openai.com/v1`
- **env var:** `OPENAI_API_KEY`

```python
provider = Provider.openai("gpt-4o-mini")
```

#### `Provider.anthropic(model, *, api_key=None)`

```python
Provider.anthropic(model: str, *, api_key: str | None = None) -> Provider
```

- **base_url:** `https://api.anthropic.com/v1`
- **env var:** `ANTHROPIC_API_KEY`

```python
provider = Provider.anthropic("claude-sonnet-4-20250514")
```

#### `Provider.openrouter(model, *, api_key=None)`

```python
Provider.openrouter(model: str, *, api_key: str | None = None) -> Provider
```

- **base_url:** `https://openrouter.ai/api/v1`
- **env var:** `OPENROUTER_API_KEY`

```python
provider = Provider.openrouter("openai/gpt-4o-mini")
```

---

## generate_text()

Generate a complete text response from the LLM (blocking call).

```python
provider.generate_text(
    prompt: str | None = None,
    *,
    system_prompt: str | None = None,
    messages: list[dict[str, str]] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    stop: str | list[str] | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    seed: int | None = None,
    response_format: dict | None = None,
    include_usage: bool = False,
) -> str | GenerateResult
```

### Parameters

| Parameter           | Type                       | Default | Description                                                                 |
|---------------------|----------------------------|---------|-----------------------------------------------------------------------------|
| `prompt`            | `str \| None`              | `None`  | User message shorthand. Ignored when `messages` is also provided.           |
| `system_prompt`     | `str \| None`              | `None`  | System prompt, prepended as a system message.                               |
| `messages`          | `list[dict] \| None`       | `None`  | Full conversation as `[{"role": ..., "content": ...}]`. Takes priority over `prompt`. |
| `temperature`       | `float \| None`            | `None`  | Sampling temperature, 0-2. API default is 1.                               |
| `max_tokens`        | `int \| None`              | `None`  | Maximum number of tokens to generate.                                       |
| `top_p`             | `float \| None`            | `None`  | Nucleus sampling threshold, 0-1. API default is 1.                          |
| `stop`              | `str \| list[str] \| None` | `None`  | Up to 4 stop sequences.                                                     |
| `frequency_penalty` | `float \| None`            | `None`  | Frequency penalty, -2 to 2. API default is 0.                              |
| `presence_penalty`  | `float \| None`            | `None`  | Presence penalty, -2 to 2. API default is 0.                               |
| `seed`              | `int \| None`              | `None`  | Random seed for deterministic generation.                                    |
| `response_format`   | `dict \| None`             | `None`  | Response format, e.g. `{"type": "json_object"}`.                            |
| `include_usage`     | `bool`                     | `False` | If `True`, returns a `GenerateResult` instead of a plain string.            |

### Returns

- **`str`** when `include_usage=False` (the default) -- the model's text response.
- **`GenerateResult`** when `include_usage=True` -- wraps the text along with token usage and metadata.

### Exceptions

| Exception         | Condition                                                  |
|-------------------|------------------------------------------------------------|
| `ConnectionError` | HTTP request failed (network error, timeout).              |
| `RuntimeError`    | API returned a non-2xx status code.                        |
| `ValueError`      | Response could not be parsed, or neither `prompt` nor `messages` was provided. |

### Message Priority

1. If `messages` is provided and non-empty, it is used. `prompt` is ignored.
2. If only `prompt` is provided, a single user message is created.
3. If `system_prompt` is provided, it is always prepended as a system message regardless of which input is used.
4. If neither `prompt` nor `messages` is provided, a `ValueError` is raised.

---

## stream_text()

Stream text from the LLM, returning an iterator of string chunks.

```python
provider.stream_text(
    prompt: str | None = None,
    *,
    system_prompt: str | None = None,
    messages: list[dict[str, str]] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    stop: str | list[str] | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    seed: int | None = None,
    response_format: dict | None = None,
    include_usage: bool = False,
) -> TextStream
```

Accepts the same parameters as [`generate_text()`](#generate_text). Always returns a `TextStream`.

When `include_usage=True`, token usage metadata is available on the `TextStream` object after the stream has been fully consumed.

### Exceptions

| Exception         | Condition                                          |
|-------------------|----------------------------------------------------|
| `ConnectionError` | Initial HTTP connection failed.                    |
| `RuntimeError`    | API returned a non-2xx status code.                |
| `ValueError`      | Neither `prompt` nor `messages` was provided.      |

---

## GenerateResult

Returned by `generate_text()` when `include_usage=True`. Wraps the generated text along with token usage statistics and metadata.

### Properties

| Property            | Type          | Description                                              |
|---------------------|---------------|----------------------------------------------------------|
| `text`              | `str`         | The model's complete text response.                      |
| `prompt_tokens`     | `int \| None` | Number of tokens in the prompt.                          |
| `completion_tokens` | `int \| None` | Number of tokens in the completion.                      |
| `total_tokens`      | `int \| None` | Total tokens used (prompt + completion).                 |
| `finish_reason`     | `str \| None` | Why the model stopped, e.g. `"stop"` or `"length"`.     |
| `model`             | `str \| None` | The model used, as reported by the API.                  |

### String Conversion

`str(result)` returns `result.text`, so a `GenerateResult` can be used anywhere a string is expected.

```python
result = provider.generate_text("Hello!", include_usage=True)
print(result.text)                # explicit access
print(result)                     # also prints the text
print(result.prompt_tokens)       # e.g. 12
print(result.completion_tokens)   # e.g. 45
print(result.total_tokens)        # e.g. 57
print(result.finish_reason)       # e.g. "stop"
print(result.model)               # e.g. "openai/gpt-4o-mini"
```

---

## TextStream

An iterator that yields `str` chunks from a streaming LLM response. Returned by `stream_text()`.

You do not construct `TextStream` directly -- it is created by `Provider.stream_text()`.

### Iterator Protocol

`TextStream` implements `__iter__` and `__next__`, so it can be used in a `for` loop:

```python
for chunk in provider.stream_text("Hello!"):
    print(chunk, end="", flush=True)
```

### Metadata Properties

When `include_usage=True` was passed to `stream_text()`, the following properties are available **after the stream has been fully consumed**:

| Property            | Type          | Description                                              |
|---------------------|---------------|----------------------------------------------------------|
| `prompt_tokens`     | `int \| None` | Number of tokens in the prompt.                          |
| `completion_tokens` | `int \| None` | Number of tokens in the completion.                      |
| `total_tokens`      | `int \| None` | Total tokens used (prompt + completion).                 |
| `finish_reason`     | `str \| None` | Why the model stopped generating.                        |
| `model`             | `str \| None` | The model used, as reported by the API.                  |

Note: These properties return `None` if `include_usage=False` (the default) or if the stream has not yet been fully consumed.

```python
stream = provider.stream_text("Hello!", include_usage=True)
for chunk in stream:
    print(chunk, end="", flush=True)

# Metadata available after full consumption
print(stream.prompt_tokens)
print(stream.completion_tokens)
print(stream.total_tokens)
print(stream.finish_reason)
print(stream.model)
```
