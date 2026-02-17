# Configuration

Reference for all configuration options, environment variables, and runtime behavior.

## API Key Resolution

The SDK resolves API keys in the following order (first match wins):

1. **Explicit `api_key` parameter** passed to the constructor or class method.
2. **Provider-specific environment variable** (see table below).
3. **`ValueError`** is raised if no key is found.

```python
from rusty_agent_sdk import Provider

# 1. Explicit key (highest priority)
provider = Provider("openai/gpt-4o-mini", api_key="sk-or-...")

# 2. Environment variable (set OPENROUTER_API_KEY before running)
provider = Provider("openai/gpt-4o-mini")
```

### Provider Environment Variables

Each provider preset reads from a different environment variable:

| Constructor                  | Environment Variable    |
|------------------------------|-------------------------|
| `Provider(model)`           | `OPENROUTER_API_KEY`    |
| `Provider.openrouter(model)`| `OPENROUTER_API_KEY`    |
| `Provider.openai(model)`    | `OPENAI_API_KEY`        |
| `Provider.anthropic(model)` | `ANTHROPIC_API_KEY`     |

---

## Runtime Environment Variables

These environment variables control timeout, retry, and backoff behavior. They are read once at `Provider` construction time.

| Variable                              | Type   | Default | Constraint | Description                                |
|---------------------------------------|--------|---------|------------|--------------------------------------------|
| `RUSTY_AGENT_REQUEST_TIMEOUT_SECS`    | `u64`  | `60`    | Must be > 0 | Timeout for the entire HTTP request (seconds). Also used as the streaming inactivity timeout. |
| `RUSTY_AGENT_CONNECT_TIMEOUT_SECS`    | `u64`  | `10`    | Must be > 0 | Timeout for establishing the TCP connection (seconds). |
| `RUSTY_AGENT_MAX_RETRIES`             | `u32`  | `2`     | Must be >= 0 | Maximum number of retry attempts after the initial request fails. |
| `RUSTY_AGENT_RETRY_BACKOFF_MS`        | `u64`  | `250`   | Must be > 0 | Base delay between retries (milliseconds). Used in exponential backoff calculation. |

```bash
# Example: increase timeouts and retries for unreliable networks
export RUSTY_AGENT_REQUEST_TIMEOUT_SECS=120
export RUSTY_AGENT_CONNECT_TIMEOUT_SECS=30
export RUSTY_AGENT_MAX_RETRIES=5
export RUSTY_AGENT_RETRY_BACKOFF_MS=500
```

Note: Invalid values (non-numeric, zero for timeout/backoff variables) cause a `ValueError` at construction time.

---

## Retry Behavior

The SDK automatically retries failed requests using exponential backoff.

### Retryable Status Codes

The following HTTP status codes trigger a retry:

| Status Code | Meaning               |
|-------------|-----------------------|
| 429         | Too Many Requests     |
| 500         | Internal Server Error |
| 502         | Bad Gateway           |
| 503         | Service Unavailable   |
| 504         | Gateway Timeout       |

All other non-2xx status codes result in an immediate `RuntimeError` without retrying.

### Retryable Errors

In addition to retryable status codes, these request-level errors also trigger retries:

- **Timeout errors** -- the request exceeded the configured timeout.
- **Connection errors** -- failed to establish a TCP connection.
- **Request errors** -- other transport-level failures.

### Exponential Backoff Formula

The delay before each retry attempt is calculated as:

```
delay = base_backoff * 2^attempt
```

Where:
- `base_backoff` is `RUSTY_AGENT_RETRY_BACKOFF_MS` (default: 250ms)
- `attempt` is the zero-indexed retry number (0, 1, 2, ...)
- The exponent is capped at 8, so the maximum multiplier is 256

**Example with defaults** (base = 250ms, max_retries = 2):

| Attempt | Delay   |
|---------|---------|
| 0       | 250 ms  |
| 1       | 500 ms  |

**Example with 5 retries** (base = 500ms):

| Attempt | Delay     |
|---------|-----------|
| 0       | 500 ms    |
| 1       | 1,000 ms  |
| 2       | 2,000 ms  |
| 3       | 4,000 ms  |
| 4       | 8,000 ms  |

---

## Timeout Behavior

### Connect Timeout

Controlled by `RUSTY_AGENT_CONNECT_TIMEOUT_SECS` (default: 10 seconds).

This is the maximum time allowed to establish a TCP connection to the API server. If the connection is not established within this window, the request fails and may be retried.

### Request Timeout

Controlled by `RUSTY_AGENT_REQUEST_TIMEOUT_SECS` (default: 60 seconds).

For `generate_text()`, this is the maximum time for the entire HTTP request-response cycle, including sending the request and receiving the full response body.

For `stream_text()`, this timeout applies to each individual chunk read. If no new data arrives within the timeout window, the stream is terminated. This acts as a streaming inactivity timeout -- as long as chunks keep arriving, the stream can run indefinitely.

---

## .env File Pattern

A common pattern is to use a `.env` file with `python-dotenv` to manage API keys and configuration:

```bash
uv add python-dotenv
```

Create a `.env` file in your project root:

```
# .env
OPENROUTER_API_KEY=sk-or-your-key-here
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key

# Optional: customize runtime behavior
RUSTY_AGENT_REQUEST_TIMEOUT_SECS=120
RUSTY_AGENT_MAX_RETRIES=3
```

Load it before creating a `Provider`:

```python
from dotenv import load_dotenv
from rusty_agent_sdk import Provider

load_dotenv()  # reads .env and sets environment variables

provider = Provider("openai/gpt-4o-mini")
response = provider.generate_text("Hello!")
print(response)
```

Note: Add `.env` to your `.gitignore` to avoid committing API keys to version control.
