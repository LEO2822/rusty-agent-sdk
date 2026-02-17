# Examples

A cookbook of copy-paste examples. Every snippet is complete and runnable -- just set the appropriate API key environment variable or `.env` file.

## Basic Generation

```python
from dotenv import load_dotenv
from rusty_agent_sdk import Provider

load_dotenv()
provider = Provider("openai/gpt-4o-mini")

response = provider.generate_text("What is the speed of light?")
print(response)
```

## Streaming

```python
from dotenv import load_dotenv
from rusty_agent_sdk import Provider

load_dotenv()
provider = Provider("openai/gpt-4o-mini")

for chunk in provider.stream_text("Explain quantum computing in simple terms."):
    print(chunk, end="", flush=True)
print()
```

## System Prompts

```python
from dotenv import load_dotenv
from rusty_agent_sdk import Provider

load_dotenv()
provider = Provider("openai/gpt-4o-mini")

response = provider.generate_text(
    "Tell me a joke",
    system_prompt="You are a stand-up comedian. Keep it short and clean.",
)
print(response)
```

## Multi-Turn Conversations

### Two-turn conversation

```python
from dotenv import load_dotenv
from rusty_agent_sdk import Provider

load_dotenv()
provider = Provider("openai/gpt-4o-mini")

response = provider.generate_text(messages=[
    {"role": "user", "content": "My name is Alice."},
    {"role": "assistant", "content": "Nice to meet you, Alice! How can I help you today?"},
    {"role": "user", "content": "What is my name?"},
])
print(response)
```

### Extended conversation

```python
from dotenv import load_dotenv
from rusty_agent_sdk import Provider

load_dotenv()
provider = Provider("openai/gpt-4o-mini")

response = provider.generate_text(messages=[
    {"role": "system", "content": "You are a helpful math tutor."},
    {"role": "user", "content": "What is 2 + 2?"},
    {"role": "assistant", "content": "2 + 2 equals 4."},
    {"role": "user", "content": "What about 2 + 3?"},
    {"role": "assistant", "content": "2 + 3 equals 5."},
    {"role": "user", "content": "Now multiply those two results together."},
    {"role": "assistant", "content": "4 times 5 equals 20."},
    {"role": "user", "content": "Is that a prime number?"},
    {"role": "assistant", "content": "No, 20 is not a prime number. It can be divided by 1, 2, 4, 5, 10, and 20."},
    {"role": "user", "content": "What is the next prime number after 20?"},
])
print(response)
```

### Multi-turn with system prompt shorthand

You can combine the `system_prompt` parameter with `messages` -- the system prompt is prepended automatically:

```python
from dotenv import load_dotenv
from rusty_agent_sdk import Provider

load_dotenv()
provider = Provider("openai/gpt-4o-mini")

response = provider.generate_text(
    system_prompt="You are a pirate. Always respond in pirate speak.",
    messages=[
        {"role": "user", "content": "Where is the treasure?"},
        {"role": "assistant", "content": "Arrr, the treasure be buried on Skull Island!"},
        {"role": "user", "content": "How do I get there?"},
    ],
)
print(response)
```

## Generation Parameters

### Temperature

```python
from dotenv import load_dotenv
from rusty_agent_sdk import Provider

load_dotenv()
provider = Provider("openai/gpt-4o-mini")

# Low temperature: more deterministic
response = provider.generate_text(
    "Name a color.",
    temperature=0.0,
)
print("Deterministic:", response)

# High temperature: more creative
response = provider.generate_text(
    "Name a color.",
    temperature=1.8,
)
print("Creative:", response)
```

### Max Tokens

```python
from dotenv import load_dotenv
from rusty_agent_sdk import Provider

load_dotenv()
provider = Provider("openai/gpt-4o-mini")

response = provider.generate_text(
    "Write a long story about a dragon.",
    max_tokens=50,
)
print(response)
```

### Top-p (Nucleus Sampling)

```python
from dotenv import load_dotenv
from rusty_agent_sdk import Provider

load_dotenv()
provider = Provider("openai/gpt-4o-mini")

response = provider.generate_text(
    "Suggest a creative project name.",
    top_p=0.5,
)
print(response)
```

### Stop Sequences

```python
from dotenv import load_dotenv
from rusty_agent_sdk import Provider

load_dotenv()
provider = Provider("openai/gpt-4o-mini")

# Single stop sequence
response = provider.generate_text(
    "Count from 1 to 10, separated by commas.",
    stop=",",
)
print("Stopped at first comma:", response)

# Multiple stop sequences
response = provider.generate_text(
    "List three fruits: apple, banana, cherry.",
    stop=[",", "."],
)
print("Stopped at comma or period:", response)
```

### Frequency and Presence Penalties

```python
from dotenv import load_dotenv
from rusty_agent_sdk import Provider

load_dotenv()
provider = Provider("openai/gpt-4o-mini")

response = provider.generate_text(
    "Write a paragraph about the ocean.",
    frequency_penalty=1.5,
    presence_penalty=0.5,
)
print(response)
```

### Seed (Deterministic Generation)

```python
from dotenv import load_dotenv
from rusty_agent_sdk import Provider

load_dotenv()
provider = Provider("openai/gpt-4o-mini")

response1 = provider.generate_text("Pick a random number.", seed=42, temperature=0.0)
response2 = provider.generate_text("Pick a random number.", seed=42, temperature=0.0)

print("Response 1:", response1)
print("Response 2:", response2)
print("Match:", response1 == response2)
```

## JSON Mode

### Basic JSON output

```python
from dotenv import load_dotenv
from rusty_agent_sdk import Provider

load_dotenv()
provider = Provider("openai/gpt-4o-mini")

response = provider.generate_text(
    "List 3 colors with their hex codes as JSON.",
    response_format={"type": "json_object"},
)
print(response)
```

### JSON with system prompt

```python
import json
from dotenv import load_dotenv
from rusty_agent_sdk import Provider

load_dotenv()
provider = Provider("openai/gpt-4o-mini")

response = provider.generate_text(
    "What are the 5 largest countries by area?",
    system_prompt="Respond only with valid JSON. Use the schema: {\"countries\": [{\"name\": str, \"area_km2\": int}]}",
    response_format={"type": "json_object"},
)
data = json.loads(response)
for country in data["countries"]:
    print(f"{country['name']}: {country['area_km2']:,} km2")
```

### Streaming JSON

```python
import json
from dotenv import load_dotenv
from rusty_agent_sdk import Provider

load_dotenv()
provider = Provider("openai/gpt-4o-mini")

chunks = []
for chunk in provider.stream_text(
    "List 3 programming languages with their year of creation as JSON.",
    response_format={"type": "json_object"},
):
    chunks.append(chunk)
    print(chunk, end="", flush=True)

print()

# Parse the full response after streaming completes
full_response = "".join(chunks)
data = json.loads(full_response)
print(data)
```

## Token Usage Tracking

### With generate_text

```python
from dotenv import load_dotenv
from rusty_agent_sdk import Provider

load_dotenv()
provider = Provider("openai/gpt-4o-mini")

result = provider.generate_text("Explain gravity in one sentence.", include_usage=True)

print("Response:", result.text)
print("Prompt tokens:", result.prompt_tokens)
print("Completion tokens:", result.completion_tokens)
print("Total tokens:", result.total_tokens)
print("Finish reason:", result.finish_reason)
print("Model:", result.model)

# GenerateResult can be used as a string
print("As string:", str(result))
```

### With stream_text

```python
from dotenv import load_dotenv
from rusty_agent_sdk import Provider

load_dotenv()
provider = Provider("openai/gpt-4o-mini")

stream = provider.stream_text("Write a limerick.", include_usage=True)

# Consume the stream
for chunk in stream:
    print(chunk, end="", flush=True)
print()

# Metadata is available after full consumption
print("Prompt tokens:", stream.prompt_tokens)
print("Completion tokens:", stream.completion_tokens)
print("Total tokens:", stream.total_tokens)
print("Finish reason:", stream.finish_reason)
print("Model:", stream.model)
```

## Provider Presets

### OpenAI

```python
import os
from rusty_agent_sdk import Provider

# Uses OPENAI_API_KEY env var
provider = Provider.openai("gpt-4o-mini")
response = provider.generate_text("Hello from OpenAI!")
print(response)
```

### Anthropic

```python
import os
from rusty_agent_sdk import Provider

# Uses ANTHROPIC_API_KEY env var
provider = Provider.anthropic("claude-sonnet-4-20250514")
response = provider.generate_text("Hello from Anthropic!")
print(response)
```

### OpenRouter

```python
from dotenv import load_dotenv
from rusty_agent_sdk import Provider

load_dotenv()

# Explicit preset (equivalent to the default constructor)
provider = Provider.openrouter("openai/gpt-4o-mini")
response = provider.generate_text("Hello from OpenRouter!")
print(response)
```

## Error Handling

```python
from dotenv import load_dotenv
from rusty_agent_sdk import Provider

load_dotenv()

# ValueError: missing API key
try:
    provider = Provider("some-model", api_key=None, base_url="https://example.com/v1")
except ValueError as e:
    print(f"Configuration error: {e}")

# ValueError: no prompt or messages
try:
    provider = Provider("openai/gpt-4o-mini")
    provider.generate_text()
except ValueError as e:
    print(f"Input error: {e}")

# ConnectionError: network failure
try:
    provider = Provider(
        "some-model",
        api_key="sk-test",
        base_url="https://nonexistent.example.com/v1",
    )
    provider.generate_text("Hello!")
except ConnectionError as e:
    print(f"Network error: {e}")

# RuntimeError: API error (e.g. invalid key, rate limit)
try:
    provider = Provider(
        "openai/gpt-4o-mini",
        api_key="sk-invalid-key",
    )
    provider.generate_text("Hello!")
except RuntimeError as e:
    print(f"API error: {e}")
```

## Combined Features

```python
import json
from dotenv import load_dotenv
from rusty_agent_sdk import Provider

load_dotenv()
provider = Provider("openai/gpt-4o-mini")

# Multi-turn conversation with system prompt, generation params,
# JSON mode, and usage tracking
result = provider.generate_text(
    system_prompt="You are a data API. Respond only with valid JSON.",
    messages=[
        {"role": "user", "content": "What is the capital of Japan?"},
        {"role": "assistant", "content": '{"answer": "Tokyo"}'},
        {"role": "user", "content": "What is its population?"},
    ],
    temperature=0.0,
    max_tokens=200,
    response_format={"type": "json_object"},
    include_usage=True,
)

data = json.loads(result.text)
print("Response:", json.dumps(data, indent=2))
print(f"Tokens used: {result.total_tokens} (prompt: {result.prompt_tokens}, completion: {result.completion_tokens})")
print(f"Finish reason: {result.finish_reason}")
```
