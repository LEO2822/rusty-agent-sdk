"""Rusty Agent SDK — a Rust-powered Python SDK for LLM text generation.

Typical usage::

    from dotenv import load_dotenv
    from rusty_agent_sdk import Provider

    load_dotenv()
    provider = Provider("openai/gpt-4o-mini")

    # Blocking
    print(provider.generate_text("Hello!"))

    # Streaming
    for chunk in provider.stream_text("Hello!"):
        print(chunk, end="", flush=True)

    # Multi-turn conversation
    response = provider.generate_text(messages=[
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Nice to meet you, Alice!"},
        {"role": "user", "content": "What is my name?"},
    ])

    # System prompt
    response = provider.generate_text(
        "Tell me a joke",
        system_prompt="You are a comedian.",
    )

    # Generation parameters
    response = provider.generate_text("Hello!", temperature=0.2, max_tokens=100)

    # JSON mode
    response = provider.generate_text(
        "List 3 colors as JSON",
        response_format={"type": "json_object"},
    )
"""

from __future__ import annotations

from typing import Any

__all__ = ["Provider", "TextStream"]

class Provider:
    """Configuration for an OpenAI-compatible LLM API provider.

    Holds the model, API key, and base URL needed to authenticate and route
    requests to any OpenAI-compatible chat completions endpoint.

    Examples:
        Using the environment variable (recommended)::

            provider = Provider("openai/gpt-4o-mini")
            print(provider.generate_text("Hello!"))

        Passing an explicit key and custom endpoint::

            provider = Provider(
                "gpt-4o-mini",
                api_key="sk-...",
                base_url="https://api.openai.com/v1",
            )
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Create a new Provider.

        Args:
            model: Model identifier, e.g. ``"openai/gpt-4o-mini"``.
            api_key: API key. Defaults to ``OPENROUTER_API_KEY`` env var.
            base_url: Base URL. Defaults to ``"https://openrouter.ai/api/v1"``.

        Raises:
            ValueError: If no API key is available.
        """
        ...

    def generate_text(
        self,
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
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Generate a complete text response (blocking).

        Args:
            prompt: The user message to send (shorthand for a single user
                message). When ``messages`` is also provided, ``prompt`` is
                ignored.
            system_prompt: Optional system prompt, prepended to the messages.
            messages: Full conversation history as a list of
                ``{"role": ..., "content": ...}`` dicts.
            temperature: Sampling temperature (0-2). Default: 1.
            max_tokens: Maximum tokens to generate.
            top_p: Nucleus sampling threshold (0-1). Default: 1.
            stop: Up to 4 stop sequences (string or list of strings).
            frequency_penalty: Frequency penalty (-2 to 2). Default: 0.
            presence_penalty: Presence penalty (-2 to 2). Default: 0.
            seed: Random seed for deterministic generation.
            response_format: Response format, e.g.
                ``{"type": "json_object"}`` or
                ``{"type": "json_schema", "json_schema": {...}}``.

        Returns:
            The model's complete text response.

        Raises:
            ConnectionError: If the HTTP request fails.
            RuntimeError: If the API returns a non-2xx status code.
            ValueError: If the response cannot be parsed, or if neither
                prompt nor messages is provided.
        """
        ...

    def stream_text(
        self,
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
        response_format: dict[str, Any] | None = None,
    ) -> TextStream:
        """Stream text from the LLM as an iterator of chunks.

        Accepts the same parameters as :meth:`generate_text`.

        Returns:
            An iterator yielding ``str`` chunks.

        Raises:
            ConnectionError: If the initial HTTP connection fails.
            RuntimeError: If the API returns a non-2xx status code.
            ValueError: If neither prompt nor messages is provided.
        """
        ...

    def __repr__(self) -> str: ...

class TextStream:
    """An iterator that yields text chunks from a streaming LLM response.

    You do not construct this directly — it is returned by
    :meth:`Provider.stream_text`.
    """

    def __iter__(self) -> TextStream: ...
    def __next__(self) -> str: ...
