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
"""

from __future__ import annotations

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

    def generate_text(self, prompt: str) -> str:
        """Generate a complete text response (blocking).

        Args:
            prompt: The user message to send.

        Returns:
            The model's complete text response.

        Raises:
            ConnectionError: If the HTTP request fails.
            RuntimeError: If the API returns a non-2xx status code.
            ValueError: If the response cannot be parsed.
        """
        ...

    def stream_text(self, prompt: str) -> TextStream:
        """Stream text from the LLM as an iterator of chunks.

        Args:
            prompt: The user message to send.

        Returns:
            An iterator yielding ``str`` chunks.

        Raises:
            ConnectionError: If the initial HTTP connection fails.
            RuntimeError: If the API returns a non-2xx status code.
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
