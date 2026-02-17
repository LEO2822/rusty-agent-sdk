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

    # Usage tracking
    result = provider.generate_text("Hello!", include_usage=True)
    print(result.text)
    print(result.prompt_tokens, result.completion_tokens, result.total_tokens)

    # Convenience constructors
    openai_provider = Provider.openai("gpt-4o-mini", api_key="sk-...")
    anthropic_provider = Provider.anthropic("claude-sonnet-4-20250514")
    openrouter_provider = Provider.openrouter("openai/gpt-4o-mini")

"""

from __future__ import annotations

from typing import Any, Literal, overload

__all__ = ["Provider", "TextStream", "GenerateResult"]

class GenerateResult:
    """Result from a text generation call when ``include_usage=True``.

    Wraps the generated text along with token usage statistics and metadata
    returned by the API.
    """

    @property
    def text(self) -> str:
        """The model's complete text response."""
        ...

    @property
    def prompt_tokens(self) -> int | None:
        """Number of tokens in the prompt, or ``None`` if not reported."""
        ...

    @property
    def completion_tokens(self) -> int | None:
        """Number of tokens in the completion, or ``None`` if not reported."""
        ...

    @property
    def total_tokens(self) -> int | None:
        """Total tokens used (prompt + completion), or ``None`` if not reported."""
        ...

    @property
    def finish_reason(self) -> str | None:
        """The reason the model stopped generating, e.g. ``"stop"`` or ``"length"``."""
        ...

    @property
    def model(self) -> str | None:
        """The model that was used for generation, as reported by the API."""
        ...

    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

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

        Using convenience constructors::

            provider = Provider.openai("gpt-4o-mini", api_key="sk-...")
            provider = Provider.anthropic("claude-sonnet-4-20250514")
            provider = Provider.openrouter("openai/gpt-4o-mini")
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

    @classmethod
    def openai(cls, model: str, *, api_key: str | None = None) -> Provider:
        """Create a Provider configured for the OpenAI API.

        Args:
            model: Model identifier, e.g. ``"gpt-4o-mini"``.
            api_key: API key. Defaults to ``OPENAI_API_KEY`` env var.

        Returns:
            A configured :class:`Provider` instance.
        """
        ...

    @classmethod
    def anthropic(cls, model: str, *, api_key: str | None = None) -> Provider:
        """Create a Provider configured for the Anthropic API.

        Args:
            model: Model identifier, e.g. ``"claude-sonnet-4-20250514"``.
            api_key: API key. Defaults to ``ANTHROPIC_API_KEY`` env var.

        Returns:
            A configured :class:`Provider` instance.
        """
        ...

    @classmethod
    def openrouter(cls, model: str, *, api_key: str | None = None) -> Provider:
        """Create a Provider configured for the OpenRouter API.

        Args:
            model: Model identifier, e.g. ``"openai/gpt-4o-mini"``.
            api_key: API key. Defaults to ``OPENROUTER_API_KEY`` env var.

        Returns:
            A configured :class:`Provider` instance.
        """
        ...

    @overload
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
        include_usage: Literal[False] = ...,
    ) -> str:
        """Generate a complete text response (blocking).

        Returns ``str`` when ``include_usage`` is ``False`` (the default).
        """
        ...

    @overload
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
        include_usage: Literal[True] = ...,
    ) -> GenerateResult:
        """Generate a complete text response (blocking).

        Returns :class:`GenerateResult` when ``include_usage`` is ``True``.
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
        include_usage: bool = False,
    ) -> str | GenerateResult:
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
            include_usage: If ``True``, return a :class:`GenerateResult` with
                token usage statistics instead of a plain string.

        Returns:
            The model's complete text response as a ``str`` when
            ``include_usage=False`` (default), or a :class:`GenerateResult`
            when ``include_usage=True``.

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
        include_usage: bool = False,
    ) -> TextStream:
        """Stream text from the LLM as an iterator of chunks.

        Accepts the same parameters as :meth:`generate_text`.

        When ``include_usage=True``, token usage statistics and metadata
        will be available on the returned :class:`TextStream` after iteration
        completes (via properties like ``prompt_tokens``, ``completion_tokens``,
        etc.).

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

    When ``include_usage=True`` was passed to :meth:`Provider.stream_text`,
    token usage statistics and metadata are available as properties after
    the stream has been fully consumed.
    """

    @property
    def prompt_tokens(self) -> int | None:
        """Number of tokens in the prompt, or ``None`` if not available."""
        ...

    @property
    def completion_tokens(self) -> int | None:
        """Number of tokens in the completion, or ``None`` if not available."""
        ...

    @property
    def total_tokens(self) -> int | None:
        """Total tokens used (prompt + completion), or ``None`` if not available."""
        ...

    @property
    def finish_reason(self) -> str | None:
        """The reason the model stopped generating, or ``None`` if not available."""
        ...

    @property
    def model(self) -> str | None:
        """The model that was used, as reported by the API."""
        ...

    def __iter__(self) -> TextStream: ...
    def __next__(self) -> str: ...
