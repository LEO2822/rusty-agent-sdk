"""Rusty Agent SDK — a Rust-powered Python SDK for LLM text generation.

This module provides high-performance functions for interacting with
OpenAI-compatible chat completions APIs. The networking and SSE parsing
are implemented in Rust for speed and reliability, while the public API
is designed to feel natural in Python.

Typical usage::

    from dotenv import load_dotenv
    from rusty_agent_sdk import Provider, generate_text, stream_text

    load_dotenv()
    provider = Provider()

    # Blocking call
    response = generate_text(provider, "openai/gpt-4o-mini", "Hello!")

    # Streaming call
    for chunk in stream_text(provider, "openai/gpt-4o-mini", "Hello!"):
        print(chunk, end="", flush=True)
"""

from __future__ import annotations

__all__ = ["Provider", "TextStream", "generate_text", "stream_text"]

class Provider:
    """Configuration for an OpenAI-compatible LLM API provider.

    Holds the API key and base URL needed to authenticate and route requests
    to any OpenAI-compatible chat completions endpoint. By default, requests
    are sent to OpenRouter (https://openrouter.ai/api/v1).

    The API key can be supplied explicitly or read from the
    ``OPENROUTER_API_KEY`` environment variable. If neither is available,
    a ``ValueError`` is raised at construction time.

    Examples:
        Using the environment variable (recommended)::

            from dotenv import load_dotenv
            from rusty_agent_sdk import Provider

            load_dotenv()  # loads OPENROUTER_API_KEY from .env
            provider = Provider()

        Passing an explicit key and custom endpoint::

            provider = Provider(
                api_key="sk-...",
                base_url="https://api.openai.com/v1",
            )

    Notes:
        - The provider is immutable after creation; create a new instance
          to change the key or URL.
        - The API key is never printed in ``repr()`` output for safety.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Create a new Provider.

        Args:
            api_key: API key for the LLM service. If ``None``, the
                ``OPENROUTER_API_KEY`` environment variable is used.
            base_url: Base URL of the OpenAI-compatible API.
                Defaults to ``"https://openrouter.ai/api/v1"``.

        Raises:
            ValueError: If no ``api_key`` is provided and the
                ``OPENROUTER_API_KEY`` environment variable is not set.
        """
        ...

    def __repr__(self) -> str: ...

class TextStream:
    """An iterator that yields text chunks from a streaming LLM response.

    ``TextStream`` implements Python's iterator protocol, so you can use it
    in a ``for`` loop or call ``next()`` on it manually. Each iteration
    yields the next content fragment as a ``str``.

    The stream is backed by a background thread that reads Server-Sent
    Events (SSE) from the API and forwards parsed content deltas through an
    internal channel. Iteration blocks until the next chunk arrives or the
    stream ends.

    You do not construct ``TextStream`` directly — it is returned by
    :func:`stream_text`.

    Examples:
        Print chunks as they arrive::

            for chunk in stream_text(provider, model, prompt):
                print(chunk, end="", flush=True)
            print()  # newline after stream ends

        Collect the full response::

            full = "".join(stream_text(provider, model, prompt))

    Raises:
        RuntimeError: If the background streaming thread encounters an error
            (e.g. a malformed SSE event or a dropped connection mid-stream),
            the error is raised on the **next** call to ``__next__``.

    Notes:
        - The iterator is single-use. Once exhausted, calling ``next()``
          will raise ``StopIteration``.
        - The background thread is automatically cleaned up when the stream
          ends or when the ``TextStream`` object is garbage-collected.
    """

    def __iter__(self) -> TextStream: ...
    def __next__(self) -> str: ...

def generate_text(provider: Provider, model: str, prompt: str) -> str:
    """Generate a complete text response from an LLM (blocking).

    Sends a single user message to the chat completions endpoint and blocks
    until the full response is available. This is the simplest way to get a
    response from a model when you don't need streaming.

    Under the hood, a temporary Tokio runtime is created for the async HTTP
    call so the function can be used from synchronous Python code without
    ``asyncio``.

    Args:
        provider: The provider configuration (API key + base URL).
        model: Model identifier, e.g. ``"openai/gpt-4o-mini"`` or
            ``"anthropic/claude-sonnet-4-5-20250514"``.
        prompt: The user message to send to the model.

    Returns:
        The model's complete text response.

    Raises:
        ConnectionError: If the HTTP request to the API fails (network error,
            DNS resolution failure, timeout, etc.).
        RuntimeError: If the API returns a non-2xx status code. The error
            message includes the HTTP status and the API's error description
            when available.
        ValueError: If the API returns a 2xx response but the body cannot be
            parsed, or if the response contains no choices.

    Examples:
        Basic usage::

            from rusty_agent_sdk import Provider, generate_text

            provider = Provider(api_key="sk-...")
            response = generate_text(provider, "openai/gpt-4o-mini", "Hello!")
            print(response)

        With a custom endpoint::

            provider = Provider(
                api_key="sk-...",
                base_url="https://api.openai.com/v1",
            )
            response = generate_text(provider, "gpt-4o-mini", "Hello!")

    Notes:
        - Only the ``user`` role is sent. System/assistant messages are not
          yet supported.
        - A new HTTP client and Tokio runtime are created per call. For
          high-throughput use cases, prefer :func:`stream_text` or batch at
          the application level.
    """
    ...

def stream_text(provider: Provider, model: str, prompt: str) -> TextStream:
    """Stream text from an LLM, returning an iterator that yields chunks in
    real time.

    Sends a chat completions request with ``stream: true`` and returns a
    :class:`TextStream` iterator. Each call to ``next()`` on the iterator
    blocks until the next content chunk arrives from the API's Server-Sent
    Events (SSE) stream.

    A dedicated background thread is spawned to handle the async HTTP
    streaming, keeping the Python thread free to process chunks as they
    arrive. Communication between the background thread and the iterator
    uses a bounded channel.

    Args:
        provider: The provider configuration (API key + base URL).
        model: Model identifier, e.g. ``"openai/gpt-4o-mini"`` or
            ``"anthropic/claude-sonnet-4-5-20250514"``.
        prompt: The user message to send to the model.

    Returns:
        An iterator yielding ``str`` chunks. Use in a ``for`` loop or call
        ``next()`` manually.

    Raises:
        ConnectionError: If the initial HTTP connection to the API fails.
        RuntimeError: If the API returns a non-2xx status code before
            streaming begins. The error is raised on the **first** call to
            ``next()`` on the returned :class:`TextStream`.

    Examples:
        Print a streaming response::

            from rusty_agent_sdk import Provider, stream_text

            provider = Provider(api_key="sk-...")
            for chunk in stream_text(provider, "openai/gpt-4o-mini", "Hello!"):
                print(chunk, end="", flush=True)
            print()

        Collect full text from a stream::

            text = "".join(stream_text(provider, "openai/gpt-4o-mini", "Hello!"))

    Notes:
        - Like :func:`generate_text`, only the ``user`` role is sent.
        - Empty content deltas from the SSE stream are silently skipped.
        - The ``[DONE]`` sentinel in the SSE stream is handled internally
          and never yielded to the caller.
    """
    ...
