"""Examples for token usage tracking and provider presets."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from rusty_agent_sdk import GenerateResult, Provider


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent / ".env")

    provider: Provider = Provider("openai/gpt-4o-mini")

    # ── Token usage with generate_text ───────────────────────────────────
    print("=== generate_text with include_usage=True ===")
    result: GenerateResult = provider.generate_text(
        "What is Rust? One sentence.", include_usage=True
    )
    print(f"Text: {result.text}")
    print(f"Prompt tokens:     {result.prompt_tokens}")
    print(f"Completion tokens: {result.completion_tokens}")
    print(f"Total tokens:      {result.total_tokens}")
    print(f"Finish reason:     {result.finish_reason}")
    print(f"Model:             {result.model}")
    print(f"str(result):       {result}")
    print()

    # ── Default behavior unchanged ───────────────────────────────────────
    print("=== generate_text default (returns str) ===")
    text: str = provider.generate_text("Say hello in one word.")
    print(f"Type: {type(text).__name__}, Value: {text}")
    print()

    # ── Token usage with stream_text ─────────────────────────────────────
    print("=== stream_text with include_usage=True ===")
    stream = provider.stream_text("Count to 5.", include_usage=True)
    for chunk in stream:
        print(chunk, end="", flush=True)
    print()
    print(f"Prompt tokens:     {stream.prompt_tokens}")
    print(f"Completion tokens: {stream.completion_tokens}")
    print(f"Total tokens:      {stream.total_tokens}")
    print(f"Finish reason:     {stream.finish_reason}")
    print(f"Model:             {stream.model}")
    print()

    # ── Provider presets ─────────────────────────────────────────────────
    print("=== Provider presets ===")
    or_provider = Provider.openrouter("openai/gpt-4o-mini")
    print(f"OpenRouter: {or_provider!r}")
    print(f"  Response: {or_provider.generate_text('Say hi.', max_tokens=10)}")
    print()

    # Uncomment if you have an OPENAI_API_KEY set:
    # oi_provider = Provider.openai("gpt-4o-mini")
    # print(f"OpenAI: {oi_provider!r}")
    # print(f"  Response: {oi_provider.generate_text('Say hi.', max_tokens=10)}")

    # Uncomment if you have an ANTHROPIC_API_KEY set:
    # an_provider = Provider.anthropic("claude-sonnet-4-20250514")
    # print(f"Anthropic: {an_provider!r}")
    # print(f"  Response: {an_provider.generate_text('Say hi.', max_tokens=10)}")


if __name__ == "__main__":
    main()
