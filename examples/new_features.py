"""Examples for Features 7, 8, 9: token usage, provider presets, and embeddings."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from rusty_agent_sdk import GenerateResult, EmbeddingResult, Provider


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent / ".env")

    provider: Provider = Provider("openai/gpt-4o-mini")

    # ── Feature 7: Token usage with generate_text ────────────────────────
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

    # ── Feature 7: Default behavior unchanged ────────────────────────────
    print("=== generate_text default (returns str) ===")
    text: str = provider.generate_text("Say hello in one word.")
    print(f"Type: {type(text).__name__}, Value: {text}")
    print()

    # ── Feature 7: Token usage with stream_text ──────────────────────────
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

    # ── Feature 8: Provider presets ──────────────────────────────────────
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

    # ── Feature 9: Single embedding ─────────────────────────────────────
    print("=== Single embedding ===")
    emb: EmbeddingResult = provider.embed("Hello, world!")
    print(f"Vectors: {len(emb.embeddings)}")
    print(f"Dimensions: {len(emb.embeddings[0])}")
    print(f"First 5 values: {emb.embeddings[0][:5]}")
    print(f"Model: {emb.model}")
    print(f"Prompt tokens: {emb.prompt_tokens}")
    print(f"Total tokens: {emb.total_tokens}")
    print()

    # ── Feature 9: Batch embeddings ─────────────────────────────────────
    print("=== Batch embeddings ===")
    batch: EmbeddingResult = provider.embed_many(
        ["Hello", "World", "Rust is fast"]
    )
    print(f"Vectors: {len(batch.embeddings)}")
    for i, vec in enumerate(batch.embeddings):
        print(f"  [{i}] dim={len(vec)}, first 3: {vec[:3]}")
    print(f"Model: {batch.model}")
    print(f"Total tokens: {batch.total_tokens}")


if __name__ == "__main__":
    main()
