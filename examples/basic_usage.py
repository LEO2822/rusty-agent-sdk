"""Basic streaming example for rusty-agent-sdk."""

from __future__ import annotations
from pathlib import Path
from dotenv import load_dotenv
from rusty_agent_sdk import Provider, TextStream

PROMPT: str = "What is Rust programming language? Answer in 2 short sentences."


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent / ".env")

    provider: Provider = Provider("openai/gpt-4o-mini")
    stream: TextStream = provider.stream_text(PROMPT)

    for chunk in stream:
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
