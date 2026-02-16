"""Basic usage of the rusty-agent-sdk."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from rusty_agent_sdk import Provider

load_dotenv(Path(__file__).resolve().parent / ".env")

provider: Provider = Provider("openai/gpt-4o-mini")

for chunk in provider.stream_text(
    "What is Rust programming language? Answer in 2 short sentences."
):
    print(chunk, end="", flush=True)
print()
