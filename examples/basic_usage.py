from pathlib import Path

from dotenv import load_dotenv
from rusty_agent_sdk import Provider, stream_text


def main() -> None:
    example_dir = Path(__file__).resolve().parent
    load_dotenv(example_dir / ".env")

    provider = Provider()
    for chunk in stream_text(
        provider,
        "openai/gpt-4o-mini",
        "What is Rust programming language? Answer in 2 short sentences.",
    ):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
