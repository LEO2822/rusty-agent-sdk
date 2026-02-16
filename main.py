from dotenv import load_dotenv

from rusty_agent_sdk import Provider, generate_text, stream_text

load_dotenv()


def main():
    provider = Provider()
    model = "openai/gpt-4o-mini"

    # --- generate_text (blocking) ---
    prompt = "What is Rust programming language? Answer in 1 sentence."
    print(f"[generate_text] Model: {model}")
    print(f"[generate_text] Prompt: {prompt}")
    print("---")
    response = generate_text(provider, model, prompt)
    print(response)
    print()

    # --- stream_text (streaming) ---
    prompt = "What is Rust programming language? Answer in 2 sentences."
    print(f"[stream_text] Model: {model}")
    print(f"[stream_text] Prompt: {prompt}")
    print("---")
    for chunk in stream_text(provider, model, prompt):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
