from dotenv import load_dotenv

from rusty_agent_sdk import Provider, generate_text

load_dotenv()


def main():
    # Reads OPENROUTER_API_KEY from .env / environment automatically
    provider = Provider()

    model = "openai/gpt-4o-mini"
    prompt = "What is Rust programming language? Answer in 1 sentences."

    print(f"Model: {model}")
    print(f"Prompt: {prompt}")
    print("---")

    response = generate_text(provider, model, prompt)
    print(response)


if __name__ == "__main__":
    main()
