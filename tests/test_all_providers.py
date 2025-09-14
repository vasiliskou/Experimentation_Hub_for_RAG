# tests/test_all_providers.py
import os
from generator import Generator  

PROVIDERS = ["openai", "gemini", "groq", "anthropic", "deepseek"]

SYSTEM_PROMPT = "You are a helpful assistant. Answer briefly which model you are."
USER_PROMPT = "Which model are you?"

def test_all_providers():
    for provider in PROVIDERS:
        print(f"\nTesting provider: {provider}")
        gen = Generator(
            provider=provider,
            max_tokens=500,
            max_retries=2,
            temperature=0.6,
            timeout=10,
            top_p=0.4,
        )
        try:
            answer = gen.generate(SYSTEM_PROMPT, USER_PROMPT)
            print(f"Response from {provider}: {answer}\n")
            assert answer and len(answer) > 0, f"No response from {provider}"
        except Exception as e:
            print(f"Error with provider {provider}: {e}")
            assert False, f"Provider {provider} failed with error: {e}"


if __name__ == "__main__":
    test_all_providers()
