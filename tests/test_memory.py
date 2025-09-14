"""
Full test suite for ConversationMemory.

Run with:
    pytest -v tests/test_memory.py

Or directly:
    python tests/test_memory.py
"""

import pytest
from memory import ConversationMemory


@pytest.fixture
def memory():
    """Fresh memory instance for each test."""
    return ConversationMemory()


def test_add_and_retrieve_messages(memory):
    memory.add_message("user", "Hello")
    memory.add_message("assistant", "Hi there!")

    history = memory.get_history()

    assert isinstance(history, list)
    assert all(isinstance(msg, tuple) for msg in history)
    assert history == [("user", "Hello"), ("assistant", "Hi there!")]


def test_format_history(memory):
    memory.add_message("user", "Hello")
    memory.add_message("assistant", "How can I help?")

    formatted = memory.format_history()

    assert formatted.startswith("Conversation History:")
    assert "User: Hello" in formatted
    assert "Assistant: How can I help?" in formatted


def test_clear_memory(memory):
    memory.add_message("user", "Hello")
    memory.add_message("assistant", "Hi!")

    memory.clear()

    assert memory.get_history() == []
    assert memory.format_history() == ""


def test_empty_memory(memory):
    assert memory.get_history() == []
    assert memory.format_history() == ""


def test_long_conversation(memory):
    """Check handling of many messages (stress test)."""
    for i in range(1000):
        memory.add_message("user", f"message {i}")
        memory.add_message("assistant", f"reply {i}")

    history = memory.get_history()
    assert len(history) == 2000
    assert history[0] == ("user", "message 0")
    assert history[1] == ("assistant", "reply 0")

    # Check formatting includes last exchange
    formatted = memory.format_history()
    assert "User: message 999" in formatted
    assert "Assistant: reply 999" in formatted


# --------------------------
# Manual Runner
# --------------------------
def main():
    """Run a manual demo of ConversationMemory without pytest."""
    memory = ConversationMemory()

    memory.add_message("user", "Hello, how are you?")
    memory.add_message("assistant", "I'm good, thanks! How about you?")
    memory.add_message("user", "Can you remind me what I asked first?")
    memory.add_message("assistant", "Sure, you asked how I am doing.")

    print("=== Raw History ===")
    print(memory.get_history())
    print()

    print("=== Formatted History ===")
    print(memory.format_history())
    print()

    # Clear memory
    memory.clear()
    print("=== After Clearing ===")
    print(memory.get_history())


if __name__ == "__main__":
    main()
