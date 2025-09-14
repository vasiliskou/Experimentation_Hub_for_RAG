"""
memory.py

Custom conversation memory implementation.
"""

from typing import List, Tuple


class ConversationMemory:
    """
    Simple conversation buffer memory.

    Stores chat history as a list of (role, message) tuples.
    Example:
        [("user", "Hi"), ("assistant", "Hello!")]
    """

    def __init__(self):
        self.history: List[Tuple[str, str]] = []

    def add_message(self, role: str, message: str):
        """Add a message to the memory."""
        self.history.append((role, message))

    def get_history(self) -> List[Tuple[str, str]]:
        """Return full conversation history."""
        return self.history

    def format_history(self) -> str:
        """Format conversation history as a string for prompts."""
        if not self.history:
            return ""
        formatted = "Conversation History:\n"
        for role, msg in self.history:
            formatted += f"{role.capitalize()}: {msg}\n"
        return formatted.strip()

    def clear(self):
        """Clear memory."""
        self.history = []
