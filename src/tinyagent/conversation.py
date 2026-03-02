"""Message history management."""

from __future__ import annotations

from tinyagent.models import Message, Role

SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Answer the user's questions clearly and concisely."
)


class Conversation:
    def __init__(self, system_prompt: str = SYSTEM_PROMPT) -> None:
        self._system = Message(role=Role.SYSTEM, content=system_prompt)
        self._messages: list[Message] = []

    def add_user(self, content: str) -> None:
        self._messages.append(Message(role=Role.USER, content=content))

    def add_assistant(self, content: str) -> None:
        self._messages.append(Message(role=Role.ASSISTANT, content=content))

    def get_messages(self) -> list[Message]:
        """Return full message list including system prompt."""
        return [self._system, *self._messages]

    def clear(self) -> None:
        self._messages.clear()

    @property
    def turn_count(self) -> int:
        return len(self._messages)
