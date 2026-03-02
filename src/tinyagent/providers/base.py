"""Abstract base for all providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from tinyagent.models import CompletionResult, Message


class Provider(ABC):
    name: str

    @abstractmethod
    def complete(self, messages: list[Message], model: str) -> CompletionResult:
        """Send messages and return a completion."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether this provider can accept requests right now."""
