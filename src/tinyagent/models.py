"""Core data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal
import time


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Complexity(str, Enum):
    TRIVIAL = "trivial"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Message:
    role: Role
    content: str


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class CompletionResult:
    content: str
    provider: str
    model: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    cost: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ModelCandidate:
    provider: str
    model: str


@dataclass
class RoutingDecision:
    complexity: Complexity
    candidates: list[ModelCandidate]
    chosen: ModelCandidate | None = None
    reason: str = ""
