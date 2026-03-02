"""Ollama (local) provider."""

from __future__ import annotations

import ollama as _ollama

from tinyagent.models import CompletionResult, Message, TokenUsage
from tinyagent.providers.base import Provider


class OllamaProvider(Provider):
    name = "ollama"

    def __init__(self, host: str | None = None) -> None:
        self._client = _ollama.Client(host=host) if host else _ollama.Client()

    def complete(self, messages: list[Message], model: str) -> CompletionResult:
        resp = self._client.chat(
            model=model,
            messages=[{"role": m.role.value, "content": m.content} for m in messages],
        )
        usage = TokenUsage(
            prompt_tokens=resp.get("prompt_eval_count", 0) or 0,
            completion_tokens=resp.get("eval_count", 0) or 0,
        )
        return CompletionResult(
            content=resp["message"]["content"],
            provider=self.name,
            model=model,
            usage=usage,
            cost=0.0,
        )

    def is_available(self) -> bool:
        try:
            self._client.list()
            return True
        except Exception:
            return False
