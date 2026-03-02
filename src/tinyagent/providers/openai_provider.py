"""OpenAI provider."""

from __future__ import annotations

import os

import openai

from tinyagent.models import CompletionResult, Message, TokenUsage
from tinyagent.pricing import compute_cost
from tinyagent.providers.base import Provider


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self) -> None:
        self._client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

    def complete(self, messages: list[Message], model: str) -> CompletionResult:
        resp = self._client.chat.completions.create(
            model=model,
            messages=[{"role": m.role.value, "content": m.content} for m in messages],
        )
        choice = resp.choices[0]
        usage_data = resp.usage
        usage = TokenUsage(
            prompt_tokens=usage_data.prompt_tokens if usage_data else 0,
            completion_tokens=usage_data.completion_tokens if usage_data else 0,
        )
        cost = compute_cost(self.name, model, usage.prompt_tokens, usage.completion_tokens)
        return CompletionResult(
            content=choice.message.content or "",
            provider=self.name,
            model=model,
            usage=usage,
            cost=cost,
        )

    def is_available(self) -> bool:
        return bool(os.environ.get("OPENAI_API_KEY"))
