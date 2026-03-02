"""Anthropic provider."""

from __future__ import annotations

import os

import anthropic

from tinyagent.models import CompletionResult, Message, Role, TokenUsage
from tinyagent.pricing import compute_cost
from tinyagent.providers.base import Provider


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(self, messages: list[Message], model: str) -> CompletionResult:
        # Anthropic requires system message passed separately
        system_text = ""
        chat_messages: list[dict] = []
        for m in messages:
            if m.role == Role.SYSTEM:
                system_text += m.content + "\n"
            else:
                chat_messages.append({"role": m.role.value, "content": m.content})

        kwargs: dict = {
            "model": model,
            "max_tokens": 4096,
            "messages": chat_messages,
        }
        if system_text.strip():
            kwargs["system"] = system_text.strip()

        resp = self._client.messages.create(**kwargs)

        usage = TokenUsage(
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
        )
        content = "".join(
            block.text for block in resp.content if hasattr(block, "text")
        )
        cost = compute_cost(self.name, model, usage.prompt_tokens, usage.completion_tokens)
        return CompletionResult(
            content=content,
            provider=self.name,
            model=model,
            usage=usage,
            cost=cost,
        )

    def is_available(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
