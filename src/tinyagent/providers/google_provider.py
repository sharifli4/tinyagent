"""Google Gemini provider."""

from __future__ import annotations

import os

from google import genai

from tinyagent.models import CompletionResult, Message, Role, TokenUsage
from tinyagent.pricing import compute_cost
from tinyagent.providers.base import Provider


class GoogleProvider(Provider):
    name = "google"

    def __init__(self) -> None:
        self._client: genai.Client | None = None

    def _get_client(self) -> genai.Client:
        if self._client is None:
            api_key = os.environ.get("GOOGLE_API_KEY", "")
            self._client = genai.Client(api_key=api_key)
        return self._client

    def complete(self, messages: list[Message], model: str) -> CompletionResult:
        # Build contents list for Gemini — system instructions handled separately
        system_text = ""
        contents: list[dict] = []
        for m in messages:
            if m.role == Role.SYSTEM:
                system_text += m.content + "\n"
            else:
                role = "user" if m.role == Role.USER else "model"
                contents.append({"role": role, "parts": [{"text": m.content}]})

        config: dict = {}
        if system_text.strip():
            config["system_instruction"] = system_text.strip()

        resp = self._get_client().models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        # Extract usage metadata
        prompt_tokens = 0
        completion_tokens = 0
        if resp.usage_metadata:
            prompt_tokens = resp.usage_metadata.prompt_token_count or 0
            completion_tokens = resp.usage_metadata.candidates_token_count or 0

        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        cost = compute_cost(self.name, model, usage.prompt_tokens, usage.completion_tokens)
        return CompletionResult(
            content=resp.text or "",
            provider=self.name,
            model=model,
            usage=usage,
            cost=cost,
        )

    def is_available(self) -> bool:
        return bool(os.environ.get("GOOGLE_API_KEY"))
