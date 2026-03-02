"""Core orchestrator — ties routing, providers, budget, and conversation together."""

from __future__ import annotations

import logging

from tinyagent.budget import Budget
from tinyagent.config import Config
from tinyagent.conversation import Conversation
from tinyagent.models import CompletionResult, ModelCandidate, RoutingDecision
from tinyagent.providers import build_registry
from tinyagent.providers.base import Provider
from tinyagent.router import Router

log = logging.getLogger(__name__)


class Agent:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.budget = Budget(limit=config.budget)
        self.conversation = Conversation()
        self.providers = build_registry()
        self.last_routing: RoutingDecision | None = None

        # Determine classifier provider/model (always local)
        classifier = self.providers.get("ollama")
        if classifier is None:
            raise RuntimeError("Ollama provider is required but not available")

        self.router = Router(
            classifier_provider=classifier,
            classifier_model=config.ollama_model,
            providers=self.providers,
            budget=self.budget,
        )

    def chat(self, user_input: str) -> CompletionResult | None:
        """Process one user turn and return the assistant response."""
        self.conversation.add_user(user_input)

        # Decide which model to use
        if self.config.force_model:
            decision = self._forced_routing()
        else:
            decision = self.router.route(user_input)

        self.last_routing = decision

        if decision.chosen is None:
            error_msg = "No model available to handle this request."
            self.conversation.add_assistant(error_msg)
            return CompletionResult(
                content=error_msg, provider="none", model="none"
            )

        provider = self.providers[decision.chosen.provider]
        messages = self.conversation.get_messages()

        try:
            result = provider.complete(messages, decision.chosen.model)
        except Exception as exc:
            log.error("Provider %s failed: %s", decision.chosen.provider, exc)
            # Try fallback to ollama
            if decision.chosen.provider != "ollama":
                ollama = self.providers.get("ollama")
                if ollama and ollama.is_available():
                    log.info("Falling back to ollama")
                    result = ollama.complete(messages, self.config.ollama_model)
                else:
                    error_msg = f"Provider error: {exc}"
                    self.conversation.add_assistant(error_msg)
                    return CompletionResult(
                        content=error_msg, provider="error", model="error"
                    )
            else:
                error_msg = f"Ollama error: {exc}"
                self.conversation.add_assistant(error_msg)
                return CompletionResult(
                    content=error_msg, provider="error", model="error"
                )

        self.budget.record(result)
        self.conversation.add_assistant(result.content)
        return result

    def _forced_routing(self) -> RoutingDecision:
        """Build a routing decision from the --model override."""
        from tinyagent.models import Complexity

        parts = self.config.force_model.split("/", 1)  # type: ignore[union-attr]
        if len(parts) == 2:
            provider_name, model_name = parts
        else:
            provider_name, model_name = "ollama", parts[0]

        candidate = ModelCandidate(provider_name, model_name)
        return RoutingDecision(
            complexity=Complexity.LOW,
            candidates=[candidate],
            chosen=candidate,
            reason=f"forced via --model {self.config.force_model}",
        )
