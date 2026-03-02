"""Complexity classifier and model selector."""

from __future__ import annotations

import json
import logging

from tinyagent.budget import Budget
from tinyagent.models import Complexity, ModelCandidate, RoutingDecision
from tinyagent.providers.base import Provider

log = logging.getLogger(__name__)

COMPLEXITY_PROMPT_TEMPLATE = (
    "Classify the complexity of the following user message for an AI assistant.\n"
    'Respond with ONLY a JSON object: {{"complexity": "<level>"}}\n'
    "where <level> is one of: trivial, low, medium, high, critical\n"
    "\n"
    "Guidelines:\n"
    "- trivial: greetings, simple facts, one-word answers\n"
    "- low: short explanations, simple lookups, basic math\n"
    "- medium: multi-step reasoning, summaries, moderate code\n"
    "- high: system design, long-form analysis, complex code\n"
    "- critical: novel research, architecture for production systems, expert-level tasks\n"
    "\n"
    "User message:\n"
    "{message}"
)

# Ordered candidate lists per complexity level
ROUTING_TABLE: dict[Complexity, list[ModelCandidate]] = {
    Complexity.TRIVIAL: [
        ModelCandidate("ollama", "llama3.2"),
    ],
    Complexity.LOW: [
        ModelCandidate("ollama", "llama3.2"),
    ],
    Complexity.MEDIUM: [
        ModelCandidate("google", "gemini-2.5-flash"),
        ModelCandidate("ollama", "llama3.2"),
    ],
    Complexity.HIGH: [
        ModelCandidate("anthropic", "claude-sonnet-4-6"),
        ModelCandidate("openai", "gpt-4o"),
        ModelCandidate("google", "gemini-2.5-pro"),
        ModelCandidate("ollama", "llama3.2"),
    ],
    Complexity.CRITICAL: [
        ModelCandidate("anthropic", "claude-opus-4-6"),
        ModelCandidate("openai", "o3"),
        ModelCandidate("anthropic", "claude-sonnet-4-6"),
        ModelCandidate("ollama", "llama3.2"),
    ],
}


class Router:
    def __init__(
        self,
        classifier_provider: Provider,
        classifier_model: str,
        providers: dict[str, Provider],
        budget: Budget,
    ) -> None:
        self._classifier = classifier_provider
        self._classifier_model = classifier_model
        self._providers = providers
        self._budget = budget

    def classify(self, user_message: str) -> Complexity:
        """Use the local model to classify message complexity."""
        from tinyagent.models import Message, Role

        prompt = COMPLEXITY_PROMPT_TEMPLATE.format(message=user_message)
        messages = [Message(role=Role.USER, content=prompt)]

        try:
            result = self._classifier.complete(messages, self._classifier_model)
            parsed = json.loads(result.content.strip())
            level = parsed.get("complexity", "low").lower()
            return Complexity(level)
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            log.warning("Classification failed (%s), defaulting to low", exc)
            return Complexity.LOW
        except Exception as exc:
            log.warning("Classifier unavailable (%s), defaulting to low", exc)
            return Complexity.LOW

    def route(self, user_message: str) -> RoutingDecision:
        """Classify and select the best available model within budget."""
        complexity = self.classify(user_message)
        candidates = ROUTING_TABLE[complexity]

        decision = RoutingDecision(complexity=complexity, candidates=candidates)

        for candidate in candidates:
            provider = self._providers.get(candidate.provider)
            if provider is None:
                continue

            # Ollama is always free, skip budget check
            if candidate.provider == "ollama":
                if provider.is_available():
                    decision.chosen = candidate
                    decision.reason = "local model (free)"
                    return decision
                continue

            if not self._budget.can_afford(candidate.provider, candidate.model):
                log.info(
                    "Skipping %s/%s — over budget", candidate.provider, candidate.model
                )
                continue

            if not provider.is_available():
                log.info(
                    "Skipping %s/%s — unavailable", candidate.provider, candidate.model
                )
                continue

            decision.chosen = candidate
            decision.reason = f"complexity={complexity.value}, within budget"
            return decision

        # Ultimate fallback — try ollama even if not in candidate list
        ollama = self._providers.get("ollama")
        if ollama and ollama.is_available():
            fallback = ModelCandidate("ollama", self._classifier_model)
            decision.chosen = fallback
            decision.reason = "fallback to local model (all external options exhausted)"
            return decision

        decision.reason = "no provider available"
        return decision
