"""Cost tracking and budget enforcement."""

from __future__ import annotations

from dataclasses import dataclass, field

from tinyagent.models import CompletionResult
from tinyagent.pricing import estimate_cost


@dataclass
class BudgetEntry:
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost: float
    timestamp: float


class Budget:
    def __init__(self, limit: float = 5.0) -> None:
        self.limit = limit
        self.spent = 0.0
        self.history: list[BudgetEntry] = []

    @property
    def remaining(self) -> float:
        return max(0.0, self.limit - self.spent)

    @property
    def is_low(self) -> bool:
        return self.remaining < self.limit * 0.15

    @property
    def is_depleted(self) -> bool:
        return self.remaining <= 0.0

    def can_afford(self, provider: str, model: str) -> bool:
        """Check if estimated cost of a call fits within remaining budget."""
        est = estimate_cost(provider, model)
        return est <= self.remaining

    def record(self, result: CompletionResult) -> None:
        """Record actual cost from a completed API call."""
        entry = BudgetEntry(
            provider=result.provider,
            model=result.model,
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            cost=result.cost,
            timestamp=result.timestamp,
        )
        self.history.append(entry)
        self.spent += result.cost

    def summary(self) -> str:
        """Return a human-readable budget summary."""
        lines = [
            f"Budget: ${self.spent:.4f} / ${self.limit:.2f} spent",
            f"Remaining: ${self.remaining:.4f}",
        ]
        if self.is_depleted:
            lines.append("Status: DEPLETED — external calls disabled")
        elif self.is_low:
            lines.append("Status: LOW — consider increasing budget")
        return "\n".join(lines)

    def history_table(self) -> list[dict]:
        """Return history as a list of dicts for display."""
        return [
            {
                "provider": e.provider,
                "model": e.model,
                "tokens": f"{e.prompt_tokens}+{e.completion_tokens}",
                "cost": f"${e.cost:.6f}",
            }
            for e in self.history
        ]
