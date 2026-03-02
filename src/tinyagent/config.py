"""Configuration: env vars -> CLI flags -> defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    budget: float = 5.0
    ollama_model: str = "llama3.2"
    ollama_host: str | None = None
    verbose: bool = False
    force_model: str | None = None  # override routing, format: "provider/model"

    @classmethod
    def from_env(cls) -> Config:
        """Build config from environment variables."""
        return cls(
            budget=float(os.environ.get("TINYAGENT_BUDGET", "5.0")),
            ollama_model=os.environ.get("TINYAGENT_OLLAMA_MODEL", "llama3.2"),
            ollama_host=os.environ.get("OLLAMA_HOST"),
            verbose=os.environ.get("TINYAGENT_VERBOSE", "").lower() in ("1", "true"),
        )
