"""Interactive REPL with rich formatting and prompt_toolkit input."""

from __future__ import annotations

import argparse
import logging
import sys

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from tinyagent.agent import Agent
from tinyagent.config import Config

console = Console()


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        prog="tinyagent",
        description="Local-first AI agent with smart delegation",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Maximum budget in dollars (default: $5.00)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Force a specific provider/model (e.g. anthropic/claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show routing decisions, complexity, and cost details",
    )
    args = parser.parse_args()

    config = Config.from_env()
    if args.budget is not None:
        config.budget = args.budget
    if args.model is not None:
        config.force_model = args.model
    if args.verbose:
        config.verbose = True

    return config


def handle_slash_command(command: str, agent: Agent) -> bool:
    """Handle slash commands. Returns True if handled."""
    cmd = command.strip().lower()

    if cmd in ("/quit", "/exit", "/q"):
        console.print("[dim]Goodbye![/dim]")
        sys.exit(0)

    if cmd == "/help":
        console.print(
            "[bold]Commands:[/bold]\n"
            "  /budget   — Show budget status\n"
            "  /history  — Show API call history\n"
            "  /clear    — Clear conversation history\n"
            "  /quit     — Exit\n"
            "  /help     — Show this help"
        )
        return True

    if cmd == "/budget":
        console.print(agent.budget.summary())
        return True

    if cmd == "/history":
        history = agent.budget.history_table()
        if not history:
            console.print("[dim]No API calls yet.[/dim]")
            return True
        table = Table(title="API Call History")
        table.add_column("Provider", style="cyan")
        table.add_column("Model", style="green")
        table.add_column("Tokens (in+out)", style="yellow")
        table.add_column("Cost", style="red")
        for row in history:
            table.add_row(row["provider"], row["model"], row["tokens"], row["cost"])
        console.print(table)
        return True

    if cmd == "/clear":
        agent.conversation.clear()
        console.print("[dim]Conversation cleared.[/dim]")
        return True

    return False


def main() -> None:
    config = parse_args()

    if config.verbose:
        logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    console.print(
        "[bold blue]tinyagent[/bold blue] — local-first AI agent\n"
        f"[dim]Budget: ${config.budget:.2f} | "
        f"Local model: {config.ollama_model} | "
        f"Type /help for commands[/dim]\n"
    )

    try:
        agent = Agent(config)
    except RuntimeError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        console.print("[dim]Make sure Ollama is running: ollama serve[/dim]")
        sys.exit(1)

    session: PromptSession = PromptSession(history=InMemoryHistory())

    while True:
        try:
            user_input = session.prompt("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            if handle_slash_command(user_input, agent):
                continue
            console.print(f"[dim]Unknown command: {user_input}[/dim]")
            continue

        with console.status("[bold green]Thinking...", spinner="dots"):
            result = agent.chat(user_input)

        if result is None:
            continue

        # Show routing info in verbose mode
        if config.verbose and agent.last_routing:
            r = agent.last_routing
            console.print(
                f"[dim]  routing: complexity={r.complexity.value} "
                f"model={r.chosen.provider}/{r.chosen.model if r.chosen else '?'} "
                f"reason={r.reason} "
                f"cost=${result.cost:.6f}[/dim]"
            )

        # Budget warning
        if agent.budget.is_low and not agent.budget.is_depleted:
            console.print("[yellow]  Warning: budget is running low[/yellow]")
        elif agent.budget.is_depleted:
            console.print(
                "[red]  Budget depleted — only local model available[/red]"
            )

        console.print()
        console.print(Markdown(result.content))
        console.print()
