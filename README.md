# tinyagent

Local-first AI agent. Runs a local Ollama model for everyday tasks and automatically delegates to cloud providers when the task demands it.

## How it works

1. You send a message
2. Your local model classifies its complexity (trivial → critical)
3. Simple stuff stays local and free. Hard stuff gets routed to the best available cloud model
4. A budget system tracks every cent and cuts off external calls when you hit your limit

```
"hi"                          → ollama/llama3.2 (free)
"explain TCP"                 → ollama/llama3.2 (free)
"write an async web scraper"  → google/gemini-2.5-flash ($0.001)
"design a payment system"     → anthropic/claude-sonnet-4-6 ($0.02)
```

## Quickstart

```bash
# requires ollama running locally
pip install -e .
tinyagent --verbose
```

Set API keys for cloud providers (all optional):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export GOOGLE_API_KEY=AIza...
```

## CLI

```
tinyagent                    # start REPL
tinyagent --verbose          # show routing decisions and costs
tinyagent --budget 1.00      # limit external spending to $1
tinyagent --model ollama/deepseek-r1   # force a specific model
```

### REPL commands

| Command    | Description              |
|------------|--------------------------|
| `/budget`  | Show spending and limit  |
| `/history` | Show all API calls       |
| `/clear`   | Reset conversation       |
| `/quit`    | Exit                     |

## Routing table

| Complexity | Models (in order of preference)                          |
|------------|----------------------------------------------------------|
| trivial    | ollama (local)                                           |
| low        | ollama (local)                                           |
| medium     | google/gemini-2.5-flash → ollama                         |
| high       | anthropic/claude-sonnet → openai/gpt-4o → google/gemini-2.5-pro → ollama |
| critical   | anthropic/claude-opus → openai/o3 → anthropic/claude-sonnet → ollama     |

If a cloud model is over budget or unavailable, it falls through to the next candidate. Ollama is always the final fallback.

## Configuration

Environment variables:

| Variable                | Default   |
|-------------------------|-----------|
| `TINYAGENT_BUDGET`      | `5.0`     |
| `TINYAGENT_OLLAMA_MODEL`| `llama3.2`|
| `TINYAGENT_VERBOSE`     | `false`   |
| `OLLAMA_HOST`           | localhost |
