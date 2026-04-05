# Contributing to AgentLoop

Thanks for your interest in contributing. AgentLoop is intentionally minimal, before adding anything, ask: *does this make the loop clearer or more powerful?*

---

## Ways to Contribute

- **Add a new provider** —> wrap a new LLM in `providers/`
- **Add a new environment** —> wrap a new compiler/profiler in `environment.py`
- **Improve the prompt** —> better prompts = smarter agent behavior
- **Write tests** —> coverage for `memory.py`, `environment.py`, and `agent.py`
- **Fix bugs** —> check [open issues](https://github.com/jagkt/agentloop/issues)

---

## Setup

```bash
git clone https://github.com/jagkt/agentloop
cd agentloop
pip install -e ".[all,dev]"
cp .env.example .env  # add your keys
```

---

## Adding a New Provider

1. Create `providers/your_provider.py`
2. Inherit from `BaseLLMProvider`
3. Implement the `generate_response(prompt: str) -> str` method
4. Register it in `loop.py` `get_provider()`

```python
# providers/your_provider.py
from providers.base import BaseLLMProvider

class YourProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "your-model"):
        ...

    def generate_response(self, prompt: str) -> str:
        ...
        return response_text
```

That's it. No other files need to change.

---

## Adding a New Environment

Subclass or modify `environment.py`. The only contract is:

```python
def run(self, modifier: Modifier) -> Feedback:
    ...
```

The agent only ever calls `environment.run()` — your internal implementation is entirely up to you.

---

## Code Style

```bash
ruff check .      # lint
ruff format .     # format
mypy .            # type check
```

All PRs should pass `ruff` without errors.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Pull Request Guidelines

- Keep PRs small and focused — one thing at a time
- Add a clear description of what changes and why
- If adding a provider, include a usage example in the PR description
- Don't break the core loop contract: `Modifier → Environment → Feedback → Memory`

---

## Project Philosophy

AgentLoop has three rules:

1. **Only `agent.py` calls the LLM** — everything else is pure logic
2. **Only `environment.py` touches the compiler/profiler** — the agent never runs code directly
3. **Memory is the agent's only state** — no hidden globals, no side effects

Any contribution that violates these rules will be declined, no matter how useful it seems.