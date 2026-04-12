# AgentLoop
[![CI](https://github.com/jagkt/agentloop/actions/workflows/ci.yaml/badge.svg)](https://github.com/jagkt/agentloop/actions/workflows/ci.yaml)

> Transforming AI from a "single-shot" responder into an autonomous, self-correcting, and goal-driven system.

AgentLoop is a minimal, observable agentic framework for code optimization tasks. It gives an LLM a goal, a tool, and an environment, then lets it loop until the goal is met or it runs out of attempts.

Built as the foundation for [KernelScope](https://github.com/jagkt/kernelscope), an AI-powered GPU kernel optimizer.

---

## How It Works

```
Task + Code + Goals
       ↓
   [Agent thinks]        ← LLM reasons about history
       ↓
   [Agent acts]          ← proposes a code modification
       ↓
   [Environment runs]    ← compiles + profiles the code
       ↓
   [Agent reflects]      ← reads feedback, updates memory
       ↓
  Goal met? → Exit
  else      → repeat
```

Each component maps directly to a module:

| Module | Responsibility |
|---|---|
| `agent.py` | Think, act, reflect (the only file that calls the LLM) |
| `environment.py` | Compile, profile, return feedback |
| `memory.py` | Accumulate iteration history across the loop |
| `modifier.py` | Data model for a proposed code change |
| `feedback.py` | Data model for environment results |
| `loop.py` | Wire everything together, entry point |

---

## Architecture

```
loop.py
  ├── Environment(goal_metric)
  ├── Memory()
  └── Agent(environment, memory, task, code, goal, llm_provider)
        │
        ├── think()       → builds prompt from task + memory.summary()
        ├── act()         → parses LLM response into Modifier
        ├── reflect()     → checks last feedback, decides to continue
        └── run()         → orchestrates the full loop
              │
              ├── environment.run(modifier)
              │     ├── _compile(code)
              │     └── _profile(code)
              │
              └── memory.store(modifier, feedback)
```

---

## Providers

AgentLoop is provider-agnostic. Switch LLMs with one word:

| Provider | Cost | Setup |
|---|---|---|
| `ollama` | Free, local | Install [Ollama](https://ollama.ai), run `ollama pull llama3` |
| `groq` | Free tier | Get key at [console.groq.com](https://console.groq.com) |
| `anthropic` | Paid | Get key at [console.anthropic.com](https://console.anthropic.com) |
| `openai` | Paid | Get key at [platform.openai.com](https://platform.openai.com) |

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/jagkt/agentloop
cd agentloop
pip install -e ".[all]"
```

### 2. Set up your environment

Copy the example env file:

```bash
cp .env.example .env
```

Add your keys to `.env`:

```
GROQ_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

### 3. Run

```bash
python loop.py
```

To switch providers, edit the last line of `loop.py`:

```python
provider="ollama"    # local, free, no key needed
provider="groq"      # free tier, fastest
provider="anthropic" # Claude
provider="openai"    # GPT-4o
```

---

## Example Output

```
[Agent] Starting loop. Max iterations: 5

[Agent] ---- Iteration 1 ----
[Agent] Thinking about iteration 1...
[Agent] Rationale: No previous attempts. Starting with shared memory tiling...
[Environment] Compiling code...
[Environment] Profiling kernel...
[Agent] Feedback: compiled=True, metric=67.3, goal_met=False
[Agent] Reflecting... metric=67.3, goal not yet met. Continuing.

[Agent] ---- Iteration 2 ----
[Agent] Thinking about iteration 2...
[Agent] Rationale: Previous attempt achieved 67.3%. Increasing tile size to improve reuse...
[Environment] Compiling code...
[Environment] Profiling kernel...
[Agent] Feedback: compiled=True, metric=84.1, goal_met=True
[Agent] Goal met at iteration 2 with metric=84.1. Stopping.

[Agent] Loop complete.
```

---

## Project Structure

```
agentloop/              ← repo root
  agentloop/            ← Python package
    providers/
      __init__.py
      base.py           # abstract LLM interface
      groq_provider.py
      ollama_provider.py
      anthropic_provider.py
      openai_provider.py
    __init__.py
    agent.py            # LLM reasoning loop
    environment.py      # mock compile + profile
    feedback.py         # data model: environment results
    loop.py             # entry point
    memory.py           # iteration history
    modifier.py         # data model: proposed code change
  .env.example
  .gitignore
  CONTRIBUTING.md
  LICENSE
  pyproject.toml
  README.md
```

---

## Building on AgentLoop
 
AgentLoop is designed to be extended. To build your own optimizer:
 
1. Install AgentLoop: `pip install agentloop`
2. Implement a custom environment:
 
```python
from agentloop.feedback import Feedback
from agentloop.modifier import Modifier
 
class MyEnvironment:
    def run(self, modifier: Modifier) -> Feedback:
        # compile, profile, return feedback
        ...
```
 
3. Wire it into the loop:
 
```python
from agentloop.agent import Agent
from agentloop.memory import Memory
 
agent = Agent(
    environment=MyEnvironment(),
    memory=Memory(),
    task="...",
    original_code="...",
    goal_description="...",
    llm_provider=GroqProvider(api_key="...")
)
agent.run(max_iterations=5)
```
 
See [KernelScope](https://github.com/jagkt/kernelscope) for a full real-world example.
 
---
 
## Roadmap
 
- [ ] **v0.2** - YAML task DSL
- [ ] **v0.3** - Structured reward signals
- [ ] **v0.4** - Parallel environment execution
- [ ] **v1.0** - Plugin system for custom environments
 
---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT
