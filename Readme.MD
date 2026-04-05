# AgentLoop

> Transforming AI from a "single-shot" responder into an autonomous, self-correcting, and goal-driven system.

AgentLoop is a minimal, observable agentic framework for code optimization tasks. It gives an LLM a goal, a tool, and an environment, then lets it loop until the goal is met or it runs out of attempts.

Built as the foundation for [KernelScope](#roadmap), an AI-powered GPU kernel optimizer.

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
git clone https://github.com/yourname/agentloop
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
agentloop/
  providers/
    __init__.py
    base.py                # abstract LLM interface
    groq_provider.py
    ollama_provider.py
    anthropic_provider.py
    openai_provider.py
  modifier.py              # data model: proposed code change
  feedback.py              # data model: environment results
  memory.py                # iteration history
  environment.py           # compile + profile
  agent.py                 # LLM reasoning loop
  loop.py                  # entry point
  .env.example
  pyproject.toml
```

---

## Roadmap

AgentLoop is the foundation. Here's where it's going:

- [ ] **v0.2** — YAML task DSL (define tasks as config files)
- [ ] **v0.3** — Structured reward signals (not just pass/fail)
- [ ] **v0.4** — Parallel environment execution
- [ ] **v1.0** — Plugin system for custom environments
- [ ] **KernelScope** — Real `nvcc` + `ncu` integration for GPU kernel optimization

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT