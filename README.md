# EnGem

EnGem is a local assistant and automation framework that connects a streaming LLM to host-side tooling and exposes a Discord bot interface. It provides:

- sub-agent execution plans and lightweight skills
- persistent conversation memory and a Chroma vector DB
- a Discord bot for chat, voice, and tool orchestration

**This README** summarizes how to get started, where key pieces live, and how to run and develop the project.

**Quick Start**

1. Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Configure required environment variables (examples):

- `DISCORD_BOT_TOKEN`: Discord bot token (see [config.py](config.py)).
- `GEMINI_API_KEY`: API key for Gemini / LLM provider.
- Optional: `DISCORD_ALLOWED_CHANNELS`, `VOICE_TOOL_TARGET_CHANNEL_NAME`, `MODEL` — see [config.py](config.py) for defaults.

3. Run the Discord bot:

```bash
python discord_bot.py
```

**Key Components**

- **Bot:** [discord_bot.py](discord_bot.py) — Discord integration, event handlers, and execution-plan progress tracking.
- **LLM integration:** [llm.py](llm.py) — model wiring, prompts, and preview/dispatch helpers.
- **Configuration:** [config.py](config.py) — environment variables and defaults.
- **Memory & embeddings:** `memory/` — channel histories under `memory/channel_history/` and embeddings at [memory/vector_db/chroma.sqlite3](memory/vector_db/chroma.sqlite3).
- **Skills:** `skills/` — modular tools the agent can invoke (e.g., [skills/run_python.py](skills/run_python.py), [skills/use_browser.py](skills/use_browser.py)).
- **Agent docs:** [agent_instructions/manager.md](agent_instructions/manager.md) — templates and guidance for sub-agents.
- **Sandbox & notebooks:** [sandbox/run_notebook.py](sandbox/run_notebook.py) and `sandbox/notebooks/` for experiments.

**Running & Development Notes**

- The project uses a plain Python entrypoint; run the bot with `python discord_bot.py` while the virtualenv is active.
- Edit code, then restart the process to pick up changes. The repo contains many small scripts under `skills/` that can be exercised independently.
- Use `requirements.txt` to reproduce the environment.

**Behavioral notes**

- The execution-plan progress tracker posts and updates a Discord message while sub-agents run. Message edits are now guarded so the bot only edits when the content actually changes, reducing unnecessary API calls and avoiding rate-limit pressure.

**Data & Storage**

- Conversation histories: `memory/channel_history/`.
- Vector DB / embeddings: [memory/vector_db/chroma.sqlite3](memory/vector_db/chroma.sqlite3).
- Generated outputs: `generated_files/`.

**Useful Commands**

```bash
# activate venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# run the Discord bot
python discord_bot.py
```