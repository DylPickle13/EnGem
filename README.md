# EnGem

EnGem is a developer-focused local assistant framework that connects a streaming LLM to host-side tooling and exposes a Discord bot interface. It's designed for experimentation with vector memory (Chroma), lightweight sub-agents, and automation scripts.

**Current project state (2026-02-25)**

- **Repository:** active; virtual environment exists at the repo root (`.venv`).
- **Python / venv:** Project uses a local virtualenv (`.venv`). Activate it with `source .venv/bin/activate` before running scripts.
- **Dependencies:** Listed in `requirements.txt`; install with `pip install -r requirements.txt` inside the venv.
- **Bot code:** `discord_bot.py` is the main entrypoint for the Discord integration.
- **LLM wiring:** `llm.py` contains the LLM provider integration and response composition logic.
- **Memory / Vector DB:** Persistent data and embeddings live under `memory/vector_db/` (includes `chroma.sqlite3`). Conversation histories are in `memory/channel_history/`.
- **Skills & helpers:** Reusable skill scripts are in the `skills/` folder (e.g., `run_python.py`, `run_google_search.py`, `use_browser.py`, `git_push.py`).
- **Agent instructions:** `agent_instructions/` contains guidance and sub-agent docs used by the system.

Quick commands

```bash
# activate venv
source .venv/bin/activate

# install deps (if needed)
pip install -r requirements.txt

# run the Discord bot
python discord_bot.py
```

Discord voice conversation (Gemini Live)

- Install voice dependencies:

```bash
pip install -r requirements.txt
```

- Ensure `GEMINI_API_KEY` is set.
- The bot now joins/moves with users in voice channels, listens to non-bot users, streams PCM audio to Gemini Live, and plays Gemini audio responses back into the same voice channel.
- Incoming voice capture requires `discord-ext-voice-recv` and PyNaCl.

Project layout (key files and folders)

- `discord_bot.py`: Discord bot entrypoint and command/event handlers.
- `llm.py`: LLM provider and response orchestration.
- `memory.py`, `history.py`: Helpers for storing/retrieving conversational memory.
- `memory/`: Persistent data (channel histories and `vector_db/`). See `memory/vector_db/chroma.sqlite3` for embeddings.
- `computer_use.py`: Opens the browser. 
- `skills/`: Modular scripts and tools the bot can invoke.
- `agent_instructions/`: Markdown instructions and job templates for sub-agents and scheduled jobs.
- `config.py`: Centralized configuration management via environment variables.
- `requirements.txt`: Python dependencies for the project.
- `README.md`: This documentation file.