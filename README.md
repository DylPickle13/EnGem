# PICKLEBOT

PICKLEBOT is a personal/local assistant framework that connects a streaming LLM to host-side tooling and a Discord bot interface. It's a developer-first codebase for experimenting with LLMs, vector memory, and lightweight sub-agents.

**Current project state (2026-02-20)**

- **Virtual environment:** `.venv` created at the repository root using Homebrew Python 3.12 (Python 3.12.12).
- **Dependencies:** Installed from `requirements.txt` into the project venv. A full install was completed; pip may show a newer version available.
- **Bot runtime:** `discord_bot.py` was started and logged in successfully as `PICKLEBOT#9572` during testing.
- **Voice support:** `PyNaCl` was installed in the venv to satisfy `discord.py` voice dependencies (the previous warning about missing `PyNaCl` was resolved).

**Quick Start / Repro steps**

- Create (if needed) and activate the project virtualenv:

```bash
# Use your preferred Python; this project uses Python 3.12
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
```

- Install dependencies (already done in this workspace):

```bash
pip install -r requirements.txt
```

- Run the bot:

```bash
source .venv/bin/activate
python discord_bot.py
```

**Notes & tips**

- If your shell does not auto-activate the venv, run `source .venv/bin/activate` before running commands in this repo.
- To recreate the venv with a specific Python version, remove `./.venv` then run the `python3.12 -m venv .venv` command above.
- Pip suggested an update at the time of installing requirements; update with `pip install --upgrade pip` if desired.

**Project layout (short)**

- `discord_bot.py` — Discord bot entrypoint and handlers.
- `llm.py` — LLM provider wiring and response composition.
- `vector_database.py` and `vector_database/` — Chroma DB helpers and persistence.
- `tools.py` — Local helper utilities used by agents.
- `credentials.py` — Credentials shim (prefer environment variables).

**Data & persistence**

- The Chroma DB and related files live under `vector_database/` (e.g., `chroma.sqlite3`). Back these up if you need to preserve embeddings.
- Conversation logs are under `memory/`.

**Next actions I can take**

- Re-run `discord_bot.py` and confirm no startup warnings (including voice).  
- Add a short VS Code dev setup note (auto-select `./.venv` interpreter).  
- Add a `Makefile` or `dev` script to simplify venv creation and run commands.

If you want any of the above, tell me which and I'll implement it.