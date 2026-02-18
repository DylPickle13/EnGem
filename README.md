# JARVIS — Local Agent / Assistant

JARVIS is a compact local agent framework that connects a streaming LLM to a Telegram bot (with optional audio transcription and code-execution helpers). It's intended as a developer-focused starting point for building and testing LLM-powered assistants locally.

**Quick Start**

- **Requirements:** Python 3.10+ and a working virtual environment
- Create and activate a venv, then install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- Provide credentials (recommended: use environment variables):
	- `TELEGRAM_BOT_TOKEN` — Telegram bot token for `main.py`
	- Optionally `HF_TOKEN` or other provider tokens if configured

- Run the agent locally:

```bash
python main.py
```

**Project Structure**

- [main.py](main.py): Telegram handlers and the program entrypoint.
- [llm.py](llm.py): LLM loader, prompt composition, and generation logic.
- [tools.py](tools.py): Utility helpers (history management, code extraction/execution, search placeholders).
- [credentials.py](credentials.py): Local credentials shim (keep out of VCS; prefer env vars).
- [history.md](history.md): Message history used to build conversational context.
- [agent_instructions/](agent_instructions/): Human-readable instruction files that shape agent behavior (e.g., [agent_instructions/instructions.md](agent_instructions/instructions.md)).
- [sandbox/memory_system/memory_system.py](sandbox/memory_system/memory_system.py): Example memory subsystem.

**How it works (high-level)**

- Incoming messages are handled in `main.py` and converted into prompts.
- `llm.py` composes prompts from `agent_instructions` + `history.md`, runs generation, and returns streamed output.
- `tools.py` can post-process model output: run searches, extract fenced Python code, execute it locally, and append outputs to the reply.

Security note: code execution is performed with the local Python interpreter. Treat model-generated code as untrusted. Use sandboxing or disable `tools.run_python()` in risky environments.

**Development Tips**

- Prefer environment variables for tokens instead of editing `credentials.py`.
- Use the virtual environment workflow above when iterating.
- To add features: add new instruction files under `agent_instructions/`, implement helper functions in `tools.py`, or swap the LLM backend in `llm.py`.