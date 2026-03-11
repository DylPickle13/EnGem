# EnGem

EnGem is a local assistant and automation framework that connects a streaming LLM (Gemini) to host-side tooling and exposes a Discord bot interface. It coordinates sub-agent execution plans, browser automation, code and notebook execution, and media generation — then captures artifacts in `generated_files/`.

This README was updated on 2026-03-11 to reflect the current repository layout and usage.

Quick start

1. Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Install Playwright browser binaries (required for browser automation):

```bash
python -m playwright install
```

3. Set required environment variables (examples):

- `DISCORD_BOT_TOKEN` — Discord bot token (required to run the bot).
- `PAID_GEMINI_API_KEY` — Gemini API key used by the LLM clients/tools.
- `DISCORD_ALLOWED_CHANNELS` — Comma-separated allowed channel names (optional).
- Optional: `REPO_PATH`, `CRON_JOB_HOUR`, `CRON_JOB_MINUTE`, `HEARTBEAT_INTERVAL_SECONDS`, `MODEL` — see `config.py` for defaults and comments.

Important: `config.py` currently contains default placeholders for convenience. Do not commit real secrets — set them in the environment or a secrets manager.

Run the Discord bot

From the repository root:

```bash
python discord_bot.py
```

This starts the bot (configured via environment variables). The bot writes per-channel conversation history files to `memory/channel_history/` and uses `llm.py` to orchestrate intent classification and sub-agent execution.

Tools and helpers

The repository contains several helper tools under `tools/` (previously documented as `skills/`). Common helpers include:

- `tools/run_google_search.py` — Google Search via Gemini (example `__main__` included).
- `tools/run_python.py` — Safely run Python snippets and capture/relocate generated artifacts into `generated_files/`.
- `tools/run_notebook.py` — Execute notebooks via Papermill; outputs are saved under `results/<timestamp>`.
- `tools/use_browser.py` — High-level browser automation that uses the `computer_use` client and Playwright.
- `tools/generate_image.py`, `tools/generate_video.py` — Media generation helpers.
- `tools/collect_generated_media.py` — List recent generated outputs.

You can run most tools as small scripts or import their main functions. Example (run the built-in example in `run_google_search`):

```bash
python tools/run_google_search.py
```

Or call functions from a Python shell:

```bash
python -c "from tools.run_google_search import run_google_search; print(run_google_search('latest AI research summary'))"
```

Data, storage, and runtime artifacts

- Conversation histories: `memory/channel_history/*.md` (one file per Discord channel).
- Vector DB: `memory/vector_db/` (ChromaDB persistent client; default collection `engem_memory`).
- Generated outputs and artifacts: `generated_files/`. Several tools also reference legacy directories like `generated_images/` and `generated_videos/`.
- Sub-agent execution orders: `sub-agents/` (JSON files written at runtime, e.g. `sub-agents/execution_order_<history_name>.json`).
- Agent instruction templates and scheduled tasks: `agent_instructions/` (contains templates such as `manager.md`, `sub_agent.md`, and `cron_jobs/`).

Key modules (what to look at)

- `discord_bot.py` — Main bot entrypoint and message handling, heartbeat and cron scheduling.
- `llm.py` — LLM orchestration: intent classification, manager-runner flow, sub-agent dispatch, texter, and media selector.
- `memory.py` — ChromaDB-backed persistent vector memory store and helper APIs.
- `history.py`, `history_cache.py` — Conversation history parsing, file I/O, and history caching for model inputs.
- `api_backoff.py` — Exponential backoff wrapper used for Gemini/API calls.
- `config.py` — Environment variable defaults and helpers (change to suit your deployment).
- `tools/` — Host-side utilities for running code, notebooks, browser automation, and media generation.
- `agent_instructions/` — Markdown templates the manager/agents use as system instructions; edit carefully.

Developer notes & conventions

- Manager/Planner: `agent_instructions/manager.md` describes how the manager creates `sub-agents/execution_order_<history>.json` files (see `llm.py`). The manager enforces a final `Reviewer` agent that must print `<yes>` when the task is complete.
- Tool discovery: LLM sub-agents call the `tools` helpers; if you add a tool, ensure its function signatures are discoverable by any function-calling logic in `llm.py`.
- Artifact relocation: `tools/run_python.py` snapshots the repo root, runs code, and moves generated artifacts into `generated_files/` to keep the repo clean.
- Playwright: required for browser automation. After `pip install -r requirements.txt`, run `python -m playwright install`.
- ChromaDB: embeddings use `sentence-transformers` by default (falls back if unavailable). Back up `memory/vector_db/` to preserve memories.

Troubleshooting

- If Playwright actions fail, ensure browsers are installed (`python -m playwright install`) and that any Playwright-related environment variables are set.
- If Gemini/Google APIs error frequently, `api_backoff.call_with_exponential_backoff` will retry on transient errors; set proper API keys in the environment.
- If the bot does not receive messages, verify `DISCORD_BOT_TOKEN` and `DISCORD_ALLOWED_CHANNELS`.


File references (short)

- `discord_bot.py` (main bot)
- `llm.py` (LLM orchestration)
- `memory.py` (ChromaDB memory store)
- `config.py` (environment and defaults)
- `tools/` (automation helpers: run_python, run_notebook, run_google_search, use_browser, generate_image, ...)
- `agent_instructions/` (manager/sub-agent templates and scheduled job definitions)
