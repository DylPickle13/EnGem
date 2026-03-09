# EnGem

EnGem is a local assistant and automation framework that connects a streaming LLM to host-side tooling and exposes a Discord bot interface. It provides:

- LLM orchestration and sub-agent execution plans
- A Discord bot (text + optional voice) with execution-plan progress tracking
- Persistent semantic memory backed by ChromaDB
- Modular skills for browser automation, running code, notebooks, and media generation
- Output capture and artifact relocation into `generated_files/`

This README shows how to get started, where important pieces live, and common development tasks.

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

- `DISCORD_BOT_TOKEN` — Discord bot token.
- `PAID_GEMINI_API_KEY` — Paid Gemini API key (required for Gemini Live / voice features).
- `DISCORD_ALLOWED_CHANNELS` — Comma-separated allowed channel names (optional).
- `VOICE_TOOL_TARGET_CHANNEL_NAME` — Text channel used to relay voice-tool messages (optional).
- Optional: `REPO_PATH`, `CRON_JOB_HOUR`, `CRON_JOB_MINUTE`, `HEARTBEAT_INTERVAL_SECONDS`, `MODEL` — see `config.py` for details.

4. Start the Discord bot from the repository root:

```bash
python discord_bot.py
```

Entry points & core modules

- `discord_bot.py` — Main bot entrypoint. Handles text messages, optional Gemini Live voice bridging, cron and heartbeat jobs, and execution-plan progress messages.
- `llm.py` — LLM orchestration: intent classification, manager, sub-agent dispatch, texter, and media selector.
- `memory.py` — ChromaDB-backed semantic memory store (default path: `memory/vector_db/`).
- `computer_use.py` & `skills/use_browser.py` — Browser automation using Playwright and Gemini ComputerUse tooling.
- `skills/run_python.py` — Run Python snippets in a subprocess; artifacts are relocated into `generated_files/`.
- `skills/run_notebook.py` — Execute notebooks via Papermill and save results under `results/<timestamp>`.
- `skills/generate_image.py` / `skills/generate_video.py` — Media generation using Gemini.
- `skills/collect_generated_media.py` — Helper to list recent generated outputs.

Data & storage

- Conversation histories: `memory/channel_history/*.md` (one file per channel).
- Vector DB: `memory/vector_db/` (ChromaDB persistent client and collection `engem_memory`).
- Generated outputs and artifacts: `generated_files/` (legacy directories: `generated_images/`, `generated_videos/`, `generated_documents/`).
- Agent instruction templates and scheduled tasks: `agent_instructions/` (see `cron_jobs/` and `heartbeat_jobs/`).

Notes & troubleshooting

- Playwright requires browser binaries; run `python -m playwright install` after installing `playwright`.
- Voice features require `PyNaCl` (`pip install pynacl`) and the `discord-ext-voice-recv` package for incoming audio capture. Set `PAID_GEMINI_API_KEY` for Gemini Live voice sessions.
- LLM API calls are wrapped with an exponential backoff helper (`api_backoff.call_with_exponential_backoff`) to handle transient errors.
- ChromaDB data lives under `memory/vector_db/`. Back up this directory if you need to preserve memories.
- `config.py` reads environment variables; avoid committing secrets into the repository. Set sensitive keys in the environment or a secure secret manager.

Developer notes

- Skills are lightweight Python modules under `skills/` and are automatically discovered for function-calling by `llm._get_function_declarations`.
- Generated artifacts from skill runs are collected in `generated_files/` and can be listed with `skills/collect_generated_media.py`.
- For development, run the bot locally after setting the required env vars and installing dependencies. Restart the process after code edits.

Repository map (short)

- [discord_bot.py](discord_bot.py) — main bot entrypoint
- [llm.py](llm.py) — LLM orchestration
- [memory.py](memory.py) — ChromaDB memory store
- [config.py](config.py) — environment variables and defaults
- [skills/](skills/) — agent skills and helper scripts
- [agent_instructions/](agent_instructions/) — instruction templates, cron/heartbeat tasks
- [generated_files/](generated_files/) — outputs captured from skills