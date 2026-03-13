# EnGem

EnGem is a local assistant and automation framework that connects a streaming LLM (Gemini) to host-side tooling and exposes a Discord bot interface. It coordinates intent classification, manager-planner flows, staged sub-agent execution, browser automation, notebook and code execution, media generation, attachment ingestion and archiving, and a ChromaDB-backed memory system. Generated artifacts and user-visible outputs are consolidated under `generated_files/`.

Last updated: 2026-03-13

Quick start

1. Create and activate a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Install Playwright browsers (required for browser automation):

```bash
python -m playwright install
```

3. Set environment variables (examples — see `config.py` for defaults):

- `DISCORD_BOT_TOKEN` — Discord bot token (required to run the bot).
- `PAID_GEMINI_API_KEY` — Paid Gemini API key used for model calls and embeddings.
- `DISCORD_ALLOWED_CHANNELS` — Optional comma-separated channel names to limit bot activity.
- `MODEL` — Default model alias; `MINIMAL_MODEL`, `LOW_MODEL`, `MEDIUM_MODEL`, and `HIGH_MODEL` are defined in `config.py`.
- Embedding/memory settings: `GEMINI_EMBEDDING_MODEL`, `GEMINI_EMBEDDING_DIM`, `GEMINI_EMBEDDING_BATCH_SIZE`, `MEMORY_SEMANTIC_COLLECTION_NAME`, `MEMORY_FILE_COLLECTION_NAME`, `MEMORY_ARCHIVE_DIR`.

Important: `config.py` ships with convenient defaults and some placeholder values. Do not commit real secrets — use environment variables or a secrets manager.

Running the Discord bot

From the repository root:

```bash
python discord_bot.py
```

The bot creates per-channel conversation history files in `memory/channel_history/`, responds to messages via `llm.py`, and can attach generated media from `generated_files/` to replies.

Bot commands

- `>commands` — List available bot commands.
- `>history length` — Show conversation history length.
- `>clear history` — Clear the current channel history file.
- `>clear memory` — Clear semantic and file memory stores (`memory.clear_all_memory_stores`).
- `>forget memories {topic}` — Semantically forget memories related to a topic.
- `>list memories [limit]` — List stored memory records.

Core concepts

- Manager / Execution Plan: The manager (driven by `agent_instructions/manager.md`) writes a JSON execution plan to `sub-agents/execution_order_<history>.json`. Plans are staged; each stage is `serial` or `parallel` and contains sub-agent instructions.
- Reviewer: The final stage must be a serial `Reviewer` agent that returns `<yes>` to indicate completion.
- Tools: Public functions in `tools/` are exposed to sub-agents via the function-calling mechanism (`llm._get_function_declarations`). To add a tool, create a top-level function (no leading underscore) in a `tools/*.py` file.
- History caching: `history_cache.py` can create cached content for Gemini so long conversation histories are not repeatedly sent to the API.

Attachments & media

- `attachments.py` ingests media attachments (images, audio, video, PDFs), extracts text/metadata using Gemini where possible, and returns extracted segments for inclusion in prompts.
- Attachments are archived content-addressed in `memory/file_archive/` and indexed into the file-memory collection via `memory.write_attachment_memory`.
- Generated outputs (images, videos, documents, reports) are stored under `generated_files/`. Legacy directories `generated_images/` and `generated_videos/` remain recognized by helper scripts for backward compatibility.
- `collect_generated_media.py` builds a catalog of generated media and implements `select_media_paths` used by the LLM to pick assets for attachment to replies; the selector system instruction is `agent_instructions/media_selector.md`.

Memory & embeddings

- Persistent memory is backed by ChromaDB in `memory/vector_db/` and managed in `memory.py`.
- Embeddings are created with the configured Gemini embedding model (`GEMINI_EMBEDDING_MODEL`) and stored alongside metadata. File records use a separate collection (see `MEMORY_FILE_COLLECTION_NAME`).
- `memory.run_memory_extraction_async` and `memory._parse_memory_extraction_response` implement automated memory extraction workflows used after manager runs for longer-term storage.

LLM orchestration & sub-agents

- `llm.py` handles intent classification, manager planning, staged execution of sub-agents (parallel/serial), invoking tools when allowed, running a final `Reviewer`, and producing a final `Texter` response.
- Execution plan JSON shape expected by the runner:

	{
		"execution_plan": [
			{"mode": "parallel|serial", "sub_agents": [{"task_name": "name", "instruction": "...", "thinking_level": "MINIMAL|LOW|MEDIUM|HIGH"}, ...]},
			...
		]
	}

- The manager must include a final serial stage with a single `Reviewer` sub-agent named `Reviewer`.

Browser automation (computer use)

- `computer_use.py` provides a Playwright-driven browser agent integrated with Gemini's `ComputerUse` tool type. It supports navigation, element extraction, clicks, typing, screenshots and returns function responses containing screenshots and action results.
- Use `tools/use_browser.py` for higher-level helper scripts that invoke the browser agent.

Calendar events

- `calendar_events.py` polls Google Workspace (via `tools/access_google_workspace.py`) for active events and can inject event descriptions into matching Discord channels through the bot's message pipeline.

Progress indicators

- `progress_indicator.py` provides `ExecutionPlanProgressIndicator` which posts an ASCII preview of the execution plan and updates progress messages in Discord channels while sub-agents run.

Tools and helpers

- Directory: `tools/` — host-side utilities and integration scripts. Notable helpers:
	- `tools/access_google_workspace.py`
	- `tools/run_google_search.py`
	- `tools/run_python.py` (snapshots execution and relocates artifacts into `generated_files/`)
	- `tools/run_notebook.py`
	- `tools/use_browser.py`
	- `tools/generate_image.py`, `tools/generate_video.py`
	- `tools/deep_research.py`

Developer notes & conventions

- Add tools as top-level functions in `tools/*.py` to make them discoverable by `llm._get_function_declarations`.
- Edit instruction templates in `agent_instructions/` carefully — they directly affect planning and agent behavior.
- Back up `memory/vector_db/` regularly to preserve memories; automatic collection-name migration is not provided.

Troubleshooting

- Playwright errors: ensure browsers are installed (`python -m playwright install`) and your environment supports headless/GUI runs as needed.
- API failures: check `PAID_GEMINI_API_KEY` and network connectivity. Transient errors are retried via `api_backoff.call_with_exponential_backoff`.
- Discord issues: verify `DISCORD_BOT_TOKEN` and `DISCORD_ALLOWED_CHANNELS` settings.

Short file references

- `discord_bot.py` — main bot and message pipeline
- `llm.py` — orchestration, planning, and sub-agent execution
- `memory.py` — ChromaDB-backed memory store and attachment archiving
- `attachments.py` — attachment ingestion and extraction
- `collect_generated_media.py` — media catalog and selection helpers
- `computer_use.py` — Playwright-backed browser agent implementation
- `calendar_events.py` — calendar polling and event injection helper
- `progress_indicator.py` — Discord execution-plan preview + progress indicator
- `tools/` — host-side automation helpers
- `agent_instructions/` — instruction templates used by manager and agents