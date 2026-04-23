# EnGem

EnGem — a local assistant and agent orchestration framework driven by a streaming
LLM (Gemini via `google-genai`) and host-side tooling. EnGem coordinates intent
classification, staged planning and execution (Planner → Execution), sub-agent
orchestration (serial and parallel stages), browser automation, notebook and
Python execution, media generation, attachment ingestion and indexing, and a
ChromaDB-backed local memory system. Generated artifacts and runtime outputs are
organized under `generated_files/`.

Last updated: 2026-04-22

Overview

EnGem is designed to run structured, multi-step workflows driven by an LLM that
can call host utilities (the `tools/` modules) and persist both semantic and
file memories locally. Typical usage is via the bundled Discord bot
integration, but the core orchestration (`llm.py`) can be used programmatically
to integrate the Planner → Execution manager flow into other interfaces. The
current repo also includes calendar-triggered prompt dispatch, periodic cron
jobs, browser CDP attach/fallback modes, Gemini-grounded search and deep-
research helpers, and a Copilot CLI smoke-test entrypoint.

Highlights

- **Planner → Execution pipeline**: Planner generates staged `planner_plan` JSON
	which the Execution manager then runs deterministically against a cached
	history snapshot.
- **Sub-agents**: Stages contain `sub_agents` with `mode` (`serial|parallel`),
	`thinking_level` (`MINIMAL|LOW|MEDIUM|HIGH`) and optional `force_tool`.
- **Cancellation-aware runs**: execution progress messages now expose a Stop
	button, and cancellation propagates through the model, browser, and tool
	layers.
- **Function-calling tools**: Host utilities in `tools/` are exposed to the
	model as callable functions for safe, auditable tool use.
- **Local memory**: ChromaDB-backed semantic and file memory with Gemini
	embeddings, skill extraction and skill DB sync from `skills/` files.
- **Attachments & media**: Attachments are ingested, optionally embedded, and
	archived; generated media is cataloged under `generated_files/` and selectable
	by the model.
- **Browser & ComputerUse**: `computer_use.py` provides a Playwright-backed
	ComputerUse tool for web automation, persistent profiles, and CDP attach to a
	running Chrome instance.
- **Integrations**: Tools include Google Workspace helpers, Gemini search and
	deep research, Copilot CLI smoke tests, image/speech/video generation,
	notebook execution, and more.

Recent changes since 2026-04-03

- Browser automation now supports cancellation-aware runs, persistent Chrome
	profiles, CDP attach, and auto-launching a companion Chrome instance when
	needed.
- The Discord bot now launches a cron-job runner for `scripts/cron_jobs/*.py`
	and a calendar poller that injects prompt files from `prompts/<event>.md`
	into matching channels.
- Google Workspace and YouTube helpers now resolve API keys from environment
	variables, files, or payloads; the YouTube helper is API-key based and meant
	for public read-only calls.
- Added `tools/run_google_search.py`, `tools/deep_research.py`, and
	`tools/debug/run_copilot.py` for Gemini-grounded search, background research,
	and Copilot CLI smoke tests.
- `agent_instructions/` was refreshed for browser setup/cancellation,
	planner/execution reviewer flow, and Veo video prompting.
- `history_cache.py` now invalidates stale or missing cached content and
	refreshes cache TTLs when needed.

Quickstart

1. Create and activate a virtual environment, then install dependencies:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

EnGem's current dependency stack builds most reliably on Python 3.13 for the
`.venv`; Python 3.14 can fail while building `tokenizers` for `chromadb`.

2. If you will use browser automation, install Playwright browsers:

```bash
python -m playwright install
```

3. Set required environment variables (see `config.py` for defaults):

- Core runtime: `DISCORD_BOT_TOKEN`, `PAID_GEMINI_API_KEY`,
	`DISCORD_ALLOWED_CHANNELS`, `DEFAULT_INFERENCE_MODE`, `BOT_RUNTIME_STATE_PATH`.
- Google services: `GOOGLE_API_KEY`, `GOOGLE_API_KEY_PATH`, `GOOGLE_CALENDAR_ID`.
	API-key calendar polling works for public calendar IDs only; private calendars
	and `primary` require OAuth via `gws auth login`.
- Memory and embeddings: `GEMINI_EMBEDDING_MODEL`, `GEMINI_EMBEDDING_DIM`,
	`GEMINI_EMBEDDING_BATCH_SIZE`, `MEMORY_SEMANTIC_COLLECTION_NAME`,
	`MEMORY_FILE_COLLECTION_NAME`, `MEMORY_ARCHIVE_DIR`, `ATTACHMENT_EMBEDDING_MODE`,
	`ATTACHMENT_EMBEDDING_MAX_BYTES`, `ATTACHMENT_MULTIMODAL_MAX_BYTES`.
- Browser automation: `BROWSER_LAUNCH_MODE`, `BROWSER_CHROME_USER_DATA_DIR`,
	`BROWSER_CHROME_PROFILE_DIRECTORY`, `BROWSER_CHROME_CHANNEL`,
	`BROWSER_CDP_ENDPOINT`, `BROWSER_CDP_PREFER_ATTACH`,
	`BROWSER_CDP_ATTACH_ON_PROFILE_LOCK`, `BROWSER_CDP_OPEN_NEW_TAB`,
	`BROWSER_CDP_AUTO_LAUNCH_COMPANION`,
	`BROWSER_CDP_CLONE_DEFAULT_PROFILE_ON_FIRST_RUN`,
	`BROWSER_CDP_COMPANION_USER_DATA_DIR`,
	`BROWSER_CDP_COMPANION_PROFILE_DIRECTORY`,
	`BROWSER_CDP_COMPANION_CHROME_BINARY`,
	`BROWSER_CDP_CONNECT_TIMEOUT_SECONDS`,
	`BROWSER_CDP_CONNECT_RETRY_INTERVAL_SECONDS`.
- Copilot CLI helper: `COPILOT_PROVIDER_BASE_URL`, `COPILOT_PROVIDER_TYPE`,
	`COPILOT_PROVIDER_API_KEY`, `COPILOT_DEFAULT_MODEL`, `COPILOT_MODEL`,
	`COPILOT_AUTOPILOT`, `COPILOT_STREAM_MODE`,
	`COPILOT_DISABLE_BUILTIN_MCPS`, `COPILOT_DISABLE_TOOL_CALLS`,
	`COPILOT_EXCLUDED_TOOLS`.

Notes: keep secret keys out of source control — use environment variables or a
secrets manager.

Running

- Start the Discord bot (recommended for interactive use):

```bash
python discord_bot.py
```

On startup the bot syncs skills, starts the calendar poller, and launches the
cron-job runner for `scripts/cron_jobs/*.py`.

- Discord convenience commands (default prefix `>`):

	- `>commands` — list available bot commands
	- `>inference mode` — show active inference mode
	- `>inference mode standard` — set standard mode
	- `>inference mode flex` — set flex mode
	- `>history length` — show length of conversation history
	- `>clear history` — clear channel history
	- `>clear memory` — clear all memory stores
	- `>forget memories {topic}` — semantically delete memories related to topic
	- `>list memories [limit]` — list stored memories

Execution-plan progress messages include a Stop button that requests
cancellation for the active run.

- Tools and helper scripts (see docstrings under `tools/`):

```bash
python tools/run_python.py --script example.py
python tools/run_notebook.py --notebook path/to/notebook.ipynb
python tools/access_google_workspace.py ...
python tools/access_youtube.py ...
python tools/debug/run_copilot.py
```

Gemini-grounded search and background research live in
`tools/run_google_search.py` and `tools/deep_research.py`.

Architecture & key concepts

- **Planner → Execution**: Planner composes a staged plan (written to
	`sub-agents/planner_order_<history>.json`); once the Planner and Planner
	Reviewer indicate readiness, the Execution manager builds an `execution_plan`
	and runs sub-agents against a stable cached history snapshot. Runs are
	cancellation-aware and can be stopped from the progress UI.
- **Sub-agents**: Each stage contains `sub_agents` (task_name, instruction,
	thinking_level, optional force_tool). Stages can be `parallel` or `serial`.
- **Function-calling**: `llm.py` exposes functions defined in `tools/` as
	callable activities the model can request; `_get_function_declarations()` is
	used to build the function-calling surface.
- **History caching**: `history_cache.py` can pre-warm and reuse Gemini cached
	content for efficient repeated model calls on the same conversation prefix,
	and now invalidates stale or missing cached content while refreshing cache
	TTLs when needed.
- **Calendar and cron jobs**: `discord_bot.py` polls Google Calendar and runs
	scripts from `scripts/cron_jobs/`; calendar events whose summary matches a
	prompt file are dispatched into the matching Discord channel.

Memory, Embeddings & Skills

- Persistent memory is provided by `memory.py` backed by ChromaDB collections
	stored under the `memory/` folder (semantic memories, file records, and a
	skill collection).
- The repo supports automatic extraction of semantic memories and reusable
	planning skills via LLM extractors (`agent_instructions/memory_extractor.md`,
	`agent_instructions/skill_extractor.md`). Skill files in `skills/` are
	synchronized into the skill DB on startup (`memory._sync_skills_from_folder()`).

Attachments & generated media

- `attachments.py` ingests attachments (images, audio, video, PDFs), extracts
	text where possible and optionally embeds attachments using Gemini
	multimodal/text embeddings.
- Generated assets (images, audio, notebooks, etc.) are placed in
	`generated_files/`. `collect_generated_media.py` builds a JSON catalog that
	the model can use to select media to attach to responses. `tools/generate_video.py`
	now uses `veo-3.1-lite-generate-preview` and accepts structured JSON prompts
	for advanced control.

Tools & integrations

- Google Workspace CLI integration: `tools/access_google_workspace.py` — planner
	guidance and helpers for invoking the `gws` CLI, including public-calendar
	API-key polling when `GOOGLE_API_KEY` is available.
- YouTube API helpers: `tools/access_youtube.py` — API-key auth for public
	read-only YouTube Data API actions.
- Search and research helpers: `tools/run_google_search.py` and
	`tools/deep_research.py`.
- Media generation: `tools/generate_image.py`, `tools/generate_speech.py`,
	`tools/generate_video.py` (model-driven media generation saved to
	`generated_files/`, with Veo 3.1 Lite preview for video).
- Notebook and script execution: `tools/run_notebook.py` and
	`tools/run_python.py` move produced artifacts into `generated_files/` for
	downstream selection.
- Browser automation / ComputerUse: `computer_use.py` integrates Playwright and
	implements a ComputerUse tool for agent-driven web interactions, screenshots,
	persistent profiles, and CDP attach.
- Copilot CLI helper: `tools/debug/run_copilot.py` runs a small local smoke test
	against the GitHub Copilot CLI with Gemini-compatible provider settings.

Developer notes

- Add new host utilities by creating top-level functions in `tools/*.py`.
	`llm._get_function_declarations()` will import these modules and expose
	functions to the model.
- Edit or add instruction templates in `agent_instructions/` to modify system
	prompts for Planner, Execution manager, Texter, Reviewers, browser setup, and
	media-generation behavior.
- Skill files (.md) placed in `skills/` are imported into the skill DB and
	become reusable planning building blocks; commit changes to `skills/` to
	update the runtime skill set (the bot synchronizes skills at startup).

Troubleshooting

- Playwright: run `python -m playwright install` and verify your environment
	supports the chosen browser mode (headless/headful). If CDP attach is flaky,
	check `BROWSER_LAUNCH_MODE` and `BROWSER_CDP_ENDPOINT`.
- Discord: confirm `DISCORD_BOT_TOKEN` and `DISCORD_ALLOWED_CHANNELS` are
	correct; check bot logs for connection errors, cron-job reports, and calendar
	polling messages.
- Google Workspace: ensure `gws` CLI is installed and authenticated when
	using `tools/access_google_workspace.py`; use a public `GOOGLE_CALENDAR_ID`
	with `GOOGLE_API_KEY`/`GOOGLE_API_KEY_PATH` for API-key polling, or run
	`gws auth login` for private calendars and `primary`.
- YouTube: `tools/access_youtube.py` now uses API-key auth, not OAuth; set
	`GOOGLE_API_KEY` or `GOOGLE_API_KEY_PATH` and stick to public read-only list
	calls.
- Copilot CLI: install and authenticate the `copilot` command before using
	`tools/debug/run_copilot.py`.