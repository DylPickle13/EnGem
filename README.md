# EnGem

EnGem — a local assistant and agent orchestration framework driven by a streaming
LLM (Gemini via `google-genai`) and host-side tooling. EnGem coordinates intent
classification, staged planning and execution (Planner → Execution), sub-agent
orchestration (serial and parallel stages), browser automation, notebook and
Python execution, media generation, attachment ingestion and indexing, and a
ChromaDB-backed local memory system. Generated artifacts and runtime outputs are
organized under `generated_files/`.

Last updated: 2026-04-03

Overview

EnGem is designed to run structured, multi-step workflows driven by an LLM that
can call host utilities (the `tools/` modules) and persist both semantic and
file memories locally. Typical usage is via the bundled Discord bot
integration, but the core orchestration (`llm.py`) can be used programmatically
to integrate the Planner → Execution manager flow into other interfaces.

Highlights

- **Planner → Execution pipeline**: Planner generates staged `planner_plan` JSON
	which the Execution manager then runs deterministically against a cached
	history snapshot.
- **Sub-agents**: Stages contain `sub_agents` with `mode` (`serial|parallel`),
	`thinking_level` (`MINIMAL|LOW|MEDIUM|HIGH`) and optional `force_tool`.
- **Function-calling tools**: Host utilities in `tools/` are exposed to the
	model as callable functions for safe, auditable tool use.
- **Local memory**: ChromaDB-backed semantic and file memory with Gemini
	embeddings, skill extraction and skill DB sync from `skills/` files.
- **Attachments & media**: Attachments are ingested, optionally embedded, and
	archived; generated media is cataloged under `generated_files/` and selectable
	by the model.
- **Browser & ComputerUse**: `computer_use.py` provides a Playwright-backed
	ComputerUse tool for web automation and screenshot-based function responses.
- **Integrations**: Tools include Google Workspace helpers, YouTube API
	support, image/speech/video generation, notebook execution, and more.

Quickstart

1. Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. If you will use browser automation, install Playwright browsers:

```bash
python -m playwright install
```

3. Set required environment variables (see `config.py` for defaults):

- `DISCORD_BOT_TOKEN` — Discord bot token (required to run the bot).
- `PAID_GEMINI_API_KEY` — Gemini API key used for model calls and embeddings.
- `DISCORD_ALLOWED_CHANNELS` — Optional comma-separated channel names to limit
	bot activity.
- `DEFAULT_INFERENCE_MODE` — `standard` or `flex` (see `config.py`).
- `BOT_RUNTIME_STATE_PATH` — Path to persisted runtime state JSON (inference
	mode persists across restarts).
- Embedding/memory settings: `GEMINI_EMBEDDING_MODEL`, `GEMINI_EMBEDDING_DIM`,
	`GEMINI_EMBEDDING_BATCH_SIZE`, `MEMORY_SEMANTIC_COLLECTION_NAME`,
	`MEMORY_FILE_COLLECTION_NAME`, `MEMORY_ARCHIVE_DIR`, `ATTACHMENT_EMBEDDING_MODE` (
	`multimodal_fallback_text|text_only|off`), `ATTACHMENT_EMBEDDING_MAX_BYTES`,
	`ATTACHMENT_MULTIMODAL_MAX_BYTES`.

Notes: keep secret keys out of source control — use environment variables or a
secrets manager.

Running

- Start the Discord bot (recommended for interactive use):

```bash
python discord_bot.py
```

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

- Tools and helper scripts (see docstrings under `tools/`):

```bash
python tools/run_python.py --script example.py
python tools/run_notebook.py --notebook path/to/notebook.ipynb
python tools/access_google_workspace.py ...
python tools/access_youtube.py ...
```

Architecture & key concepts

- **Planner → Execution**: Planner composes a staged plan (written to
	`sub-agents/planner_order_<history>.json`); once the Planner and Planner
	Reviewer indicate readiness, the Execution manager builds an `execution_plan`
	and runs sub-agents against a stable cached history snapshot.
- **Sub-agents**: Each stage contains `sub_agents` (task_name, instruction,
	thinking_level, optional force_tool). Stages can be `parallel` or `serial`.
- **Function-calling**: `llm.py` exposes functions defined in `tools/` as
	callable activities the model can request; `_get_function_declarations()` is
	used to build the function-calling surface.
- **History caching**: `history_cache.py` can pre-warm and reuse Gemini cached
	content for efficient repeated model calls on the same conversation prefix.

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
	the model can use to select media to attach to responses.

Tools & integrations

- Google Workspace CLI integration: `tools/access_google_workspace.py` — planner
	guidance and helpers for invoking the `gws` CLI.
- YouTube API helpers: `tools/access_youtube.py` — OAuth helpers and a planner
	interface for mapping user requests to YouTube Data API actions.
- Media generation: `tools/generate_image.py`, `tools/generate_speech.py`,
	`tools/generate_video.py` (model-driven media generation saved to
	`generated_files/`).
- Notebook and script execution: `tools/run_notebook.py` and
	`tools/run_python.py` move produced artifacts into `generated_files/` for
	downstream selection.
- Browser automation / ComputerUse: `computer_use.py` integrates Playwright and
	implements a ComputerUse tool for agent-driven web interactions and
	screenshots.

Developer notes

- Add new host utilities by creating top-level functions in `tools/*.py`.
	`llm._get_function_declarations()` will import these modules and expose
	functions to the model.
- Edit or add instruction templates in `agent_instructions/` to modify system
	prompts for Planner, Execution manager, Texter, and Reviewers.
- Skill files (.md) placed in `skills/` are imported into the skill DB and
	become reusable planning building blocks; commit changes to `skills/` to
	update the runtime skill set (the bot synchronizes skills at startup).

Troubleshooting

- Playwright: run `python -m playwright install` and verify your environment
	supports the chosen browser mode (headless/headful).
- Discord: confirm `DISCORD_BOT_TOKEN` and `DISCORD_ALLOWED_CHANNELS` are
	correct; check bot logs for connection errors.
- Google Workspace: ensure `gws` CLI is installed and authenticated when
	using `tools/access_google_workspace.py` — follow the CLI's login flow.
- YouTube: tools require OAuth credentials; see `tools/access_youtube.py` for
	guidance on `client_secrets` and credential storage (`youtube_token.json`).

Short references

- [discord_bot.py](discord_bot.py)
- [llm.py](llm.py)
- [memory.py](memory.py)
- [attachments.py](attachments.py)
- [collect_generated_media.py](collect_generated_media.py)
- [progress_indicator.py](progress_indicator.py)
- [history_cache.py](history_cache.py)
- [tools/](tools/)
- [agent_instructions/](agent_instructions/)
- [skills/](skills/)
- [generated_files/](generated_files/)
