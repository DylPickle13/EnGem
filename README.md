# EnGem

EnGem is a local assistant and automation framework that connects a streaming LLM (Gemini) to host-side tooling and exposes a Discord bot interface. It coordinates intent classification, a two-phase planning/execution manager flow, staged sub-agent execution, browser automation, notebook and code execution, media generation, attachment ingestion and archiving, and a ChromaDB-backed memory system. Generated artifacts and user-visible outputs are consolidated under `generated_files/`.

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

3. Set environment variables (see `config.py` for defaults):

- `DISCORD_BOT_TOKEN` — Discord bot token (required to run the bot).
- `PAID_GEMINI_API_KEY` — Paid Gemini API key used for model calls and embeddings.
- `DISCORD_ALLOWED_CHANNELS` — Optional comma-separated channel names to limit bot activity.
- `MODEL` — Default model alias. Several levels (MINIMAL_MODEL, LOW_MODEL, MEDIUM_MODEL, HIGH_MODEL) are defined in `config.py`.
- Embedding/memory settings: `GEMINI_EMBEDDING_MODEL`, `GEMINI_EMBEDDING_DIM`, `GEMINI_EMBEDDING_BATCH_SIZE`, `MEMORY_SEMANTIC_COLLECTION_NAME`, `MEMORY_FILE_COLLECTION_NAME`, `MEMORY_ARCHIVE_DIR`.

Important: `config.py` ships with convenient defaults and placeholder values. Do not commit real secrets — use environment variables or a secrets manager.

Running the Discord bot

From the repository root:

```bash
python discord_bot.py
```

The bot creates per-channel conversation history files in `memory/channel_history/`, responds to messages via [llm.py](llm.py), and can attach generated media from `generated_files/` to replies.

Bot commands

- `>commands` — List available bot commands.
- `>history length` — Show conversation history length.
- `>clear history` — Clear the current channel history file.
- `>clear memory` — Clear semantic and file memory stores (`memory.clear_all_memory_stores`).
- `>forget memories {topic}` — Semantically forget memories related to a topic.
- `>list memories [limit]` — List stored memory records.

Core concepts

- **Two-phase Manager Flow**: EnGem runs a Planner phase followed by an Execution phase.
  - **PlannerManager**: Generates a structured planner artifact (JSON) describing staged sub-agents. Planner output is written to `sub-agents/planner_order_<history>.json` and uses the top-level key `planner_plan`.
  - **Planner Reviewer**: The planner finishes when the planner reviewer agent returns the literal `<ready>` indicating the plan is sufficient.
  - **Post-Planner Cache Freeze**: After the planner returns `<ready>`, EnGem rebuilds the conversation `HistoryContextCache` so the Execution phase runs from a stable cached prelude (avoids sending long mutable histories repeatedly).
  - **ExecutionManager**: Uses the planner's outcome to produce an execution plan (top-level key `execution_plan`) and writes `sub-agents/execution_order_<history>.json`. Execution runs staged sub-agents and must finish with a final serial `Reviewer` that returns the literal `<yes>` to indicate completion.

- **Execution plan shape**: Both planner and execution JSON files are arrays of staged objects. Example shapes:

Planner example

```json
{
  "planner_plan": [
    {"mode": "parallel", "sub_agents": [{"task_name": "Research", "instruction": "...", "thinking_level": "HIGH"}]},
    {"mode": "serial", "sub_agents": [{"task_name": "PlannerReviewer", "instruction": "...", "thinking_level": "LOW"}]}
  ]
}
```

Execution example

```json
{
  "execution_plan": [
    {"mode": "serial", "sub_agents": [{"task_name": "DoSteps", "instruction": "...", "thinking_level": "MEDIUM"}]},
    {"mode": "serial", "sub_agents": [{"task_name": "Reviewer", "instruction": "...", "thinking_level": "LOW"}]}
  ]
}
```

- **Stages**: Each stage has a `mode` of `serial` or `parallel` and a `sub_agents` array. Each sub-agent entry includes `task_name`, `instruction` (plain text system-style instruction), and `thinking_level` (one of `MINIMAL|LOW|MEDIUM|HIGH`).

- **Tools**: Public functions in the `tools/` directory are exposed to sub-agents via the function-calling mechanism implemented in [llm.py](llm.py). Add new tools as top-level functions (no leading underscore) in `tools/*.py` to make them discoverable.

- **History caching**: See [history_cache.py](history_cache.py). Core primitives (e.g., `create_history_context_cache`, `CachedContentProfile`, `resolve_history_cached_prompt`) are used to create cached conversation preludes — EnGem explicitly rebuilds this cache after the Planner phase.

Attachments & media

- **Attachments ingestion**: [attachments.py](attachments.py) ingests images, audio, video and documents, extracts text/metadata, and returns segments for inclusion in prompts.
- **Media catalog / selection**: [collect_generated_media.py](collect_generated_media.py) builds a catalog used by the LLM to select assets for attaching to replies.
- **Generated assets**: Created assets are stored in `generated_files/`.

Memory & embeddings

- **Persistent memory**: Managed by [memory.py](memory.py) and stored locally in ChromaDB under `memory/memories_vector_db/` (semantic file memory) and `memory/skills_vector_db/` (skill records).
- **Embeddings**: Created with the configured Gemini embedding model (`GEMINI_EMBEDDING_MODEL`). Embedding parameters are controlled by environment variables described above.

LLM orchestration & sub-agents

- **Orchestration entrypoint**: [llm.py](llm.py) handles intent classification, the two-phase planner/execution pipeline, staged sub-agent execution (parallel/serial), tool invocation via function-calling, a final `Reviewer` gate, and a `Texter` that composes the final reply.
- **Instruction templates**: Edit `agent_instructions/` templates carefully — they guide the planner and manager behavior. Notable instruction files include `agent_instructions/planner.md`, `agent_instructions/execution_manager.md`, `agent_instructions/planner_reviewer.md`, and `agent_instructions/execution_reviewer.md` (the former `reviewer.md` was renamed to make phase intent explicit).

Progress indicators

- **Preview & progress**: [progress_indicator.py](progress_indicator.py) provides `ExecutionPlanProgressIndicator` which posts an ASCII preview of the active plan and updates a progress message while sub-agents run. It is phase-aware (planner vs execution) and anchors progress to the manager role to produce meaningful previews.

Tools and helpers

- Directory: `tools/` — host-side utilities and integration scripts. Notable helpers:
  - `tools/access_google_workspace.py`
  - `tools/run_google_search.py`
  - `tools/run_python.py` (captures execution output and relocates artifacts to `generated_files/`)
  - `tools/run_notebook.py`
  - `tools/use_browser.py`
  - `tools/generate_image.py`, `tools/generate_video.py`
  - `tools/deep_research.py`

Developer notes & conventions

- **Instruction templates**: Templates in `agent_instructions/` are effectively system messages for managers and sub-agents. Small edits can significantly change planner behavior — iterate carefully and test.
- **Plan file conventions**: Planner output files are `sub-agents/planner_order_<history>.json` (key `planner_plan`) and execution files are `sub-agents/execution_order_<history>.json` (key `execution_plan`). These artifacts are useful for debugging and replay.
- **Adding tools**: To expose a new tool to sub-agents, add a top-level function in `tools/*.py` and ensure its signature is compatible with the function-calling mechanism used by the LLM runner.

Troubleshooting

- Playwright errors: ensure browsers are installed (`python -m playwright install`) and your environment supports headless/GUI runs as needed.
- API failures: check `PAID_GEMINI_API_KEY` and network connectivity. Transient errors are retried via `api_backoff.py`.
- Discord issues: verify `DISCORD_BOT_TOKEN` and `DISCORD_ALLOWED_CHANNELS` settings.

Short file references

- [discord_bot.py](discord_bot.py)
- [llm.py](llm.py)
- [memory.py](memory.py)
- [attachments.py](attachments.py)
- [collect_generated_media.py](collect_generated_media.py)
- [progress_indicator.py](progress_indicator.py)
- [history_cache.py](history_cache.py)
- [agent_instructions/](agent_instructions/)
- [`tools/`](tools/)