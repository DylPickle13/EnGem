# EnGem

EnGem — a local assistant and automation framework that connects a streaming LLM (Gemini) to host-side tooling and exposes a Discord bot interface. It coordinates intent classification, a two-phase planning/execution manager flow, staged sub-agent execution, browser automation, notebook and code execution, media generation, attachment ingestion and archiving, and a ChromaDB-backed local memory system. Generated artifacts and user-visible outputs are consolidated under `generated_files/`.

Last updated: 2026-03-15

Table of contents

- Overview
- Quickstart
- Configuration
- Running
- Architecture & key concepts
- Files & folders
- Troubleshooting
- Contributing

Overview

EnGem orchestrates structured multi-step workflows using a Planner → Execution manager pattern. It runs staged sub-agents (serial or parallel), exposes host-side helper functions (in `tools/`) to the LLM via function-calling, and stores conversation and semantic memories locally using ChromaDB.

Quickstart

1. Create and activate a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. (Optional) Install Playwright browsers if you plan to use browser automation:

```bash
python -m playwright install
```

3. Configure environment variables (see `config.py` for defaults):

- `DISCORD_BOT_TOKEN` — Discord bot token (required to run the bot).
- `PAID_GEMINI_API_KEY` — Gemini API key used for model calls and embeddings.
- `DISCORD_ALLOWED_CHANNELS` — Optional comma-separated channel names to limit bot activity.
- `MODEL` — Default model alias (configured in `config.py` with presets like `MINIMAL_MODEL`, `LOW_MODEL`, `MEDIUM_MODEL`, `HIGH_MODEL`).
- Embedding/memory settings: `GEMINI_EMBEDDING_MODEL`, `GEMINI_EMBEDDING_DIM`, `GEMINI_EMBEDDING_BATCH_SIZE`, `MEMORY_SEMANTIC_COLLECTION_NAME`, `MEMORY_FILE_COLLECTION_NAME`, `MEMORY_ARCHIVE_DIR`, `ATTACHMENT_EMBEDDING_MODE` (`multimodal_fallback_text|text_only|off`), `ATTACHMENT_EMBEDDING_MAX_BYTES`, `ATTACHMENT_MULTIMODAL_MAX_BYTES`.
- Optional metrics flag: `CACHE_METRICS_ENABLED=1` to emit cache and embedding metrics to `logs/cache_metrics.jsonl`.

Important: `config.py` contains convenience defaults. Keep secrets out of source control — use environment variables or a secrets manager.

Running

- Start the Discord bot:

```bash
python discord_bot.py
```

- Many helper operations are available under `tools/` and can be run directly (see the script docstrings):

```bash
python tools/run_python.py --script example.py
python tools/run_notebook.py --notebook demo.ipynb
```

Architecture & key concepts

- Two-phase manager flow: Planner → Execution. The Planner produces a structured `planner_plan` JSON describing staged work; after the planner reaches a reviewer-confirmed ready state the Execution manager runs the plan against a frozen history cache.
- Planner and Execution artifacts are written to `sub-agents/` as `planner_order_<id>.json` and `execution_order_<id>.json` (keys: `planner_plan`, `execution_plan`).
- Sub-agents: Each stage has `mode` (`serial` or `parallel`) and a `sub_agents` list. Sub-agents include `task_name`, `instruction`, and `thinking_level` (`MINIMAL|LOW|MEDIUM|HIGH`).
- Tools: Host-side helper functions in `tools/` are exposed for function-calling from the LLM runner implemented in `llm.py` — add new top-level functions in `tools/*.py` to make them available.
- History caching: `history_cache.py` builds cached conversation preludes so the Execution phase can run from a stable cached context.

Files & folders (high level)

- [discord_bot.py](discord_bot.py): Discord integration, message routing, and per-channel history management.
- [llm.py](llm.py): Core orchestration — intent classification, planner/execution managers, sub-agent orchestration, and the `Texter`/`Reviewer` roles.
- [config.py](config.py): Defaults and environment-driven configuration values.
- [memory.py](memory.py) & `memory/`: Local semantic and file memory backed by ChromaDB (`memory/memories_vector_db/`, `memory/skills_vector_db/`) and per-channel histories (`memory/channel_history/`).
- [attachments.py](attachments.py): Attachment ingestion and preprocessing.
- [collect_generated_media.py](collect_generated_media.py): Builds a catalog of generated assets for selection and attachment.
- [progress_indicator.py](progress_indicator.py): ASCII plan previews and live progress for plan execution.
- [history_cache.py](history_cache.py): Utilities for building and resolving cached history preludes.
- [sub-agents/](sub-agents/): Planner and execution JSON artifacts and helpers.
- [agent_instructions/](agent_instructions/): Instruction templates used by planner, execution manager, and reviewers.
- [tools/](tools/): Host utilities and integrations (Google Workspace, YouTube access, browser automation, media generation, notebook and Python runners).
- `generated_files/`: Output artifacts created by tools and sub-agents (images, audio, notebooks, scripts).

Memory & embeddings

- Persistent memory is managed locally via `memory.py` and persisted in the `memory/` DB folders. Embeddings are generated using the configured Gemini embedding model and parameters from the environment.

Attachments & media

- Attachments are processed and segmented by `attachments.py` for inclusion in prompts. Generated assets are collected and indexed by `collect_generated_media.py` and stored in `generated_files/`.

Developer notes

- Add new tools by creating top-level functions in `tools/*.py`.
- Edit or add instruction templates in `agent_instructions/` to influence planner and execution behavior — small changes can have outsized effects, iterate carefully.

Troubleshooting

- Playwright: run `python -m playwright install` and ensure your environment supports headless/GUI operation.
- Discord: verify `DISCORD_BOT_TOKEN` and `DISCORD_ALLOWED_CHANNELS` to ensure the bot can connect and respond.
- LLM/API: check `PAID_GEMINI_API_KEY` and network access; `api_backoff.py` provides retry logic for transient failures.

Contributing

Contributions and improvements are welcome. Please open issues or pull requests with focused changes. When adding new tools or instruction templates, include examples or a short integration note.

License

This repository does not include a license file by default — add one if you intend to make the project public.

Short references

- [agent_instructions/](agent_instructions/)
- [tools/](tools/)
- [discord_bot.py](discord_bot.py)
- [llm.py](llm.py)