# Local LLM — Jarvis

A small local assistant that connects a streaming LLM to a Telegram bot and optional Whisper transcription. It wraps a local LLM loader, a simple toolset for executing and testing Python code blocks, and history tracking for conversational context.

**Key features:**
- **Telegram bot interface:** message and voice handling via `main.py`.
- **Streaming LLM integration:** model loading and generation in `llm.py` (uses `mlx-lm`).
- **Audio transcription:** optional Whisper transcription for voice messages.
- **Execution tools:** `tools.py` can extract and run Python code blocks and append outputs to responses.
- **Conversation history:** `history.md` stores message history used as context by the LLM.

**Important:** this project contains a `credentials.py` file for convenience during local development. Do NOT commit real tokens to version control.

**Quick start**

Prerequisites:
- Python 3.10+ recommended
- Install dependencies:

```bash
pip install -r requirements.txt
```

Configure credentials (one of the following):
- Set environment variable `TELEGRAM_BOT_TOKEN` (recommended), or
- Edit [credentials.py](credentials.py) to add your `TELEGRAM_BOT_TOKEN` and optionally `HF_TOKEN` for Hugging Face access.

Run the bot locally:

```bash
python main.py
```

The bot runs in polling mode and will respond to text and voice messages sent to the configured Telegram bot.

**Files of interest**
- [main.py](main.py) — entrypoint and Telegram handlers.
- [llm.py](llm.py) — loads the model, composes prompts, and runs generation.
- [tools.py](tools.py) — helper utilities: history management, code execution, and a placeholder search.
- [history.md](history.md) — conversational history log used as context.
- [agent_instructions/instructions.md](agent_instructions/instructions.md) — in-repo agent behavior and tool usage guidelines.
- [credentials.py](credentials.py) — local credentials (do not commit secrets).

**Design notes & behavior**
- `llm.generate_response()` composes prompts from `agent_instructions` and `history.md`, runs the model, invokes `tools.search()` and `tools.run_python()` on generated content, and runs a reviewer pass.
- `tools.run_python()` extracts fenced Python blocks from model output and runs them in a temporary subprocess, appending captured stdout/stderr to the response.
- The project uses lazy loading for heavyweight models (LLM + Whisper) to avoid allocating memory at import time.

**Security & privacy**
- Never commit `credentials.py` with real tokens. Prefer environment variables or a secure secrets manager.
- `tools.run_python()` executes code blocks with the local Python interpreter — treat model-generated code as untrusted. Consider sandboxing or disabling code execution in untrusted environments.

**Development & testing**
- To iterate locally, run `python main.py` and send messages to the Telegram bot.
- `tools.py` contains a `__main__` Playwright demo — only run it if you have Playwright installed and configured.