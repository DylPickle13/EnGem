import shutil
from pathlib import Path
from datetime import datetime, timezone

# History file path located in memory/history.md alongside this module
HISTORY_FILE = Path(__file__).parent / "memory" / "history.md"
HISTORY_MAX_CHARS = 100_000


def get_conversation_history() -> str:
    """Get the conversation history as raw text."""
    try:
        return HISTORY_FILE.read_text(encoding="utf-8")
    except Exception:
        return "No history available."
    

def clear_history() -> None:
    """Clear the conversation history by clearing the contents of the history file."""
    try:
        HISTORY_FILE.write_text("", encoding="utf-8")
    except Exception:
        return "Failed to clear history file."


def append_history(role: str, text: str) -> None:
    """Append a single message to the history file synchronously.

    Role should be something like 'user' or 'llm'.
    """
    try:
        ts = datetime.now(timezone.utc).isoformat()
        with HISTORY_FILE.open("a", encoding="utf-8") as f:
            f.write(f"## {ts} - {role}\n\n")
            f.write(text.rstrip() + "\n\n---\n\n")
        _prune_history()
    except Exception:
        return "Failed to append to history file."


def _prune_history(max_chars: int = HISTORY_MAX_CHARS) -> None:
    """Trim history.md from the top when it exceeds max_chars.

    Keeps the most recent history content and attempts to align to the next
    message boundary ("\n\n---\n\n") so entries are not cut mid-block.
    """
    try:
        if max_chars <= 0:
            return

        if not HISTORY_FILE.exists():
            return

        history_text = HISTORY_FILE.read_text(encoding="utf-8")
        if len(history_text) <= max_chars:
            return

        trimmed = history_text[-max_chars:]
        boundary = "\n\n---\n\n"
        boundary_index = trimmed.find(boundary)

        if boundary_index != -1:
            trimmed = trimmed[boundary_index + len(boundary) :]

        HISTORY_FILE.write_text(trimmed.lstrip(), encoding="utf-8")
    except Exception:
        return "Failed to prune history file."