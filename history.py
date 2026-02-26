from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

CHANNEL_HISTORY_DIR = Path(__file__).parent / "memory" / "channel_history"
HISTORY_MAX_CHARS = 200_000
TORONTO_TZ = ZoneInfo("America/Toronto")


def _resolve_history_file(history_file: str = "default") -> Path:
    safe_name = (history_file or "default").strip()
    if not safe_name:
        safe_name = "default"
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in safe_name)
    if not safe_name.endswith(".md"):
        safe_name += ".md"
    CHANNEL_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    return CHANNEL_HISTORY_DIR / safe_name


def get_conversation_history(history_file: str = "default") -> str:
    """Get the conversation history as raw text."""
    target_file = _resolve_history_file(history_file)
    try:
        return target_file.read_text(encoding="utf-8")
    except Exception:
        return "No history available."
    

def clear_history(history_file: str = "default") -> None:
    """Clear the conversation history by clearing the contents of the history file."""
    target_file = _resolve_history_file(history_file)
    try:
        target_file.write_text("", encoding="utf-8")
    except Exception:
        return "Failed to clear history file."


def append_history(role: str, text: str, history_file: str = "default") -> None:
    """Append a single message to the history file synchronously.

    Role should be something like 'user' or 'llm'.
    """
    target_file = _resolve_history_file(history_file)
    try:
        ts = datetime.now(TORONTO_TZ).isoformat()
        with target_file.open("a", encoding="utf-8") as f:
            f.write(f"## {ts} - {role}\n\n")
            f.write(text.rstrip() + "\n\n---\n\n")
        _prune_history(history_file=history_file)
    except Exception:
        return "Failed to append to history file."


def _prune_history(history_file: str = "default", max_chars: int = HISTORY_MAX_CHARS) -> None:
    """Trim history.md from the top when it exceeds max_chars.

    Keeps the most recent history content and attempts to align to the next
    message boundary ("\n\n---\n\n") so entries are not cut mid-block.
    """
    target_file = _resolve_history_file(history_file)
    try:
        if max_chars <= 0:
            return

        if not target_file.exists():
            return

        history_text = target_file.read_text(encoding="utf-8")
        if len(history_text) <= max_chars:
            return

        trimmed = history_text[-max_chars:]
        boundary = "\n\n---\n\n"
        boundary_index = trimmed.find(boundary)

        if boundary_index != -1:
            trimmed = trimmed[boundary_index + len(boundary) :]

        target_file.write_text(trimmed.lstrip(), encoding="utf-8")
    except Exception:
        return "Failed to prune history file."