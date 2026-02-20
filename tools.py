import shutil
from pathlib import Path
from datetime import datetime, timezone

# History file path located in memory/history.md alongside this module
HISTORY_FILE = Path(__file__).parent / "memory" / "history.md"


def get_conversation_history() -> str:
    """Get the conversation history as raw text."""
    try:
        return HISTORY_FILE.read_text(encoding="utf-8")
    except Exception:
        return "No history available."


def init_history() -> None:
    """Create or truncate the history file when the bot starts."""
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with HISTORY_FILE.open("w", encoding="utf-8") as f:
            f.write("")  # Start with an empty file
    except Exception:
        return "Failed to initialize history file."


def append_history(role: str, text: str) -> None:
    """Append a single message to the history file synchronously.

    Role should be something like 'user' or 'llm'.
    """
    try:
        ts = datetime.now(timezone.utc).isoformat()
        with HISTORY_FILE.open("a", encoding="utf-8") as f:
            f.write(f"## {ts} - {role}\n\n")
            f.write(text.rstrip() + "\n\n---\n\n")
    except Exception:
        return "Failed to append to history file."

def archive_history() -> None:
    """Archive history by copying into memory/conversations and renaming the copy."""
    try:
        if HISTORY_FILE.exists():
            if not HISTORY_FILE.read_text(encoding="utf-8").strip():
                return

            archive_dir = Path(__file__).parent / "memory" / "conversations"
            archive_dir.mkdir(parents=True, exist_ok=True)

            copied_file = archive_dir / HISTORY_FILE.name
            shutil.copy2(HISTORY_FILE, copied_file)

            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            archive_name = archive_dir / f"history_{ts}.md"
            copied_file.rename(archive_name)
    except Exception:
        return "Failed to archive history file."