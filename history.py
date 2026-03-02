from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import re
from typing import List, Dict, Optional

CHANNEL_HISTORY_DIR = Path(__file__).parent / "memory" / "channel_history"
HISTORY_MAX_CHARS = 100_000
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


def parse_history(history_text: str) -> List[Dict[str, Optional[str]]]:
    """Parse a conversation history string into a list of message parts.

    Tries two common formats:
    - Markdown blocks written by `append_history`:
        ## 2026-03-01T12:00:00 - role\n\nmessage text\n\n---\n\n
    - Simple prefixed lines like `User: ...` and `Assistant: ...` where blocks
      start with `Name:` at the start of a line and may span multiple lines.

    Returns a list of dicts with keys: `speaker`, `text`, and optional
    `timestamp` (only for the markdown format).
    """
    messages: List[Dict[str, Optional[str]]] = []

    if not history_text:
        return messages

    # Try the markdown timestamped blocks first (append_history format)
    # ISO-8601 timestamp (basic validation) to avoid matching truncated headers
    iso_ts = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:\d{2}|Z)?"
    md_pattern = re.compile(
        rf"(?m)^##\s*(?P<timestamp>{iso_ts})\s+-\s+(?P<speaker>[^\n]+)\n\n(?P<text>.*?)(?=(?:\n\n---\n\n)|(?:^##\\s*{iso_ts}\s+-\s+[^\n]+)|\Z)",
        re.DOTALL | re.MULTILINE,
    )

    md_matches = list(md_pattern.finditer(history_text))
    if md_matches:
        for m in md_matches:
            messages.append(
                {
                    "speaker": m.group("speaker").strip(),
                    "text": m.group("text").strip(),
                    "timestamp": m.group("timestamp").strip(),
                }
            )
        return messages

    # Fallback: simple "Name: text" blocks (colon at line-start)
    colon_pattern = re.compile(
        r"(?ms)^(?P<speaker>[A-Za-z0-9 _\-\[\]\(\)]+):\s*(?P<text>.*?)(?=^[A-Za-z0-9 _\-\[\]\(\)]+:\s|\Z)",
        re.MULTILINE,
    )

    for m in colon_pattern.finditer(history_text):
        messages.append(
            {"speaker": m.group("speaker").strip(), "text": m.group("text").strip(), "timestamp": None}
        )

    return messages


def parse_history_file(history_file: str = "default") -> List[Dict[str, Optional[str]]]:
    """Read the history file and parse it into parts using `parse_history`."""
    raw = get_conversation_history(history_file)
    if not raw or raw == "No history available.":
        return []
    return parse_history(raw)


if __name__ == "__main__":    # Example usage
    history_file = "engem-chat3"

    history = parse_history_file(history_file)
    for msg in history:
        print(f"{msg['speaker']}:")