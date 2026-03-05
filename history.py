from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import re
import threading
from typing import List, Dict, Optional

CHANNEL_HISTORY_DIR = Path(__file__).parent / "memory" / "channel_history"
TORONTO_TZ = ZoneInfo("America/Toronto")
_HISTORY_FILE_LOCK = threading.RLock()


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
        with _HISTORY_FILE_LOCK:
            return target_file.read_text(encoding="utf-8")
    except Exception:
        return "No history available."
    

def clear_history(history_file: str = "default") -> None:
    """Clear the conversation history by clearing the contents of the history file."""
    target_file = _resolve_history_file(history_file)
    try:
        with _HISTORY_FILE_LOCK:
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
        with _HISTORY_FILE_LOCK:
            with target_file.open("a", encoding="utf-8") as f:
                f.write(f"## {ts} - {role}\n\n")
                f.write(text.rstrip() + "\n\n---\n\n")
    except Exception:
        return "Failed to append to history file."


def _format_message_block(message: Dict[str, Optional[str]]) -> str:
    timestamp = (message.get("timestamp") or datetime.now(TORONTO_TZ).isoformat()).strip()
    speaker = (message.get("speaker") or "unknown").strip() or "unknown"
    text = (message.get("text") or "").rstrip()
    return f"## {timestamp} - {speaker}\n\n{text}\n\n---\n\n"


def _find_latest_speaker_index(messages: List[Dict[str, Optional[str]]], speaker: str) -> int:
    target_speaker = (speaker or "").strip().lower()
    if not target_speaker:
        return -1

    for idx in range(len(messages) - 1, -1, -1):
        current_speaker = (messages[idx].get("speaker") or "").strip().lower()
        if current_speaker == target_speaker:
            return idx
    return -1


def get_history_before_latest_role(history_file: str = "default", role: str = "user") -> str:
    """Return history text that occurred before the latest message for `role`."""
    messages = parse_history_file(history_file)
    if not messages:
        return ""

    latest_role_index = _find_latest_speaker_index(messages, role)
    if latest_role_index <= 0:
        return ""

    return "".join(_format_message_block(msg) for msg in messages[:latest_role_index]).strip()


def get_history_before_latest_user(history_file: str = "default") -> str:
    """Return history text that occurred before the latest user message."""
    return get_history_before_latest_role(history_file=history_file, role="user")


def get_history_before_latest_manager(history_file: str = "default") -> str:
    """Return history text that occurred before the latest manager response."""
    return get_history_before_latest_role(history_file=history_file, role="manager")


def rewrite_history_with_summary_before_latest_role(
    summary_text: str,
    history_file: str = "default",
    pivot_role: str = "user",
    summary_role: str = "ConversationSummary",
) -> None:
    """Rewrite history so summary is first, then latest `pivot_role` message and following messages."""
    cleaned_summary = (summary_text or "").strip()
    if not cleaned_summary:
        return

    target_file = _resolve_history_file(history_file)
    with _HISTORY_FILE_LOCK:
        try:
            raw_history = target_file.read_text(encoding="utf-8")
        except Exception:
            return

        messages = parse_history(raw_history)
        if not messages:
            return

        latest_pivot_index = _find_latest_speaker_index(messages, pivot_role)
        if latest_pivot_index <= 0:
            return

        summary_message: Dict[str, Optional[str]] = {
            "speaker": summary_role,
            "text": cleaned_summary,
            "timestamp": datetime.now(TORONTO_TZ).isoformat(),
        }
        rewritten = [summary_message] + messages[latest_pivot_index:]
        payload = "".join(_format_message_block(msg) for msg in rewritten)
        target_file.write_text(payload, encoding="utf-8")


def rewrite_history_with_summary_before_latest_user(
    summary_text: str,
    history_file: str = "default",
    summary_role: str = "ConversationSummary",
) -> None:
    """Rewrite history so first message is summary, then latest user request and following messages."""
    rewrite_history_with_summary_before_latest_role(
        summary_text=summary_text,
        history_file=history_file,
        pivot_role="user",
        summary_role=summary_role,
    )


def rewrite_history_with_summary_before_latest_manager(
    summary_text: str,
    history_file: str = "default",
    summary_role: str = "ConversationSummary",
) -> None:
    """Rewrite history so first message is summary, then latest manager response and following messages."""
    rewrite_history_with_summary_before_latest_role(
        summary_text=summary_text,
        history_file=history_file,
        pivot_role="manager",
        summary_role=summary_role,
    )


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