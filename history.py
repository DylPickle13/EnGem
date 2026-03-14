from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import re
import threading
from importlib import import_module
from typing import List, Dict, Optional

from config import MINIMAL_MODEL

CHANNEL_HISTORY_DIR = Path(__file__).parent / "memory" / "channel_history"
HISTORY_SUMMARIZER_SYSTEM = Path(__file__).parent / "agent_instructions" / "history_summarizer.md"
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


def get_history_after_latest_role(history_file: str = "default", role: str = "user") -> str:
    """Return history text that occurred after the latest message for `role`."""
    messages = parse_history_file(history_file)
    if not messages:
        return ""

    latest_role_index = _find_latest_speaker_index(messages, role)
    if latest_role_index < 0 or latest_role_index >= len(messages) - 1:
        return ""

    return "".join(_format_message_block(msg) for msg in messages[latest_role_index + 1 :]).strip()


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


def rewrite_history_with_summary_after_latest_role(
    summary_text: str,
    history_file: str = "default",
    anchor_role: str = "user",
    summary_role: str = "ConversationSummary",
) -> None:
    """Rewrite history so summary replaces content after the latest `anchor_role` message."""
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

        latest_anchor_index = _find_latest_speaker_index(messages, anchor_role)
        if latest_anchor_index < 0 or latest_anchor_index >= len(messages) - 1:
            return

        summary_message: Dict[str, Optional[str]] = {
            "speaker": summary_role,
            "text": cleaned_summary,
            "timestamp": datetime.now(TORONTO_TZ).isoformat(),
        }
        rewritten = messages[: latest_anchor_index + 1] + [summary_message]
        payload = "".join(_format_message_block(msg) for msg in rewritten)
        target_file.write_text(payload, encoding="utf-8")


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


def run_history_summarization(
    history_file: str,
    temperature: float,
    pivot_role: str = "user",
    summarize_after_latest_role: str | None = None,
    history_cache: object | None = None,
    current_history_text: str | None = None,
) -> None:
    try:
        llm = import_module("llm")
        run_model_api = llm._run_model_api

        summarize_after_mode = isinstance(summarize_after_latest_role, str) and bool(summarize_after_latest_role.strip())
        effective_after_role = summarize_after_latest_role.strip() if summarize_after_mode else None

        if history_cache is not None:
            if summarize_after_mode:
                summary_prompt = (
                    f"Summarize only the conversation history after the latest '{effective_after_role}' message. "
                    f"Do not include that latest '{effective_after_role}' message or anything before it."
                )
            else:
                summary_prompt = (
                    f"Summarize only the conversation history before the latest '{pivot_role}' message. "
                    f"Do not include that latest '{pivot_role}' message or anything after it."
                )

            summary = run_model_api(
                text=summary_prompt,
                system_instructions=HISTORY_SUMMARIZER_SYSTEM.read_text(encoding="utf-8"),
                model=MINIMAL_MODEL,
                tool_use_allowed=False,
                force_tool=False,
                temperature=temperature,
                thinking_level="low",
                history_cache=history_cache,
                current_history_text=current_history_text,
            )
        else:
            if summarize_after_mode:
                target_history = get_history_after_latest_role(
                    history_file=history_file,
                    role=effective_after_role,
                )
            else:
                target_history = get_history_before_latest_role(history_file=history_file, role=pivot_role)

            if not target_history:
                return

            summary = run_model_api(
                target_history,
                HISTORY_SUMMARIZER_SYSTEM.read_text(encoding="utf-8"),
                MINIMAL_MODEL,
                tool_use_allowed=False,
                force_tool=False,
                temperature=temperature,
                thinking_level="low",
            )

        cleaned_summary = (summary or "").strip()
        if not cleaned_summary:
            return

        if summarize_after_mode:
            rewrite_history_with_summary_after_latest_role(
                summary_text=cleaned_summary,
                history_file=history_file,
                anchor_role=effective_after_role,
            )
        else:
            rewrite_history_with_summary_before_latest_role(
                summary_text=cleaned_summary,
                history_file=history_file,
                pivot_role=pivot_role,
            )
    except Exception as exc:
        print(f"Error running history summarization: {exc}")
    finally:
        if history_cache is not None:
            history_cache.release()


def run_history_summarization_async(
    history_file: str,
    temperature: float,
    pivot_role: str = "user",
    summarize_after_latest_role: str | None = None,
    history_cache: object | None = None,
    current_history_text: str | None = None,
) -> None:
    def _worker() -> None:
        run_history_summarization(
            history_file=history_file,
            temperature=temperature,
            pivot_role=pivot_role,
            summarize_after_latest_role=summarize_after_latest_role,
            history_cache=history_cache,
            current_history_text=current_history_text,
        )

    threading.Thread(target=_worker, daemon=True).start()