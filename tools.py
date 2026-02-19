import subprocess
import tempfile
import sys
import ast
import shutil
from pathlib import Path
from datetime import datetime, timezone

# History file path located in memory/history.md alongside this module
HISTORY_FILE = Path(__file__).parent / "memory" / "history.md"

def run_python(code: str):
    """Run Python code safely and return stdout and stderr."""
    try:
        ast.parse(code, mode="exec")
    except SyntaxError as e:
        return "", f"SyntaxError: {e}"

    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
        f.write(code.encode())
        filename = f.name

    result = subprocess.run(
        [sys.executable, filename],
        capture_output=True,
        text=True,
        timeout=10
    )
    
    if result.stderr:
        return result.stderr
    return result.stdout


def get_conversation_history() -> str:
    """Get the conversation history as raw text."""
    try:
        return HISTORY_FILE.read_text(encoding="utf-8")
    except Exception:
        return ""


def init_history() -> None:
    """Create or truncate the history file when the bot starts."""
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with HISTORY_FILE.open("w", encoding="utf-8") as f:
            f.write("# Conversation history\n\n")
    except Exception:
        # Don't crash if history file can't be initialized
        pass


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
        # Swallow file-write errors to avoid breaking the flow
        pass

def archive_history() -> None:
    """Archive history by copying into memory/conversations and renaming the copy."""
    try:
        if HISTORY_FILE.exists():
            archive_dir = Path(__file__).parent / "memory" / "conversations"
            archive_dir.mkdir(parents=True, exist_ok=True)

            copied_file = archive_dir / HISTORY_FILE.name
            shutil.copy2(HISTORY_FILE, copied_file)

            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            archive_name = archive_dir / f"history_{ts}.md"
            copied_file.rename(archive_name)
    except Exception:
        # Swallow errors to avoid breaking the flow
        pass