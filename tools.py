import subprocess
import tempfile
import sys
import ast
import re
from pathlib import Path
from typing import List, Dict
from datetime import datetime, timezone
from playwright.sync_api import Page, expect, sync_playwright

# History file path (same directory as this module)
HISTORY_FILE = Path(__file__).parent / "history.md"

def _is_python_syntax(code: str, mode: str = "exec") -> bool:
    try:
        ast.parse(code, mode=mode)
        return True
    except SyntaxError:
        return False


def _run_code(code):
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
        f.write(code.encode())
        filename = f.name

    result = subprocess.run(
        [sys.executable, filename],
        capture_output=True,
        text=True,
        timeout=10
    )
    return result.stdout, result.stderr


def get_conversation_history() -> List[Dict[str, str]]:
    """Read and parse the conversation history file into chat-style messages.

    If `history_file` is not provided, this function will look for a
    `history.md` file in the caller's package directory.
    Returns a list of dicts with `role` and `content` keys suitable for
    passing to chat-formatting utilities.
    """
    messages: List[Dict[str, str]] = []

    try:
        hf = HISTORY_FILE
        if hf.exists():
            raw = hf.read_text(encoding="utf-8")
            parts = [p.strip() for p in raw.split('---') if p.strip()]
            for part in parts:
                lines = [l for l in part.splitlines() if l.strip()]
                if not lines:
                    continue
                header = lines[0]
                m = re.match(r"##\s+[^\s]+\s*-\s*(\w+)", header)
                if m:
                    role_raw = m.group(1).lower()
                else:
                    role_raw = "system"

                content = "\n".join(lines[1:]).strip()
                if not content:
                    continue

                if role_raw == "user":
                    role = "user"
                elif role_raw in ("llm", "assistant", "bot"):
                    role = "assistant"
                else:
                    role = "system"

                messages.append({"role": role, "content": content})
    except Exception:
        messages = []

    return messages


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


def run_python(result_text: str, verbose: bool = False) -> str:
    """Extract Python code blocks from markdown, run them combined, and append output.

    Returns the original text with an appended "Output:" block containing
    the combined stdout/stderr from executing all syntactically valid Python blocks.
    """
    pattern = re.compile(r"```(?:python)?\n(.*?)```", flags=re.DOTALL | re.IGNORECASE)
    matches = list(pattern.finditer(result_text))

    if not matches:
        return ""

    python_blocks = []
    skipped_blocks = []
    for idx, match in enumerate(matches, start=1):
        code_block = match.group(1)
        if _is_python_syntax(code_block):
            python_blocks.append(code_block)
        else:
            skipped_blocks.append(idx)

    if not python_blocks:
        return result_text

    combined_code = "\n\n".join(python_blocks)

    try:
        stdout, stderr = _run_code(combined_code)
        if stderr:
            combined_output = f"{stdout}\n\nCode error:\n{stderr}"
        else:
            combined_output = f"{stdout}"
        if verbose:
            print("Code output:\n" + combined_output)
    except Exception as e:
        combined_output = f"[code_runner error] {e}"

    output_block = f"\n\nOutput:\n{combined_output}"
    return output_block


def search(result_text: str) -> str:
    """Perform a web search for the given query and return a summary of results."""

    pattern = re.compile(r"<search>(.*?)</search>", flags=re.DOTALL | re.IGNORECASE)
    match = pattern.search(result_text)
    if not match:
        return ""
    
    # concatenate all search queries if there are multiple <search> blocks
    queries = [m.group(1).strip() for m in pattern.finditer(result_text)]
    full_query = "\n".join(queries)

    # Placeholder implementation - in a real implementation, this would call an API like Bing Search or Google Custom Search
    return f"\n\nSearch results for: {full_query}"

if __name__ == "__main__":
    with sync_playwright() as p:
        browser = p.webkit.launch()
        page = browser.new_page()
        page.goto("https://playwright.dev/")
        clickable_selector = """
            a,
            button,
            input[type="button"],
            input[type="submit"],
            input[type="reset"],
            [role="button"],
            [onclick],
            [tabindex]
        """

        elements = page.locator(clickable_selector)
        count = elements.count()

        print(f"\nFound {count} clickable elements:\n")

        for i in range(count):
            el = elements.nth(i)

            text = el.inner_text().strip() if el.inner_text() else ""
            tag = el.evaluate("el => el.tagName")
            href = el.get_attribute("href")

            print(f"{i+1}. <{tag}> Text: '{text}' Href: {href}")

        browser.close()