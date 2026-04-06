from pathlib import Path
import sys
import threading
import computer_use

# Ensure repository root is on sys.path so top-level modules (like config)
# can be imported when this module is run directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from config import MINIMAL_MODEL as MINIMAL_MODEL

# Browser Summarizer file located alongside this module
BROWSER_SUMMARIZER_FILE = Path(__file__).parent.parent / "agent_instructions/browser_summarizer.md"

def use_browser(prompt: str, cancellation_event: threading.Event | None = None) -> str:
    """
    Runs actions based on the provided prompt using the computer_use agent.
    Returns the output of the agent's actions as a string.
    """
    if cancellation_event is not None and cancellation_event.is_set():
        return ""

    output = ""
    client = computer_use.create_client()
    _playwright, _browser, page = computer_use.setup_browser()

    try:
        output = computer_use.run_agent_loop(
            client,
            page,
            prompt=prompt,
            cancellation_event=cancellation_event,
        )
    except computer_use.BrowserRunCancelledError:
        output = ""
    except Exception as e:
        output = f"An error occurred: {str(e)}"
    finally:
        try:
            _browser.close()
        except Exception:
            pass
        try:
            _playwright.stop()
        except Exception:
            pass
    return output