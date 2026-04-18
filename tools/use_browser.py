from pathlib import Path
import sys
import threading

# Ensure repository root is on sys.path so top-level modules (like config)
# can be imported when this module is run directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import computer_use

_TOOL_RUNTIME_CONTEXT = threading.local()


def _set_tool_cancellation_event(cancellation_event: threading.Event | None) -> None:
    _TOOL_RUNTIME_CONTEXT.cancellation_event = cancellation_event


def _clear_tool_cancellation_event() -> None:
    if hasattr(_TOOL_RUNTIME_CONTEXT, "cancellation_event"):
        delattr(_TOOL_RUNTIME_CONTEXT, "cancellation_event")


def _get_tool_cancellation_event() -> threading.Event | None:
    event = getattr(_TOOL_RUNTIME_CONTEXT, "cancellation_event", None)
    return event if isinstance(event, threading.Event) else None


def use_browser(prompt: str) -> str:
    """
    Runs actions based on the provided prompt using the computer_use agent.
    Returns the output of the agent's actions as a string.
    """
    cancellation_event = _get_tool_cancellation_event()

    if cancellation_event is not None and cancellation_event.is_set():
        return ""

    client = computer_use.create_client()
    playwright = None
    browser = None
    page = None

    try:
        playwright, browser, page = computer_use.setup_browser()
        return computer_use.run_agent_loop(
            client,
            page,
            prompt=prompt,
            cancellation_event=cancellation_event,
        )
    except computer_use.BrowserRunCancelledError:
        return ""
    except Exception as exc:
        return f"An error occurred: {exc}"
    finally:
        try:
            computer_use.close_page_handle(page)
        except Exception:
            pass
        try:
            computer_use.close_browser_handle(browser)
        except Exception:
            pass
        try:
            if playwright is not None:
                playwright.stop()
        except Exception:
            pass