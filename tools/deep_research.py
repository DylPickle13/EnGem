import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import get_paid_gemini_api_key as get_paid_gemini_api_key
from api_backoff import call_with_exponential_backoff
from google import genai


def deep_research(query: str) -> str:
    """
    Run a background deep research interaction and return the final text result.
    """
    client = genai.Client(api_key=get_paid_gemini_api_key())

    try:
        interaction = call_with_exponential_backoff(
            lambda: client.interactions.create(
                input=query,
                agent="deep-research-preview-04-2026",
                background=True,
            ),
            description="Gemini deep research start",
        )
    except Exception as exc:
        return f"Failed to start deep research: {exc}"

    start_message = f"Research started: {interaction.id}"

    while True:
        try:
            interaction = call_with_exponential_backoff(
                lambda: client.interactions.get(interaction.id),
                description="Gemini deep research poll",
            )
        except Exception as exc:
            return f"{start_message}\nError polling research status: {exc}"

        status = getattr(interaction, "status", "")
        if status == "completed":
            outputs = getattr(interaction, "outputs", None) or []
            if outputs:
                last_output = outputs[-1]
                text = getattr(last_output, "text", None)
                if text:
                    return f"{start_message}\n\n{text}"
            return f"{start_message}\nResearch completed, but no output text was returned."

        if status == "failed":
            error = getattr(interaction, "error", "Unknown error")
            return f"{start_message}\nResearch failed: {error}"

        time.sleep(10)