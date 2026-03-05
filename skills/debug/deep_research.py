import os
import time
from config import get_paid_gemini_api_key as get_paid_gemini_api_key
from google import genai


def deep_research(query: str) -> str:
    """
    Run a background deep research interaction and return the final text result.
    """
    client = genai.Client(api_key=get_paid_gemini_api_key())

    try:
        interaction = client.interactions.create(
            input=query,
            agent="deep-research-pro-preview-12-2025",
            background=True,
        )
    except Exception as exc:
        return f"Failed to start deep research: {exc}"

    start_message = f"Research started: {interaction.id}"

    while True:
        try:
            interaction = client.interactions.get(interaction.id)
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