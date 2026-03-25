import os
import sys
from pathlib import Path

# Ensure repository root is on sys.path so top-level modules (like config)
# can be imported when this module is run directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

import types
from config import get_paid_gemini_api_key as get_paid_gemini_api_key, MINIMAL_MODEL as MINIMAL_MODEL
from api_backoff import call_with_exponential_backoff
from google import genai
from google.genai import types


def run_google_search(query: str) -> str:
    """
    Run a Google Search using the Gemini API's Google Search tool.
    """
    client = genai.Client(api_key=get_paid_gemini_api_key())
    grounding_tool = types.Tool(
        google_search=types.GoogleSearch()
    )
    config = types.GenerateContentConfig(
        tools=[grounding_tool]
    )

    response = call_with_exponential_backoff(
        lambda: client.models.generate_content(
            model=MINIMAL_MODEL,
            contents=query,
            config=config,
        ),
        description="Gemini Google Search",
    )
    
    return str(response.candidates[0]) or ""