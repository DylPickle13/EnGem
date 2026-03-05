import os
import sys
from pathlib import Path

# Ensure repository root is on sys.path so top-level modules (like config)
# can be imported when this module is run directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

import types
from config import get_paid_gemini_api_key as get_paid_gemini_api_key
from config import model as model
from google import genai
from google.genai import types


def run_google_search(query: str) -> str:
    """
    Run a Google Search using the Gemini API's Google Search tool.
    Only returns summarized search results, not urls or article titles.
    """
    client = genai.Client(api_key=get_paid_gemini_api_key())
    grounding_tool = types.Tool(
        google_search=types.GoogleSearch()
    )
    config = types.GenerateContentConfig(
        tools=[grounding_tool]
    )

    while True:
        try:
            response = client.models.generate_content(
                model=model,
                contents=query,
                config=config,
            )
            break
        except Exception as e:
            print(f"Error running Google Search: {e}")
        print("Retrying Google Search...")
    
    return str(response.candidates[0]) or ""


if __name__ == "__main__":
    query = "What are the latest advancements in AI research as of June 2024?"
    results = run_google_search(query)