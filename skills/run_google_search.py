import os
import types
from config import GEMINI_API_KEY as GEMINI_API_KEY
from config import model as model
from google import genai
from google.genai import types


def run_google_search(query: str) -> str:
    """Run a Google Search using the Gemini API's Google Search tool."""
    os.environ.setdefault("GEMINI_API_KEY", GEMINI_API_KEY)

    client = genai.Client()
    grounding_tool = types.Tool(
        google_search=types.GoogleSearch()
    )
    config = types.GenerateContentConfig(
        tools=[grounding_tool]
    )

    response = client.models.generate_content(
        model=model,
        contents=query,
        config=config,
    )
    return response.candidates[0].content.parts[0].text or ""