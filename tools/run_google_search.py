import os
import sys
from pathlib import Path

# Ensure repository root is on sys.path so top-level modules (like config)
# can be imported when this module is run directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

import types
from config import (
    FLEX_REQUEST_TIMEOUT_MS as FLEX_REQUEST_TIMEOUT_MS,
    INFERENCE_MODE_FLEX as INFERENCE_MODE_FLEX,
    MINIMAL_MODEL as MINIMAL_MODEL,
    get_paid_gemini_api_key as get_paid_gemini_api_key,
)
from api_backoff import call_with_exponential_backoff
from google import genai
from google.genai import types


_FLEX_SUPPORTED_MODELS = {
    "gemini-3.1-flash-lite-preview",
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "gemini-3-pro-image-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-image",
    "gemini-2.5-flash-lite",
}


def run_google_search(query: str) -> str:
    """
    Run a Google Search using the Gemini API's Google Search tool.
    """
    client = genai.Client(api_key=get_paid_gemini_api_key())
    model_name = str(MINIMAL_MODEL).strip()
    use_flex = model_name.lower() in _FLEX_SUPPORTED_MODELS
    grounding_tool = types.Tool(
        google_search=types.GoogleSearch()
    )
    config_kwargs = {
        "tools": [grounding_tool],
    }
    if use_flex:
        config_kwargs["http_options"] = types.HttpOptions(
            timeout=FLEX_REQUEST_TIMEOUT_MS,
            extra_body={"service_tier": INFERENCE_MODE_FLEX},
        )
    config = types.GenerateContentConfig(**config_kwargs)

    response = call_with_exponential_backoff(
        lambda: client.models.generate_content(
            model=model_name,
            contents=query,
            config=config,
        ),
        description="Gemini Google Search",
    )
    
    return str(response.candidates[0]) or ""