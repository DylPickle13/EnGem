import os
import time
import re
import sys
from pathlib import Path

# When running this module directly (python skills/generate_image.py), the
# script's directory becomes the first entry in sys.path which prevents
# importing top-level modules like `config.py`. Ensure the repository root
# is on sys.path so `config` can be imported reliably.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import GEMINI_API_KEY as GEMINI_API_KEY
from google import genai
from google.genai import types


def generate_image(prompt: str) -> str:
    """
    Generate an image from `prompt`, save it under the repository root
    in a `generated_images/` folder, and return the filesystem path to
    the saved image. Returns an empty string on failure. 
    Can only create one image at a time. 
    """
    os.environ.setdefault("GEMINI_API_KEY", GEMINI_API_KEY)

    client = genai.Client()

    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-image-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["Text", "Image"],
                image_config=types.ImageConfig(aspect_ratio="16:9"),
                tools=[{"google_search": {}}],
            ),
        )
    except Exception:
        return ""

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "generated_images"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return ""

    for part in getattr(response, "parts", []) or []:
        # Prefer image parts
        try:
            image_obj = part.as_image()
        except Exception:
            image_obj = None

        if image_obj:
            # Create a safe filename based on the prompt
            def _slugify(text: str) -> str:
                text = text.lower()
                text = re.sub(r"\s+", "_", text)
                text = re.sub(r"[^a-z0-9_\-]", "", text)
                return text.strip("_-")

            safe = _slugify(prompt)[:60] or "image"
            filename = f"{safe}_{int(time.time())}.png"
            out_path = out_dir / filename
            try:
                image_obj.save(str(out_path))
                return str(out_path)
            except Exception:
                return ""

    # No image parts found
    return ""