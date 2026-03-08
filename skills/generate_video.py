import os
import time
import sys
import re
from pathlib import Path

# Ensure repository root is on sys.path so top-level modules (like config)
# can be imported when this module is run directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from config import get_paid_gemini_api_key as get_paid_gemini_api_key
from api_backoff import call_with_exponential_backoff
from google import genai
from google.genai import types


def generate_video(prompt: str) -> str:
  """
  Generate a short video from `prompt`, save it under the repository root
  in the `generated_files/` folder, and return the filesystem path to the
  saved video file. Returns an empty string on failure.
  """
  client = genai.Client(api_key=get_paid_gemini_api_key())

  def _get_operation_status(operation_ref: object) -> object:
    try:
      return client.operations.get(operation_ref)
    except Exception:
      return client.operations.get(getattr(operation_ref, "name", operation_ref))

  try:
    operation = call_with_exponential_backoff(
      lambda: client.models.generate_videos(
        model="veo-3.1-generate-preview",
        prompt=prompt,
        config=types.GenerateVideosConfig(aspect_ratio="16:9"),
      ),
      description="Gemini video generation",
    )
  except Exception as exc:
    import traceback
    print("Failed to start video generation:", exc)
    print(traceback.format_exc())
    return f"error: {exc}"

  # Poll until the long-running operation completes.
  # Support either passing the operation object or its name/id to client.operations.get().
  try:
    op_ref = operation
    while not getattr(op_ref, "done", False):
      time.sleep(10)
      op_ref = call_with_exponential_backoff(
        lambda: _get_operation_status(op_ref),
        description="Gemini video status poll",
      )
    operation = op_ref
  except Exception as exc:
    import traceback
    print("Polling failed:", exc)
    print(traceback.format_exc())
    return f"error: {exc}"

  # Extract generated video metadata
  generated_videos = getattr(operation.response or {}, "generated_videos", None) or getattr(operation.response, "generated_videos", None) if operation.response else None
  if not generated_videos:
    # Try alternative access path and dump operation for debugging
    try:
      generated_videos = operation.response.generated_videos
    except Exception:
      print("No generated_videos found on operation.response. Operation:", operation)
      try:
        print("Operation response:", operation.response)
      except Exception:
        pass
      return "error: no generated_videos in operation response"

  if not generated_videos:
    return "error: empty generated_videos"

  generated_video = generated_videos[0]

  # Ensure output directory exists
  repo_root = Path(__file__).resolve().parent.parent
  out_dir = repo_root / "generated_files"
  try:
    out_dir.mkdir(parents=True, exist_ok=True)
  except Exception:
    return ""

  # Download and save the video file
  try:
    call_with_exponential_backoff(
      lambda: client.files.download(file=generated_video.video),
      description="Gemini video download",
    )
  except Exception as exc:
    import traceback
    print("Failed to download video file:", exc)
    print(traceback.format_exc())
    return f"error: {exc}"

  # Create a safe filename based on the prompt
  def _slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_\-]", "", text)
    return text.strip("_-")

  safe = _slugify(prompt)[:60] or "video"
  filename = f"{safe}_{int(time.time())}.mp4"
  out_path = out_dir / filename
  try:
    # generated_video.video is expected to have a .save(path) method
    generated_video.video.save(str(out_path))
    return str(out_path)
  except Exception as exc:
    import traceback
    print("Failed to save video to disk:", exc)
    print(traceback.format_exc())
    return f"error: {exc}"