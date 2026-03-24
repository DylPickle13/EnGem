import json
import time
import sys
import re
import base64
import mimetypes
import shutil
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


_DEFAULT_VIDEO_MODEL = "veo-3.1-fast-generate-preview"
_DEFAULT_ASPECT_RATIO = "16:9"
_ALLOWED_ASPECT_RATIOS = {"16:9", "9:16"}
_ALLOWED_RESOLUTIONS = {"720p", "1080p", "4k"}
_ALLOWED_DURATIONS = {4, 6, 8}
_ALLOWED_REFERENCE_TYPES = {"asset": "ASSET", "style": "STYLE"}


def _first_present(payload: dict, *keys: str) -> object:
  for key in keys:
    if key in payload:
      return payload.get(key)
  return None


def _extract_generated_videos(operation: object) -> list:
  response = getattr(operation, "response", None)
  if response is None:
    return []

  # Common SDK response shape.
  by_attr = getattr(response, "generated_videos", None)
  if isinstance(by_attr, list):
    return by_attr

  # Dict-like response shape fallback.
  if isinstance(response, dict):
    by_key = response.get("generated_videos")
    if isinstance(by_key, list):
      return by_key
    nested = response.get("response")
    if isinstance(nested, dict):
      nested_videos = nested.get("generated_videos")
      if isinstance(nested_videos, list):
        return nested_videos

  # Protobuf-like payloads can sometimes expose to_dict().
  to_dict = getattr(response, "to_dict", None)
  if callable(to_dict):
    try:
      response_dict = to_dict()
      if isinstance(response_dict, dict):
        by_key = response_dict.get("generated_videos")
        if isinstance(by_key, list):
          return by_key
    except Exception:
      pass

  return []


def _extract_json_payload(raw_prompt: str) -> dict:
  text = (raw_prompt or "").strip()
  if not text:
    return {}

  candidates: list[str] = [text]

  fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.IGNORECASE | re.DOTALL)
  if fenced_match:
    candidates.append(fenced_match.group(1).strip())

  first_brace = text.find("{")
  last_brace = text.rfind("}")
  if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
    candidates.append(text[first_brace:last_brace + 1].strip())

  for candidate in candidates:
    try:
      parsed = json.loads(candidate)
    except Exception:
      continue
    if isinstance(parsed, dict):
      return parsed

  return {}


def _coerce_int(value: object) -> int | None:
  if isinstance(value, bool):
    return None
  if isinstance(value, int):
    return value
  if isinstance(value, float):
    return int(value)
  if isinstance(value, str):
    text = value.strip()
    if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
      try:
        return int(text)
      except Exception:
        return None
  return None


def _coerce_bool(value: object) -> bool | None:
  if isinstance(value, bool):
    return value
  if isinstance(value, str):
    text = value.strip().lower()
    if text in {"true", "1", "yes", "y"}:
      return True
    if text in {"false", "0", "no", "n"}:
      return False
  return None


def _normalize_value(value: object) -> str:
  if value is None:
    return ""
  return str(value).strip()


def _guess_mime_type(path: str, default_mime: str) -> str:
  guessed, _ = mimetypes.guess_type(path)
  return guessed or default_mime


def _image_from_spec(image_spec: object) -> types.Image | None:
  if isinstance(image_spec, str):
    image_spec = {"path": image_spec}

  if not isinstance(image_spec, dict):
    return None

  gcs_uri = _normalize_value(
    _first_present(image_spec, "gcs_uri", "gcsUri", "uri")
  )
  if gcs_uri:
    return types.Image(gcs_uri=gcs_uri)

  image_b64 = _first_present(image_spec, "image_base64", "imageBase64")
  if isinstance(image_b64, str) and image_b64.strip():
    try:
      image_bytes = base64.b64decode(image_b64)
      mime_type = _normalize_value(_first_present(image_spec, "mime_type", "mimeType")) or "image/png"
      return types.Image(image_bytes=image_bytes, mime_type=mime_type)
    except Exception:
      return None

  image_path = _normalize_value(_first_present(image_spec, "path"))
  if not image_path:
    return None

  image_file = Path(image_path).expanduser()
  if not image_file.is_absolute():
    image_file = (_REPO_ROOT / image_file).resolve()
  if not image_file.exists() or not image_file.is_file():
    return None

  try:
    image_bytes = image_file.read_bytes()
  except Exception:
    return None

  mime_type = _normalize_value(_first_present(image_spec, "mime_type", "mimeType")) or _guess_mime_type(str(image_file), "image/png")
  return types.Image(image_bytes=image_bytes, mime_type=mime_type)


def _video_from_spec(video_spec: object) -> types.Video | None:
  if isinstance(video_spec, str):
    video_spec = {"path": video_spec}

  if not isinstance(video_spec, dict):
    return None

  video_uri = _normalize_value(_first_present(video_spec, "uri"))
  if video_uri:
    return types.Video(uri=video_uri)

  # SDK currently rejects encodedVideo for Gemini API video generation in this setup.
  # To avoid INVALID_ARGUMENT errors, only URI-based extension inputs are allowed.
  return None


def _build_video_request(raw_prompt: str) -> dict:
  request_payload = _extract_json_payload(raw_prompt)

  if not request_payload:
    # Backward compatible plain-text mode.
    return {
      "model": _DEFAULT_VIDEO_MODEL,
      "prompt": raw_prompt,
      "image": None,
      "video": None,
      "config": types.GenerateVideosConfig(aspect_ratio=_DEFAULT_ASPECT_RATIO),
    }

  prompt_text = request_payload.get("prompt")
  if not isinstance(prompt_text, str):
    prompt_text = ""
  prompt_text = prompt_text.strip()

  model_name = _normalize_value(request_payload.get("model")) or _DEFAULT_VIDEO_MODEL

  config_kwargs: dict = {}

  aspect_ratio = _normalize_value(_first_present(request_payload, "aspect_ratio", "aspectRatio"))
  if aspect_ratio in _ALLOWED_ASPECT_RATIOS:
    config_kwargs["aspect_ratio"] = aspect_ratio
  elif not aspect_ratio:
    config_kwargs["aspect_ratio"] = _DEFAULT_ASPECT_RATIO

  resolution = _normalize_value(request_payload.get("resolution")).lower()
  if resolution in _ALLOWED_RESOLUTIONS:
    config_kwargs["resolution"] = resolution

  duration_seconds = _coerce_int(_first_present(request_payload, "duration_seconds", "durationSeconds"))
  if duration_seconds in _ALLOWED_DURATIONS:
    config_kwargs["duration_seconds"] = duration_seconds

  number_of_videos = _coerce_int(_first_present(request_payload, "number_of_videos", "numberOfVideos"))
  if number_of_videos == 1:
    config_kwargs["number_of_videos"] = 1

  negative_prompt = _normalize_value(_first_present(request_payload, "negative_prompt", "negativePrompt"))
  if negative_prompt:
    config_kwargs["negative_prompt"] = negative_prompt

  enhance_prompt = _coerce_bool(_first_present(request_payload, "enhance_prompt", "enhancePrompt"))
  if isinstance(enhance_prompt, bool):
    config_kwargs["enhance_prompt"] = enhance_prompt

  last_frame = _image_from_spec(_first_present(request_payload, "last_frame", "lastFrame"))
  if last_frame is not None:
    config_kwargs["last_frame"] = last_frame

  reference_images_raw = _first_present(request_payload, "reference_images", "referenceImages")
  if isinstance(reference_images_raw, list):
    reference_images: list[types.VideoGenerationReferenceImage] = []
    for reference in reference_images_raw[:3]:
      if isinstance(reference, str):
        reference = {"path": reference}
      if not isinstance(reference, dict):
        continue
      image_obj = _image_from_spec(_first_present(reference, "image") or reference)
      if image_obj is None:
        continue
      ref_type = _normalize_value(_first_present(reference, "reference_type", "referenceType")).lower()
      ref_type = _ALLOWED_REFERENCE_TYPES.get(ref_type, "ASSET")
      reference_images.append(
        types.VideoGenerationReferenceImage(
          image=image_obj,
          reference_type=ref_type,
        )
      )
    if reference_images:
      config_kwargs["reference_images"] = reference_images

  first_image = _image_from_spec(
    _first_present(request_payload, "first_image", "firstImage", "image")
  )
  input_video = _video_from_spec(_first_present(request_payload, "video"))

  # Enforce Veo request constraints to avoid invalid parameter combinations.
  if input_video is not None:
    # Extension requests support 720p only.
    config_kwargs["resolution"] = "720p"
    # Image and video inputs are different generation modes; prefer extension input.
    first_image = None

  if "reference_images" in config_kwargs or input_video is not None:
    # Reference-image generation and extension require 8 seconds.
    config_kwargs["duration_seconds"] = 8

  if config_kwargs.get("resolution") in {"1080p", "4k"}:
    # 1080p and 4k are supported for 8-second generations only.
    config_kwargs["duration_seconds"] = 8

  if first_image is None:
    # last_frame is only valid when using an initial image input.
    config_kwargs.pop("last_frame", None)

  return {
    "model": model_name,
    "prompt": prompt_text,
    "image": first_image,
    "video": input_video,
    "config": types.GenerateVideosConfig(**config_kwargs),
    "request_payload": request_payload,
  }


def generate_video(prompt: str) -> str:
  """
  Generate a video and save it under `generated_files/`.

  Backward compatibility:
  - Plain text input works as before and is treated as the prompt.

  Advanced mode:
  - Input can be a JSON object encoded as a string with fields such as:
    prompt, model, aspectRatio/aspect_ratio, resolution,
    durationSeconds/duration_seconds, numberOfVideos/number_of_videos,
    negativePrompt/negative_prompt, enhancePrompt/enhance_prompt,
    firstImage/first_image/image, lastFrame/last_frame,
    referenceImages/reference_images, and video.

  Returns an absolute video path on success, or an `error: ...` message on failure.
  """
  client = genai.Client(api_key=get_paid_gemini_api_key())

  def _get_operation_status(operation_ref: object) -> object:
    try:
      return client.operations.get(operation_ref)
    except Exception:
      return client.operations.get(getattr(operation_ref, "name", operation_ref))

  request = _build_video_request(prompt)
  prompt_text = _normalize_value(request.get("prompt"))
  if not prompt_text:
    return "error: prompt is required"

  try:
    operation = call_with_exponential_backoff(
      lambda: client.models.generate_videos(
        model=request["model"],
        prompt=prompt_text,
        image=request.get("image"),
        video=request.get("video"),
        config=request["config"],
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

  # Extract generated video metadata.
  generated_videos = _extract_generated_videos(operation)
  if not generated_videos:
    print("No generated_videos found on operation response. Operation:", operation)
    try:
      print("Operation response:", operation.response)
    except Exception:
      pass
    return "error: empty generated_videos"

  generated_video = generated_videos[0]
  video_uri = ""
  try:
    video_obj = getattr(generated_video, "video", None)
    if video_obj is not None:
      video_uri = _normalize_value(getattr(video_obj, "uri", None))
    if not video_uri and isinstance(generated_video, dict):
      nested = generated_video.get("video")
      if isinstance(nested, dict):
        video_uri = _normalize_value(nested.get("uri"))
  except Exception:
    video_uri = ""

  # Ensure output directory exists
  repo_root = Path(__file__).resolve().parent.parent
  out_dir = repo_root / "generated_files"
  try:
    out_dir.mkdir(parents=True, exist_ok=True)
  except Exception:
    return ""

  # Download and save the video file
  downloaded_file = None
  try:
    downloaded_file = call_with_exponential_backoff(
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

  safe = _slugify(prompt_text)[:60] or "video"
  filename = f"{safe}_{int(time.time())}.mp4"
  out_path = out_dir / filename
  try:
    # Prefer saving the downloaded file handle if available.
    save_target = downloaded_file if downloaded_file is not None else generated_video.video
    save_fn = getattr(save_target, "save", None)
    if not callable(save_fn):
      save_fn = getattr(generated_video.video, "save", None)
    if not callable(save_fn):
      return "error: generated video object has no save(path) method"

    save_fn(str(out_path))

    # Enforce final location under generated_files even if SDK saved elsewhere.
    if not out_path.exists():
      cwd_candidate = Path.cwd() / filename
      if cwd_candidate.exists() and cwd_candidate.is_file():
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(cwd_candidate), str(out_path))

    if not out_path.exists() or not out_path.is_file():
      return "error: failed to save downloaded video file to generated_files"

    out_path_str = str(out_path.resolve())
    uri_line = f"VIDEO_URI:{video_uri}" if video_uri else ""

    # If the request used a JSON payload, include it in the returned text
    original_payload = request.get("request_payload") if isinstance(request, dict) else None
    if isinstance(original_payload, dict) and original_payload:
      try:
        json_input = json.dumps(original_payload, ensure_ascii=False)
        extras = []
        if uri_line:
          extras.append(uri_line)
        extras.append(f"JSON_INPUT:{json_input}")
        return f"{out_path_str}\n\n" + "\n".join(extras)
      except Exception:
        if uri_line:
          return f"{out_path_str}\n\n{uri_line}"
        return out_path_str

    if uri_line:
      return f"{out_path_str}\n\n{uri_line}"
    return out_path_str
  except Exception as exc:
    import traceback
    print("Failed to save video to disk:", exc)
    print(traceback.format_exc())
    return f"error: {exc}"