import time
import json
import re
import sys
import mimetypes
import base64
from pathlib import Path

# When running this module directly (python skills/generate_image.py), the
# script's directory becomes the first entry in sys.path which prevents
# importing top-level modules like `config.py`. Ensure the repository root
# is on sys.path so `config` can be imported reliably.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import (
    FLEX_REQUEST_TIMEOUT_MS as FLEX_REQUEST_TIMEOUT_MS,
    INFERENCE_MODE_FLEX as INFERENCE_MODE_FLEX,
    get_paid_gemini_api_key as get_paid_gemini_api_key,
)
from api_backoff import call_with_exponential_backoff
from google import genai
from google.genai import types


_DEFAULT_IMAGE_MODEL = "gemini-3.1-flash-image-preview"
_DEFAULT_IMAGE_ASPECT_RATIO = "16:9"
_DEFAULT_RESPONSE_MODALITIES = ["Text", "Image"]
_ALLOWED_IMAGE_ASPECT_RATIOS = {
    "1:1", "1:4", "1:8", "2:3", "3:2", "3:4", "4:1", "4:3",
    "4:5", "5:4", "8:1", "9:16", "16:9", "21:9",
}
_ALLOWED_IMAGE_SIZES = {"512", "1K", "2K", "4K"}
_ALLOWED_PERSON_GENERATION = {"allow_all", "allow_adult", "dont_allow"}
_ALLOWED_OUTPUT_MIME_TYPES = {"image/png", "image/jpeg", "image/webp"}
_ALLOWED_MODALITIES = {"TEXT", "IMAGE"}
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


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


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


def _coerce_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
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


def _guess_mime_type(path: str, default_mime: str) -> str:
    guessed, _ = mimetypes.guess_type(path)
    return guessed or default_mime


def _part_from_image_spec(image_spec: object) -> types.Part | None:
    if isinstance(image_spec, str):
        image_spec = {"path": image_spec}

    if not isinstance(image_spec, dict):
        return None

    uri = _normalize_text(image_spec.get("uri") or image_spec.get("gcs_uri"))
    if uri:
        mime_type = _normalize_text(image_spec.get("mime_type")) or "image/png"
        try:
            return types.Part.from_uri(file_uri=uri, mime_type=mime_type)
        except Exception:
            return None

    image_b64 = image_spec.get("image_base64")
    if isinstance(image_b64, str) and image_b64.strip():
        try:
            image_bytes = base64.b64decode(image_b64)
        except Exception:
            return None
        mime_type = _normalize_text(image_spec.get("mime_type")) or "image/png"
        try:
            return types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        except Exception:
            return None

    image_path = _normalize_text(image_spec.get("path"))
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

    mime_type = _normalize_text(image_spec.get("mime_type")) or _guess_mime_type(str(image_file), "image/png")
    try:
        return types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    except Exception:
        return None


def _normalize_modalities(raw_modalities: object) -> list[str]:
    if not isinstance(raw_modalities, list):
        return list(_DEFAULT_RESPONSE_MODALITIES)

    normalized: list[str] = []
    for value in raw_modalities:
        item = _normalize_text(value).upper()
        if item in _ALLOWED_MODALITIES and item not in normalized:
            normalized.append(item)

    return normalized or list(_DEFAULT_RESPONSE_MODALITIES)


def _build_image_request(raw_prompt: str) -> dict:
    request_payload = _extract_json_payload(raw_prompt)

    if not request_payload:
        # Backward compatible plain-text mode.
        model_name = _DEFAULT_IMAGE_MODEL
        config_kwargs = {
            "response_modalities": list(_DEFAULT_RESPONSE_MODALITIES),
            "image_config": types.ImageConfig(aspect_ratio=_DEFAULT_IMAGE_ASPECT_RATIO),
            "tools": [{"google_search": {}}],
        }
        if model_name.lower() in _FLEX_SUPPORTED_MODELS:
            config_kwargs["http_options"] = types.HttpOptions(
                timeout=FLEX_REQUEST_TIMEOUT_MS,
                extra_body={"service_tier": INFERENCE_MODE_FLEX},
            )
        return {
            "model": model_name,
            "contents": [raw_prompt],
            "prompt_for_filename": raw_prompt,
            "config": types.GenerateContentConfig(**config_kwargs),
        }

    prompt_text = request_payload.get("prompt")
    if not isinstance(prompt_text, str):
        prompt_text = ""
    prompt_text = prompt_text.strip()

    model_name = _normalize_text(request_payload.get("model")) or _DEFAULT_IMAGE_MODEL

    contents: list = [prompt_text]
    reference_images_raw = request_payload.get("reference_images")
    if isinstance(reference_images_raw, list):
        for image_spec in reference_images_raw[:14]:
            part = _part_from_image_spec(image_spec)
            if part is not None:
                contents.append(part)

    image_config_kwargs: dict = {}

    aspect_ratio = _normalize_text(request_payload.get("aspect_ratio"))
    if aspect_ratio in _ALLOWED_IMAGE_ASPECT_RATIOS:
        image_config_kwargs["aspect_ratio"] = aspect_ratio
    elif not aspect_ratio:
        image_config_kwargs["aspect_ratio"] = _DEFAULT_IMAGE_ASPECT_RATIO

    image_size = _normalize_text(request_payload.get("image_size")).upper()
    if image_size in _ALLOWED_IMAGE_SIZES:
        image_config_kwargs["image_size"] = image_size

    person_generation = _normalize_text(request_payload.get("person_generation")).lower()
    if person_generation in _ALLOWED_PERSON_GENERATION:
        image_config_kwargs["person_generation"] = person_generation

    output_mime_type = _normalize_text(request_payload.get("output_mime_type")).lower()
    if output_mime_type in _ALLOWED_OUTPUT_MIME_TYPES:
        image_config_kwargs["output_mime_type"] = output_mime_type

    output_compression_quality = _coerce_int(request_payload.get("output_compression_quality"))
    if isinstance(output_compression_quality, int) and 0 <= output_compression_quality <= 100:
        image_config_kwargs["output_compression_quality"] = output_compression_quality

    config_kwargs = {
        "response_modalities": _normalize_modalities(request_payload.get("response_modalities")),
        "image_config": types.ImageConfig(**image_config_kwargs),
    }

    temperature = _coerce_float(request_payload.get("temperature"))
    if isinstance(temperature, float):
        config_kwargs["temperature"] = temperature

    seed = _coerce_int(request_payload.get("seed"))
    if isinstance(seed, int):
        config_kwargs["seed"] = seed

    use_google_search = _coerce_bool(request_payload.get("use_google_search"))
    if use_google_search is None:
        use_google_search = True
    if use_google_search:
        config_kwargs["tools"] = [{"google_search": {}}]

    if model_name.lower() in _FLEX_SUPPORTED_MODELS:
        config_kwargs["http_options"] = types.HttpOptions(
            timeout=FLEX_REQUEST_TIMEOUT_MS,
            extra_body={"service_tier": INFERENCE_MODE_FLEX},
        )

    return {
        "model": model_name,
        "contents": contents,
        "prompt_for_filename": prompt_text,
        "config": types.GenerateContentConfig(**config_kwargs),
        "request_payload": request_payload,
    }


def generate_image(prompt: str) -> str:
    """
    Generate an image and save it under `generated_files/`.

    Backward compatibility:
    - Plain text input works as before and is treated as the prompt.

    Advanced mode:
    - Input can be a JSON object encoded as a string with fields such as:
      prompt, model, aspect_ratio, image_size, person_generation,
      output_mime_type, output_compression_quality, response_modalities,
      temperature, seed, use_google_search, and reference_images.

    Returns the absolute saved image path on success, or an empty string on failure.
    """
    client = genai.Client(api_key=get_paid_gemini_api_key())
    request = _build_image_request(prompt)
    prompt_for_filename = _normalize_text(request.get("prompt_for_filename"))
    if not prompt_for_filename:
        return ""

    try:
        response = call_with_exponential_backoff(
            lambda: client.models.generate_content(
                model=request["model"],
                contents=request["contents"],
                config=request["config"],
            ),
            description="Gemini image generation",
        )
    except Exception:
        return ""

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "generated_files"
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

            safe = _slugify(prompt_for_filename)[:60] or "image"
            filename = f"{safe}_{int(time.time())}.png"
            out_path = out_dir / filename
            try:
                image_obj.save(str(out_path))
                out_path_str = str(out_path)
                original_payload = request.get("request_payload") if isinstance(request, dict) else None
                if isinstance(original_payload, dict) and original_payload:
                    try:
                        json_input = json.dumps(original_payload, ensure_ascii=False)
                        return f"{out_path_str}\n\nJSON_INPUT:{json_input}"
                    except Exception:
                        return out_path_str
                return out_path_str
            except Exception:
                return ""

    # No image parts found
    return ""