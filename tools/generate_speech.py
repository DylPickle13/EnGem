import base64
import json
import re
import sys
import time
import wave
from pathlib import Path

# Ensure repository root is on sys.path so top-level modules (like config)
# can be imported when this module is run directly.
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


_DEFAULT_TTS_MODEL = "gemini-2.5-flash-preview-tts"
_DEFAULT_VOICE_NAME = "Kore"
_DEFAULT_SAMPLE_RATE_HZ = 24000
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


def _normalize_stop_sequences(value: object) -> list[str] | None:
	if not isinstance(value, list):
		return None
	results: list[str] = []
	for item in value:
		text = _normalize_text(item)
		if text:
			results.append(text)
	return results or None


def _build_single_voice_config(voice_name: str) -> types.VoiceConfig:
	resolved_voice_name = _normalize_text(voice_name) or _DEFAULT_VOICE_NAME
	return types.VoiceConfig(
		prebuilt_voice_config=types.PrebuiltVoiceConfig(
			voice_name=resolved_voice_name,
		)
	)


def _build_multi_speaker_voice_config(speakers_payload: object) -> types.MultiSpeakerVoiceConfig | None:
	if not isinstance(speakers_payload, list):
		return None

	speaker_voice_configs: list[types.SpeakerVoiceConfig] = []
	for speaker_payload in speakers_payload[:2]:
		if not isinstance(speaker_payload, dict):
			continue

		speaker_name = _normalize_text(speaker_payload.get("speaker"))
		voice_name = _normalize_text(speaker_payload.get("voice_name"))
		if not speaker_name or not voice_name:
			continue

		speaker_voice_configs.append(
			types.SpeakerVoiceConfig(
				speaker=speaker_name,
				voice_config=_build_single_voice_config(voice_name),
			)
		)

	if not speaker_voice_configs:
		return None

	return types.MultiSpeakerVoiceConfig(speaker_voice_configs=speaker_voice_configs)


def _build_speech_request(raw_prompt: str, fallback_voice_name: str = _DEFAULT_VOICE_NAME) -> dict:
	request_payload = _extract_json_payload(raw_prompt)

	if not request_payload:
		# Backward compatible plain-text mode.
		prompt_text = _normalize_text(raw_prompt)
		model_name = _DEFAULT_TTS_MODEL
		config_kwargs = {
			"response_modalities": ["AUDIO"],
			"speech_config": types.SpeechConfig(
				voice_config=_build_single_voice_config(fallback_voice_name),
			),
		}
		if model_name.lower() in _FLEX_SUPPORTED_MODELS:
			config_kwargs["http_options"] = types.HttpOptions(
				timeout=FLEX_REQUEST_TIMEOUT_MS,
				extra_body={"service_tier": INFERENCE_MODE_FLEX},
			)
		return {
			"model": model_name,
			"contents": prompt_text,
			"prompt_for_filename": prompt_text,
			"sample_rate_hz": _DEFAULT_SAMPLE_RATE_HZ,
			"config": types.GenerateContentConfig(**config_kwargs),
			"request_payload": {},
		}

	prompt_text = request_payload.get("prompt")
	if not isinstance(prompt_text, str):
		prompt_text = ""
	prompt_text = prompt_text.strip()

	model_name = _normalize_text(request_payload.get("model")) or _DEFAULT_TTS_MODEL

	sample_rate_hz = _coerce_int(request_payload.get("sample_rate_hz"))
	if not isinstance(sample_rate_hz, int) or sample_rate_hz <= 0:
		sample_rate_hz = _DEFAULT_SAMPLE_RATE_HZ

	speech_config_kwargs: dict = {}

	language_code = _normalize_text(request_payload.get("language_code"))
	if language_code:
		speech_config_kwargs["language_code"] = language_code

	multi_speaker_voice_config = _build_multi_speaker_voice_config(request_payload.get("speakers"))
	if multi_speaker_voice_config is not None:
		speech_config_kwargs["multi_speaker_voice_config"] = multi_speaker_voice_config
	else:
		resolved_voice_name = _normalize_text(request_payload.get("voice_name")) or _normalize_text(fallback_voice_name) or _DEFAULT_VOICE_NAME
		speech_config_kwargs["voice_config"] = _build_single_voice_config(resolved_voice_name)

	config_kwargs: dict = {
		"response_modalities": ["AUDIO"],
		"speech_config": types.SpeechConfig(**speech_config_kwargs),
	}

	temperature = _coerce_float(request_payload.get("temperature"))
	if isinstance(temperature, float):
		config_kwargs["temperature"] = temperature

	top_p = _coerce_float(request_payload.get("top_p"))
	if isinstance(top_p, float):
		config_kwargs["top_p"] = top_p

	top_k = _coerce_int(request_payload.get("top_k"))
	if isinstance(top_k, int):
		config_kwargs["top_k"] = top_k

	candidate_count = _coerce_int(request_payload.get("candidate_count"))
	if isinstance(candidate_count, int) and candidate_count > 0:
		config_kwargs["candidate_count"] = candidate_count

	max_output_tokens = _coerce_int(request_payload.get("max_output_tokens"))
	if isinstance(max_output_tokens, int) and max_output_tokens > 0:
		config_kwargs["max_output_tokens"] = max_output_tokens

	seed = _coerce_int(request_payload.get("seed"))
	if isinstance(seed, int):
		config_kwargs["seed"] = seed

	stop_sequences = _normalize_stop_sequences(request_payload.get("stop_sequences"))
	if stop_sequences is not None:
		config_kwargs["stop_sequences"] = stop_sequences

	if model_name.lower() in _FLEX_SUPPORTED_MODELS:
		config_kwargs["http_options"] = types.HttpOptions(
			timeout=FLEX_REQUEST_TIMEOUT_MS,
			extra_body={"service_tier": INFERENCE_MODE_FLEX},
		)

	return {
		"model": model_name,
		"contents": prompt_text,
		"prompt_for_filename": prompt_text,
		"sample_rate_hz": sample_rate_hz,
		"config": types.GenerateContentConfig(**config_kwargs),
		"request_payload": request_payload,
	}


def _slugify(text: str) -> str:
	text = text.lower()
	text = re.sub(r"\s+", "_", text)
	text = re.sub(r"[^a-z0-9_\-]", "", text)
	return text.strip("_-")


def _extract_audio_bytes(response: object) -> bytes:
	candidates = getattr(response, "candidates", None) or []
	for candidate in candidates:
		content = getattr(candidate, "content", None)
		parts = getattr(content, "parts", None) or []
		for part in parts:
			inline_data = getattr(part, "inline_data", None)
			data = getattr(inline_data, "data", None) if inline_data else None
			if isinstance(data, (bytes, bytearray)) and data:
				return bytes(data)
			if isinstance(data, str) and data:
				try:
					return base64.b64decode(data)
				except Exception:
					continue

	# Fallback for response shapes that expose parts directly.
	for part in getattr(response, "parts", []) or []:
		inline_data = getattr(part, "inline_data", None)
		data = getattr(inline_data, "data", None) if inline_data else None
		if isinstance(data, (bytes, bytearray)) and data:
			return bytes(data)
		if isinstance(data, str) and data:
			try:
				return base64.b64decode(data)
			except Exception:
				continue

	return b""


def _write_wave_file(output_path: Path, pcm_data: bytes, sample_rate_hz: int = 24000) -> float:
	with wave.open(str(output_path), "wb") as wf:
		wf.setnchannels(1)
		wf.setsampwidth(2)  # 16-bit PCM
		wf.setframerate(sample_rate_hz)
		wf.writeframes(pcm_data)

	sample_count = len(pcm_data) // 2
	return sample_count / float(sample_rate_hz)


def generate_speech(
	prompt: str,
	voice_name: str = "Kore"
) -> str:
	"""
	Convert text in `prompt` to speech audio and save it as a WAV file under
	the repository `generated_files/` folder.

	Backward compatibility:
	- Plain text input works as before and is treated as a single-speaker prompt.

	Advanced mode:
	- Input can be a JSON object encoded as a string with fields such as:
	  prompt, model, voice_name, language_code, speakers (multi-speaker),
	  sample_rate_hz, temperature, top_p, top_k, candidate_count,
	  max_output_tokens, seed, and stop_sequences.

	Returns an error string prefixed with "error:" when generation fails.
	"""
	client = genai.Client(api_key=get_paid_gemini_api_key())
	request = _build_speech_request(prompt, fallback_voice_name=voice_name)
	prompt_for_filename = _normalize_text(request.get("prompt_for_filename"))
	if not prompt_for_filename:
		return "error: prompt is required"

	try:
		response = call_with_exponential_backoff(
			lambda: client.models.generate_content(
				model=request["model"],
				contents=request["contents"],
				config=request["config"],
			),
			description="Gemini speech generation",
		)
	except Exception as exc:
		return f"error: {exc}"

	audio_bytes = _extract_audio_bytes(response)
	if not audio_bytes:
		return "error: no audio data returned by model"

	repo_root = Path(__file__).resolve().parent.parent
	out_dir = repo_root / "generated_files"
	try:
		out_dir.mkdir(parents=True, exist_ok=True)
	except Exception as exc:
		return f"error: failed creating output directory: {exc}"

	safe = _slugify(prompt_for_filename)[:60] or "speech"
	file_name = f"{safe}_{int(time.time())}.wav"
	out_path = out_dir / file_name

	try:
		sample_rate_hz = _coerce_int(request.get("sample_rate_hz")) or _DEFAULT_SAMPLE_RATE_HZ
		duration_sec = _write_wave_file(out_path, audio_bytes, sample_rate_hz=sample_rate_hz)
		out_path_str = str(out_path)
		duration_line = f"DURATION_SEC:{duration_sec:.6f}"

		original_payload = request.get("request_payload") if isinstance(request, dict) else None
		if isinstance(original_payload, dict) and original_payload:
			try:
				json_input = json.dumps(original_payload, ensure_ascii=False)
				return f"{out_path_str}\n{duration_line}\n\nJSON_INPUT:{json_input}"
			except Exception:
				return f"{out_path_str}\n{duration_line}"

		return f"{out_path_str}\n{duration_line}"
	except Exception as exc:
		return f"error: failed writing wav file: {exc}"
