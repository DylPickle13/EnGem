import base64
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

from config import get_paid_gemini_api_key as get_paid_gemini_api_key
from api_backoff import call_with_exponential_backoff
from google import genai
from google.genai import types


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


def _write_wave_file(output_path: Path, pcm_data: bytes, sample_rate_hz: int = 24000) -> None:
	with wave.open(str(output_path), "wb") as wf:
		wf.setnchannels(1)
		wf.setsampwidth(2)  # 16-bit PCM
		wf.setframerate(sample_rate_hz)
		wf.writeframes(pcm_data)


def generate_speech(
	prompt: str,
	voice_name: str = "Kore"
) -> str:
	"""
	Convert text in `prompt` to single-speaker speech audio, save it as a WAV file
	under the repository `generated_files/` folder, and return the filesystem path.

	Returns an error string prefixed with "error:" when generation fails.
	"""
	client = genai.Client(api_key=get_paid_gemini_api_key())
	model = "gemini-2.5-flash-preview-tts"

	try:
		response = call_with_exponential_backoff(
			lambda: client.models.generate_content(
				model=model,
				contents=prompt,
				config=types.GenerateContentConfig(
					response_modalities=["AUDIO"],
					speech_config=types.SpeechConfig(
						voice_config=types.VoiceConfig(
							prebuilt_voice_config=types.PrebuiltVoiceConfig(
								voice_name=voice_name,
							)
						)
					),
				),
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

	safe = _slugify(prompt)[:60] or "speech"
	file_name = f"{safe}_{int(time.time())}.wav"
	out_path = out_dir / file_name

	try:
		_write_wave_file(out_path, audio_bytes)
		return str(out_path)
	except Exception as exc:
		return f"error: failed writing wav file: {exc}"
