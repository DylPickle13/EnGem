import warnings
# Silence the audioop deprecation warning (Python 3.13 deprecates audioop).
# We only suppress DeprecationWarnings that mention "audioop" so other
# deprecation warnings remain visible.
warnings.filterwarnings("ignore", message=".*audioop.*", category=DeprecationWarning)
import asyncio
import audioop
import array
import datetime
import logging
import queue
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Optional

from config import (
	CRON_JOB_HOUR,
	CRON_JOB_MINUTE,
	DISCORD_BOT_CHANNELS,
	DISCORD_BOT_TOKEN,
    HEARTBEAT_INTERVAL_SECONDS,
)

import discord
from discord.opus import Decoder, OpusError
from google import genai
from google.genai import types
import history
import llm
import memory as memory
from config import GEMINI_API_KEY as GEMINI_API_KEY
from config import VOICE_TOOL_TARGET_CHANNEL_NAME as VOICE_TOOL_TARGET_CHANNEL_NAME

VOICE_INSTRUCTIONS = Path(__file__).parent / "agent_instructions" / "voice_instructions.md"

try:
	from discord.ext import voice_recv
	VOICE_RECV_AVAILABLE = True
except Exception:
	voice_recv = None
	VOICE_RECV_AVAILABLE = False


def _apply_voice_recv_compat_patch() -> None:
	if not VOICE_RECV_AVAILABLE or voice_recv is None:
		return

	voice_client_cls = getattr(voice_recv, "VoiceRecvClient", None)
	if voice_client_cls is None:
		return

	original_remove_ssrc = getattr(voice_client_cls, "_remove_ssrc", None)
	if original_remove_ssrc is None:
		return

	if getattr(original_remove_ssrc, "_engem_guarded", False):
		return

	def _safe_remove_ssrc(self, *, user_id):
		try:
			return original_remove_ssrc(self, user_id=user_id)
		except AttributeError as exc:
			reader = getattr(self, "_reader", None)
			if reader is None or not hasattr(reader, "speaking_timer"):
				logging.warning(
					"voice_recv race detected while dropping SSRC for user_id=%s; skipping unsafe _remove_ssrc call.",
					user_id,
				)
				return None
			raise exc

	setattr(_safe_remove_ssrc, "_engem_guarded", True)
	setattr(voice_client_cls, "_remove_ssrc", _safe_remove_ssrc)


_apply_voice_recv_compat_patch()

CRON_JOBS_DIR = Path(__file__).parent / "agent_instructions/cron_jobs"
HEARTBEAT_JOBS_DIR = Path(__file__).parent / "agent_instructions/heartbeat_jobs"
DISCORD_MESSAGE_LIMIT = 2000
DISCORD_MAX_ATTACHMENTS_PER_MESSAGE = 10
DISCORD_ATTACHMENT_BATCH_MAX_BYTES = 24 * 1024 * 1024
DISCORD_DEFAULT_UPLOAD_LIMIT_BYTES = 8 * 1024 * 1024
EXECUTION_PLAN_PROGRESS_UPDATE_INTERVAL_SECONDS = 1
EXECUTION_PLAN_WAITING_EMOJI = "⏳"
EXECUTION_PLAN_IN_PROGRESS_EMOJI = "🔄"
EXECUTION_PLAN_COMPLETED_EMOJI = "✅"
EXECUTION_PLAN_MESSAGE_HEADER = "Sub-agent execution plan progress"
MESSAGE_WORKER_CONCURRENCY = 3
CHANNEL_HISTORY_DIR = Path(__file__).parent / "memory" / "channel_history"
RELAY_PREFIX = ">"
GEMINI_LIVE_MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
SEND_DISCORD_TEXT_MESSAGE_TOOL = {
	"name": "send_discord_text_message",
	"description": "Send a text message to the Discord channel associated with the current voice conversation. ",
	"parameters": {
		"type": "object",
		"properties": {
			"message": {
				"type": "string",
				"description": "The message content to send.",
			},
		},
		"required": ["message"],
	},
}
GEMINI_LIVE_CONFIG = {
	"response_modalities": ["AUDIO"],
	"enable_affective_dialog": True,
	"system_instruction": VOICE_INSTRUCTIONS.read_text(encoding="utf-8") + str(memory.get_default_store().read_all_memories()),
	"tools": [
		{"google_search": {}},
		{"function_declarations": [SEND_DISCORD_TEXT_MESSAGE_TOOL]},
	],
	"speech_config": types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(
               voice_name='Leda',
            )
        ),
    ),
	"context_window_compression": types.ContextWindowCompressionConfig(
		sliding_window=types.SlidingWindow(),
	),
}
DISCORD_VOICE_SAMPLE_RATE = 48000
GEMINI_INPUT_SAMPLE_RATE = 16000
GEMINI_OUTPUT_SAMPLE_RATE = 24000
DISCORD_FRAME_SIZE_BYTES = 3840
VOICE_OUTPUT_QUEUE_MAX_FRAMES = 1200
VOICE_PLAYBACK_PREBUFFER_FRAMES = 12
VOICE_PLAYBACK_GET_TIMEOUT_SECONDS = 0.03
VOICE_PLAYBACK_MAX_CONCEALMENT_FRAMES = 8
VOICE_RECV_START_DELAY_SECONDS = 0.45
VOICE_RECV_START_RETRIES = 2
MIN_UTTERANCE_BYTES_BEFORE_END = 6400
SPEECH_STOP_DEBOUNCE_SECONDS = 0.65


class _QueueAudioSource(discord.AudioSource):
	def __init__(self, audio_frames: "queue.Queue[bytes]") -> None:
		super().__init__()
		self._audio_frames = audio_frames
		self._closed = False
		self._started = False
		self._min_buffer_frames = VOICE_PLAYBACK_PREBUFFER_FRAMES
		self._last_frame = b"\x00" * DISCORD_FRAME_SIZE_BYTES
		self._consecutive_starves = 0
		self._max_concealment_frames = VOICE_PLAYBACK_MAX_CONCEALMENT_FRAMES
		self._last_left_sample = 0
		self._last_right_sample = 0
		self._apply_declick_next_frame = True
		self._crossfade_pairs = 0
		self._prev_tail: list[int] | None = None

	def _declick(self, frame: bytes) -> bytes:
		if len(frame) < 8:
			return frame

		samples = array.array("h")
		samples.frombytes(frame)
		if len(samples) < 8:
			return frame

		ramp_pairs = min(24, len(samples) // 2)
		for i in range(ramp_pairs):
			alpha_num = i + 1
			alpha_den = ramp_pairs + 1
			target_left = samples[2 * i]
			target_right = samples[2 * i + 1]
			samples[2 * i] = int((self._last_left_sample * (alpha_den - alpha_num) + target_left * alpha_num) / alpha_den)
			samples[2 * i + 1] = int((self._last_right_sample * (alpha_den - alpha_num) + target_right * alpha_num) / alpha_den)

		self._last_left_sample = samples[-2]
		self._last_right_sample = samples[-1]

		if self._prev_tail is not None and len(samples) >= 2:
			cross_pairs = min(self._crossfade_pairs, len(samples) // 2, len(self._prev_tail) // 2)
			for i in range(cross_pairs):
				alpha_num = i + 1
				alpha_den = cross_pairs + 1
				prev_left = self._prev_tail[2 * i]
				prev_right = self._prev_tail[2 * i + 1]
				cur_left = samples[2 * i]
				cur_right = samples[2 * i + 1]
				samples[2 * i] = int((prev_left * (alpha_den - alpha_num) + cur_left * alpha_num) / alpha_den)
				samples[2 * i + 1] = int((prev_right * (alpha_den - alpha_num) + cur_right * alpha_num) / alpha_den)

		tail_pairs = min(self._crossfade_pairs, len(samples) // 2)
		self._prev_tail = list(samples[-2 * tail_pairs:]) if tail_pairs > 0 else None
		return samples.tobytes()

	def read(self) -> bytes:
		if self._closed:
			return b""

		if not self._started:
			if self._audio_frames.qsize() < self._min_buffer_frames:
				return b"\x00" * DISCORD_FRAME_SIZE_BYTES
			self._started = True

		try:
			frame = self._audio_frames.get(timeout=VOICE_PLAYBACK_GET_TIMEOUT_SECONDS)
		except queue.Empty:
			self._consecutive_starves += 1
			self._apply_declick_next_frame = True
			if self._consecutive_starves <= self._max_concealment_frames:
				return self._last_frame
			self._started = False
			return b"\x00" * DISCORD_FRAME_SIZE_BYTES

		if frame == b"":
			self._closed = True
			return b""

		if len(frame) < DISCORD_FRAME_SIZE_BYTES:
			frame = frame + (b"\x00" * (DISCORD_FRAME_SIZE_BYTES - len(frame)))
		elif len(frame) > DISCORD_FRAME_SIZE_BYTES:
			frame = frame[:DISCORD_FRAME_SIZE_BYTES]

		self._last_frame = frame
		self._consecutive_starves = 0
		if self._apply_declick_next_frame:
			self._apply_declick_next_frame = False
			return self._declick(frame)
		return frame

	def cleanup(self) -> None:
		self._closed = True

	def is_opus(self) -> bool:
		return False


class _GeminiVoiceConversation:
	def __init__(self, client: discord.Client, command_prefix: str = ">") -> None:
		self._discord_client = client
		self.command_prefix = command_prefix
		self._genai_client = genai.Client(api_key=GEMINI_API_KEY, http_options={"api_version": "v1alpha"})
		self._task: asyncio.Task[None] | None = None
		self._stop_event = asyncio.Event()
		self._voice_client: object | None = None
		self._live_session: object | None = None
		self._input_queue: asyncio.Queue[dict[str, bytes | str]] = asyncio.Queue(maxsize=64)
		self._output_frames: queue.Queue[bytes] = queue.Queue(maxsize=VOICE_OUTPUT_QUEUE_MAX_FRAMES)
		self._out_buffer = bytearray()
		self._in_rate_state: tuple[float, ...] | None = None
		self._out_rate_state: tuple[float, ...] | None = None
		self._last_activity_end_sent_at = 0.0
		self._bytes_since_activity_end = 0
		self._processed_tool_call_ids: set[str] = set()
		self._last_tool_message_key: tuple[int, str] | None = None
		self._last_tool_message_sent_at = 0.0

	def _queue_discord_frame(self, frame: bytes) -> None:
		if not frame:
			return

		mono = audioop.tomono(frame, 2, 0.5, 0.5)
		converted, self._in_rate_state = audioop.ratecv(
			mono,
			2,
			1,
			DISCORD_VOICE_SAMPLE_RATE,
			GEMINI_INPUT_SAMPLE_RATE,
			self._in_rate_state,
		)

		if not converted:
			return

		self._bytes_since_activity_end += len(converted)

		item: dict[str, bytes | str] = {"data": converted, "mime_type": "audio/pcm;rate=16000"}
		try:
			self._input_queue.put_nowait(item)
		except asyncio.QueueFull:
			try:
				self._input_queue.get_nowait()
			except asyncio.QueueEmpty:
				pass
			try:
				self._input_queue.put_nowait(item)
			except asyncio.QueueFull:
				pass

	def push_discord_audio(self, frame: bytes) -> None:
		loop = self._discord_client.loop
		if loop.is_closed():
			return
		loop.call_soon_threadsafe(self._queue_discord_frame, frame)

	def _push_gemini_audio(self, data: bytes) -> None:
		if not data:
			return

		resampled, self._out_rate_state = audioop.ratecv(
			data,
			2,
			1,
			GEMINI_OUTPUT_SAMPLE_RATE,
			DISCORD_VOICE_SAMPLE_RATE,
			self._out_rate_state,
		)
		stereo = audioop.tostereo(resampled, 2, 1.0, 1.0)
		stereo = audioop.mul(stereo, 2, 0.66)
		self._out_buffer.extend(stereo)

		while len(self._out_buffer) >= DISCORD_FRAME_SIZE_BYTES:
			chunk = bytes(self._out_buffer[:DISCORD_FRAME_SIZE_BYTES])
			del self._out_buffer[:DISCORD_FRAME_SIZE_BYTES]
			try:
				self._output_frames.put_nowait(chunk)
			except queue.Full:
				break

	def _flush_output_tail(self) -> None:
		if not self._out_buffer:
			return

		tail = bytes(self._out_buffer)
		self._out_buffer.clear()

		if len(tail) % 2 == 1:
			tail += b"\x00"

		samples = array.array("h")
		samples.frombytes(tail)
		stereo_pairs = len(samples) // 2
		if stereo_pairs > 0:
			fade_pairs = min(192, stereo_pairs)
			for i in range(fade_pairs):
				pair_index = stereo_pairs - fade_pairs + i
				gain_num = fade_pairs - i
				gain_den = fade_pairs
				samples[2 * pair_index] = int(samples[2 * pair_index] * gain_num / gain_den)
				samples[2 * pair_index + 1] = int(samples[2 * pair_index + 1] * gain_num / gain_den)

		chunk = samples.tobytes()
		if len(chunk) < DISCORD_FRAME_SIZE_BYTES:
			chunk += b"\x00" * (DISCORD_FRAME_SIZE_BYTES - len(chunk))
		elif len(chunk) > DISCORD_FRAME_SIZE_BYTES:
			chunk = chunk[:DISCORD_FRAME_SIZE_BYTES]

		try:
			self._output_frames.put_nowait(chunk)
		except queue.Full:
			pass

	async def _send_audio_loop(self, live_session: object) -> None:
			while not self._stop_event.is_set():
				try:
					audio = await asyncio.wait_for(self._input_queue.get(), timeout=1.0)
				except asyncio.TimeoutError:
					continue

				try:
					await live_session.send_realtime_input(audio=audio)
				except asyncio.CancelledError:
					raise
				except Exception as exc:
					logging.exception("Gemini Live send loop error: %s", exc)
					# Stop the session to avoid an unhandled exception propagating out of TaskGroup
					self._stop_event.set()
					return

	async def _send_audio_stream_end(self) -> None:
		live_session = self._live_session
		if live_session is None:
			return

		try:
			await live_session.send_realtime_input(audio_stream_end=True)
		except Exception as exc:
			logging.exception("Failed sending audio_stream_end to Gemini Live: %s", exc)

	def notify_speech_stopped(self) -> None:
		now = time.monotonic()
		if now - self._last_activity_end_sent_at < SPEECH_STOP_DEBOUNCE_SECONDS:
			return
		if self._bytes_since_activity_end < MIN_UTTERANCE_BYTES_BEFORE_END:
			return
		self._last_activity_end_sent_at = now
		self._bytes_since_activity_end = 0

		loop = self._discord_client.loop
		if loop.is_closed():
			return
		loop.call_soon_threadsafe(lambda: asyncio.create_task(self._send_audio_stream_end()))

	async def _resolve_vc_text_channel(self) -> discord.TextChannel | None:
		target_name = str(VOICE_TOOL_TARGET_CHANNEL_NAME or "").strip()
		if not target_name:
			return None

		for guild in self._discord_client.guilds:
			for channel in guild.text_channels:
				if channel.name == target_name:
					return channel

		target_name_folded = target_name.casefold()
		for guild in self._discord_client.guilds:
			for channel in guild.text_channels:
				if channel.name.casefold() == target_name_folded:
					return channel
		return None

	async def _tool_send_discord_text_message(self, args: object) -> dict[str, object]:
		if not isinstance(args, dict):
			return {"ok": False, "error": "Invalid arguments payload."}

		message_text = str(args.get("message") or "").strip()
		if not message_text:
			return {"ok": False, "error": "message is required."}
		if not message_text.startswith(RELAY_PREFIX):
			message_text = f"{RELAY_PREFIX}{message_text}"

		channel = await self._resolve_vc_text_channel()
		if channel is None:
			return {
				"ok": False,
				"error": f"Text channel not found for channel_name='{VOICE_TOOL_TARGET_CHANNEL_NAME}'"
			}

		now = time.monotonic()
		message_key = (channel.id, message_text)
		if self._last_tool_message_key == message_key and (now - self._last_tool_message_sent_at) < 1.5:
			logging.warning("Suppressed duplicate send_discord_text_message for channel %s", channel.name)
			return {
				"ok": True,
				"channel_id": channel.id,
				"channel_name": channel.name,
				"sent_parts": 0,
				"duplicate_ignored": True,
			}

		sent_parts = 0
		for start in range(0, len(message_text), DISCORD_MESSAGE_LIMIT):
			part = message_text[start : start + DISCORD_MESSAGE_LIMIT]
			if not part:
				continue
			await channel.send(part)
			sent_parts += 1

		self._last_tool_message_key = message_key
		self._last_tool_message_sent_at = now

		return {
			"ok": True,
			"channel_id": channel.id,
			"channel_name": channel.name,
			"sent_parts": sent_parts,
		}

	async def _handle_tool_calls(self, live_session: object, response: object) -> None:
		tool_call = getattr(response, "tool_call", None)
		if tool_call is None:
			return

		function_calls = getattr(tool_call, "function_calls", None) or []
		function_responses: list[types.FunctionResponse] = []
		for function_call in function_calls:
			call_id = getattr(function_call, "id", None)
			name = getattr(function_call, "name", "")
			args = getattr(function_call, "args", {})

			if call_id and call_id in self._processed_tool_call_ids:
				result = {"ok": True, "duplicate_ignored": True, "call_id": call_id}
				if call_id and name:
					function_responses.append(
						types.FunctionResponse(
							id=call_id,
							name=name,
							response=result,
						)
					)
				continue

			if name == "send_discord_text_message":
				try:
					result = await self._tool_send_discord_text_message(args)
					if isinstance(result, dict) and not result.get("ok", False):
						logging.warning("send_discord_text_message returned non-ok result: %s", result)
				except Exception as exc:
					logging.exception("send_discord_text_message tool failed: %s", exc)
					result = {"ok": False, "error": f"Failed to send message: {exc}"}
			else:
				result = {"ok": False, "error": f"Unknown tool: {name}"}

			if call_id:
				self._processed_tool_call_ids.add(call_id)

			if call_id and name:
				function_responses.append(
					types.FunctionResponse(
						id=call_id,
						name=name,
						response=result,
					)
				)

		if function_responses:
			await live_session.send_tool_response(function_responses=function_responses)

	async def _receive_audio_loop(self, live_session: object) -> None:
			while not self._stop_event.is_set():
				try:
					turn = live_session.receive()
					async for response in turn:
						if self._stop_event.is_set():
							return

						await self._handle_tool_calls(live_session, response)

						server_content = getattr(response, "server_content", None)
						model_turn = getattr(server_content, "model_turn", None) if server_content else None
						if not model_turn:
							continue

						for part in model_turn.parts:
							inline_data = getattr(part, "inline_data", None)
							if inline_data and isinstance(inline_data.data, bytes):
								self._push_gemini_audio(inline_data.data)
					self._flush_output_tail()
				except asyncio.CancelledError:
					raise
				except Exception as exc:
					logging.exception("Gemini Live receive loop error: %s", exc)
					# Stop the session gracefully to avoid bubbling the exception out of the TaskGroup
					self._stop_event.set()
					return

	async def _run(self) -> None:
		try:
			async with self._genai_client.aio.live.connect(
				model=GEMINI_LIVE_MODEL,
				config=GEMINI_LIVE_CONFIG,
			) as live_session:
				self._live_session = live_session
				logging.info("Connected to Gemini Live voice session.")
				async with asyncio.TaskGroup() as tg:
					tg.create_task(self._send_audio_loop(live_session))
					tg.create_task(self._receive_audio_loop(live_session))
		except asyncio.CancelledError:
			raise
		except Exception as exc:
			logging.exception("Gemini Live voice session failed: %s", exc)
		finally:
			self._live_session = None

	async def start(self, voice_client: object) -> None:
		if self._task is not None and not self._task.done():
			return

		self._voice_client = voice_client
		self._stop_event.clear()
		self._input_queue = asyncio.Queue(maxsize=64)
		self._output_frames = queue.Queue(maxsize=VOICE_OUTPUT_QUEUE_MAX_FRAMES)
		self._out_buffer = bytearray()
		self._in_rate_state = None
		self._out_rate_state = None
		self._last_activity_end_sent_at = 0.0
		self._bytes_since_activity_end = 0
		self._processed_tool_call_ids.clear()
		self._last_tool_message_key = None
		self._last_tool_message_sent_at = 0.0

		audio_source = _QueueAudioSource(self._output_frames)
		if hasattr(voice_client, "is_playing") and not voice_client.is_playing():
			voice_client.play(audio_source)

		self._task = asyncio.create_task(self._run())

		def _log_task_result(task: asyncio.Task[None]) -> None:
			try:
				task.result()
			except asyncio.CancelledError:
				return
			except Exception as exc:
				logging.exception("Voice conversation task crashed: %s", exc)

		self._task.add_done_callback(_log_task_result)

	async def stop(self) -> None:
		self._stop_event.set()
		if self._task is not None:
			self._task.cancel()
			await asyncio.gather(self._task, return_exceptions=True)
			self._task = None

		try:
			self._output_frames.put_nowait(b"")
		except queue.Full:
			pass

		voice_client = self._voice_client
		if voice_client is not None and hasattr(voice_client, "is_playing") and voice_client.is_playing():
			voice_client.stop()
		self._voice_client = None


if VOICE_RECV_AVAILABLE:
	class _DiscordPCMInputSink(voice_recv.AudioSink):
		def __init__(self, conversation: _GeminiVoiceConversation) -> None:
			super().__init__()
			self._conversation = conversation
			self._closed = False
			self._decoders: dict[int, Decoder] = {}

		def wants_opus(self) -> bool:
			return True

		def write(self, user: object, data: object) -> None:
			if self._closed:
				return

			if getattr(user, "bot", False):
				return

			packet = getattr(data, "packet", None)
			opus = getattr(data, "opus", None)
			if not isinstance(opus, (bytes, bytearray)) or not opus:
				return

			ssrc = getattr(packet, "ssrc", None)
			if not isinstance(ssrc, int):
				return

			decoder = self._decoders.get(ssrc)
			if decoder is None:
				decoder = Decoder()
				self._decoders[ssrc] = decoder

			try:
				pcm = decoder.decode(bytes(opus), fec=False)
			except OpusError:
				return

			if pcm:
				self._conversation.push_discord_audio(pcm)

		def cleanup(self) -> None:
			self._closed = True
			self._decoders.clear()

		@voice_recv.AudioSink.listener()
		def on_voice_member_speaking_stop(self, member: object) -> None:
			if getattr(member, "bot", False):
				return
			self._conversation.notify_speech_stopped()

def _sanitize_history_filename_component(value: str | None) -> str:
	value = (value or "").strip()
	if not value:
		return "unnamed"
	safe = "".join(
		ch if ch.isalnum() or ch in ("-", "_") else "_"
		for ch in value
	).strip("_")
	return safe or "unnamed"

def _get_channel_history_file(channel: discord.TextChannel) -> Path:
	channel_name = _sanitize_history_filename_component(channel.name)
	filename = f"{channel_name}.md"
	return CHANNEL_HISTORY_DIR / filename

def _get_history_file_key_for_channel(channel: object) -> str:
	channel_name = _sanitize_history_filename_component(getattr(channel, "name", None))
	return channel_name or "default"

def ensure_history_files_for_text_channels(channels: Iterable[discord.TextChannel]) -> int:
	channel_list = list(channels)
	if not channel_list:
		return 0

	CHANNEL_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
	for channel in channel_list:
		try:
			_get_channel_history_file(channel).touch(exist_ok=True)
		except OSError as exc:
			logging.exception(
				"Failed to create history file for channel '%s' (%s): %s",
				channel.name,
				channel.id,
				exc,
			)
	return len(channel_list)


class DiscordBotWrapper:
	def __init__(
		self,
		responder: Optional[Callable[[str, discord.Message], Awaitable[llm.LLMResponse | str]]] = None,
		command_prefix: str = ">",
	) -> None:
		self.token = DISCORD_BOT_TOKEN
		self.command_prefix = command_prefix
		self.allowed_channels = {
			channel.strip()
			for channel in str(DISCORD_BOT_CHANNELS).split(",")
			if channel.strip()
		}
		self._message_queue: asyncio.Queue[discord.Message] = asyncio.Queue()
		self._worker_tasks: set[asyncio.Task[None]] = set()
		self._worker_start_lock = asyncio.Lock()
		self._channel_processing_locks: dict[int, asyncio.Lock] = {}
		self._cron_task: asyncio.Task[None] | None = None
		self._cron_stop_event: asyncio.Event | None = None
		self._heartbeat_task: asyncio.Task[None] | None = None
		self._heartbeat_stop_event: asyncio.Event | None = None
		self._execution_plan_progress_tasks: dict[str, asyncio.Task[None]] = {}
		self._execution_plan_progress_messages: dict[str, discord.Message] = {}
		self._voice_conversation: _GeminiVoiceConversation | None = None
		self._voice_input_sink: object | None = None


		intents = discord.Intents.default()
		intents.message_content = True
		intents.voice_states = True

		# Detect whether voice support (PyNaCl) is available and record it.
		try:
			import nacl  # type: ignore
			self.voice_available = True
		except Exception:
			self.voice_available = False

		self.client = discord.Client(intents=intents)
		if self.voice_available and VOICE_RECV_AVAILABLE and GEMINI_API_KEY:
			self._voice_conversation = _GeminiVoiceConversation(self.client, self.command_prefix)
		else:
			self._voice_conversation = None
		self.responder = responder or self._default_responder
		self._register_events()

	async def _default_responder(self, text: str, message: discord.Message) -> llm.LLMResponse:
		history_file = _get_history_file_key_for_channel(message.channel)
		media_payloads = await self._read_media_attachments(message)
		execution_plan_notifier = self._build_execution_plan_notifier(message.channel, history_file)
		return await asyncio.to_thread(
			llm.generate_response,
			text,
			False,
			history_file,
			media_payloads,
			execution_plan_notifier,
		)

	def _build_execution_plan_notifier(
		self,
		channel: discord.abc.Messageable,
		history_file: str,
	) -> Callable[[str, list[dict[str, Any]], int, bool], None]:
		loop = self.client.loop

		def _notifier(
			diagram_text: str,
			execution_plan: list[dict[str, Any]],
			attempt_number: int = 1,
			reset_previous_preview: bool = False,
		) -> None:
			if not diagram_text:
				return
			if loop.is_closed():
				return
			if not execution_plan:
				return

			try:
				future = asyncio.run_coroutine_threadsafe(
					self._start_execution_plan_progress_tracker(
						channel=channel,
						history_file=history_file,
						execution_plan=execution_plan,
						attempt_number=attempt_number,
						reset_previous_preview=reset_previous_preview,
					),
					loop,
				)

				def _log_send_result(done_future: object) -> None:
					try:
						exception = done_future.exception()  # type: ignore[attr-defined]
						if exception is not None:
							logging.exception(
								"Failed sending execution plan preview for history '%s': %s",
								history_file,
								exception,
							)
					except Exception:
						pass

				future.add_done_callback(_log_send_result)
			except Exception as exc:
				logging.exception(
					"Failed scheduling execution plan preview for history '%s': %s",
					history_file,
					exc,
				)

		return _notifier

	async def _start_execution_plan_progress_tracker(
		self,
		channel: discord.abc.Messageable,
		history_file: str,
		execution_plan: list[dict[str, Any]],
		attempt_number: int,
		reset_previous_preview: bool,
	) -> None:
		tracker_key = f"{id(channel)}::{history_file}"
		existing_task = self._execution_plan_progress_tasks.get(tracker_key)
		if existing_task is not None and not existing_task.done():
			existing_task.cancel()
			await asyncio.gather(existing_task, return_exceptions=True)

		if reset_previous_preview:
			previous_message = self._execution_plan_progress_messages.pop(tracker_key, None)
			if previous_message is not None:
				try:
					await previous_message.delete()
				except Exception:
					pass

		task = asyncio.create_task(
			self._run_execution_plan_progress_tracker(
				channel=channel,
				history_file=history_file,
				execution_plan=execution_plan,
				attempt_number=attempt_number,
			)
		)
		self._execution_plan_progress_tasks[tracker_key] = task

		def _cleanup_tracker(done_task: asyncio.Task[None]) -> None:
			current_task = self._execution_plan_progress_tasks.get(tracker_key)
			if current_task is done_task:
				self._execution_plan_progress_tasks.pop(tracker_key, None)

		task.add_done_callback(_cleanup_tracker)

	async def _run_execution_plan_progress_tracker(
		self,
		channel: discord.abc.Messageable,
		history_file: str,
		execution_plan: list[dict[str, Any]],
		attempt_number: int,
	) -> None:
		progress_message: discord.Message | None = None
		last_sent_content: str | None = None
		tracker_key = f"{id(channel)}::{history_file}"

		while True:
			message_content, all_completed = await self._build_execution_plan_progress_message(
				history_file=history_file,
				execution_plan=execution_plan,
				attempt_number=attempt_number,
			)

			try:
				if progress_message is None:
					progress_message = await channel.send(message_content)
					last_sent_content = message_content
					self._execution_plan_progress_messages[tracker_key] = progress_message
				elif message_content != last_sent_content:
					await progress_message.edit(content=message_content)
					last_sent_content = message_content
			except Exception as exc:
				logging.exception(
					"Failed to send/edit execution progress message for history '%s': %s",
					history_file,
					exc,
				)
				return

			if all_completed:
				return

			await asyncio.sleep(EXECUTION_PLAN_PROGRESS_UPDATE_INTERVAL_SECONDS)

	async def _clear_execution_plan_progress_message(
		self,
		channel: discord.abc.Messageable,
		history_file: str,
	) -> None:
		tracker_key = f"{id(channel)}::{history_file}"

		task = self._execution_plan_progress_tasks.get(tracker_key)
		if task is not None and not task.done():
			task.cancel()
			await asyncio.gather(task, return_exceptions=True)

		self._execution_plan_progress_tasks.pop(tracker_key, None)

		progress_message = self._execution_plan_progress_messages.pop(tracker_key, None)
		if progress_message is not None:
			try:
				await progress_message.delete()
			except Exception:
				pass

	async def _build_execution_plan_progress_message(
		self,
		history_file: str,
		execution_plan: list[dict[str, Any]],
		attempt_number: int,
	) -> tuple[str, bool]:
		history_entries = await asyncio.to_thread(history.parse_history_file, history_file)

		latest_manager_index = -1
		for index in range(len(history_entries) - 1, -1, -1):
			role = str(history_entries[index].get("speaker") or "").strip()
			if role.casefold() == "manager":
				latest_manager_index = index
				break

		if latest_manager_index >= 0:
			relevant_entries = history_entries[latest_manager_index + 1 :]
		else:
			relevant_entries = history_entries

		role_counts: dict[str, int] = {}
		for entry in relevant_entries:
			role = str(entry.get("speaker") or "").strip()
			if not role:
				continue
			role_counts[role] = role_counts.get(role, 0) + 1

		flattened_agents: list[dict[str, Any]] = []
		for stage_index, stage in enumerate(execution_plan):
			mode = str(stage.get("mode", "serial"))
			sub_agents = stage.get("sub_agents", []) if isinstance(stage.get("sub_agents"), list) else []
			for agent_index, agent in enumerate(sub_agents):
				if not isinstance(agent, dict):
					continue
				flattened_agents.append(
					{
						"stage_index": stage_index,
						"agent_index": agent_index,
						"mode": mode,
						"task_name": str(agent.get("task_name", "unnamed_task")),
						"instruction": str(agent.get("instruction", "")),
					}
				)

		reviewer_stage_index = len(execution_plan)
		reviewer_agent = {
			"stage_index": reviewer_stage_index,
			"agent_index": 0,
			"mode": "serial",
			"task_name": "Reviewer",
			"instruction": "Validates whether execution can exit with <yes>.",
		}
		flattened_agents.append(reviewer_agent)

		consumed_counts: dict[str, int] = {}
		completed_keys: set[tuple[int, int]] = set()
		for item in flattened_agents:
			task_name = item["task_name"]
			seen = consumed_counts.get(task_name, 0)
			if seen < role_counts.get(task_name, 0):
				completed_keys.add((item["stage_index"], item["agent_index"]))
				consumed_counts[task_name] = seen + 1

		stage_completion: list[bool] = []
		for stage_index, stage in enumerate(execution_plan):
			sub_agents = stage.get("sub_agents", []) if isinstance(stage.get("sub_agents"), list) else []
			stage_keys = {
				(stage_index, agent_index)
				for agent_index, agent in enumerate(sub_agents)
				if isinstance(agent, dict)
			}
			if not stage_keys:
				stage_completion.append(True)
				continue
			stage_completion.append(stage_keys.issubset(completed_keys))

		reviewer_key = (reviewer_stage_index, 0)
		reviewer_completed = reviewer_key in completed_keys
		stage_completion.append(reviewer_completed)

		active_stage_index: int | None = None
		for idx, done in enumerate(stage_completion):
			if not done:
				active_stage_index = idx
				break

		in_progress_keys: set[tuple[int, int]] = set()
		if active_stage_index is not None:
			if active_stage_index < len(execution_plan):
				active_stage = execution_plan[active_stage_index]
				active_mode = str(active_stage.get("mode", "serial"))
				active_sub_agents = active_stage.get("sub_agents", []) if isinstance(active_stage.get("sub_agents"), list) else []
				incomplete_keys = [
					(active_stage_index, agent_index)
					for agent_index, agent in enumerate(active_sub_agents)
					if isinstance(agent, dict) and (active_stage_index, agent_index) not in completed_keys
				]
				if active_mode == "parallel":
					in_progress_keys.update(incomplete_keys)
				elif incomplete_keys:
					in_progress_keys.add(incomplete_keys[0])
			else:
				in_progress_keys.add(reviewer_key)

		lines: list[str] = []
		title_line = f"{EXECUTION_PLAN_MESSAGE_HEADER} ({history_file})"
		attempt_label = max(1, int(attempt_number))
		lines.append(title_line)
		lines.append(f"Attempt: {attempt_label}")
		lines.append("")

		for stage_index, stage in enumerate(execution_plan, start=1):
			mode = str(stage.get("mode", "serial"))
			sub_agents = stage.get("sub_agents", []) if isinstance(stage.get("sub_agents"), list) else []
			lines.append(f"Stage {stage_index} [{mode}]")

			for agent_index, agent in enumerate(sub_agents, start=1):
				if not isinstance(agent, dict):
					continue

				key = (stage_index - 1, agent_index - 1)
				if key in completed_keys:
					emoji = EXECUTION_PLAN_COMPLETED_EMOJI
				elif key in in_progress_keys:
					emoji = EXECUTION_PLAN_IN_PROGRESS_EMOJI
				else:
					emoji = EXECUTION_PLAN_WAITING_EMOJI

				task_name = str(agent.get("task_name", "unnamed_task"))
				instruction = " ".join(str(agent.get("instruction", "")).split())
				if len(instruction) > 200:
					instruction = instruction[:200] + "..."

				lines.append(f"  {emoji} {task_name}")
				if key in in_progress_keys:
					lines.append(f"      instruction: {instruction}")

			lines.append("")

		if reviewer_completed:
			reviewer_emoji = EXECUTION_PLAN_COMPLETED_EMOJI
		elif reviewer_key in in_progress_keys:
			reviewer_emoji = EXECUTION_PLAN_IN_PROGRESS_EMOJI
		else:
			reviewer_emoji = EXECUTION_PLAN_WAITING_EMOJI

		lines.append("Final Stage [serial]")
		lines.append(f"  {reviewer_emoji} Reviewer")
		if reviewer_key in in_progress_keys:
			lines.append("      instruction: Validates whether execution can exit with <yes>.")
		lines.append("")

		text_body = "\n".join(lines).strip()
		wrapped_content = f"```\n{text_body}\n```"
		if len(wrapped_content) > DISCORD_MESSAGE_LIMIT:
			max_body_length = DISCORD_MESSAGE_LIMIT - len("```\n\n```") - 3
			truncated_body = text_body[:max_body_length] + "..."
			wrapped_content = f"```\n{truncated_body}\n```"

		all_completed = bool(stage_completion) and all(stage_completion)
		if not execution_plan:
			all_completed = True

		return wrapped_content, all_completed

	async def _send_long_message(self, channel: discord.abc.Messageable, text: str) -> None:
		if not text:
			return
		for start in range(0, len(text), DISCORD_MESSAGE_LIMIT):
			await channel.send(text[start : start + DISCORD_MESSAGE_LIMIT])

	@staticmethod
	def _normalize_response_payload(response: llm.LLMResponse | str) -> llm.LLMResponse:
		if isinstance(response, llm.LLMResponse):
			return response

		if isinstance(response, str):
			return llm.LLMResponse(text=response, media_paths=[])

		return llm.LLMResponse(text=str(response), media_paths=[])

	async def _send_media_attachments(self, channel: discord.abc.Messageable, media_paths: list[str]) -> None:
		if not media_paths:
			return

		upload_limit_bytes = self._get_channel_upload_limit_bytes(channel)

		valid_paths: list[Path] = []
		skipped_paths: list[str] = []
		oversized_paths: dict[str, int] = {}
		for media_path in media_paths:
			path = Path(media_path)
			if not path.exists() or not path.is_file():
				skipped_paths.append(str(path))
				continue
			try:
				file_size = int(path.stat().st_size)
				if file_size > upload_limit_bytes:
					logging.warning(
						"Skipping media '%s': file size %d exceeds upload limit %d bytes.",
						path,
						file_size,
						upload_limit_bytes,
					)
					oversized_paths[str(path)] = file_size
					skipped_paths.append(str(path))
					continue
				valid_paths.append(path)
			except Exception as exc:
				logging.warning("Failed to attach media '%s': %s", path, exc)
				skipped_paths.append(str(path))

		if not valid_paths:
			if oversized_paths:
				await self._send_oversized_media_warning(channel, oversized_paths, upload_limit_bytes)
			non_oversized_skips = [p for p in skipped_paths if p not in oversized_paths]
			if non_oversized_skips:
				await self._send_long_message(
					channel,
					"Could not attach media files (missing/unreadable files).",
				)
			return

		batches: list[list[Path]] = []
		current_batch: list[Path] = []
		current_batch_bytes = 0

		for path in valid_paths:
			try:
				file_size = int(path.stat().st_size)
			except Exception:
				skipped_paths.append(str(path))
				continue

			would_exceed_count = len(current_batch) >= DISCORD_MAX_ATTACHMENTS_PER_MESSAGE
			would_exceed_bytes = current_batch and (current_batch_bytes + file_size > DISCORD_ATTACHMENT_BATCH_MAX_BYTES)

			if would_exceed_count or would_exceed_bytes:
				batches.append(current_batch)
				current_batch = []
				current_batch_bytes = 0

			current_batch.append(path)
			current_batch_bytes += file_size

		if current_batch:
			batches.append(current_batch)

		for batch in batches:
			try:
				batch_skipped = await self._send_media_batch(channel, batch)
				for skipped in batch_skipped:
					if skipped not in oversized_paths:
						try:
							oversized_paths[skipped] = int(Path(skipped).stat().st_size)
						except Exception:
							oversized_paths[skipped] = 0
				skipped_paths.extend(batch_skipped)
			except Exception as exc:
				logging.exception("Failed sending media batch: %s", exc)
				skipped_paths.extend(str(path) for path in batch)

		if oversized_paths:
			await self._send_oversized_media_warning(channel, oversized_paths, upload_limit_bytes)

		if skipped_paths:
			unique_skipped = list(dict.fromkeys(skipped_paths))
			non_oversized_skips = [path for path in unique_skipped if path not in oversized_paths]
			skipped_count = len(non_oversized_skips)
			if skipped_count == 0:
				return
			if skipped_count == 1:
				await self._send_long_message(channel, f"Skipped 1 media file that Discord would not accept: {non_oversized_skips[0]}")
			else:
				await self._send_long_message(channel, f"Skipped {skipped_count} media files that Discord would not accept.")

	async def _send_oversized_media_warning(
		self,
		channel: discord.abc.Messageable,
		oversized_paths: dict[str, int],
		upload_limit_bytes: int,
	) -> None:
		if not oversized_paths:
			return

		entries = list(dict.fromkeys(oversized_paths.keys()))
		display_entries = entries[:5]
		lines: list[str] = []
		for media_path in display_entries:
			size_bytes = oversized_paths.get(media_path, 0)
			if size_bytes > 0:
				size_text = f"{size_bytes / (1024 * 1024):.1f}MB"
			else:
				size_text = "unknown size"
			lines.append(f"- {Path(media_path).name} ({size_text})")

		remaining = len(entries) - len(display_entries)
		limit_text = f"{upload_limit_bytes / (1024 * 1024):.1f}MB"
		message = "⚠️ Some media files are too large for this Discord channel upload limit "
		message += f"({limit_text}):\n" + "\n".join(lines)
		if remaining > 0:
			message += f"\n...and {remaining} more file(s)."

		await self._send_long_message(channel, message)

	@staticmethod
	def _get_channel_upload_limit_bytes(channel: discord.abc.Messageable) -> int:
		guild = getattr(channel, "guild", None)
		limit = getattr(guild, "filesize_limit", None)
		if isinstance(limit, int) and limit > 0:
			return limit
		return DISCORD_DEFAULT_UPLOAD_LIMIT_BYTES

	async def _send_media_batch(self, channel: discord.abc.Messageable, batch_paths: list[Path]) -> list[str]:
		if not batch_paths:
			return []

		files: list[discord.File] = []
		try:
			for path in batch_paths:
				files.append(discord.File(str(path), filename=path.name))
			await channel.send(files=files)
			return []
		except discord.HTTPException as exc:
			if getattr(exc, "code", None) == 40005:
				if len(batch_paths) > 1:
					skipped: list[str] = []
					for path in batch_paths:
						skipped.extend(await self._send_media_batch(channel, [path]))
					return skipped

				single_path = batch_paths[0]
				logging.warning("Skipping media '%s': Discord rejected the file as too large.", single_path)
				return [str(single_path)]
			raise
		finally:
			for file_obj in files:
				try:
					file_obj.close()
				except Exception:
					pass

	@staticmethod
	def _build_prompt(content: str, attachment_text: str) -> str:
		if not content:
			return attachment_text
		return f"{content}\n\n{attachment_text}".strip()

	async def _respond_and_send(self, channel: discord.abc.Messageable, prompt: str, message: discord.Message) -> None:
		async with channel.typing():
			response = await self.responder(prompt, message)

		payload = self._normalize_response_payload(response)
		await self._send_long_message(channel, payload.text)
		history_file = _get_history_file_key_for_channel(message.channel)
		await self._clear_execution_plan_progress_message(channel, history_file)
		await self._send_media_attachments(channel, payload.media_paths)

	async def _shutdown_background_tasks(self) -> None:
		worker_tasks = [task for task in self._worker_tasks if not task.done()]
		execution_plan_tasks = [task for task in self._execution_plan_progress_tasks.values() if not task.done()]
		tasks = [
			task
			for task in (self._cron_task, self._heartbeat_task, *worker_tasks, *execution_plan_tasks)
			if task is not None and not task.done()
		]
		for task in tasks:
			task.cancel()

		if tasks:
			await asyncio.gather(*tasks, return_exceptions=True)

		if self._voice_conversation is not None:
			await self._voice_conversation.stop()

		self._cron_task = None
		self._heartbeat_task = None
		self._worker_tasks.clear()
		self._execution_plan_progress_tasks.clear()
		self._execution_plan_progress_messages.clear()
		self._voice_input_sink = None

	async def _read_text_attachment(self, attachment: discord.Attachment) -> str:
		data = await attachment.read()
		try:
			return data.decode("utf-8").strip()
		except UnicodeDecodeError:
			return data.decode("utf-8", errors="replace").strip()

	@staticmethod
	def _is_image_attachment(attachment: discord.Attachment) -> bool:
		content_type = (attachment.content_type or "").lower()
		filename = (attachment.filename or "").lower()
		return content_type.startswith("image/") or filename.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"))

	@staticmethod
	def _is_video_attachment(attachment: discord.Attachment) -> bool:
		content_type = (attachment.content_type or "").lower()
		filename = (attachment.filename or "").lower()
		return content_type.startswith("video/") or filename.endswith((".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"))

	def _is_media_attachment(self, attachment: discord.Attachment) -> bool:
		return self._is_image_attachment(attachment) or self._is_video_attachment(attachment)

	async def _read_media_attachments(self, message: discord.Message) -> list[dict[str, bytes | str]]:
		media_payloads: list[dict[str, bytes | str]] = []
		for attachment in message.attachments:
			if len(media_payloads) >= DISCORD_MAX_ATTACHMENTS_PER_MESSAGE:
				break
			if not self._is_media_attachment(attachment):
				continue

			try:
				attachment_bytes = await attachment.read()
			except Exception as exc:
				logging.warning("Failed reading attachment '%s': %s", attachment.filename, exc)
				continue

			if not attachment_bytes:
				continue

			default_name = "attachment"
			if self._is_video_attachment(attachment):
				default_name = "video"
			elif self._is_image_attachment(attachment):
				default_name = "image"

			media_payloads.append(
				{
					"data": attachment_bytes,
					"mime_type": attachment.content_type or "application/octet-stream",
					"filename": attachment.filename or default_name,
				}
			)

		return media_payloads

	@staticmethod
	def _seconds_until_next_daily_run(target_hour: int, target_minute: int) -> float:
		now = datetime.datetime.now()
		next_run = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
		if now >= next_run:
			next_run += datetime.timedelta(days=1)
		return (next_run - now).total_seconds()

	def _find_channel_by_name(self, channel_name: str) -> Optional[discord.abc.Messageable]:
		for guild in self.client.guilds:
			for channel in guild.text_channels:
				if channel.name == channel_name:
					return channel
		return None

	@staticmethod
	def _get_task_channel_name(task_file: Path) -> str:
		return task_file.stem.replace("_", "-")

	def _load_cron_tasks(self) -> list[tuple[str, str, str]]:
		tasks: list[tuple[str, str, str]] = []

		for task_file in sorted(CRON_JOBS_DIR.glob("*.md")):
			try:
				task_prompt = task_file.read_text(encoding="utf-8").strip()
			except Exception as exc:
				logging.exception("Error reading scheduled task file '%s': %s", task_file, exc)
				continue

			if task_prompt:
				tasks.append((task_file.name, self._get_task_channel_name(task_file), task_prompt))

		return tasks

	def _load_heartbeat_tasks(self) -> list[tuple[str, str, str]]:
		tasks: list[tuple[str, str, str]] = []

		for task_file in sorted(HEARTBEAT_JOBS_DIR.glob("*.md")):
			try:
				task_prompt = task_file.read_text(encoding="utf-8").strip()
			except Exception as exc:
				logging.exception("Error reading heartbeat task file '%s': %s", task_file, exc)
				continue

			if task_prompt:
				tasks.append((task_file.name, self._get_task_channel_name(task_file), task_prompt))

		return tasks

	def _list_cron_task_names(self) -> list[str]:
		return [f"{task_name} -> #{channel_name}" for task_name, channel_name, _ in self._load_cron_tasks()]

	def _list_heartbeat_task_names(self) -> list[str]:
		return [f"{task_name} -> #{channel_name}" for task_name, channel_name, _ in self._load_heartbeat_tasks()]

	def _start_cron_task_if_needed(self, *, run_immediately: bool = False) -> bool:
		if self._cron_task is not None and not self._cron_task.done():
			return False

		self._cron_stop_event = asyncio.Event()
		self._cron_task = asyncio.create_task(self._run_cron_scheduler(run_immediately=run_immediately))
		return True

	def _start_heartbeat_task_if_needed(self, *, run_immediately: bool = False) -> bool:
		if self._heartbeat_task is not None and not self._heartbeat_task.done():
			return False

		self._heartbeat_stop_event = asyncio.Event()
		self._heartbeat_task = asyncio.create_task(self._run_heartbeat_scheduler(run_immediately=run_immediately))
		return True

	async def _stop_cron_task_if_running(self) -> bool:
		if self._cron_task is None or self._cron_task.done():
			self._cron_task = None
			self._cron_stop_event = None
			return False

		if self._cron_stop_event is not None:
			self._cron_stop_event.set()

		self._cron_task.cancel()
		await asyncio.gather(self._cron_task, return_exceptions=True)
		self._cron_task = None
		self._cron_stop_event = None
		return True

	async def _stop_heartbeat_task_if_running(self) -> bool:
		if self._heartbeat_task is None or self._heartbeat_task.done():
			self._heartbeat_task = None
			self._heartbeat_stop_event = None
			return False

		if self._heartbeat_stop_event is not None:
			self._heartbeat_stop_event.set()

		self._heartbeat_task.cancel()
		await asyncio.gather(self._heartbeat_task, return_exceptions=True)
		self._heartbeat_task = None
		self._heartbeat_stop_event = None
		return True

	async def _run_cron_scheduler(self, run_immediately: bool = False) -> None:
		async def send_cron_jobs() -> None:
			tasks = self._load_cron_tasks()
			if not tasks:
				logging.warning("No cron job tasks found in '%s'.", CRON_JOBS_DIR)
				return

			for task_name, channel_name, task_prompt in tasks:
				channel = self._find_channel_by_name(channel_name)
				if channel is None:
					logging.warning(
						"Channel '%s' not found for scheduled task '%s'; skipping.",
						channel_name,
						task_name,
					)
					continue

				try:
					channel_lock = self._get_channel_processing_lock(channel)
					async with channel_lock:
						async with channel.typing():
							execution_plan_notifier = self._build_execution_plan_notifier(channel, channel_name)
							response = await asyncio.to_thread(
								llm.generate_response,
								task_prompt,
								True,
								channel_name,
								None,
								execution_plan_notifier,
							)
					payload = self._normalize_response_payload(response)
					await self._send_long_message(channel, payload.text)
					await self._clear_execution_plan_progress_message(channel, channel_name)
					await self._send_media_attachments(channel, payload.media_paths)
				except Exception as exc:
					logging.exception("Error running cron job task '%s': %s", task_name, exc)

		if run_immediately:
			await send_cron_jobs()

		while True:
			try:
				seconds_until_next_run = self._seconds_until_next_daily_run(CRON_JOB_HOUR, CRON_JOB_MINUTE)
				if self._cron_stop_event is not None:
					await asyncio.wait_for(
						self._cron_stop_event.wait(),
						timeout=seconds_until_next_run,
					)
					break
				await asyncio.sleep(seconds_until_next_run)
			except asyncio.TimeoutError:
				pass
			except asyncio.CancelledError:
				break

			await send_cron_jobs()

	async def _run_heartbeat_scheduler(self, run_immediately: bool = False) -> None:
		async def send_heartbeat_jobs() -> None:
			tasks = self._load_heartbeat_tasks()
			if not tasks:
				logging.warning("No heartbeat tasks found in '%s'.", HEARTBEAT_JOBS_DIR)
				return

			for task_name, channel_name, task_prompt in tasks:
				channel = self._find_channel_by_name(channel_name)
				if channel is None:
					logging.warning(
						"Channel '%s' not found for heartbeat task '%s'; skipping.",
						channel_name,
						task_name,
					)
					continue

				try:
					channel_lock = self._get_channel_processing_lock(channel)
					async with channel_lock:
						async with channel.typing():
							execution_plan_notifier = self._build_execution_plan_notifier(channel, channel_name)
							response = await asyncio.to_thread(
								llm.generate_response,
								task_prompt,
								True,
								channel_name,
								None,
								execution_plan_notifier,
							)
					payload = self._normalize_response_payload(response)
					await self._send_long_message(channel, payload.text)
					await self._clear_execution_plan_progress_message(channel, channel_name)
					await self._send_media_attachments(channel, payload.media_paths)
				except Exception as exc:
					logging.exception("Error running heartbeat task '%s': %s", task_name, exc)

		if run_immediately:
			await send_heartbeat_jobs()

		while True:
			try:
				if self._heartbeat_stop_event is not None:
					await asyncio.wait_for(
						self._heartbeat_stop_event.wait(),
						timeout=HEARTBEAT_INTERVAL_SECONDS,
					)
					break
				await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
			except asyncio.TimeoutError:
				pass
			except asyncio.CancelledError:
				break

			await send_heartbeat_jobs()

	async def _ensure_worker_pool(self) -> None:
		async with self._worker_start_lock:
			self._worker_tasks = {task for task in self._worker_tasks if not task.done()}

			workers_to_start = MESSAGE_WORKER_CONCURRENCY - len(self._worker_tasks)
			for _ in range(max(0, workers_to_start)):
				worker = asyncio.create_task(self._queue_worker())
				self._worker_tasks.add(worker)

	async def _enqueue_message(self, message: discord.Message) -> None:
		await self._ensure_worker_pool()
		await self._message_queue.put(message)

	def _get_channel_processing_lock(self, channel: object) -> asyncio.Lock:
		channel_id = getattr(channel, "id", None)
		if not isinstance(channel_id, int):
			channel_id = id(channel)
		lock = self._channel_processing_locks.get(channel_id)
		if lock is None:
			lock = asyncio.Lock()
			self._channel_processing_locks[channel_id] = lock
		return lock

	async def _queue_worker(self) -> None:
		while True:
			try:
				message = await self._message_queue.get()
			except asyncio.CancelledError:
				break

			channel_lock = self._get_channel_processing_lock(message.channel)
			try:
				async with channel_lock:
					await self._process_message(message)
			except asyncio.CancelledError:
				raise
			except Exception as exc:
				logging.exception("Error handling Discord message: %s", exc)
				await message.channel.send("Error processing your message: " + str(exc))
			finally:
				self._message_queue.task_done()

	async def _try_handle_command(self, message: discord.Message) -> bool:
		content = (message.content or "").strip()
		history_file = _get_history_file_key_for_channel(message.channel)

		if content == f"{self.command_prefix}commands":
			await message.channel.send(
				"Available commands:\n"
				f"- {self.command_prefix}commands\n"
				f"- {self.command_prefix}history length\n"
				f"- {self.command_prefix}clear history\n"
				f"- {self.command_prefix}clear memory\n"
				f"- {self.command_prefix}forget memories {{topic}}\n"
				f"- {self.command_prefix}list memories [limit]\n"
				f"- {self.command_prefix}list cron jobs\n"
				f"- {self.command_prefix}list heartbeat jobs\n"
				f"- {self.command_prefix}start heartbeat\n"
				f"- {self.command_prefix}stop heartbeat\n"
			)
			return True

		if content == f"{self.command_prefix}history length":
			history_text = history.get_conversation_history(history_file=history_file)
			await message.channel.send(f"Conversation history length: {len(history_text)}")
			return True
		if content == f"{self.command_prefix}clear history":
			history.clear_history(history_file=history_file)
			await message.channel.send("Conversation history cleared.")
			return True
		if content in {f"{self.command_prefix}clear memory", f"{self.command_prefix}clear_memory"}:
			store = memory.get_default_store()
			cleared_count = store.clear_memories()
			await message.channel.send(f"Memory cleared. Removed {cleared_count} entr{'y' if cleared_count == 1 else 'ies'}.")
			return True
		if content.startswith(f"{self.command_prefix}forget memories"):
			topic = content[len(f"{self.command_prefix}forget memories"):].strip()
			if not topic:
				await message.channel.send(f"Usage: {self.command_prefix}forget memories {{topic}}")
				return True

			result = await asyncio.to_thread(memory.forget_memories, topic)
			await self._send_long_message(message.channel, result)
			return True
		if content.startswith(f"{self.command_prefix}list memories"):
			parts = content.split()
			limit = None
			if len(parts) >= 3:
				try:
					limit = int(parts[2])
				except Exception:
					await message.channel.send(
						f"Usage: {self.command_prefix}list memories [limit] — limit must be an integer."
					)
					return True
			store = memory.get_default_store()
			memories = store.read_all_memories(limit=limit)
			if not memories:
				await message.channel.send("No memories stored.")
				return True
			formatted = []
			for m in memories:
				text = (m.text or "").strip().replace("\n", " ")
				if len(text) > 300:
					text = text[:297] + "..."
				formatted.append(f"- {text}")
			await self._send_long_message(message.channel, "Memories:\n" + "\n".join(formatted))
			return True
		if content == f"{self.command_prefix}list cron jobs":
			task_names = self._list_cron_task_names()
			if not task_names:
				await message.channel.send(f"No .md cron job tasks found in {CRON_JOBS_DIR}.")
				return True

			formatted_tasks = "\n".join(f"- {name}" for name in task_names)
			await message.channel.send("Cron job tasks (run in this order):\n" + formatted_tasks)
			return True
		if content == f"{self.command_prefix}list heartbeat jobs":
			task_names = self._list_heartbeat_task_names()
			if not task_names:
				await message.channel.send(f"No .md heartbeat tasks found in {HEARTBEAT_JOBS_DIR}.")
				return True

			formatted_tasks = "\n".join(f"- {name}" for name in task_names)
			await message.channel.send("Heartbeat tasks (run in this order):\n" + formatted_tasks)
			return True
		if content == f"{self.command_prefix}start heartbeat":
			started = self._start_heartbeat_task_if_needed(run_immediately=True)
			if started:
				await message.channel.send("Heartbeat scheduler started and ran immediately.")
			else:
				await message.channel.send("Heartbeat scheduler is already running.")
			return True
		if content == f"{self.command_prefix}stop heartbeat":
			stopped = await self._stop_heartbeat_task_if_running()
			if stopped:
				await message.channel.send("Heartbeat scheduler stopped.")
			else:
				await message.channel.send("Heartbeat scheduler is not running.")
			return True
		# Reload command removed
		return False

	async def _process_message(self, message: discord.Message) -> None:
		content = (message.content or "").strip()
		is_self_relay_message = (
			message.author.bot
			and self.client.user is not None
			and message.author.id == self.client.user.id
			and content.startswith(RELAY_PREFIX)
		)

		if is_self_relay_message:
			content = content[len(RELAY_PREFIX):].strip()

		if not is_self_relay_message and await self._try_handle_command(message):
			return

		text_attachment = next(
			(
				attachment
				for attachment in message.attachments
				if (attachment.filename or "").lower().endswith(".txt")
			),
			None,
		)
		has_media_attachment = any(
			self._is_media_attachment(attachment)
			for attachment in message.attachments
		)

		if not content and text_attachment is None and not has_media_attachment:
			return

		if text_attachment is not None:
			async with message.channel.typing():
				attachment_text = await self._read_text_attachment(text_attachment)

			if not attachment_text:
				await message.channel.send("Could not read .txt attachment (empty result).")
				return

			prompt = self._build_prompt(content, attachment_text)
			await self._respond_and_send(message.channel, prompt, message)
			return

		await self._respond_and_send(message.channel, content, message)

	def _ensure_channel_history_files(self) -> None:
		channels = [
			channel
			for guild in self.client.guilds
			for channel in guild.text_channels
		]
		if not channels:
			return

		created = ensure_history_files_for_text_channels(channels)
		if created:
			logging.info(
				"Ensured history files for %d Discord channel%s in %s.",
				created,
				"" if created == 1 else "s",
				CHANNEL_HISTORY_DIR,
			)

	async def _start_voice_conversation_if_possible(self, voice_client: object) -> None:
		if self._voice_conversation is None:
			return

		if not VOICE_RECV_AVAILABLE:
			return

		if not hasattr(voice_client, "listen"):
			logging.warning(
				"Voice receive client is unavailable. Reconnect with discord-ext-voice-recv installed."
			)
			return

		await self._voice_conversation.stop()

		last_error: Exception | None = None
		for attempt in range(VOICE_RECV_START_RETRIES):
			self._voice_input_sink = _DiscordPCMInputSink(self._voice_conversation)
			try:
				if hasattr(voice_client, "is_listening") and voice_client.is_listening():
					voice_client.stop_listening()
				await asyncio.sleep(VOICE_RECV_START_DELAY_SECONDS + (attempt * 0.15))
				voice_client.listen(self._voice_input_sink)
				await self._voice_conversation.start(voice_client)
				logging.info("Started Discord voice <-> Gemini Live conversation bridge.")
				return
			except Exception as exc:
				last_error = exc
				logging.warning(
					"Voice bridge startup attempt %d/%d failed: %s",
					attempt + 1,
					VOICE_RECV_START_RETRIES,
					exc,
				)

		logging.exception("Failed to start voice conversation bridge after retries: %s", last_error)

	async def _stop_voice_conversation_if_running(self) -> None:
		if self._voice_conversation is None:
			return

		await self._voice_conversation.stop()
		self._voice_input_sink = None
		logging.info("Stopped Discord voice <-> Gemini Live conversation bridge.")

	def _register_events(self) -> None:
		@self.client.event
		async def on_ready() -> None:
			logging.info("Discord bot logged in as %s", self.client.user)
			if not getattr(self, "voice_available", False):
				logging.warning("PyNaCl not found: voice features will be unavailable. Install with: pip install pynacl")
			if getattr(self, "voice_available", False) and not VOICE_RECV_AVAILABLE:
				logging.warning(
					"discord-ext-voice-recv not found: incoming Discord voice audio cannot be captured. Install with: pip install discord-ext-voice-recv"
				)
			if not GEMINI_API_KEY:
				logging.warning("GEMINI_API_KEY is not set: Gemini Live voice conversation is disabled.")
			self._ensure_channel_history_files()
			self._start_cron_task_if_needed()
			logging.info("Heartbeat scheduler is idle. Use '%sstart heartbeat' to start it.", self.command_prefix)

		@self.client.event
		async def on_message(message: discord.Message) -> None:
			content = (message.content or "").strip()
			is_self_relay_message = (
				message.author.bot
				and self.client.user is not None
				and message.author.id == self.client.user.id
				and content.startswith(RELAY_PREFIX)
			)

			# Ignore bot messages, except self-authored relay messages prefixed with the command prefix.
			if message.author.bot and not is_self_relay_message:
				return

			# If allowed_channels are set, ignore messages not in those channels
			if self.allowed_channels and getattr(message.channel, "name", None) not in self.allowed_channels:
				return

			if not is_self_relay_message and await self._try_handle_command(message):
				return

			await self._enqueue_message(message)

		@self.client.event
		async def on_voice_state_update(member: discord.Member, before: discord.VoiceState, after: discord.VoiceState) -> None:
			# Skip voice handling if PyNaCl (voice support) is not installed.
			if not getattr(self, "voice_available", False):
				logging.warning("Skipping voice state handling: PyNaCl not installed.")
				return
			# Handle joins, moves, and leaves so the bot follows users and leaves empty channels.
			try:
				if member.bot:
					return

				# If nothing changed about channel membership, ignore (e.g., mute/unmute)
				if before.channel == after.channel:
					return

				guild = getattr(member, "guild", None)
				if guild is None:
					return

				voice_client = getattr(guild, "voice_client", None)

				# If the member joined or moved to a channel, connect or move the bot there.
				if after.channel is not None:
					if voice_client is None:
						connect_kwargs = {}
						if VOICE_RECV_AVAILABLE:
							connect_kwargs["cls"] = voice_recv.VoiceRecvClient
						await after.channel.connect(**connect_kwargs)
						voice_client = getattr(guild, "voice_client", None)
						logging.info(
							"Connected to voice channel %s in guild %s",
							getattr(after.channel, "name", None),
							getattr(guild, "name", None),
						)
					else:
						if voice_client.channel != after.channel:
							await voice_client.move_to(after.channel)
							logging.info(
								"Moved voice client to %s in guild %s",
								getattr(after.channel, "name", None),
								getattr(guild, "name", None),
							)

					if voice_client is not None:
						await self._start_voice_conversation_if_possible(voice_client)

				# If the member left a channel (after.channel is None) or moved away, check if the previous channel is empty.
				if before.channel is not None:
					try:
						non_bot_members = [m for m in before.channel.members if not m.bot]
						if not non_bot_members:
							# If the bot is connected to that channel, disconnect.
							vc = getattr(guild, "voice_client", None)
							if vc is not None and vc.channel == before.channel:
								if hasattr(vc, "is_listening") and vc.is_listening():
									vc.stop_listening()
								await self._stop_voice_conversation_if_running()
								await vc.disconnect()
								logging.info(
									"Disconnected from empty voice channel %s in guild %s",
									getattr(before.channel, "name", None),
									getattr(guild, "name", None),
								)
					except Exception:
						# If reading members fails for any reason, just log and continue.
						logging.exception(
							"Failed checking members for channel %s",
							getattr(before.channel, "name", None),
						)
			except Exception as exc:
				logging.exception("Error handling voice state update: %s", exc)

	def run(self) -> None:
		if not self.token:
			raise ValueError("Set DISCORD_BOT_TOKEN in environment configuration")

		logging.basicConfig(
			format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
			level=logging.WARNING,
		)
		logging.getLogger("discord.ext.voice_recv.opus").setLevel(logging.ERROR)
		logging.getLogger("discord.ext.voice_recv.reader").setLevel(logging.CRITICAL)
		self.client.run(self.token, log_level=logging.WARNING)


def _clear_sub_agents_directory() -> None:
	sub_agents_dir = Path("sub-agents")
	if sub_agents_dir.exists() and sub_agents_dir.is_dir():
		for file in sub_agents_dir.iterdir():
			try:
				if file.is_file():
					file.unlink()
			except Exception:
				logging.warning("Could not unlink sub-agent file '%s'; it may be in use. Skipping.", file)


if __name__ == "__main__":
	if not Path(memory.DEFAULT_DB_PATH).exists():
		memory.get_default_store()

	_clear_sub_agents_directory()

	print("Starting EnGem...")
	DiscordBotWrapper().run()