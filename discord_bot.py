import asyncio
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Awaitable, Callable, Optional

from config import (
	DISCORD_BOT_CHANNEL,
	DISCORD_BOT_TOKEN,
	DISCORD_UPDATES_CHANNEL,
	DISCORD_UPDATES_ENABLED,
	DISCORD_UPDATES_INTERVAL_SECONDS,
)

import discord
import tools
import llm
import whisper
import skills.vector_database as vector_database

# Tasker file located alongside this module
TASKER_FILE = Path(__file__).parent / "agent_instructions/tasker.md"
DISCORD_MESSAGE_LIMIT = 2000
DEFAULT_UPDATES_INTERVAL_SECONDS = 86400.0


class DiscordBotWrapper:
	def __init__(
		self,
		responder: Optional[Callable[[str, discord.Message], Awaitable[str]]] = None,
		command_prefix: str = ">",
	) -> None:
		self.token = DISCORD_BOT_TOKEN
		self.command_prefix = command_prefix
		self.allowed_channel = DISCORD_BOT_CHANNEL
		self.updates_channel = DISCORD_UPDATES_CHANNEL
		self.updates_enabled = DISCORD_UPDATES_ENABLED
		self.updates_prompt = TASKER_FILE.read_text(encoding="utf-8")
		self.updates_interval_seconds = DISCORD_UPDATES_INTERVAL_SECONDS
		self.whisper_model = None
		self._message_queue: asyncio.Queue[discord.Message] = asyncio.Queue()
		self._worker_task: asyncio.Task[None] | None = None
		self._updates_task: asyncio.Task[None] | None = None
		self._queue_lock = asyncio.Lock()

		intents = discord.Intents.default()
		intents.message_content = True

		self.client = discord.Client(intents=intents)
		self.responder = responder or self._default_responder
		self._register_events()

	async def _default_responder(self, text: str, _: discord.Message) -> str:
		return await asyncio.to_thread(llm.generate_response, text)

	async def _send_long_message(self, channel: discord.abc.Messageable, text: str) -> None:
		for start in range(0, len(text), DISCORD_MESSAGE_LIMIT):
			await channel.send(text[start : start + DISCORD_MESSAGE_LIMIT])

	@staticmethod
	def _to_bool(value: object) -> bool:
		return str(value).strip().lower() in {"1", "true", "yes", "on"}

	@staticmethod
	def _build_prompt(content: str, attachment_text: str) -> str:
		if not content:
			return attachment_text
		return f"{content}\n\n{attachment_text}".strip()

	async def _respond_and_send(self, channel: discord.abc.Messageable, prompt: str, message: discord.Message) -> None:
		async with channel.typing():
			reply = await self.responder(prompt, message)
		await self._send_long_message(channel, reply)

	async def _get_whisper_model(self):
		if self.whisper_model is None:
			self.whisper_model = await asyncio.to_thread(whisper.load_model, "base")
		return self.whisper_model

	async def _transcribe_attachment(self, attachment: discord.Attachment) -> str:
		if shutil.which("ffmpeg") is None:
			raise RuntimeError("ffmpeg is not installed. Install it with: brew install ffmpeg")

		suffix = Path(attachment.filename or "audio.ogg").suffix or ".ogg"
		with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
			tmp_path = tmp.name

		try:
			await attachment.save(tmp_path)
			model = await self._get_whisper_model()
			result = await asyncio.to_thread(model.transcribe, tmp_path, fp16=False)
			return (result.get("text") or "").strip()
		finally:
			Path(tmp_path).unlink(missing_ok=True)

	async def _read_text_attachment(self, attachment: discord.Attachment) -> str:
		data = await attachment.read()
		try:
			return data.decode("utf-8").strip()
		except UnicodeDecodeError:
			return data.decode("utf-8", errors="replace").strip()

	def _get_updates_interval_seconds(self) -> float:
		try:
			interval_seconds = float(self.updates_interval_seconds)
		except (ValueError, TypeError):
			logging.warning(
				"Invalid DISCORD_UPDATES_INTERVAL_SECONDS '%s'; defaulting to %s",
				self.updates_interval_seconds,
				int(DEFAULT_UPDATES_INTERVAL_SECONDS),
			)
			interval_seconds = DEFAULT_UPDATES_INTERVAL_SECONDS

		if interval_seconds <= 0:
			logging.warning(
				"DISCORD_UPDATES_INTERVAL_SECONDS must be > 0; defaulting to %s",
				int(DEFAULT_UPDATES_INTERVAL_SECONDS),
			)
			interval_seconds = DEFAULT_UPDATES_INTERVAL_SECONDS

		return interval_seconds

	def _updates_are_enabled(self) -> bool:
		return self._to_bool(self.updates_enabled)

	def _find_channel_by_name(self, channel_name: str) -> Optional[discord.abc.Messageable]:
		for guild in self.client.guilds:
			for channel in guild.text_channels:
				if channel.name == channel_name:
					return channel
		return None

	async def _run_updates_scheduler(self) -> None:
		while True:
			await asyncio.sleep(self._get_updates_interval_seconds())

			channel = self._find_channel_by_name(self.updates_channel)
			if channel is None:
				logging.warning("Updates channel '%s' not found; skipping scheduled message.", self.updates_channel)
				continue

			try:
				tools.archive_history()
				tools.init_history()
				reply = await asyncio.to_thread(llm.generate_response, self.updates_prompt)
				await self._send_long_message(channel, reply)
			except Exception as exc:
				logging.exception("Error running scheduled updates message: %s", exc)

	async def _enqueue_message(self, message: discord.Message) -> None:
		async with self._queue_lock:
			if self._worker_task is None or self._worker_task.done():
				self._worker_task = asyncio.create_task(self._queue_worker())

			await self._message_queue.put(message)

	async def _queue_worker(self) -> None:
		while True:
			message = await self._message_queue.get()
			try:
				await self._process_message(message)
			except Exception as exc:
				logging.exception("Error handling Discord message: %s", exc)
				await message.channel.send("Error processing your message: " + str(exc))
			finally:
				self._message_queue.task_done()

	async def _process_message(self, message: discord.Message) -> None:
		content = (message.content or "").strip()
		audio_attachment = next(
			(
				attachment
				for attachment in message.attachments
				if (attachment.content_type or "").startswith("audio/")
			),
			None,
		)
		text_attachment = next(
			(
				attachment
				for attachment in message.attachments
				if (attachment.filename or "").lower().endswith(".txt")
			),
			None,
		)

		if content == f"{self.command_prefix}history":
			history = tools.get_conversation_history()
			if len(history) == 0:
				history = "No conversation history available."
			await message.channel.send(history)
			return
		elif content == f"{self.command_prefix}clear history":
			tools.clear_history()
			await message.channel.send("Conversation history cleared.")
			return

		if not content and audio_attachment is None and text_attachment is None:
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

		if audio_attachment is not None and not content:
			async with message.channel.typing():
				transcription = await self._transcribe_attachment(audio_attachment)

			if not transcription:
				await message.channel.send("Could not transcribe audio (empty result).")
				return

			await self._respond_and_send(message.channel, transcription, message)
			return

		await self._respond_and_send(message.channel, content, message)

	def _register_events(self) -> None:
		@self.client.event
		async def on_ready() -> None:
			logging.info("Discord bot logged in as %s", self.client.user)
			if shutil.which("ffmpeg") is None:
				logging.warning("ffmpeg not found: voice transcription will be unavailable until installed.")
			if self._updates_are_enabled():
				if self._updates_task is None or self._updates_task.done():
					self._updates_task = asyncio.create_task(self._run_updates_scheduler())
			else:
				logging.info("Automatic updates are disabled by DISCORD_UPDATES_ENABLED")

		@self.client.event
		async def on_message(message: discord.Message) -> None:
			# Ignore messages from bots (including itself) and messages not in the allowed channel
			if message.author.bot:
				return

			if getattr(message.channel, "name", None) == self.updates_channel:
				return

			# If allowed_channel is set, ignore messages not in that channel
			if getattr(message.channel, "name", None) != self.allowed_channel:
				return

			await self._enqueue_message(message)

	def run(self) -> None:
		if not self.token:
			raise ValueError("Set DISCORD_BOT_TOKEN in environment configuration")

		logging.basicConfig(
			format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
			level=logging.WARNING,
		)
		self.client.run(self.token, log_level=logging.WARNING)


if __name__ == "__main__":
	# if vector_db is not there, create it
	if not Path(vector_database.DEFAULT_DB_PATH).exists():
		logging.info("Creating vector database...")
		vector_database.get_default_store()

	print("Starting PICKLEBOT...")
	DiscordBotWrapper().run()