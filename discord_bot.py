import asyncio
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Awaitable, Callable, Optional

from config import (
	DISCORD_BOT_CHANNELS,
	DISCORD_BOT_TOKEN,
	DISCORD_UPDATES_CHANNEL,
	DISCORD_UPDATES_ENABLED,
	DISCORD_UPDATES_INTERVAL_SECONDS,
)

import discord
import history
import llm
import whisper
import memory as memory

CRON_JOBS_DIR = Path(__file__).parent / "agent_instructions/cron_jobs"
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
		self.allowed_channels = {
			channel.strip()
			for channel in str(DISCORD_BOT_CHANNELS).split(",")
			if channel.strip()
		}
		self.updates_channel = DISCORD_UPDATES_CHANNEL
		self.updates_enabled = DISCORD_UPDATES_ENABLED
		self.updates_interval_seconds = DISCORD_UPDATES_INTERVAL_SECONDS
		self.whisper_model = None
		self._message_queue: asyncio.Queue[discord.Message] = asyncio.Queue()
		self._worker_task: asyncio.Task[None] | None = None
		self._updates_task: asyncio.Task[None] | None = None
		self._updates_stop_event: asyncio.Event | None = None
		self._restart_requested = False
		self._queue_lock = asyncio.Lock()

		intents = discord.Intents.default()
		intents.message_content = True

		self.client = discord.Client(intents=intents)
		self.responder = responder or self._default_responder
		self._register_events()

	async def _default_responder(self, text: str, _: discord.Message) -> str:
		return await asyncio.to_thread(llm.generate_response, text, False)

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

	async def _shutdown_background_tasks(self) -> None:
		tasks = [task for task in (self._updates_task, self._worker_task) if task is not None and not task.done()]
		for task in tasks:
			task.cancel()

		if tasks:
			await asyncio.gather(*tasks, return_exceptions=True)

		self._updates_task = None
		self._worker_task = None

	async def _request_reload(self) -> None:
		if self._restart_requested:
			return

		self._restart_requested = True
		await self._shutdown_background_tasks()
		await self.client.close()

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

	@staticmethod
	def _get_task_channel_name(task_file: Path) -> str:
		return task_file.stem.replace("_", "-")

	def _load_update_tasks(self) -> list[tuple[str, str, str]]:
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

	def _list_update_task_names(self) -> list[str]:
		return [f"{task_name} -> #{channel_name}" for task_name, channel_name, _ in self._load_update_tasks()]

	def _start_updates_task_if_needed(self, *, run_immediately: bool = False) -> bool:
		if self._updates_task is not None and not self._updates_task.done():
			return False

		self._updates_stop_event = asyncio.Event()
		self._updates_task = asyncio.create_task(self._run_updates_scheduler(run_immediately=run_immediately))
		return True

	async def _stop_updates_task_if_running(self) -> bool:
		if self._updates_task is None or self._updates_task.done():
			self._updates_task = None
			self._updates_stop_event = None
			return False

		if self._updates_stop_event is not None:
			self._updates_stop_event.set()

		self._updates_task.cancel()
		await asyncio.gather(self._updates_task, return_exceptions=True)
		self._updates_task = None
		self._updates_stop_event = None
		return True

	async def _run_updates_scheduler(self, run_immediately: bool = False) -> None:
		async def send_scheduled_update() -> None:
			tasks = self._load_update_tasks()
			if not tasks:
				logging.warning("No scheduled update tasks found in '%s'.", CRON_JOBS_DIR)
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
					async with channel.typing():
						reply = await asyncio.to_thread(llm.generate_response, task_prompt, True)
					await self._send_long_message(channel, reply)
				except Exception as exc:
					logging.exception("Error running scheduled updates task '%s': %s", task_name, exc)

		if run_immediately:
			await send_scheduled_update()

		while True:
			try:
				if self._updates_stop_event is not None:
					await asyncio.wait_for(
						self._updates_stop_event.wait(),
						timeout=self._get_updates_interval_seconds(),
					)
					break
				await asyncio.sleep(self._get_updates_interval_seconds())
			except asyncio.TimeoutError:
				pass
			except asyncio.CancelledError:
				break

			await send_scheduled_update()

	async def _enqueue_message(self, message: discord.Message) -> None:
		async with self._queue_lock:
			if self._worker_task is None or self._worker_task.done():
				self._worker_task = asyncio.create_task(self._queue_worker())

			await self._message_queue.put(message)

	async def _queue_worker(self) -> None:
		while True:
			try:
				message = await self._message_queue.get()
			except asyncio.CancelledError:
				break

			try:
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

		if content == f"{self.command_prefix}commands":
			await message.channel.send(
				"Available commands:\n"
				f"- {self.command_prefix}commands\n"
				f"- {self.command_prefix}history length\n"
				f"- {self.command_prefix}clear history\n"
				f"- {self.command_prefix}clear memory\n"
				f"- {self.command_prefix}list updates tasks\n"
				f"- {self.command_prefix}start updates\n"
				f"- {self.command_prefix}stop updates\n"
				f"- {self.command_prefix}reload"
			)
			return True

		if content == f"{self.command_prefix}history length":
			history_text = history.get_conversation_history()
			await message.channel.send(f"Conversation history length: {len(history_text)}")
			return True
		if content == f"{self.command_prefix}clear history":
			history.clear_history()
			await message.channel.send("Conversation history cleared.")
			return True
		if content in {f"{self.command_prefix}clear memory", f"{self.command_prefix}clear_memory"}:
			store = memory.get_default_store()
			cleared_count = store.clear_memories()
			await message.channel.send(f"Memory cleared. Removed {cleared_count} entr{'y' if cleared_count == 1 else 'ies'}.")
			return True
		if content == f"{self.command_prefix}list updates tasks":
			task_names = self._list_update_task_names()
			if not task_names:
				await message.channel.send(f"No .md update tasks found in {CRON_JOBS_DIR}.")
				return True

			formatted_tasks = "\n".join(f"- {name}" for name in task_names)
			await message.channel.send("Scheduled update tasks (run in this order):\n" + formatted_tasks)
			return True
		if content == f"{self.command_prefix}reload":
			await message.channel.send("Reloading bot...")
			await self._request_reload()
			return True
		if content == f"{self.command_prefix}start updates":
			started = self._start_updates_task_if_needed(run_immediately=True)
			if started:
				await message.channel.send("Updates scheduler started and ran immediately.")
			else:
				await message.channel.send("Updates scheduler is already running.")
			return True
		if content == f"{self.command_prefix}stop updates":
			stopped = await self._stop_updates_task_if_running()
			if stopped:
				await message.channel.send("Updates scheduler stopped.")
			else:
				await message.channel.send("Updates scheduler is not running.")
			return True

		return False

	async def _process_message(self, message: discord.Message) -> None:
		content = (message.content or "").strip()
		if await self._try_handle_command(message):
			return

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
				self._start_updates_task_if_needed()
			else:
				logging.info("Automatic updates are disabled by DISCORD_UPDATES_ENABLED")

		@self.client.event
		async def on_message(message: discord.Message) -> None:
			# Ignore messages from bots (including itself) and messages not in the allowed channel
			if message.author.bot:
				return

			if getattr(message.channel, "name", None) == self.updates_channel:
				return

			# If allowed_channels are set, ignore messages not in those channels
			if self.allowed_channels and getattr(message.channel, "name", None) not in self.allowed_channels:
				return

			if await self._try_handle_command(message):
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
		if self._restart_requested:
			os.execv(sys.executable, [sys.executable, *sys.argv])


if __name__ == "__main__":
	# if vector_db is not there, create it
	if not Path(memory.DEFAULT_DB_PATH).exists():
		logging.info("Creating vector database...")
		memory.get_default_store()

	print("Starting PICKLEBOT...")
	DiscordBotWrapper().run()
	print("PICKLEBOT stopped.")