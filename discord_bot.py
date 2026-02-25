import asyncio
import datetime
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Awaitable, Callable, Iterable, Optional

from config import (
	CRON_JOB_HOUR,
	CRON_JOB_MINUTE,
	DISCORD_BOT_CHANNELS,
	DISCORD_BOT_TOKEN,
)

import discord
import history
import llm
import whisper
import memory as memory

CRON_JOBS_DIR = Path(__file__).parent / "agent_instructions/cron_jobs"
HEARTBEAT_JOBS_DIR = Path(__file__).parent / "agent_instructions/heartbeat_jobs"
DISCORD_MESSAGE_LIMIT = 2000
HEARTBEAT_INTERVAL_SECONDS = 30 * 60
MESSAGE_WORKER_CONCURRENCY = 3
CHANNEL_HISTORY_DIR = Path(__file__).parent / "memory" / "channel_history"

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
		self.whisper_model = None
		self._message_queue: asyncio.Queue[discord.Message] = asyncio.Queue()
		self._worker_tasks: set[asyncio.Task[None]] = set()
		self._worker_start_lock = asyncio.Lock()
		self._cron_task: asyncio.Task[None] | None = None
		self._cron_stop_event: asyncio.Event | None = None
		self._heartbeat_task: asyncio.Task[None] | None = None
		self._heartbeat_stop_event: asyncio.Event | None = None
		self._restart_requested = False

		intents = discord.Intents.default()
		intents.message_content = True

		self.client = discord.Client(intents=intents)
		self.responder = responder or self._default_responder
		self._register_events()

	async def _default_responder(self, text: str, message: discord.Message) -> str:
		history_file = _get_history_file_key_for_channel(message.channel)
		return await asyncio.to_thread(llm.generate_response, text, False, history_file)

	async def _send_long_message(self, channel: discord.abc.Messageable, text: str) -> None:
		for start in range(0, len(text), DISCORD_MESSAGE_LIMIT):
			await channel.send(text[start : start + DISCORD_MESSAGE_LIMIT])

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
		worker_tasks = [task for task in self._worker_tasks if not task.done()]
		tasks = [
			task
			for task in (self._cron_task, self._heartbeat_task, *worker_tasks)
			if task is not None and not task.done()
		]
		for task in tasks:
			task.cancel()

		if tasks:
			await asyncio.gather(*tasks, return_exceptions=True)

		self._cron_task = None
		self._heartbeat_task = None
		self._worker_tasks.clear()

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
					async with channel.typing():
						reply = await asyncio.to_thread(llm.generate_response, task_prompt, True, channel_name)
					await self._send_long_message(channel, reply)
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
					async with channel.typing():
						reply = await asyncio.to_thread(llm.generate_response, task_prompt, True, channel_name)
					await self._send_long_message(channel, reply)
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
		history_file = _get_history_file_key_for_channel(message.channel)

		if content == f"{self.command_prefix}commands":
			await message.channel.send(
				"Available commands:\n"
				f"- {self.command_prefix}commands\n"
				f"- {self.command_prefix}history length\n"
				f"- {self.command_prefix}clear history\n"
				f"- {self.command_prefix}clear memory\n"
				f"- {self.command_prefix}list cron jobs\n"
				f"- {self.command_prefix}list heartbeat jobs\n"
				f"- {self.command_prefix}start heartbeat\n"
				f"- {self.command_prefix}stop heartbeat\n"
				f"- {self.command_prefix}reload"
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
		if content == f"{self.command_prefix}reload":
			await message.channel.send("Reloading bot...")
			await self._request_reload()
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

	def _register_events(self) -> None:
		@self.client.event
		async def on_ready() -> None:
			logging.info("Discord bot logged in as %s", self.client.user)
			if shutil.which("ffmpeg") is None:
				logging.warning("ffmpeg not found: voice transcription will be unavailable until installed.")
			self._ensure_channel_history_files()
			self._start_cron_task_if_needed()
			logging.info("Heartbeat scheduler is idle. Use '%sstart heartbeat' to start it.", self.command_prefix)

		@self.client.event
		async def on_message(message: discord.Message) -> None:
			# Ignore messages from bots (including itself) and messages not in the allowed channel
			if message.author.bot:
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