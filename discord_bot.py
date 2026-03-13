import asyncio
import logging
import mimetypes
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Optional

from config import (
	DISCORD_BOT_CHANNELS,
	DISCORD_BOT_TOKEN,
)

import discord
import calendar_events
import history
import llm
import memory as memory
import progress_indicator

DISCORD_MESSAGE_LIMIT = progress_indicator.DISCORD_MESSAGE_LIMIT
DISCORD_MAX_ATTACHMENTS_PER_MESSAGE = 10
DISCORD_ATTACHMENT_BATCH_MAX_BYTES = 24 * 1024 * 1024
DISCORD_DEFAULT_UPLOAD_LIMIT_BYTES = 8 * 1024 * 1024
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
		self._progress_indicator = progress_indicator.ExecutionPlanProgressIndicator(
			message_limit=DISCORD_MESSAGE_LIMIT
		)

		intents = discord.Intents.default()
		intents.message_content = True

		self.client = discord.Client(intents=intents)
		self.responder = responder or self._default_responder
		self._register_events()

	async def _default_responder(self, text: str, message: discord.Message) -> llm.LLMResponse:
		history_file = _get_history_file_key_for_channel(message.channel)
		media_payloads = await self._read_media_attachments(message)
		execution_plan_notifier = self._progress_indicator.build_execution_plan_notifier(
			loop=self.client.loop,
			channel=message.channel,
			history_file=history_file,
		)
		return await asyncio.to_thread(
			llm.generate_response,
			text,
			False,
			history_file,
			media_payloads,
			execution_plan_notifier,
		)

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
						"Skipping attachment '%s': file size %d exceeds upload limit %d bytes.",
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
					"Could not attach files (missing or unreadable files).",
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
				await self._send_long_message(channel, f"Skipped 1 file that Discord would not accept: {non_oversized_skips[0]}")
			else:
				await self._send_long_message(channel, f"Skipped {skipped_count} files that Discord would not accept.")

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
		message = "⚠️ Some files are too large for this Discord channel upload limit "
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
				logging.warning("Skipping attachment '%s': Discord rejected the file as too large.", single_path)
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
		await self._progress_indicator.clear_execution_plan_progress_message(channel, history_file)
		await self._send_media_attachments(channel, payload.media_paths)

	async def _shutdown_background_tasks(self) -> None:
		worker_tasks = [task for task in self._worker_tasks if not task.done()]
		execution_plan_tasks = self._progress_indicator.get_active_tasks()
		tasks = [
			task
			for task in (*worker_tasks, *execution_plan_tasks)
			if task is not None and not task.done()
		]
		for task in tasks:
			task.cancel()

		if tasks:
			await asyncio.gather(*tasks, return_exceptions=True)

		self._worker_tasks.clear()
		self._progress_indicator.clear_state()

	async def _read_text_attachment(self, attachment: discord.Attachment) -> str:
		data = await attachment.read()
		try:
			return data.decode("utf-8").strip()
		except UnicodeDecodeError:
			return data.decode("utf-8", errors="replace").strip()

	@staticmethod
	def _get_attachment_mime_type(attachment: discord.Attachment) -> str:
		content_type = (attachment.content_type or "").strip().lower()
		if content_type:
			return content_type

		guessed_content_type, _ = mimetypes.guess_type(attachment.filename or "")
		if guessed_content_type:
			return guessed_content_type.lower()

		return "application/octet-stream"

	@staticmethod
	def _is_image_attachment(attachment: discord.Attachment) -> bool:
		return DiscordBotWrapper._get_attachment_mime_type(attachment).startswith("image/")

	@staticmethod
	def _is_video_attachment(attachment: discord.Attachment) -> bool:
		return DiscordBotWrapper._get_attachment_mime_type(attachment).startswith("video/")

	@staticmethod
	def _is_audio_attachment(attachment: discord.Attachment) -> bool:
		return DiscordBotWrapper._get_attachment_mime_type(attachment).startswith("audio/")

	@staticmethod
	def _is_pdf_attachment(attachment: discord.Attachment) -> bool:
		return DiscordBotWrapper._get_attachment_mime_type(attachment) == "application/pdf"

	def _is_media_attachment(self, attachment: discord.Attachment) -> bool:
		return (
			self._is_image_attachment(attachment)
			or self._is_video_attachment(attachment)
			or self._is_audio_attachment(attachment)
			or self._is_pdf_attachment(attachment)
		)

	@staticmethod
	def _default_attachment_name(mime_type: str) -> str:
		if mime_type.startswith("image/"):
			return "image"
		if mime_type.startswith("video/"):
			return "video"
		if mime_type.startswith("audio/"):
			return "audio"
		if mime_type == "application/pdf":
			return "document"
		return "attachment"

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

			mime_type = self._get_attachment_mime_type(attachment)
			default_name = self._default_attachment_name(mime_type)

			media_payloads.append(
				{
					"data": attachment_bytes,
					"mime_type": mime_type,
					"filename": attachment.filename or default_name,
				}
			)

		return media_payloads

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
			cleared_counts = memory.clear_all_memory_stores()
			await message.channel.send(
				"Memory cleared. Removed "
				f"{cleared_counts['total']} entr{'y' if cleared_counts['total'] == 1 else 'ies'} "
				f"({cleared_counts['semantic']} semantic, {cleared_counts['files']} file, {cleared_counts.get('skills', 0)} skill)."
			)
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
			memories = memory.read_all_memory_records(limit=limit)
			if not memories:
				await message.channel.send("No memories stored.")
				return True
			formatted = []
			for m in memories:
				text = (m.text or "").strip().replace("\n", " ")
				if len(text) > 300:
					text = text[:297] + "..."
				record_type = m.metadata.get("record_type", "semantic_memory")
				formatted.append(f"- [{record_type}] {text}")
			await self._send_long_message(message.channel, "Memories:\n" + "\n".join(formatted))
			return True
		# Reload command removed
		return False

	async def _process_message(self, message: discord.Message) -> None:
		content = (message.content or "").strip()
		if await self._try_handle_command(message):
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

	def _register_events(self) -> None:
		@self.client.event
		async def on_ready() -> None:
			logging.info("Discord bot logged in as %s", self.client.user)
			self._ensure_channel_history_files()

		@self.client.event
		async def on_message(message: discord.Message) -> None:
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


def _clear_sub_agents_directory() -> None:
	sub_agents_dir = Path("sub-agents")
	if sub_agents_dir.exists() and sub_agents_dir.is_dir():
		for file in sub_agents_dir.iterdir():
			try:
				if file.is_file():
					file.unlink()
			except Exception:
				logging.warning("Could not unlink sub-agent file '%s'; it may be in use. Skipping.", file)


def _process_calendar_event(event: dict[str, Any]) -> bool:
	event_name = str(event.get("summary") or "").strip()
	if not event_name:
		logging.info("Skipping calendar event without summary: %s", event.get("id"))
		return False

	description = str(event.get("description") or "").strip()
	if not description:
		logging.info("Skipping calendar event '%s': empty description.", event_name)
		return False

	matching_channel = next(
		(
			channel
			for guild in bot.client.guilds
			for channel in guild.text_channels
			if channel.name == event_name
		),
		None,
	)
	if matching_channel is None:
		logging.info("Skipping calendar event '%s': no matching Discord channel.", event_name)
		return False

	class _InjectedCalendarMessage:
		def __init__(self, channel: discord.TextChannel, content: str) -> None:
			self.channel = channel
			self.content = content
			self.attachments: list[discord.Attachment] = []

	async def _dispatch_as_normal_message() -> None:
		injected_message = _InjectedCalendarMessage(matching_channel, description)
		channel_lock = bot._get_channel_processing_lock(matching_channel)
		async with channel_lock:
			await bot._process_message(injected_message)

	try:
		future = asyncio.run_coroutine_threadsafe(_dispatch_as_normal_message(), bot.client.loop)
		future.result(timeout=900)
		return True
	except Exception:
		logging.exception("Failed to process calendar event '%s' through Discord message pipeline.", event_name)
		return False


if __name__ == "__main__":
	if not Path(memory.DEFAULT_DB_PATH).exists():
		memory.get_default_store()

	_clear_sub_agents_directory()

	print("Starting EnGem...")
	bot = DiscordBotWrapper()

	calendar_thread, calendar_stop_event = calendar_events.check_active_events(
		poll_interval_seconds=5.0,
		daemon=False,
		event_processor=_process_calendar_event,
	)
	try:
		bot.run()
	finally:
		calendar_stop_event.set()
		calendar_thread.join(5)