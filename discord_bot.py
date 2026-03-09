import asyncio
import datetime
import logging
import mimetypes
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
import history
import llm
import memory as memory

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
THINKING_LEVEL_TO_EMOJI = {
	"MINIMAL": "⚪",
	"LOW": "🟢",
	"MEDIUM": "🟡",
	"HIGH": "🔴",
}
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
		self._cron_task: asyncio.Task[None] | None = None
		self._cron_stop_event: asyncio.Event | None = None
		self._heartbeat_task: asyncio.Task[None] | None = None
		self._heartbeat_stop_event: asyncio.Event | None = None
		self._execution_plan_progress_tasks: dict[str, asyncio.Task[None]] = {}
		self._execution_plan_progress_messages: dict[str, discord.Message] = {}

		intents = discord.Intents.default()
		intents.message_content = True

		self.client = discord.Client(intents=intents)
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
		tracker_key = f"{id(channel)}::{history_file}"
		progress_message: discord.Message | None = self._execution_plan_progress_messages.get(tracker_key)
		last_sent_content: str | None = None

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
					try:
						await progress_message.edit(content=message_content)
					except discord.NotFound:
						progress_message = await channel.send(message_content)
						self._execution_plan_progress_messages[tracker_key] = progress_message
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
				thinking_level = str(agent.get("thinking_level", "MEDIUM")).strip().upper()
				thinking_emoji = THINKING_LEVEL_TO_EMOJI.get(thinking_level, THINKING_LEVEL_TO_EMOJI["MEDIUM"])
				instruction = " ".join(str(agent.get("instruction", "")).split())
				if len(instruction) > 200:
					instruction = instruction[:200] + "..."

				lines.append(f"  {emoji} {task_name} {thinking_emoji}")
				if key in in_progress_keys:
					lines.append(f"      instruction: {instruction}")

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

		self._cron_task = None
		self._heartbeat_task = None
		self._worker_tasks.clear()
		self._execution_plan_progress_tasks.clear()
		self._execution_plan_progress_messages.clear()

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
			self._start_cron_task_if_needed()
			logging.info("Heartbeat scheduler is idle. Use '%sstart heartbeat' to start it.", self.command_prefix)

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


if __name__ == "__main__":
	if not Path(memory.DEFAULT_DB_PATH).exists():
		memory.get_default_store()

	_clear_sub_agents_directory()

	print("Starting EnGem...")
	DiscordBotWrapper().run()