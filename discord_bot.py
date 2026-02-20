import asyncio
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Awaitable, Callable, Optional
from config import DISCORD_BOT_TOKEN as DISCORD_BOT_TOKEN, DISCORD_BOT_CHANNEL as DISCORD_BOT_CHANNEL

import discord
import tools
import llm
import whisper
import skills.vector_database as vector_database


class DiscordBotWrapper:
	def __init__(
		self,
		responder: Optional[Callable[[str, discord.Message], Awaitable[str]]] = None,
		command_prefix: str = "!",
	) -> None:
		self.token = DISCORD_BOT_TOKEN
		self.command_prefix = command_prefix
		self.allowed_channel = DISCORD_BOT_CHANNEL
		self.whisper_model = None

		intents = discord.Intents.default()
		intents.message_content = True

		self.client = discord.Client(intents=intents)
		self.responder = responder or self._default_responder
		self._register_events()

	async def _default_responder(self, text: str, _: discord.Message) -> str:
		return await asyncio.to_thread(llm.generate_response, text)

	async def _send_long_message(self, channel: discord.abc.Messageable, text: str) -> None:
		max_len = 2000
		for start in range(0, len(text), max_len):
			await channel.send(text[start : start + max_len])

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

	def _register_events(self) -> None:
		@self.client.event
		async def on_ready() -> None:
			logging.info("Discord bot logged in as %s", self.client.user)
			if shutil.which("ffmpeg") is None:
				logging.warning("ffmpeg not found: voice transcription will be unavailable until installed.")

		@self.client.event
		async def on_message(message: discord.Message) -> None:
			if message.author.bot:
				return

			if getattr(message.channel, "name", None) != self.allowed_channel:
				return

			content = (message.content or "").strip()
			audio_attachment = next(
				(
					attachment
					for attachment in message.attachments
					if (attachment.content_type or "").startswith("audio/")
				),
				None,
			)

			if content == f"{self.command_prefix}ping":
				await message.channel.send("pong")
				return

			if not content and audio_attachment is None:
				return

			try:
				if audio_attachment is not None and not content:
					async with message.channel.typing():
						transcription = await self._transcribe_attachment(audio_attachment)

					if not transcription:
						await message.channel.send("Could not transcribe audio (empty result).")
						return

					async with message.channel.typing():
						reply = await self.responder(transcription, message)
					await self._send_long_message(message.channel, reply)
					return

				async with message.channel.typing():
					reply = await self.responder(content, message)
				await self._send_long_message(message.channel, reply)
			except Exception as exc:
				logging.exception("Error handling Discord message: %s", exc)
				await message.channel.send("Error processing your message: " + str(exc))

	def run(self) -> None:
		if not self.token:
			raise ValueError("Set DISCORD_BOT_TOKEN in env or credentials.py")

		logging.basicConfig(
			format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
			level=logging.INFO,
		)
		self.client.run(self.token, log_level=logging.WARNING)


if __name__ == "__main__":
	tools.archive_history()
	tools.init_history()

	# if vector_db is not there, create it
	if not Path(vector_database.DEFAULT_DB_PATH).exists():
		logging.info("Creating vector database...")
		vector_database.get_default_store()

	DiscordBotWrapper().run()