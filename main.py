import logging
import asyncio
import tempfile
from pathlib import Path
import whisper
from credentials import TELEGRAM_BOT_TOKEN as TELEGRAM_BOT_TOKEN

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    filters,
)

import llm, tools
from telegram.constants import ChatAction
import contextlib


# Whisper model will be loaded lazily on first audio/voice message
WHISPER_MODEL = None

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Send response
    async def send_long_message(message_obj, text: str) -> None:
        try:
            MAX_LEN = 4096
            for i in range(0, len(text), MAX_LEN):
                await message_obj.reply_text(text[i:i + MAX_LEN])
        except Exception as e:
            logging.error(f"Error sending long message: {e}")

    async def _typing_indicator(chat_id: int) -> None:
        """Send periodic typing actions while `llm.llm_running` is True."""
        try:
            # Wait briefly for the LLM thread to set the running flag.
            waited = 0.0
            while not getattr(llm, "llm_running", False) and waited < 5.0:
                await asyncio.sleep(0.1)
                waited += 0.1

            while getattr(llm, "llm_running", False):
                try:
                    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
                except Exception:
                    # Ignore failures to send chat action
                    pass
                await asyncio.sleep(3)
        except asyncio.CancelledError:
            # Task cancelled — just exit
            return
    if update.message and update.message.text:
        user_text = update.message.text
        try:
            # Start typing indicator while the (blocking) LLM runs in a thread
            typing_task = asyncio.create_task(_typing_indicator(update.effective_chat.id))
            try:
                llm_reply = await asyncio.to_thread(llm.generate_response, user_text)
            finally:
                typing_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await typing_task
        except Exception as e:
            # Log and store the error as the LLM response
            err_msg = f"Error generating response: {e}"
            return

        await send_long_message(update.message, llm_reply)

    if update.message and update.message.voice:
        file_obj = update.message.voice

        try:
            tg_file = await file_obj.get_file()
            
            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
                tmp_path = tmp.name
            
            await tg_file.download_to_drive(tmp_path)

            global WHISPER_MODEL
            if WHISPER_MODEL is None:
                try:
                    WHISPER_MODEL = await asyncio.to_thread(whisper.load_model, "base")
                except Exception as e:
                    err_msg = f"Failed to load speech model: {e}"
                    await update.message.reply_text(err_msg)
                    Path(tmp_path).unlink(missing_ok=True)
                    return

            result = await asyncio.to_thread(WHISPER_MODEL.transcribe, tmp_path)
            transcription = result["text"].strip()

            Path(tmp_path).unlink(missing_ok=True)

            if not transcription:
                await update.message.reply_text("Could not transcribe audio (empty result).")
                return
            
            # Send to LLM (show typing while generation runs)
            typing_task = asyncio.create_task(_typing_indicator(update.effective_chat.id))
            try:
                llm_reply = await asyncio.to_thread(llm.generate_response, transcription)
            finally:
                typing_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await typing_task

            await send_long_message(update.message, llm_reply)

        except Exception as e:
            err_msg = f"Error processing audio: {e}"

def main() -> None:
    token = TELEGRAM_BOT_TOKEN
    if not token:
        print("Error: set TELEGRAM_BOT_TOKEN in credentials.py")
        return

    # suppress verbose logging from libraries we use, while still showing warnings and errors from our own code
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    app = ApplicationBuilder().token(token).build()

    # Handle audio/voice messages
    app.add_handler(MessageHandler(filters.TEXT | filters.VOICE, handle_message))

    print("Jarvis is starting!")
    tools.init_history()
    app.run_polling()

if __name__ == "__main__":
    main()