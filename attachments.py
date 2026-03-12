from __future__ import annotations

import mimetypes

from google import genai
from google.genai import types

from api_backoff import call_with_exponential_backoff
from config import LOW_MODEL, get_paid_gemini_api_key
import memory

MAX_INPUT_ATTACHMENTS = 10

_client = genai.Client(api_key=get_paid_gemini_api_key())


def normalize_attachments(
    attachments: dict[str, bytes | str] | list[dict[str, bytes | str]] | None,
    max_items: int = MAX_INPUT_ATTACHMENTS,
) -> list[dict[str, bytes | str]]:
    normalized: list[dict[str, bytes | str]] = []
    if not attachments:
        return normalized
    if isinstance(attachments, dict):
        normalized = [attachments]
    elif isinstance(attachments, list):
        normalized = [item for item in attachments if isinstance(item, dict)]
    return normalized[:max_items]


def ingest_attachments_for_memory(
    attachments: list[dict[str, bytes | str]],
    history_file: str,
) -> tuple[str, str]:
    if not attachments:
        return "", ""

    extracted_segments: list[str] = []
    attachment_memory_contexts: list[str] = []

    for index, attachment in enumerate(attachments, start=1):
        extracted_text = convert_single_attachment_to_text(attachment)

        indexed_item = None
        try:
            indexed_item = memory.write_attachment_memory(
                attachment=attachment,
                history_file=history_file,
                extracted_text=extracted_text,
            )
        except Exception as exc:
            print(f"Error writing attachment memory: {exc}")

        filename = attachment.get("filename")
        if extracted_text and isinstance(filename, str) and filename:
            extracted_segments.append(f"[Attachment {index}: {filename}]\n{extracted_text}")
        elif extracted_text:
            extracted_segments.append(f"[Attachment {index}]\n{extracted_text}")

        if indexed_item is not None:
            attachment_memory_contexts.append(memory.render_memory_for_prompt(indexed_item))

    return "\n\n".join(extracted_segments), "\n\n".join(attachment_memory_contexts)


def convert_single_attachment_to_text(attachment: dict[str, bytes | str]) -> str:
    if not attachment:
        return ""

    attachment_bytes = attachment.get("data")
    filename = attachment.get("filename")

    if not isinstance(attachment_bytes, bytes) or not attachment_bytes:
        return ""

    mime_type = normalize_attachment_mime_type(attachment)

    if not isinstance(filename, str) or not filename:
        filename = default_attachment_name_for_mime_type(mime_type)

    prompt = build_attachment_extraction_prompt(mime_type)

    try:
        response = call_with_exponential_backoff(
            lambda: _client.models.generate_content(
                model=LOW_MODEL,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(
                                text=(
                                    "Attachment filename: "
                                    f"{filename}\nAttachment MIME type: {mime_type}\n{prompt}"
                                )
                            ),
                            types.Part.from_bytes(data=attachment_bytes, mime_type=mime_type),
                        ],
                    )
                ],
                config=types.GenerateContentConfig(temperature=0.2),
            ),
            description="Gemini attachment extraction",
        )
        return (getattr(response, "text", "") or "").strip()
    except Exception as exc:
        print(f"Error converting attachment to text: {exc}")

    return ""


def normalize_attachment_mime_type(attachment: dict[str, bytes | str]) -> str:
    mime_type = attachment.get("mime_type")
    if isinstance(mime_type, str) and mime_type.strip():
        return mime_type.strip().lower()

    filename = attachment.get("filename")
    if isinstance(filename, str) and filename:
        guessed_mime_type, _ = mimetypes.guess_type(filename)
        if guessed_mime_type:
            return guessed_mime_type.lower()

    return "application/octet-stream"


def default_attachment_name_for_mime_type(mime_type: str) -> str:
    if mime_type.startswith("image/"):
        return "image"
    if mime_type.startswith("video/"):
        return "video"
    if mime_type.startswith("audio/"):
        return "audio"
    if mime_type == "application/pdf":
        return "document"
    return "attachment"


def build_attachment_extraction_prompt(mime_type: str) -> str:
    if mime_type.startswith("image/"):
        return (
            "Extract the useful content from this image. Return plain text only, concise but complete. "
            "Include a description of important visual elements, any visible on-screen text, and relevant contextual details."
        )
    if mime_type.startswith("video/"):
        return (
            "Extract the useful content from this video. Return plain text only, concise but complete. "
            "Include a brief description of important visual events, any visible on-screen text, and a transcript or summary of spoken audio when present."
        )
    if mime_type.startswith("audio/"):
        return (
            "Extract the useful content from this audio clip. Return plain text only, concise but complete. "
            "Transcribe spoken words when possible and summarize relevant non-speech audio if it matters."
        )
    if mime_type == "application/pdf":
        return (
            "Extract the useful content from this PDF document. Return plain text only, concise but complete. "
            "Preserve important wording, headings, lists, and key structured details when they matter to the user's request."
        )
    return (
        "Extract the useful content from this attachment. Return plain text only, concise but complete. "
        "Include readable text and summarize any relevant non-text content."
    )
