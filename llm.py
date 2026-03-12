from google import genai
from google.genai import types
import json
import inspect
import mimetypes
import re
import threading
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable
from config import get_paid_gemini_api_key as get_paid_gemini_api_key
from config import MINIMAL_MODEL as MINIMAL_MODEL, LOW_MODEL as LOW_MODEL, MEDIUM_MODEL as MEDIUM_MODEL, HIGH_MODEL as HIGH_MODEL
from api_backoff import call_with_exponential_backoff
from history_cache import CachedContentProfile, HistoryContextCache, compose_uncached_history_prompt, create_cached_content_profile, create_history_context_cache, resolve_history_cached_prompt
import history
import memory as memory

# Memory Retriever file located alongside this module
MEMORY_RETRIEVER_FILE = Path(__file__).parent / "agent_instructions/memory_retriever.md"

# Intent file located alongside this module
INTENT_FILE = Path(__file__).parent / "agent_instructions/intent.md"

# Manager file located alongside this module
MANAGER_FILE = Path(__file__).parent / "agent_instructions/manager.md"

# Sub-agent file located alongside this module
SUB_AGENT_FILE = Path(__file__).parent / "agent_instructions/sub_agent.md"

# Reviewer file located alongside this module
REVIEWER_FILE = Path(__file__).parent / "agent_instructions/reviewer.md"

# Summarize file located alongside this module
TEXTER_FILE = Path(__file__).parent / "agent_instructions/texter.md"

# Media selector file located alongside this module
MEDIA_SELECTOR_FILE = Path(__file__).parent / "agent_instructions/media_selector.md"

# Memory Extractor file located alongside this module
MEMORY_EXTRACTOR_FILE = Path(__file__).parent / "agent_instructions/memory_extractor.md"

# History summarizer system instructions located alongside this module
HISTORY_SUMMARIZER_SYSTEM = Path(__file__).parent / "agent_instructions/history_summarizer.md"


GENERATED_IMAGES_DIR = (Path(__file__).parent / "generated_images").resolve()
GENERATED_VIDEOS_DIR = (Path(__file__).parent / "generated_videos").resolve()
GENERATED_FILES_DIR = (Path(__file__).parent / "generated_files").resolve()
GENERATED_DOCUMENTS_DIR = (Path(__file__).parent / "generated_documents").resolve()
RESTRICTED_OUTPUT_DIRECTORIES = (
    GENERATED_IMAGES_DIR,
    GENERATED_VIDEOS_DIR,
)
ALLOWED_OUTPUT_DIRECTORIES = (
    GENERATED_IMAGES_DIR,
    GENERATED_VIDEOS_DIR,
    GENERATED_FILES_DIR,
    GENERATED_DOCUMENTS_DIR,
)
SUPPORTED_OUTPUT_FILE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff",
    ".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v",
    ".pdf", ".txt", ".md", ".csv", ".json", ".xml", ".yaml", ".yml",
    ".html", ".htm", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip",
}
MAX_INPUT_ATTACHMENTS = 10
SUB_AGENT_INSTRUCTION_PREVIEW_CHARS = 200
REVIEWER_TASK_NAME = "Reviewer"
VALID_PLAN_THINKING_LEVELS = {"MINIMAL", "LOW", "MEDIUM", "HIGH"}
THINKING_LEVEL_TO_MODEL = {
    "MINIMAL": MINIMAL_MODEL,
    "LOW": LOW_MODEL,
    "MEDIUM": MEDIUM_MODEL,
    "HIGH": HIGH_MODEL,
}
THINKING_LEVEL_TO_API_LEVEL = {
    "MINIMAL": "low",
    "LOW": "low",
    "MEDIUM": "medium",
    "HIGH": "high",
}

client = genai.Client(api_key=get_paid_gemini_api_key())


@dataclass
class LLMResponse:
    text: str = ""
    media_paths: list[str] = field(default_factory=list)


def generate_response(
    user_message: str,
    job: bool,
    history_file: str,
    image: dict[str, bytes | str] | list[dict[str, bytes | str]] | None = None,
    execution_plan_notifier: Callable[[str, list[dict], int, bool], None] | None = None,
) -> LLMResponse:
    """
    Main function to generate a response from the model based on the user's message,
    conversation history, and optionally multimodal attachments.
    This function handles the entire flow of generating a response, including intent classification, sub-agent execution, and final response generation.
    If execution_plan_notifier is provided, it is called asynchronously with an
    ASCII diagram of the active execution plan whenever a new plan is detected.
    """

    exit_string = ""
    default_temperature = 1.0
    temperature = default_temperature
    attempt_number = 1
    normalized_attachments = _normalize_attachments(image)
    attachment_text, attachment_memory_context = _ingest_attachments_for_memory(
        normalized_attachments,
        history_file,
    )
    if attachment_text:
        if user_message:
            user_message = f"{user_message}\n\nAttachment text:\n{attachment_text}"
        else:
            user_message = f"Attachment text:\n{attachment_text}"
    history.append_history(role="user", text=user_message, history_file=history_file)
    active_history_cache = create_history_context_cache(
        history_file=history_file,
        history_text=history.get_conversation_history(history_file=history_file),
        client=client,
        backoff_call=call_with_exponential_backoff,
    )

    if not job:
        current_history_text = history.get_conversation_history(history_file=history_file)
        relevant_memories_text = _build_relevant_memories_text(current_history_text, semantic_limit=5, file_limit=3)

        intent_response = ""
        try:
            intent_response = _run_model_api(
                text="Classify the latest user request using the conversation context.",
                system_instructions=INTENT_FILE.read_text(encoding="utf-8") + relevant_memories_text,
                model=LOW_MODEL,
                tool_use_allowed=False,
                force_tool=False,
                temperature=default_temperature,
                thinking_level="low",
                history_cache=active_history_cache,
                current_history_text=current_history_text,
            )
        except Exception as e:
            print(f"Error generating intent response: {e}")

        if "<complex>" not in (intent_response or ""):
            history.append_history(role="IntentClassifier", text=intent_response, history_file=history_file)
            if not job:
                active_history_cache.release()
                post_intent_cache = create_history_context_cache(
                    history_file=history_file,
                    history_text=history.get_conversation_history(history_file=history_file),
                    client=client,
                    backoff_call=call_with_exponential_backoff,
                )
                try:
                    _run_history_summarization_async(
                        history_file=history_file,
                        temperature=default_temperature,
                        history_cache=post_intent_cache.retain(),
                    )
                    summarized_history = history.get_conversation_history(history_file=history_file)
                    relevant_memories_history = _build_relevant_memories_text(
                        summarized_history,
                        semantic_limit=10,
                        file_limit=4,
                    )
                    _run_memory_extraction_async(
                        history_file=history_file,
                        temperature=default_temperature,
                        relevant_memories_text=relevant_memories_history,
                        attachment_context_text=attachment_memory_context,
                        history_cache=post_intent_cache.retain(),
                    )
                finally:
                    post_intent_cache.release()
            return LLMResponse(text=intent_response, media_paths=[])

    while True:
        exit_string = ""
        # Execution order file path
        EXECUTION_ORDER_FILE = Path(__file__).parent / f"sub-agents/execution_order_{history_file}.json"

        # Always clear stale execution-order output so each manager run writes a fresh plan.
        if EXECUTION_ORDER_FILE.exists():
            try:
                EXECUTION_ORDER_FILE.unlink()
            except Exception as e:
                print(f"Error clearing execution order file before manager run: {e}")
                continue

        manager_response = ""
        # Get the manager's response based on the conversation history and the new user message
        try:
            manager_response = _run_model_api(
                text="Review the conversation context and create the execution plan JSON file.",
                system_instructions=MANAGER_FILE.read_text(encoding="utf-8") + history_file,
                model=MEDIUM_MODEL,
                tool_use_allowed=True,
                force_tool=True,
                temperature=temperature,
                thinking_level="high",
                history_cache=active_history_cache,
                current_history_text=history.get_conversation_history(history_file=history_file),
            )
            history.append_history(role="Manager", text=f"{EXECUTION_ORDER_FILE} created.", history_file=history_file)
        except Exception as e:
            print(f"Error generating manager response: {e}")

        # read the execution order from the .json file
        if not EXECUTION_ORDER_FILE.exists():
            print("No execution order file found. ")
            continue
        else:
            try:
                with EXECUTION_ORDER_FILE.open("r", encoding="utf-8") as f:
                    execution_order_dict = json.load(f)
            except Exception as e:
                print(f"Error reading execution order file: {e}")
                continue

        execution_plan = _normalize_execution_plan(execution_order_dict)
        if not execution_plan:
            print("No valid execution plan found in execution order file.")
            continue
        if not _has_final_reviewer_agent(execution_plan):
            print("Execution plan is missing a final serial Reviewer agent.")
            continue

        _dispatch_execution_plan_preview_async(
            execution_plan,
            history_file,
            execution_plan_notifier,
            attempt_number,
            attempt_number > 1,
        )

        sub_agent_system_instructions = SUB_AGENT_FILE.read_text(encoding="utf-8")
        for stage_index, stage in enumerate(execution_plan):
            mode = stage["mode"]
            agents = stage["sub_agents"]

            if mode == "parallel" and len(agents) > 1:
                base_history = history.get_conversation_history(history_file=history_file)

                def _run_parallel_agent(agent: dict, stage_history: str) -> str:
                    model_name, api_thinking_level = _resolve_sub_agent_model_config(agent)
                    return _run_model_api(
                        text=agent["instruction"],
                        system_instructions=sub_agent_system_instructions,
                        model=model_name,
                        tool_use_allowed=True,
                        force_tool=False,
                        temperature=temperature,
                        thinking_level=api_thinking_level,
                        history_cache=active_history_cache,
                        current_history_text=stage_history,
                    )

                with ThreadPoolExecutor(max_workers=min(len(agents), 8)) as executor:
                    future_to_index = {
                        executor.submit(_run_parallel_agent, agent, base_history): idx
                        for idx, agent in enumerate(agents)
                    }

                    for future in as_completed(future_to_index):
                        idx = future_to_index[future]
                        agent = agents[idx]
                        sub_agent_response = ""
                        try:
                            sub_agent_response = future.result()
                        except Exception as e:
                            print(f"Error generating response for sub-agent '{agent['task_name']}': {e}")

                        history.append_history(role=agent["task_name"], text=sub_agent_response, history_file=history_file)

            else:
                for agent_index, agent in enumerate(agents):
                    sub_agent_response = ""
                    try:
                        if _is_final_execution_agent(execution_plan, stage_index, agent_index):
                            sub_agent_response = _run_final_reviewer(
                                history_file,
                                user_message,
                                default_temperature,
                                history_cache=active_history_cache,
                            )
                            exit_string = sub_agent_response
                            history.append_history(role=REVIEWER_TASK_NAME, text=sub_agent_response, history_file=history_file)
                        else:
                            model_name, api_thinking_level = _resolve_sub_agent_model_config(agent)
                            sub_agent_response = _run_model_api(
                                text=agent["instruction"],
                                system_instructions=sub_agent_system_instructions,
                                model=model_name,
                                tool_use_allowed=True,
                                force_tool=False,
                                temperature=temperature,
                                thinking_level=api_thinking_level,
                                history_cache=active_history_cache,
                                current_history_text=history.get_conversation_history(history_file=history_file),
                            )
                            history.append_history(role=agent["task_name"], text=sub_agent_response, history_file=history_file)
                    except Exception as e:
                        print(f"Error generating response for sub-agent '{agent['task_name']}': {e}")

        if not exit_string:
            print("Final Reviewer agent did not produce an exit string.")
            continue
        if exit_string == "<yes>":
            break
        else:
            _run_history_summarization(
                history_file=history_file,
                temperature=default_temperature,
                pivot_role="manager",
                history_cache=active_history_cache.retain(),
                current_history_text=history.get_conversation_history(history_file=history_file),
            )
            active_history_cache.release()
            active_history_cache = create_history_context_cache(
                history_file=history_file,
                history_text=history.get_conversation_history(history_file=history_file),
                client=client,
                backoff_call=call_with_exponential_backoff,
            )
            if temperature < 2.0:
                temperature += 0.1
            attempt_number += 1

    text_response = ""
    media_paths: list[str] = []

    active_history_cache.release()
    post_review_cache = create_history_context_cache(
        history_file=history_file,
        history_text=history.get_conversation_history(history_file=history_file),
        client=client,
        backoff_call=call_with_exponential_backoff,
    )
    _run_history_summarization_async(
        history_file=history_file,
        temperature=default_temperature,
        history_cache=post_review_cache.retain(),
    )

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            text_future = executor.submit(
                _run_model_api,
                text="Use the conversation context to answer the user's latest request.",
                system_instructions=TEXTER_FILE.read_text(encoding="utf-8"),
                model=MINIMAL_MODEL,
                tool_use_allowed=False,
                force_tool=False,
                temperature=default_temperature,
                thinking_level="high",
                history_cache=post_review_cache,
                current_history_text=post_review_cache.history_text,
            )
            media_future = executor.submit(
                _select_media_paths,
                history_file,
                user_message,
                default_temperature,
                post_review_cache,
            )

            try:
                text_response = text_future.result()
                history.append_history(role="Texter", text=text_response, history_file=history_file)
            except Exception as e:
                print(f"Error generating texter response: {e}")

            try:
                media_paths = media_future.result()
                history.append_history(
                    role="MediaSelector",
                    text=json.dumps({"media_paths": media_paths}, ensure_ascii=False),
                    history_file=history_file,
                )
            except Exception as e:
                print(f"Error selecting media paths: {e}")
                media_paths = []

        if not job:
            relevant_memories_history = _build_relevant_memories_text(
                history.get_conversation_history(history_file=history_file),
                semantic_limit=10,
                file_limit=4,
            )
            _run_memory_extraction_async(
                history_file=history_file,
                temperature=default_temperature,
                relevant_memories_text=relevant_memories_history,
                attachment_context_text=attachment_memory_context,
                history_cache=post_review_cache.retain(),
            )
    finally:
        post_review_cache.release()
    return LLMResponse(text=text_response, media_paths=media_paths)


def _dispatch_execution_plan_preview_async(
    execution_plan: list[dict],
    history_file: str,
    execution_plan_notifier: Callable[[str, list[dict], int, bool], None] | None,
    attempt_number: int,
    reset_previous_preview: bool,
) -> None:
    if execution_plan_notifier is None or not execution_plan:
        return

    def _worker() -> None:
        try:
            diagram = _build_execution_plan_ascii_diagram(execution_plan, history_file)
            if diagram:
                try:
                    execution_plan_notifier(diagram, execution_plan, attempt_number, reset_previous_preview)
                except TypeError:
                    execution_plan_notifier(diagram, execution_plan)  # type: ignore[misc, call-arg]
        except Exception as e:
            print(f"Error dispatching execution plan preview: {e}")

    threading.Thread(target=_worker, daemon=True).start()


def _build_execution_plan_ascii_diagram(execution_plan: list[dict], history_file: str) -> str:
    lines: list[str] = []
    lines.append(f"+-- Execution Plan ({history_file})")

    for stage_index, stage in enumerate(execution_plan, start=1):
        mode = str(stage.get("mode", "serial"))
        sub_agents = stage.get("sub_agents", []) if isinstance(stage.get("sub_agents"), list) else []
        lines.append(f"|-- Stage {stage_index} [{mode}]")

        for agent_index, agent in enumerate(sub_agents, start=1):
            if not isinstance(agent, dict):
                continue

            task_name = str(agent.get("task_name", "unnamed_task"))
            instruction = str(agent.get("instruction", ""))
            plan_thinking_level = _normalize_plan_thinking_level(agent.get("thinking_level"))
            preview = _truncate_instruction_preview(instruction, SUB_AGENT_INSTRUCTION_PREVIEW_CHARS)
            lines.append(f"|   |-- Agent {agent_index}: {task_name}")
            lines.append(f"|   |   instruction: {preview}")
            lines.append(f"|   |   thinking_level: {plan_thinking_level}")

    return "\n".join(lines)


def _truncate_instruction_preview(instruction: str, limit: int) -> str:
    compact = " ".join((instruction or "").split())
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "..."


def _select_media_paths(
    history_file: str,
    user_message: str,
    temperature: float,
    history_cache: HistoryContextCache | None = None,
) -> list[str]:
    try:
        from collect_generated_media import get_generated_media, parse_selected_media_paths
    except Exception as e:
        print(f"Error importing media selection skill: {e}")
        return []

    catalog_json = get_generated_media("120")
    selector_input = (
        "Latest user request:\n"
        f"{user_message}\n\n"
        "Use the conversation history and this generated media catalog JSON:\n"
        f"{catalog_json}"
    )

    if history_cache is not None:
        selector_response = _run_model_api(
            text=selector_input,
            system_instructions=MEDIA_SELECTOR_FILE.read_text(encoding="utf-8"),
            model=MINIMAL_MODEL,
            tool_use_allowed=False,
            force_tool=False,
            temperature=temperature,
            thinking_level="low",
            history_cache=history_cache,
            current_history_text=history_cache.history_text,
        )
    else:
        selector_response = _run_model_api(
            text=selector_input,
            system_instructions=MEDIA_SELECTOR_FILE.read_text(encoding="utf-8"),
            model=MINIMAL_MODEL,
            tool_use_allowed=False,
            force_tool=False,
            temperature=temperature,
            thinking_level="low",
            current_history_text=history.get_conversation_history(history_file=history_file),
        )
    return parse_selected_media_paths(selector_response)


def _normalize_attachments(attachments: dict[str, bytes | str] | list[dict[str, bytes | str]] | None) -> list[dict[str, bytes | str]]:
    normalized_attachments: list[dict[str, bytes | str]] = []
    if not attachments:
        return normalized_attachments
    if isinstance(attachments, dict):
        normalized_attachments = [attachments]
    elif isinstance(attachments, list):
        normalized_attachments = [item for item in attachments if isinstance(item, dict)]
    return normalized_attachments[:MAX_INPUT_ATTACHMENTS]


def _ingest_attachments_for_memory(
    attachments: list[dict[str, bytes | str]],
    history_file: str,
) -> tuple[str, str]:
    if not attachments:
        return "", ""

    extracted_segments: list[str] = []
    attachment_memory_contexts: list[str] = []

    for index, attachment in enumerate(attachments, start=1):
        extracted_text = _convert_single_attachment_to_text(attachment)

        indexed_item = None
        try:
            indexed_item = memory.write_attachment_memory(
                attachment=attachment,
                history_file=history_file,
                extracted_text=extracted_text,
            )
        except Exception as e:
            print(f"Error writing attachment memory: {e}")

        filename = attachment.get("filename")
        if extracted_text and isinstance(filename, str) and filename:
            extracted_segments.append(f"[Attachment {index}: {filename}]\n{extracted_text}")
        elif extracted_text:
            extracted_segments.append(f"[Attachment {index}]\n{extracted_text}")

        if indexed_item is not None:
            attachment_memory_contexts.append(memory.render_memory_for_prompt(indexed_item))

    return "\n\n".join(extracted_segments), "\n\n".join(attachment_memory_contexts)


def _convert_attachments_to_text(attachments: dict[str, bytes | str] | list[dict[str, bytes | str]] | None) -> str:
    normalized_attachments = _normalize_attachments(attachments)
    extracted_segments: list[str] = []

    for index, attachment in enumerate(normalized_attachments, start=1):
        extracted_text = _convert_single_attachment_to_text(attachment)
        if not extracted_text:
            continue

        filename = attachment.get("filename")
        if isinstance(filename, str) and filename:
            extracted_segments.append(f"[Attachment {index}: {filename}]\n{extracted_text}")
        else:
            extracted_segments.append(f"[Attachment {index}]\n{extracted_text}")

    return "\n\n".join(extracted_segments)


def _build_relevant_memories_text(query: str, semantic_limit: int, file_limit: int) -> str:
    relevant_memories = memory.search_all_memories(
        query,
        semantic_limit=semantic_limit,
        file_limit=file_limit,
    )
    return "\n\n".join(memory.render_memory_for_prompt(item) for item in relevant_memories)


def _convert_single_attachment_to_text(attachment: dict[str, bytes | str]) -> str:
    if not attachment:
        return ""

    attachment_bytes = attachment.get("data")
    filename = attachment.get("filename")

    if not isinstance(attachment_bytes, bytes) or not attachment_bytes:
        return ""

    mime_type = _normalize_attachment_mime_type(attachment)

    if not isinstance(filename, str) or not filename:
        filename = _default_attachment_name_for_mime_type(mime_type)

    prompt = _build_attachment_extraction_prompt(mime_type)

    try:
        response = call_with_exponential_backoff(
            lambda: client.models.generate_content(
                model=LOW_MODEL,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(text=f"Attachment filename: {filename}\nAttachment MIME type: {mime_type}\n{prompt}"),
                            types.Part.from_bytes(data=attachment_bytes, mime_type=mime_type),
                        ],
                    )
                ],
                config=types.GenerateContentConfig(temperature=0.2),
            ),
            description="Gemini attachment extraction",
        )
        return (getattr(response, "text", "") or "").strip()
    except Exception as e:
        print(f"Error converting attachment to text: {e}")

    return ""


def _normalize_attachment_mime_type(attachment: dict[str, bytes | str]) -> str:
    mime_type = attachment.get("mime_type")
    if isinstance(mime_type, str) and mime_type.strip():
        return mime_type.strip().lower()

    filename = attachment.get("filename")
    if isinstance(filename, str) and filename:
        guessed_mime_type, _ = mimetypes.guess_type(filename)
        if guessed_mime_type:
            return guessed_mime_type.lower()

    return "application/octet-stream"


def _default_attachment_name_for_mime_type(mime_type: str) -> str:
    if mime_type.startswith("image/"):
        return "image"
    if mime_type.startswith("video/"):
        return "video"
    if mime_type.startswith("audio/"):
        return "audio"
    if mime_type == "application/pdf":
        return "document"
    return "attachment"


def _build_attachment_extraction_prompt(mime_type: str) -> str:
    if mime_type.startswith("image/"):
        return  (
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


def _run_memory_extraction_async(
    history_file: str,
    temperature: float,
    relevant_memories_text: str = "",
    attachment_context_text: str = "",
    history_cache: HistoryContextCache | None = None,
) -> None:
    def _worker() -> None:
        try:
            extraction_context = "Relevant memories:\n\n" + relevant_memories_text
            if attachment_context_text.strip():
                extraction_context += "\n\nRecent attachment memories:\n\n" + attachment_context_text.strip()

            if history_cache is not None:
                memory_extractor_response = _run_model_api(
                    text=extraction_context,
                    system_instructions=MEMORY_EXTRACTOR_FILE.read_text(encoding="utf-8"),
                    model=MINIMAL_MODEL,
                    tool_use_allowed=False,
                    force_tool=False,
                    temperature=temperature,
                    thinking_level="low",
                    history_cache=history_cache,
                    current_history_text=history_cache.history_text,
                )
            else:
                extraction_input = (
                    "History: "
                    + history.get_conversation_history(history_file=history_file)
                    + "\n\n"
                    + extraction_context
                )
                memory_extractor_response = _run_model_api(
                    text=extraction_input,
                    system_instructions=MEMORY_EXTRACTOR_FILE.read_text(encoding="utf-8"),
                    model=MINIMAL_MODEL,
                    tool_use_allowed=False,
                    force_tool=False,
                    temperature=temperature,
                    thinking_level="low"
                )

            raw_response = (memory_extractor_response or "").strip()
            parsed_candidates = _parse_memory_extraction_response(raw_response)
            if parsed_candidates:
                for candidate in parsed_candidates:
                    metadata = {
                        "source_type": "memory_extractor",
                        "history_file": history_file,
                        "memory_category": candidate["category"],
                    }
                    if candidate["related_file_ids"]:
                        metadata["related_file_ids_json"] = json.dumps(candidate["related_file_ids"], ensure_ascii=False)
                    memory.write_semantic_memory(candidate["memory"], metadata=metadata)
                history.append_history(
                    role="MemoryExtractor",
                    text=json.dumps({"memories": parsed_candidates}, ensure_ascii=False),
                    history_file=history_file,
                )
            elif raw_response and raw_response != "<NO_MEMORY>":
                memory.write_semantic_memory(
                    raw_response,
                    metadata={
                        "source_type": "memory_extractor_fallback",
                        "history_file": history_file,
                    },
                )
                history.append_history(role="MemoryExtractor", text=raw_response, history_file=history_file)
        except Exception as e:
            print(f"Error generating memory extractor response: {e}")
        finally:
            if history_cache is not None:
                history_cache.release()

    threading.Thread(target=_worker, daemon=True).start()


def _parse_memory_extraction_response(response_text: str) -> list[dict[str, Any]]:
    cleaned_text = (response_text or "").strip()
    if not cleaned_text or cleaned_text == "<NO_MEMORY>":
        return []

    def _try_parse_json(text: str) -> Any:
        try:
            return json.loads(text)
        except Exception:
            return None

    parsed = _try_parse_json(cleaned_text)
    if parsed is None:
        match = re.search(r"(\{.*\}|\[.*\])", cleaned_text, re.S)
        if match:
            parsed = _try_parse_json(match.group(1))

    if isinstance(parsed, dict):
        candidates = parsed.get("memories")
    else:
        candidates = parsed

    if not isinstance(candidates, list):
        return []

    normalized_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue

        memory_text = str(candidate.get("memory") or candidate.get("text") or "").strip()
        if not memory_text:
            continue

        category = str(candidate.get("category") or "general").strip() or "general"
        raw_related_file_ids = candidate.get("related_file_ids") or []
        if not isinstance(raw_related_file_ids, list):
            raw_related_file_ids = []

        normalized_candidates.append(
            {
                "memory": memory_text,
                "category": category,
                "related_file_ids": [str(file_id) for file_id in raw_related_file_ids if str(file_id).strip()],
            }
        )

    return normalized_candidates


def _run_history_summarization(
    history_file: str,
    temperature: float,
    pivot_role: str = "user",
    history_cache: HistoryContextCache | None = None,
    current_history_text: str | None = None,
) -> None:
    try:
        if history_cache is not None:
            summary = _run_model_api(
                text=(
                    f"Summarize only the conversation history before the latest '{pivot_role}' message. "
                    f"Do not include that latest '{pivot_role}' message or anything after it."
                ),
                system_instructions=HISTORY_SUMMARIZER_SYSTEM.read_text(encoding="utf-8"),
                model=LOW_MODEL,
                tool_use_allowed=False,
                force_tool=False,
                temperature=temperature,
                thinking_level="low",
                history_cache=history_cache,
                current_history_text=current_history_text,
            )
        else:
            prior_history = history.get_history_before_latest_role(history_file=history_file, role=pivot_role)
            if not prior_history:
                return

            summary = _run_model_api(
                prior_history,
                HISTORY_SUMMARIZER_SYSTEM.read_text(encoding="utf-8"),
                LOW_MODEL,
                tool_use_allowed=False,
                force_tool=False,
                temperature=temperature,
                thinking_level="low",
            )

        cleaned_summary = (summary or "").strip()
        if not cleaned_summary:
            return

        history.rewrite_history_with_summary_before_latest_role(
            summary_text=cleaned_summary,
            history_file=history_file,
            pivot_role=pivot_role,
        )
    except Exception as e:
        print(f"Error running history summarization: {e}")
    finally:
        if history_cache is not None:
            history_cache.release()


def _run_history_summarization_async(
    history_file: str,
    temperature: float,
    pivot_role: str = "user",
    history_cache: HistoryContextCache | None = None,
    current_history_text: str | None = None,
) -> None:
    def _worker() -> None:
        _run_history_summarization(
            history_file=history_file,
            temperature=temperature,
            pivot_role=pivot_role,
            history_cache=history_cache,
            current_history_text=current_history_text,
        )

    threading.Thread(target=_worker, daemon=True).start()


def _get_function_declarations(client: genai.Client = None) -> list[types.FunctionDeclaration]:
    """
    Helper function to return a list of available tools for the agent to use, based on the functions defined in the skills directory.
    """
    # get all modules in the tools directory and return only functions defined in those modules
    function_declarations = []
    tools_dir = Path(__file__).parent / "tools"
    for tool_file in tools_dir.glob("*.py"):
        module_name = tool_file.stem
        try:
            module = __import__(f"tools.{module_name}", fromlist=[module_name])
            for _, attr in inspect.getmembers(module, inspect.isfunction):
                if attr.__module__ == module.__name__ and not attr.__name__.startswith("_"):
                    function_declarations.append(types.FunctionDeclaration.from_callable(client=client, callable=attr))
        except Exception as e:
            print(f"Error importing tool module '{module_name}': {e}")
    return function_declarations


def _normalize_execution_plan(execution_order_dict: dict) -> list[dict]:
    """Normalize execution order payload into staged execution format.

    Required format:
    - {"execution_plan": [{"mode": "parallel|serial", "sub_agents": [...]}, ...]}
    - Each stage must explicitly include "mode" as "parallel" or "serial".
    - The final stage must be serial and end with a Reviewer sub-agent.
    - Each sub-agent should include "thinking_level" (MINIMAL/LOW/MEDIUM/HIGH).
    - Missing or invalid thinking levels are normalized to MEDIUM for backward compatibility.
    """
    normalized_plan: list[dict] = []

    if isinstance(execution_order_dict, dict) and isinstance(execution_order_dict.get("execution_plan"), list):
        for stage in execution_order_dict["execution_plan"]:
            if not isinstance(stage, dict):
                continue

            mode = stage.get("mode")
            if mode not in ("parallel", "serial"):
                continue
            sub_agents = stage.get("sub_agents", [])
            if not isinstance(sub_agents, list):
                continue

            cleaned_agents = []
            for agent in sub_agents:
                if not isinstance(agent, dict):
                    continue
                task_name = agent.get("task_name")
                instruction = agent.get("instruction")
                if not isinstance(task_name, str) or not isinstance(instruction, str):
                    continue
                thinking_level = _normalize_plan_thinking_level(agent.get("thinking_level"))
                cleaned_agents.append(
                    {
                        "task_name": task_name,
                        "instruction": instruction,
                        "thinking_level": thinking_level,
                    }
                )

            if cleaned_agents:
                normalized_plan.append({"mode": mode, "sub_agents": cleaned_agents})

        return normalized_plan

    return []


def _get_skill(function_name: str, function_args: dict) -> str:
    """Helper function to execute a tool function based on its name and arguments, and return the output as a string."""
    function_output = ""
    for tool_file in (Path(__file__).parent / "tools").glob("*.py"):
        module_name = tool_file.stem
        module = __import__(f"tools.{module_name}", fromlist=[module_name])
        for _, attr in inspect.getmembers(module, inspect.isfunction):
            if attr.__module__ == module.__name__ and attr.__name__ == function_name:
                result = attr(function_args[next(iter(function_args))])
                function_output += result
    return function_output


def _has_final_reviewer_agent(execution_plan: list[dict]) -> bool:
    if not execution_plan:
        return False

    final_stage = execution_plan[-1]
    if final_stage.get("mode") != "serial":
        return False

    final_agents = final_stage.get("sub_agents")
    if not isinstance(final_agents, list) or len(final_agents) != 1:
        return False

    final_agent = final_agents[0]
    return isinstance(final_agent, dict) and final_agent.get("task_name") == REVIEWER_TASK_NAME


def _is_final_execution_agent(execution_plan: list[dict], stage_index: int, agent_index: int) -> bool:
    if not execution_plan:
        return False

    if stage_index != len(execution_plan) - 1:
        return False

    final_stage_agents = execution_plan[-1].get("sub_agents", [])
    if not isinstance(final_stage_agents, list):
        return False

    return agent_index == len(final_stage_agents) - 1


def _run_final_reviewer(
    history_file: str,
    user_message: str,
    temperature: float,
    history_cache: HistoryContextCache | None = None,
) -> str:
    if history_cache is not None:
        return _run_model_api(
            text="Review the conversation context and determine whether the user's request is complete.",
            system_instructions=REVIEWER_FILE.read_text(encoding="utf-8") + user_message,
            model=LOW_MODEL,
            tool_use_allowed=False,
            force_tool=False,
            temperature=temperature,
            thinking_level="low",
            history_cache=history_cache,
            current_history_text=history.get_conversation_history(history_file=history_file),
        )

    return _run_model_api(
        text=history.get_conversation_history(history_file=history_file),
        system_instructions=REVIEWER_FILE.read_text(encoding="utf-8") + user_message,
        model=LOW_MODEL,
        tool_use_allowed=False,
        force_tool=False,
        temperature=temperature,
        thinking_level="low",
    )


def _normalize_plan_thinking_level(raw_level: object) -> str:
    if isinstance(raw_level, str):
        normalized = raw_level.strip().upper()
        if normalized in VALID_PLAN_THINKING_LEVELS:
            return normalized
    return "MEDIUM"


def _normalize_api_thinking_level(raw_level: object) -> str:
    if isinstance(raw_level, str):
        normalized = raw_level.strip().lower()
        if normalized in {"low", "medium", "high"}:
            return normalized
    return "high"


def _resolve_sub_agent_model_config(agent: dict) -> tuple[str, str]:
    plan_thinking_level = _normalize_plan_thinking_level(agent.get("thinking_level"))
    model_name = THINKING_LEVEL_TO_MODEL.get(plan_thinking_level, MEDIUM_MODEL)
    api_thinking_level = THINKING_LEVEL_TO_API_LEVEL.get(plan_thinking_level, "medium")
    return model_name, api_thinking_level


def _run_model_api(
    text: str,
    system_instructions: str,
    model: str,
    tool_use_allowed: bool = True,
    force_tool: bool = False,
    temperature: float = 1,
    thinking_level: str = "high",
    history_cache: HistoryContextCache | None = None,
    current_history_text: str | None = None,
) -> str:
    """
    Helper function to call the model API with the given text and system instructions, and return the generated response.
    text: the input text to generate a response for
    system_instructions: the system instructions to provide to the model for this generation
    tool_use_allowed: whether to allow the model to use tools for this generation (default: True)
    temperature: the temperature to use for this generation (default: 1)
    """

    agent_tools = types.Tool(function_declarations=_get_function_declarations(client=client))
    tool_config = types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(
            mode="ANY", allowed_function_names=["run_python"]
        )
    )
    cache_tools = [agent_tools] if tool_use_allowed else None
    cache_tool_config = tool_config if force_tool else None

    if model == MINIMAL_MODEL and "2.5" in MINIMAL_MODEL:
        thinking_config = types.ThinkingConfig(thinking_budget=16384)
    else:
        thinking_config = types.ThinkingConfig(thinking_level=_normalize_api_thinking_level(thinking_level))

    prompt_text = text
    cached_content_name = None
    cache_profile: CachedContentProfile | None = None
    if history_cache is not None:
        cache_profile = create_cached_content_profile(
            model=model,
            system_instruction=system_instructions,
            tools=cache_tools,
            tool_config=cache_tool_config,
        )
        prompt_text, cached_content_name = resolve_history_cached_prompt(
            history_cache=history_cache,
            profile=cache_profile,
            model=model,
            dynamic_text=text,
            current_history_text=current_history_text,
        )
    elif current_history_text is not None:
        prompt_text = compose_uncached_history_prompt(current_history_text, text)

    if cached_content_name:
        config = types.GenerateContentConfig(
            temperature=temperature,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=force_tool),
            thinking_config=thinking_config,
            cached_content=cached_content_name,
        )
    else:
        config = types.GenerateContentConfig(
            system_instruction=system_instructions,
            tool_config=cache_tool_config,
            tools=cache_tools or [],
            temperature=temperature,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=force_tool),
            thinking_config=thinking_config,
            cached_content=None,
        )

    function_output = ""

    response = call_with_exponential_backoff(
        lambda: client.models.generate_content(
            model=model,
            config=config,
            contents=prompt_text,
        ),
        description=f"Gemini generate_content ({model})",
    )

    if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if part.function_call:
                function_output += _get_skill(part.function_call.name, part.function_call.args)
                    
    output = ""

    if function_output == "":
        output = response.text or ""
    else:
        output = response.text or ""
        output += "\n\n" + function_output
    return output