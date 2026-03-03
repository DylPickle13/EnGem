from google import genai
from google.genai import types
import os
import json
import inspect
import threading
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable
from config import GEMINI_API_KEY as GEMINI_API_KEY
from config import model as model
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

# Image Extractor file located alongside this module
IMAGE_EXTRACTOR_FILE = Path(__file__).parent / "agent_instructions/image_extractor.md"

GENERATED_IMAGES_DIR = (Path(__file__).parent / "generated_images").resolve()
GENERATED_VIDEOS_DIR = (Path(__file__).parent / "generated_videos").resolve()
SUPPORTED_MEDIA_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff",
    ".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v",
}
MAX_INPUT_ATTACHMENTS = 10
SUB_AGENT_INSTRUCTION_PREVIEW_CHARS = 200


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
    conversation history, and optionally image/video attachments.
    This function handles the entire flow of generating a response, including intent classification, sub-agent execution, and final response generation.
    If execution_plan_notifier is provided, it is called asynchronously with an
    ASCII diagram of the active execution plan whenever a new plan is detected.
    """

    exit_string = ""
    default_temperature = 1.0
    temperature = default_temperature
    last_notified_execution_plan = ""
    attempt_number = 1
    attachment_text = _convert_attachments_to_text(image)
    if attachment_text:
        if user_message:
            user_message = f"{user_message}\n\nAttachment text:\n{attachment_text}"
        else:
            user_message = f"Attachment text:\n{attachment_text}"
    history.append_history(role="user", text=user_message, history_file=history_file)

    if not job:
        relevant_memories = memory.get_default_store().search_memories(history.get_conversation_history(history_file=history_file), limit=5)
        relevant_memories_text = "\n\n".join([f"Memory: {memory.text}\nMetadata: {json.dumps(memory.metadata)}" for memory in relevant_memories])

        intent_response = ""
        try:
            intent_response = _run_model_api(history.get_conversation_history(history_file=history_file), INTENT_FILE.read_text(encoding="utf-8") + relevant_memories_text, tool_use_allowed=False, force_tool=False, temperature=default_temperature)
        except Exception as e:
            print(f"Error generating intent response: {e}")

        if intent_response != "<complex>":
            history.append_history(role="IntentClassifier", text=intent_response, history_file=history_file)
            if not job:
                _run_history_summarization_async(history_file=history_file, temperature=default_temperature)
                extraction_input = history.get_conversation_history(history_file=history_file)
                _run_memory_extraction_async(extraction_input, history_file, default_temperature)
            return LLMResponse(text=intent_response, media_paths=[])

    while True:
        # Execution order file path
        EXECUTION_ORDER_FILE = Path(__file__).parent / f"sub-agents/execution_order_{history_file}.json"

        manager_response = ""
        # Get the manager's response based on the conversation history and the new user message
        try:
            manager_response = _run_model_api(history.get_conversation_history(history_file=history_file), MANAGER_FILE.read_text(encoding="utf-8") + history_file, tool_use_allowed=True, force_tool=True, temperature=temperature)
            history.append_history(role="Manager", text=manager_response, history_file=history_file)
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

        try:
            serialized_execution_plan = json.dumps(execution_plan, ensure_ascii=False, sort_keys=True)
        except Exception:
            serialized_execution_plan = ""

        if serialized_execution_plan and serialized_execution_plan != last_notified_execution_plan:
            _dispatch_execution_plan_preview_async(
                execution_plan,
                history_file,
                execution_plan_notifier,
                attempt_number,
                attempt_number > 1,
            )
            last_notified_execution_plan = serialized_execution_plan

        cannot_proceed = False
        for stage in execution_plan:
            mode = stage["mode"]
            agents = stage["sub_agents"]

            if mode == "parallel" and len(agents) > 1:
                base_history = history.get_conversation_history(history_file=history_file)

                def _run_parallel_agent(agent: dict, stage_history: str) -> str:
                    return _run_model_api(
                        stage_history + "\n\n" + agent["instruction"],
                        system_instructions=SUB_AGENT_FILE.read_text(encoding="utf-8"),
                        tool_use_allowed=True,
                        force_tool=False,
                        temperature=temperature,
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
                        if sub_agent_response.strip() == "<CANNOT_PROCEED>":
                            cannot_proceed = True

                if cannot_proceed:
                    break

            else:
                for agent in agents:
                    sub_agent_response = ""
                    try:
                        sub_agent_response = _run_model_api(
                            history.get_conversation_history(history_file=history_file) + "\n\n" + agent["instruction"],
                            system_instructions=SUB_AGENT_FILE.read_text(encoding="utf-8"),
                            tool_use_allowed=True,
                            force_tool=False,
                            temperature=temperature,
                        )
                        history.append_history(role=agent["task_name"], text=sub_agent_response, history_file=history_file)
                        if sub_agent_response.strip() == "<CANNOT_PROCEED>":
                            cannot_proceed = True
                            break
                    except Exception as e:
                        print(f"Error generating response for sub-agent '{agent['task_name']}': {e}")

                if cannot_proceed:
                    break

        if cannot_proceed:
            continue

        try:
            exit_string = _run_model_api(history.get_conversation_history(history_file=history_file), REVIEWER_FILE.read_text(encoding="utf-8") + user_message, tool_use_allowed=False, force_tool=False, temperature=default_temperature)
            history.append_history(role="Reviewer", text=exit_string, history_file=history_file)
        except Exception as e:
            print(f"Error generating reviewer response: {e}")
        if exit_string == "<yes>":
            # clear the 'sub-agents/execution_order_{history_file}.json' file at the start of each loop iteration
            if EXECUTION_ORDER_FILE.exists():
                try:
                    EXECUTION_ORDER_FILE.unlink()
                except Exception as e:
                    print(f"Error clearing execution order file: {e}")
            break
        else:
            if temperature < 2.0:
                temperature += 0.1
            attempt_number += 1

    text_response = ""
    media_paths: list[str] = []

    _run_history_summarization_async(history_file=history_file, temperature=default_temperature)

    with ThreadPoolExecutor(max_workers=2) as executor:
        text_future = executor.submit(
            _run_model_api,
            history.get_conversation_history(history_file=history_file),
            TEXTER_FILE.read_text(encoding="utf-8"),
            False,
            False,
            default_temperature,
        )
        media_future = executor.submit(
            _select_media_paths,
            history_file,
            user_message,
            default_temperature,
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
        relevant_memories_history = "\n\n".join([f"Memory: {memory.text}" for memory in memory.get_default_store().search_memories(history.get_conversation_history(history_file=history_file), limit=10)])
        extraction_input = "History: " + history.get_conversation_history(history_file=history_file) + "\n\nRelevant memories: \n\n" + relevant_memories_history
        _run_memory_extraction_async(extraction_input, history_file, default_temperature)
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
            preview = _truncate_instruction_preview(instruction, SUB_AGENT_INSTRUCTION_PREVIEW_CHARS)
            lines.append(f"|   |-- Agent {agent_index}: {task_name}")
            lines.append(f"|   |   instruction: {preview}")

    return "\n".join(lines)


def _truncate_instruction_preview(instruction: str, limit: int) -> str:
    compact = " ".join((instruction or "").split())
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "..."


def _select_media_paths(history_file: str, user_message: str, temperature: float) -> list[str]:
    try:
        from skills.select_generated_media import collect_generated_media
    except Exception as e:
        print(f"Error importing media selection skill: {e}")
        return []

    catalog_json = collect_generated_media("120")
    selector_input = (
        "Latest user request:\n"
        f"{user_message}\n\n"
        "Conversation history:\n"
        f"{history.get_conversation_history(history_file=history_file)}\n\n"
        "Generated media catalog JSON:\n"
        f"{catalog_json}"
    )

    selector_response = _run_model_api(
        selector_input,
        MEDIA_SELECTOR_FILE.read_text(encoding="utf-8"),
        tool_use_allowed=False,
        force_tool=False,
        temperature=temperature,
    )
    return _parse_selected_media_paths(selector_response)


def _parse_selected_media_paths(selector_response: str) -> list[str]:
    text = (selector_response or "").strip()
    if not text:
        return []

    payload = _extract_json_payload(text)
    if not isinstance(payload, dict):
        return []

    raw_paths = payload.get("media_paths", [])
    if not isinstance(raw_paths, list):
        return []

    normalized: list[str] = []
    seen: set[str] = set()

    for item in raw_paths:
        if not isinstance(item, str):
            continue
        safe_path = _normalize_media_path(item)
        if not safe_path or safe_path in seen:
            continue
        seen.add(safe_path)
        normalized.append(safe_path)
        if len(normalized) >= 10:
            break

    return normalized


def _extract_json_payload(text: str) -> dict | None:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        parsed = json.loads(text[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


def _normalize_media_path(raw_path: str) -> str | None:
    try:
        path = Path(raw_path).expanduser().resolve()
    except Exception:
        return None

    if not path.exists() or not path.is_file():
        return None
    if path.suffix.lower() not in SUPPORTED_MEDIA_EXTENSIONS:
        return None
    if not _is_under_directory(path, GENERATED_IMAGES_DIR) and not _is_under_directory(path, GENERATED_VIDEOS_DIR):
        return None
    return str(path)


def _is_under_directory(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except Exception:
        return False


def _convert_attachments_to_text(attachments: dict[str, bytes | str] | list[dict[str, bytes | str]] | None) -> str:
    if not attachments:
        return ""

    normalized_attachments: list[dict[str, bytes | str]] = []
    if isinstance(attachments, dict):
        normalized_attachments = [attachments]
    elif isinstance(attachments, list):
        normalized_attachments = [item for item in attachments if isinstance(item, dict)]
    else:
        return ""

    extracted_segments: list[str] = []
    for index, attachment in enumerate(normalized_attachments[:MAX_INPUT_ATTACHMENTS], start=1):
        extracted_text = _convert_single_attachment_to_text(attachment)
        if not extracted_text:
            continue

        filename = attachment.get("filename")
        if isinstance(filename, str) and filename:
            extracted_segments.append(f"[Attachment {index}: {filename}]\n{extracted_text}")
        else:
            extracted_segments.append(f"[Attachment {index}]\n{extracted_text}")

    return "\n\n".join(extracted_segments)


def _convert_single_attachment_to_text(attachment: dict[str, bytes | str]) -> str:
    if not attachment:
        return ""

    attachment_bytes = attachment.get("data")
    mime_type = attachment.get("mime_type")
    filename = attachment.get("filename")

    if not isinstance(attachment_bytes, bytes) or not attachment_bytes:
        return ""

    if not isinstance(mime_type, str) or not mime_type:
        mime_type = "application/octet-stream"

    if not isinstance(filename, str) or not filename:
        filename = "image"

    prompt = IMAGE_EXTRACTOR_FILE.read_text(encoding="utf-8")

    os.environ.setdefault("GEMINI_API_KEY", GEMINI_API_KEY)
    client = genai.Client()

    while True:
        try:
            response = client.models.generate_content(
                model=model,
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
            )
            return (getattr(response, "text", "") or "").strip()
        except Exception as e:
            print(f"Error converting image to text: {e}")
            break

    return ""


def _run_memory_extraction_async(extraction_input: str, history_file: str, temperature: float) -> None:
    def _worker() -> None:
        try:
            memory_extractor_response = _run_model_api(
                extraction_input,
                MEMORY_EXTRACTOR_FILE.read_text(encoding="utf-8"),
                tool_use_allowed=False,
                force_tool=False,
                temperature=temperature,
            )
            cleaned_response = memory_extractor_response.strip()
            if cleaned_response and cleaned_response != "<NO_MEMORY>":
                memory.get_default_store().write_memory(cleaned_response)
                history.append_history(role="MemoryExtractor", text=cleaned_response, history_file=history_file)
        except Exception as e:
            print(f"Error generating memory extractor response: {e}")

    threading.Thread(target=_worker, daemon=True).start()


def _run_history_summarization_async(history_file: str, temperature: float) -> None:
    def _worker() -> None:
        try:
            prior_history = history.get_history_before_latest_user(history_file=history_file)
            if not prior_history:
                return

            summary = _run_model_api(
                prior_history,
                HISTORY_SUMMARIZER_SYSTEM.read_text(encoding="utf-8"),
                tool_use_allowed=False,
                force_tool=False,
                temperature=temperature,
            )
            cleaned_summary = (summary or "").strip()
            if not cleaned_summary:
                return

            history.rewrite_history_with_summary_before_latest_user(
                summary_text=cleaned_summary,
                history_file=history_file,
            )
        except Exception as e:
            print(f"Error running history summarization: {e}")

    threading.Thread(target=_worker, daemon=True).start()


def _get_function_declarations(client: genai.Client = None) -> list[types.FunctionDeclaration]:
    """
    Helper function to return a list of available tools for the agent to use, based on the functions defined in the skills directory.
    """
    # get all modules in the skills directory and return only functions defined in those modules
    function_declarations = []
    skills_dir = Path(__file__).parent / "skills"
    for skill_file in skills_dir.glob("*.py"):
        module_name = skill_file.stem
        try:
            module = __import__(f"skills.{module_name}", fromlist=[module_name])
            for _, attr in inspect.getmembers(module, inspect.isfunction):
                if attr.__module__ == module.__name__ and not attr.__name__.startswith("_"):
                    function_declarations.append(types.FunctionDeclaration.from_callable(client=client, callable=attr))
        except Exception as e:
            print(f"Error importing skill module '{module_name}': {e}")
    return function_declarations


def _normalize_execution_plan(execution_order_dict: dict) -> list[dict]:
    """Normalize execution order payload into staged execution format.

    Required format:
    - {"execution_plan": [{"mode": "parallel|serial", "sub_agents": [...]}, ...]}
    - Each stage must explicitly include "mode" as "parallel" or "serial".
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
                cleaned_agents.append({"task_name": task_name, "instruction": instruction})

            if cleaned_agents:
                normalized_plan.append({"mode": mode, "sub_agents": cleaned_agents})

        return normalized_plan

    return []


def _get_skill(function_name: str, function_args: dict) -> str:
    """Helper function to execute a tool function based on its name and arguments, and return the output as a string."""
    function_output = ""
    for skill_file in (Path(__file__).parent / "skills").glob("*.py"):
        module_name = skill_file.stem
        module = __import__(f"skills.{module_name}", fromlist=[module_name])
        for _, attr in inspect.getmembers(module, inspect.isfunction):
            if attr.__module__ == module.__name__ and attr.__name__ == function_name:
                result = attr(function_args[next(iter(function_args))])
                function_output += result
    return function_output


def _run_model_api(text: str, system_instructions: str, tool_use_allowed: bool = True, force_tool: bool = False, temperature: float = 1) -> str:
    """
    Helper function to call the model API with the given text and system instructions, and return the generated response.
    text: the input text to generate a response for
    system_instructions: the system instructions to provide to the model for this generation
    tool_use_allowed: whether to allow the model to use tools for this generation (default: True)
    temperature: the temperature to use for this generation (default: 1)
    """
    os.environ.setdefault("GEMINI_API_KEY", GEMINI_API_KEY)

    client = genai.Client()
    agent_tools = types.Tool(function_declarations=_get_function_declarations(client=client))
    tool_config = types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(
            mode="ANY", allowed_function_names=["run_python"]
        )
    )
    config = types.GenerateContentConfig(
        system_instruction=system_instructions,
        tool_config=tool_config if force_tool else None,
        tools=[agent_tools] if tool_use_allowed else [],
        temperature=temperature,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=force_tool)
    )

    function_output = ""

    while True:
        try:
            response = client.models.generate_content(
                model=model,
                config=config,
                contents=text,
            )
            break
        except Exception as e:
            print(f"Error calling model API: {e}")
        print("Retrying...")

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