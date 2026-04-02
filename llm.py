from google import genai
from google.genai import types
import json
import inspect
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable
from progress_indicator import dispatch_execution_plan_preview_async as _dispatch_execution_plan_preview_async
from progress_indicator import normalize_plan_thinking_level as _normalize_plan_thinking_level
import attachments
import collect_generated_media
from config import get_paid_gemini_api_key as get_paid_gemini_api_key
from config import MINIMAL_MODEL as MINIMAL_MODEL, LOW_MODEL as LOW_MODEL, MEDIUM_MODEL as MEDIUM_MODEL, HIGH_MODEL as HIGH_MODEL
from api_backoff import call_with_exponential_backoff
from history_cache import CachedContentProfile, HistoryContextCache, compose_uncached_history_prompt, create_cached_content_profile, create_history_context_cache, emit_cache_metric, resolve_history_cached_prompt
import history
import memory as memory

# Memory Retriever file located alongside this module
MEMORY_RETRIEVER_FILE = Path(__file__).parent / "agent_instructions/memory_retriever.md"

# Intent file located alongside this module
INTENT_FILE = Path(__file__).parent / "agent_instructions/intent.md"

# Planner file located alongside this module
PLANNER_FILE = Path(__file__).parent / "agent_instructions/planner.md"

# Execution manager file located alongside this module
EXECUTION_MANAGER_FILE = Path(__file__).parent / "agent_instructions/execution_manager.md"

# Sub-agent file located alongside this module
SUB_AGENT_FILE = Path(__file__).parent / "agent_instructions/sub_agent.md"

# Reviewer file located alongside this module
REVIEWER_FILE = Path(__file__).parent / "agent_instructions/execution_reviewer.md"

# Planner reviewer file located alongside this module
PLANNER_REVIEWER_FILE = Path(__file__).parent / "agent_instructions/planner_reviewer.md"

# Summarize file located alongside this module
TEXTER_FILE = Path(__file__).parent / "agent_instructions/texter.md"

# Tool-specific guidance files used when a tool is explicitly forced
TOOL_INSTRUCTIONS_DIR = Path(__file__).parent / "agent_instructions/tools"

PLANNER_MANAGER_TASK_NAME = "PlannerManager"
EXECUTION_MANAGER_TASK_NAME = "ExecutionManager"
PLANNER_REVIEWER_TASK_NAME = "PlannerReviewer"
REVIEWER_TASK_NAME = "Reviewer"
PLANNER_REVIEWER_STAGE_INSTRUCTION = "Review whether planning has enough information. Print only <ready> if complete; otherwise print missing checks."
REVIEWER_STAGE_INSTRUCTION = "Review whether the user's latest request is complete. Print only <yes> if complete; otherwise print what is missing."
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
    execution_plan_notifier: Callable[..., None] | None = None,
) -> LLMResponse:
    """
    Main function to generate a response from the model based on the user's message,
    conversation history, and optionally multimodal attachments.
    This function handles the entire flow of generating a response, including intent classification, sub-agent execution, and final response generation.
    If execution_plan_notifier is provided, it is called asynchronously with an
    ASCII diagram of the active planner/execution plan whenever a new plan is detected.
    """

    exit_string = ""
    default_temperature = 1.0
    temperature = default_temperature
    attempt_number = 1
    normalized_attachments = attachments.normalize_attachments(image)
    attachment_text, attachment_memory_context = attachments.ingest_attachments_for_memory(
        normalized_attachments,
        history_file,
    )
    if attachment_text:
        if user_message:
            user_message = f"{user_message}\n\nAttachment text:\n{attachment_text}"
        else:
            user_message = f"Attachment text:\n{attachment_text}"

    current_history_text = history.get_conversation_history(history_file=history_file)

    def _append_history_and_update(role: str, text: str) -> None:
        nonlocal current_history_text

        appended_block = history.append_history(role=role, text=text, history_file=history_file)
        if appended_block:
            if current_history_text and current_history_text != "No history available.":
                current_history_text += appended_block
            else:
                current_history_text = appended_block
            return

        # Fallback only if append helper failed to return a block.
        current_history_text = history.get_conversation_history(history_file=history_file)

    _append_history_and_update(role="user", text=user_message)

    retrieval_context = memory.create_retrieval_context()
    retrieval_query = (user_message or "").strip() or current_history_text
    skills_query = (user_message or "").strip() or retrieval_query

    active_history_cache = create_history_context_cache(
        history_file=history_file,
        history_text=current_history_text,
        client=client,
        backoff_call=call_with_exponential_backoff,
    )

    if not job:
        relevant_memories_text = memory.build_relevant_memories_text(
            retrieval_query,
            semantic_limit=5,
            file_limit=3,
            retrieval_context=retrieval_context,
        )
        relevant_skills_text = memory.build_relevant_skills_text(
            skills_query,
            limit=5,
            retrieval_context=retrieval_context,
        )

        intent_system_instructions = INTENT_FILE.read_text(encoding="utf-8") + relevant_memories_text
        if relevant_skills_text:
            intent_system_instructions += "\n\nRelevant reusable planning skills:\n\n" + relevant_skills_text

        intent_response = ""
        try:
            intent_response = _run_model_api(
                text="Classify the latest user request using the conversation context.",
                system_instructions=intent_system_instructions,
                model=LOW_MODEL,
                tool_use_allowed=False,
                force_tool="",
                temperature=default_temperature,
                thinking_level="low",
                history_cache=active_history_cache,
                current_history_text=current_history_text,
            )
        except Exception as e:
            print(f"Error generating intent response: {e}")

        if "<complex>" not in (intent_response or ""):
            _append_history_and_update(role="IntentClassifier", text=intent_response)
            if not job:
                active_history_cache.release()
                post_intent_cache = create_history_context_cache(
                    history_file=history_file,
                    history_text=current_history_text,
                    client=client,
                    backoff_call=call_with_exponential_backoff,
                )
                try:
                    relevant_memories_history = memory.build_relevant_memories_text(
                        retrieval_query,
                        semantic_limit=10,
                        file_limit=4,
                        retrieval_context=retrieval_context,
                    )
                    memory.run_memory_extraction_async(
                        history_file=history_file,
                        temperature=default_temperature,
                        relevant_memories_text=relevant_memories_history,
                        attachment_context_text=attachment_memory_context,
                        history_cache=post_intent_cache.retain(),
                    )
                finally:
                    post_intent_cache.release()
            return LLMResponse(text=intent_response, media_paths=[])

    def _advance_attempt(
        pivot_role: str,
        summarize_after_latest_role: str | None = None,
        summarize_history: bool = True,
    ) -> None:
        nonlocal temperature
        nonlocal attempt_number
        nonlocal active_history_cache
        nonlocal current_history_text

        _refresh_active_history_cache()

        if summarize_history:
            history.run_history_summarization(
                history_file=history_file,
                temperature=default_temperature,
                pivot_role=pivot_role,
                summarize_after_latest_role=summarize_after_latest_role,
                history_cache=active_history_cache.retain(),
                current_history_text=current_history_text,
            )
            current_history_text = history.get_conversation_history(history_file=history_file)
        _refresh_active_history_cache()
        if temperature < 2.0:
            temperature += 0.1
        attempt_number += 1

    def _refresh_active_history_cache() -> None:
        nonlocal active_history_cache

        active_history_cache.release()
        active_history_cache = create_history_context_cache(
            history_file=history_file,
            history_text=current_history_text,
            client=client,
            backoff_call=call_with_exponential_backoff,
        )

    planner_phase_ready = False

    while True:
        exit_string = ""
        planner_exit_string = ""
        planner_order_file = Path(__file__).parent / f"sub-agents/planner_order_{history_file}.json"
        execution_order_file = Path(__file__).parent / f"sub-agents/execution_order_{history_file}.json"

        plan_files_to_clear: list[tuple[Path, str]] = [
            (execution_order_file, "execution order"),
        ]
        if not planner_phase_ready:
            plan_files_to_clear.insert(0, (planner_order_file, "planner order"))

        cleared_plan_files = True
        for plan_file, label in plan_files_to_clear:
            if plan_file.exists():
                try:
                    plan_file.unlink()
                except Exception as e:
                    print(f"Error clearing {label} file before manager run: {e}")
                    cleared_plan_files = False
                    break

        if not cleared_plan_files:
            if planner_phase_ready:
                _advance_attempt(
                    EXECUTION_MANAGER_TASK_NAME,
                    summarize_after_latest_role=PLANNER_REVIEWER_TASK_NAME,
                )
            else:
                _advance_attempt(PLANNER_MANAGER_TASK_NAME, summarize_history=False)
            continue

        if not planner_phase_ready:
            try:
                planner_history_text = current_history_text
                planner_skill_names = memory.build_skill_names_text(
                    query=skills_query,
                    limit=10,
                    retrieval_context=retrieval_context,
                )
                planner_system_instructions = PLANNER_FILE.read_text(encoding="utf-8") + history_file
                if planner_skill_names:
                    planner_system_instructions += "\n\n" + planner_skill_names

                _run_model_api(
                    text="Review the conversation context and create the planner plan JSON file.",
                    system_instructions=planner_system_instructions,
                    model=MEDIUM_MODEL,
                    tool_use_allowed=True,
                    force_tool="run_python",
                    temperature=temperature,
                    thinking_level="high",
                    history_cache=active_history_cache,
                    current_history_text=planner_history_text,
                )
                _append_history_and_update(
                    role=PLANNER_MANAGER_TASK_NAME,
                    text=f"{planner_order_file} created. \nRelevant skills: {planner_skill_names}",
                )
            except Exception as e:
                print(f"Error generating planner manager response: {e}")

            if not planner_order_file.exists():
                print("No planner order file found.")
                _advance_attempt(PLANNER_MANAGER_TASK_NAME, summarize_history=False)
                continue

            try:
                with planner_order_file.open("r", encoding="utf-8") as f:
                    planner_order_dict = json.load(f)
            except Exception as e:
                print(f"Error reading planner order file: {e}")
                _advance_attempt(PLANNER_MANAGER_TASK_NAME, summarize_history=False)
                continue

            planner_plan = _normalize_execution_plan(planner_order_dict, plan_key="planner_plan")
            if not planner_plan:
                print("No valid planner plan found in planner order file.")
                _advance_attempt(PLANNER_MANAGER_TASK_NAME, summarize_history=False)
                continue

            planner_plan = _ensure_final_named_agent(
                planner_plan,
                PLANNER_REVIEWER_TASK_NAME,
                PLANNER_REVIEWER_STAGE_INSTRUCTION,
            )

            _dispatch_execution_plan_preview_async(
                planner_plan,
                history_file,
                execution_plan_notifier,
                attempt_number,
                attempt_number > 1,
                plan_kind="planner",
            )

            planner_exit_string, current_history_text = _run_sub_agent_plan(
                execution_plan=planner_plan,
                history_file=history_file,
                temperature=temperature,
                history_cache=active_history_cache,
                final_agent_task_name=PLANNER_REVIEWER_TASK_NAME,
                final_agent_runner=lambda latest_history_text: _run_planner_reviewer(
                    history_file,
                    user_message,
                    default_temperature,
                    history_cache=active_history_cache,
                    current_history_text=latest_history_text,
                ),
            )

            if "<ready>" not in (planner_exit_string or "").lower():
                print("Planner phase did not report readiness. Retrying planner attempt without summarization.")
                _advance_attempt(PLANNER_MANAGER_TASK_NAME, summarize_history=False)
                continue

            planner_phase_ready = True

            # Planner results are now stable context; rebuild history cache so execution
            # manager/sub-agents can reuse a static cached prefix from this point onward.
            _refresh_active_history_cache()

        try:
            execution_history_text = current_history_text
            execution_manager_system_instructions = EXECUTION_MANAGER_FILE.read_text(encoding="utf-8") + history_file
            execution_manager_prompt = "Use the planner findings in the conversation context to create the execution plan JSON file."

            latest_conversation_summary = _get_latest_history_message_by_role(
                execution_history_text,
                "ConversationSummary",
            )
            if latest_conversation_summary:
                execution_manager_prompt += (
                    "\n\nLatest ConversationSummary feedback to apply when updating the execution plan:\n"
                    + latest_conversation_summary
                )

            latest_reviewer_feedback = _get_latest_history_message_by_role(
                execution_history_text,
                REVIEWER_TASK_NAME,
            )
            if latest_reviewer_feedback:
                execution_manager_prompt += (
                    "\n\nLatest Reviewer feedback to address in this plan:\n"
                    + latest_reviewer_feedback
                )

            _run_model_api(
                text=execution_manager_prompt,
                system_instructions=execution_manager_system_instructions,
                model=MEDIUM_MODEL,
                tool_use_allowed=True,
                force_tool="run_python",
                temperature=temperature,
                thinking_level="high",
                history_cache=active_history_cache,
                current_history_text=execution_history_text,
            )
            _append_history_and_update(
                role=EXECUTION_MANAGER_TASK_NAME,
                text=f"{execution_order_file} created.",
            )
        except Exception as e:
            print(f"Error generating execution manager response: {e}")

        if not execution_order_file.exists():
            print("No execution order file found.")
            _advance_attempt(
                EXECUTION_MANAGER_TASK_NAME,
                summarize_after_latest_role=PLANNER_REVIEWER_TASK_NAME,
            )
            continue

        try:
            with execution_order_file.open("r", encoding="utf-8") as f:
                execution_order_dict = json.load(f)
        except Exception as e:
            print(f"Error reading execution order file: {e}")
            _advance_attempt(
                EXECUTION_MANAGER_TASK_NAME,
                summarize_after_latest_role=PLANNER_REVIEWER_TASK_NAME,
            )
            continue

        execution_plan = _normalize_execution_plan(execution_order_dict, plan_key="execution_plan")
        if not execution_plan:
            print("No valid execution plan found in execution order file.")
            _advance_attempt(
                EXECUTION_MANAGER_TASK_NAME,
                summarize_after_latest_role=PLANNER_REVIEWER_TASK_NAME,
            )
            continue

        execution_plan = _ensure_final_named_agent(
            execution_plan,
            REVIEWER_TASK_NAME,
            REVIEWER_STAGE_INSTRUCTION,
        )

        _dispatch_execution_plan_preview_async(
            execution_plan,
            history_file,
            execution_plan_notifier,
            attempt_number,
            attempt_number > 1,
            plan_kind="execution",
        )

        exit_string, current_history_text = _run_sub_agent_plan(
            execution_plan=execution_plan,
            history_file=history_file,
            temperature=temperature,
            history_cache=active_history_cache,
            final_agent_task_name=REVIEWER_TASK_NAME,
            final_agent_runner=lambda latest_history_text: _run_final_reviewer(
                history_file,
                user_message,
                default_temperature,
                history_cache=active_history_cache,
                current_history_text=latest_history_text,
            ),
        )

        if not exit_string:
            print("Final Reviewer agent did not produce an exit string.")
            _advance_attempt(
                EXECUTION_MANAGER_TASK_NAME,
                summarize_after_latest_role=PLANNER_REVIEWER_TASK_NAME,
            )
            continue
        if "<yes>" in (exit_string or "").lower():
            break

        _advance_attempt(
            EXECUTION_MANAGER_TASK_NAME,
            summarize_after_latest_role=PLANNER_REVIEWER_TASK_NAME,
        )

    text_response = ""
    media_paths: list[str] = []

    active_history_cache.release()
    post_review_cache = create_history_context_cache(
        history_file=history_file,
        history_text=current_history_text,
        client=client,
        backoff_call=call_with_exponential_backoff,
    )
    history.run_history_summarization_async(
        history_file=history_file,
        temperature=default_temperature,
        history_cache=post_review_cache.retain(),
        current_history_text=current_history_text,
    )

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            text_future = executor.submit(
                _run_model_api,
                text="Use the conversation context to answer the user's latest request.",
                system_instructions=TEXTER_FILE.read_text(encoding="utf-8"),
                model=LOW_MODEL,
                tool_use_allowed=False,
                force_tool="",
                temperature=default_temperature,
                thinking_level="low",
                history_cache=post_review_cache,
                current_history_text=post_review_cache.history_text,
            )
            media_future = executor.submit(
                collect_generated_media.select_media_paths,
                history_file,
                user_message,
                default_temperature,
                post_review_cache,
            )

            try:
                text_response = text_future.result()
                _append_history_and_update(role="Texter", text=text_response)
            except Exception as e:
                print(f"Error generating texter response: {e}")

            try:
                media_paths = media_future.result()
                _append_history_and_update(
                    role="MediaSelector",
                    text=json.dumps({"media_paths": media_paths}, ensure_ascii=False),
                )
            except Exception as e:
                print(f"Error selecting media paths: {e}")
                media_paths = []

        if not job:
            relevant_memories_history = memory.build_relevant_memories_text(
                retrieval_query,
                semantic_limit=10,
                file_limit=4,
                retrieval_context=retrieval_context,
            )
            memory.run_memory_extraction_async(
                history_file=history_file,
                temperature=default_temperature,
                relevant_memories_text=relevant_memories_history,
                attachment_context_text=attachment_memory_context,
                history_cache=post_review_cache.retain(),
            )
            memory.run_skill_extraction_async(
                history_file=history_file,
                temperature=default_temperature,
                relevant_memories_text=relevant_memories_history,
                attachment_context_text=attachment_memory_context,
                history_cache=post_review_cache.retain(),
            )
    finally:
        post_review_cache.release()
    return LLMResponse(text=text_response, media_paths=media_paths)


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


def _normalize_execution_plan(execution_order_dict: dict, plan_key: str = "execution_plan") -> list[dict]:
    """Normalize execution order payload into staged execution format.

    Required format:
    - {"<plan_key>": [{"mode": "parallel|serial", "sub_agents": [...]}, ...]}
    - Each stage must explicitly include "mode" as "parallel" or "serial".
    - Each sub-agent should include "thinking_level" (MINIMAL/LOW/MEDIUM/HIGH).
    - Each sub-agent should include "force_tool" as a string tool name; empty string means not forced.
    - Missing or invalid thinking levels are normalized to MEDIUM for backward compatibility.
    """
    normalized_plan: list[dict] = []

    if not isinstance(execution_order_dict, dict):
        return []

    plan_stages = execution_order_dict.get(plan_key)
    if not isinstance(plan_stages, list) and plan_key != "execution_plan":
        plan_stages = execution_order_dict.get("execution_plan")

    if isinstance(plan_stages, list):
        for stage in plan_stages:
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
                force_tool = _normalize_force_tool_name(agent.get("force_tool", ""))
                cleaned_agents.append(
                    {
                        "task_name": task_name,
                        "instruction": instruction,
                        "thinking_level": thinking_level,
                        "force_tool": force_tool,
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


def _has_final_named_agent(execution_plan: list[dict], task_name: str) -> bool:
    if not execution_plan:
        return False

    final_stage = execution_plan[-1]
    if final_stage.get("mode") != "serial":
        return False

    final_agents = final_stage.get("sub_agents")
    if not isinstance(final_agents, list) or len(final_agents) != 1:
        return False

    final_agent = final_agents[0]
    return isinstance(final_agent, dict) and final_agent.get("task_name") == task_name


def _has_final_reviewer_agent(execution_plan: list[dict]) -> bool:
    return _has_final_named_agent(execution_plan, REVIEWER_TASK_NAME)


def _ensure_final_named_agent(execution_plan: list[dict], task_name: str, instruction: str) -> list[dict]:
    if not execution_plan:
        return execution_plan

    plan_without_named_agent: list[dict] = []
    for stage in execution_plan:
        sub_agents = stage.get("sub_agents")
        if not isinstance(sub_agents, list):
            continue

        filtered_sub_agents = [
            agent
            for agent in sub_agents
            if isinstance(agent, dict) and agent.get("task_name") != task_name
        ]
        if not filtered_sub_agents:
            continue

        plan_without_named_agent.append(
            {
                "mode": stage.get("mode"),
                "sub_agents": filtered_sub_agents,
            }
        )

    plan_without_named_agent.append(
        {
            "mode": "serial",
            "sub_agents": [
                {
                    "task_name": task_name,
                    "instruction": instruction,
                    "thinking_level": "LOW",
                    "force_tool": "",
                }
            ],
        }
    )
    return plan_without_named_agent


def _get_latest_history_message_by_role(history_text: str | None, role: str) -> str:
    if not history_text:
        return ""

    target_role = (role or "").strip().lower()
    if not target_role:
        return ""

    try:
        messages = history.parse_history(history_text)
    except Exception:
        return ""

    for message in reversed(messages):
        speaker = (message.get("speaker") or "").strip().lower()
        if speaker == target_role:
            return (message.get("text") or "").strip()

    return ""


def _is_final_plan_agent(
    execution_plan: list[dict],
    stage_index: int,
    agent_index: int,
    final_task_name: str,
) -> bool:
    if not execution_plan:
        return False

    if stage_index != len(execution_plan) - 1:
        return False

    final_stage_agents = execution_plan[-1].get("sub_agents", [])
    if not isinstance(final_stage_agents, list) or not final_stage_agents:
        return False

    if agent_index != len(final_stage_agents) - 1:
        return False

    final_agent = final_stage_agents[agent_index]
    return isinstance(final_agent, dict) and final_agent.get("task_name") == final_task_name


def _run_sub_agent_plan(
    execution_plan: list[dict[str, Any]],
    history_file: str,
    temperature: float,
    history_cache: HistoryContextCache,
    final_agent_task_name: str,
    final_agent_runner: Callable[[str], str],
) -> tuple[str, str]:
    sub_agent_system_instructions = SUB_AGENT_FILE.read_text(encoding="utf-8")
    final_agent_output = ""
    current_history_text = history_cache.history_text

    def _append_history_and_update(role: str, text: str) -> None:
        nonlocal current_history_text

        appended_block = history.append_history(role=role, text=text, history_file=history_file)
        if appended_block:
            if current_history_text and current_history_text != "No history available.":
                current_history_text += appended_block
            else:
                current_history_text = appended_block
            return

        # Fallback only if append helper failed to return a block.
        current_history_text = history.get_conversation_history(history_file=history_file)

    for stage_index, stage in enumerate(execution_plan):
        mode = stage["mode"]
        agents = stage["sub_agents"]

        if mode == "parallel" and len(agents) > 1:
            base_history = current_history_text
            _prewarm_parallel_stage_cache_profiles(
                agents=agents,
                stage_history=base_history,
                sub_agent_system_instructions=sub_agent_system_instructions,
                history_cache=history_cache,
            )

            def _run_parallel_agent(agent: dict[str, Any], stage_history: str) -> str:
                model_name, api_thinking_level, force_tool_name = _resolve_sub_agent_model_config(agent)
                return _run_model_api(
                    text=agent["instruction"],
                    system_instructions=sub_agent_system_instructions,
                    model=model_name,
                    tool_use_allowed=True,
                    force_tool=force_tool_name,
                    temperature=temperature,
                    thinking_level=api_thinking_level,
                    history_cache=history_cache,
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

                    _append_history_and_update(role=agent["task_name"], text=sub_agent_response)
        else:
            for agent_index, agent in enumerate(agents):
                sub_agent_response = ""
                try:
                    if _is_final_plan_agent(execution_plan, stage_index, agent_index, final_agent_task_name):
                        sub_agent_response = final_agent_runner(current_history_text)
                        final_agent_output = sub_agent_response
                    else:
                        model_name, api_thinking_level, force_tool_name = _resolve_sub_agent_model_config(agent)
                        sub_agent_response = _run_model_api(
                            text=agent["instruction"],
                            system_instructions=sub_agent_system_instructions,
                            model=model_name,
                            tool_use_allowed=True,
                            force_tool=force_tool_name,
                            temperature=temperature,
                            thinking_level=api_thinking_level,
                            history_cache=history_cache,
                            current_history_text=current_history_text,
                        )

                    _append_history_and_update(role=agent["task_name"], text=sub_agent_response)
                except Exception as e:
                    print(f"Error generating response for sub-agent '{agent['task_name']}': {e}")

    return final_agent_output, current_history_text


def _run_planner_reviewer(
    history_file: str,
    user_message: str,
    temperature: float,
    history_cache: HistoryContextCache | None = None,
    current_history_text: str | None = None,
) -> str:
    if history_cache is not None:
        return _run_model_api(
            text="Review whether planning has gathered enough information to proceed to execution.",
            system_instructions=PLANNER_REVIEWER_FILE.read_text(encoding="utf-8") + user_message,
            model=LOW_MODEL,
            tool_use_allowed=False,
            force_tool="",
            temperature=temperature,
            thinking_level="low",
            history_cache=history_cache,
            current_history_text=current_history_text,
        )

    return _run_model_api(
        text=history.get_conversation_history(history_file=history_file),
        system_instructions=PLANNER_REVIEWER_FILE.read_text(encoding="utf-8") + user_message,
        model=LOW_MODEL,
        tool_use_allowed=False,
        force_tool="",
        temperature=temperature,
        thinking_level="low",
    )


def _run_final_reviewer(
    history_file: str,
    user_message: str,
    temperature: float,
    history_cache: HistoryContextCache | None = None,
    current_history_text: str | None = None,
) -> str:
    if history_cache is not None:
        return _run_model_api(
            text="Review the conversation context and determine whether the user's request is complete.",
            system_instructions=REVIEWER_FILE.read_text(encoding="utf-8") + user_message,
            model=LOW_MODEL,
            tool_use_allowed=False,
            force_tool="",
            temperature=temperature,
            thinking_level="low",
            history_cache=history_cache,
            current_history_text=current_history_text,
        )

    return _run_model_api(
        text=history.get_conversation_history(history_file=history_file),
        system_instructions=REVIEWER_FILE.read_text(encoding="utf-8") + user_message,
        model=LOW_MODEL,
        tool_use_allowed=False,
        force_tool="",
        temperature=temperature,
        thinking_level="low",
    )


def _normalize_api_thinking_level(raw_level: object) -> str:
    if isinstance(raw_level, str):
        normalized = raw_level.strip().lower()
        if normalized in {"low", "medium", "high"}:
            return normalized
    return "high"


def _normalize_force_tool_name(raw_force_tool: object) -> str:
    if not isinstance(raw_force_tool, str):
        return ""
    normalized = raw_force_tool.strip()
    return normalized if normalized else ""


def _get_forced_tool_instructions(forced_tool_name: str) -> str:
    normalized_tool_name = _normalize_force_tool_name(forced_tool_name)
    if not normalized_tool_name:
        return ""

    guidance_file = TOOL_INSTRUCTIONS_DIR / f"{normalized_tool_name}.md"
    if not guidance_file.exists():
        return ""

    try:
        guidance_text = guidance_file.read_text(encoding="utf-8").strip()
    except Exception as e:
        print(f"Error reading tool guidance file '{guidance_file}': {e}")
        return ""

    if not guidance_text:
        return ""

    return (
        "\n\n"
        f"Tool-specific instructions for forced tool '{normalized_tool_name}':\n"
        f"{guidance_text}"
    )


def _resolve_sub_agent_model_config(agent: dict) -> tuple[str, str, str]:
    plan_thinking_level = _normalize_plan_thinking_level(agent.get("thinking_level"))
    model_name = THINKING_LEVEL_TO_MODEL.get(plan_thinking_level, MEDIUM_MODEL)
    api_thinking_level = THINKING_LEVEL_TO_API_LEVEL.get(plan_thinking_level, "medium")
    force_tool = _normalize_force_tool_name(agent.get("force_tool"))
    return model_name, api_thinking_level, force_tool

def _build_cache_profile_settings(
    system_instructions: str,
    tool_use_allowed: bool,
    force_tool: str,
) -> tuple[str, list[types.Tool] | None, types.ToolConfig | None, str]:
    forced_tool_name = _normalize_force_tool_name(force_tool) if tool_use_allowed else ""
    effective_system_instructions = system_instructions
    if forced_tool_name:
        effective_system_instructions += _get_forced_tool_instructions(forced_tool_name)

    cache_tools = None
    cache_tool_config = None
    if tool_use_allowed:
        agent_tools = types.Tool(function_declarations=_get_function_declarations(client=client))
        cache_tools = [agent_tools]
        if forced_tool_name:
            cache_tool_config = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY",
                    allowed_function_names=[forced_tool_name],
                )
            )

    return effective_system_instructions, cache_tools, cache_tool_config, forced_tool_name


def _prewarm_parallel_stage_cache_profiles(
    agents: list[dict[str, Any]],
    stage_history: str,
    sub_agent_system_instructions: str,
    history_cache: HistoryContextCache,
) -> None:
    unique_profiles: dict[str, tuple[str, CachedContentProfile]] = {}

    for agent in agents:
        try:
            model_name, _, force_tool_name = _resolve_sub_agent_model_config(agent)
            effective_system_instructions, cache_tools, cache_tool_config, _ = _build_cache_profile_settings(
                system_instructions=sub_agent_system_instructions,
                tool_use_allowed=True,
                force_tool=force_tool_name,
            )
            profile = create_cached_content_profile(
                model=model_name,
                system_instruction=effective_system_instructions,
                tools=cache_tools,
                tool_config=cache_tool_config,
            )
            unique_profiles.setdefault(profile.profile_key, (model_name, profile))
        except Exception as e:
            print(f"Error preparing cache prewarm profile for parallel agent '{agent.get('task_name', 'unknown')}': {e}")

    if not unique_profiles:
        return

    emit_cache_metric(
        "parallel_stage_cache_prewarm_start",
        profile_count=len(unique_profiles),
        stage_history_chars=len(stage_history or ""),
    )

    for model_name, profile in unique_profiles.values():
        try:
            history_cache.get_cached_content_entry(
                model=model_name,
                profile=profile,
                current_history_text=stage_history,
            )
        except Exception as e:
            print(f"Error prewarming parallel cache profile '{profile.profile_label}' for model '{model_name}': {e}")

    emit_cache_metric(
        "parallel_stage_cache_prewarm_end",
        profile_count=len(unique_profiles),
    )


def _extract_usage_metadata(response: object) -> dict[str, Any]:
    usage_metadata = getattr(response, "usage_metadata", None)
    if usage_metadata is None:
        return {}

    if isinstance(usage_metadata, dict):
        return usage_metadata

    snapshot: dict[str, Any] = {}
    for field_name in (
        "cached_content_token_count",
        "prompt_token_count",
        "total_token_count",
        "candidates_token_count",
        "input_token_count",
        "output_token_count",
    ):
        field_value = getattr(usage_metadata, field_name, None)
        if field_value is not None:
            snapshot[field_name] = field_value

    if snapshot:
        return snapshot

    to_dict = getattr(usage_metadata, "to_dict", None)
    if callable(to_dict):
        try:
            dict_value = to_dict()
            if isinstance(dict_value, dict):
                return dict_value
        except Exception:
            pass

    return {"raw": str(usage_metadata)}


def _run_model_api(
    text: str,
    system_instructions: str,
    model: str,
    tool_use_allowed: bool = True,
    force_tool: str = "",
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

    effective_system_instructions, cache_tools, cache_tool_config, forced_tool_name = _build_cache_profile_settings(
        system_instructions=system_instructions,
        tool_use_allowed=tool_use_allowed,
        force_tool=force_tool,
    )

    if model == MINIMAL_MODEL and "2.5" in MINIMAL_MODEL:
        thinking_config = types.ThinkingConfig(thinking_budget=24576)
    else:
        thinking_config = types.ThinkingConfig(thinking_level=_normalize_api_thinking_level(thinking_level))

    prompt_text = text
    cached_content_name = None
    cache_profile: CachedContentProfile | None = None
    if history_cache is not None:
        cache_profile = create_cached_content_profile(
            model=model,
            system_instruction=effective_system_instructions,
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
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=bool(forced_tool_name)),
            thinking_config=thinking_config,
            cached_content=cached_content_name,
        )
    else:
        config = types.GenerateContentConfig(
            system_instruction=effective_system_instructions,
            tool_config=cache_tool_config,
            tools=cache_tools or [],
            temperature=temperature,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=bool(forced_tool_name)),
            thinking_config=thinking_config,
            cached_content=None,
        )

    function_output = ""

    emit_cache_metric(
        "generate_content_request",
        model=model,
        cached_content_used=bool(cached_content_name),
        cached_content_name=cached_content_name or "",
        tool_use_allowed=tool_use_allowed,
        forced_tool_name=forced_tool_name,
        prompt_chars=len(prompt_text or ""),
    )

    response = call_with_exponential_backoff(
        lambda: client.models.generate_content(
            model=model,
            config=config,
            contents=prompt_text,
        ),
        description=f"Gemini generate_content ({model})",
    )

    emit_cache_metric(
        "generate_content_response",
        model=model,
        cached_content_used=bool(cached_content_name),
        usage_metadata=_extract_usage_metadata(response),
    )

    # The API may return a candidate whose `content` is None (no function-calling parts).
    # Guard against that case before iterating `parts`.
    parts = None
    if getattr(response, "candidates", None):
        candidate0 = response.candidates[0]
        if candidate0 is not None:
            content = getattr(candidate0, "content", None)
            parts = getattr(content, "parts", None) if content is not None else None

    if parts:
        for part in parts:
            if getattr(part, "function_call", None):
                function_output += _get_skill(part.function_call.name, part.function_call.args)
                    
    output = ""

    if function_output == "":
        output = response.text or ""
    else:
        output = response.text or ""
        output += "\n\n" + function_output
    return output