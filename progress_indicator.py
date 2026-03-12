from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Callable, TYPE_CHECKING

import history

if TYPE_CHECKING:
    import discord

DISCORD_MESSAGE_LIMIT = 2000
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
SUB_AGENT_INSTRUCTION_PREVIEW_CHARS = 200
VALID_PLAN_THINKING_LEVELS = {"MINIMAL", "LOW", "MEDIUM", "HIGH"}


def normalize_plan_thinking_level(raw_level: object) -> str:
    if isinstance(raw_level, str):
        normalized = raw_level.strip().upper()
        if normalized in VALID_PLAN_THINKING_LEVELS:
            return normalized
    return "MEDIUM"


def truncate_instruction_preview(instruction: str, limit: int = SUB_AGENT_INSTRUCTION_PREVIEW_CHARS) -> str:
    compact = " ".join((instruction or "").split())
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "..."


def build_execution_plan_ascii_diagram(execution_plan: list[dict], history_file: str) -> str:
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
            plan_thinking_level = normalize_plan_thinking_level(agent.get("thinking_level"))
            preview = truncate_instruction_preview(instruction)
            lines.append(f"|   |-- Agent {agent_index}: {task_name}")
            lines.append(f"|   |   instruction: {preview}")
            lines.append(f"|   |   thinking_level: {plan_thinking_level}")

    return "\n".join(lines)


def dispatch_execution_plan_preview_async(
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
            diagram = build_execution_plan_ascii_diagram(execution_plan, history_file)
            if diagram:
                try:
                    execution_plan_notifier(diagram, execution_plan, attempt_number, reset_previous_preview)
                except TypeError:
                    execution_plan_notifier(diagram, execution_plan)  # type: ignore[misc, call-arg]
        except Exception as e:
            print(f"Error dispatching execution plan preview: {e}")

    threading.Thread(target=_worker, daemon=True).start()


class ExecutionPlanProgressIndicator:
    def __init__(self, message_limit: int = DISCORD_MESSAGE_LIMIT) -> None:
        self._message_limit = message_limit
        self._execution_plan_progress_tasks: dict[str, asyncio.Task[None]] = {}
        self._execution_plan_progress_messages: dict[str, discord.Message] = {}

    def get_active_tasks(self) -> list[asyncio.Task[None]]:
        return [
            task
            for task in self._execution_plan_progress_tasks.values()
            if not task.done()
        ]

    def clear_state(self) -> None:
        self._execution_plan_progress_tasks.clear()
        self._execution_plan_progress_messages.clear()

    def build_execution_plan_notifier(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        channel: discord.abc.Messageable,
        history_file: str,
    ) -> Callable[[str, list[dict[str, Any]], int, bool], None]:
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
        _ = reset_previous_preview
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
        import discord

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

    async def clear_execution_plan_progress_message(
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
                active_sub_agents = (
                    active_stage.get("sub_agents", [])
                    if isinstance(active_stage.get("sub_agents"), list)
                    else []
                )
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
        if len(wrapped_content) > self._message_limit:
            max_body_length = self._message_limit - len("```\n\n```") - 3
            truncated_body = text_body[:max_body_length] + "..."
            wrapped_content = f"```\n{truncated_body}\n```"

        all_completed = bool(stage_completion) and all(stage_completion)
        if not execution_plan:
            all_completed = True

        return wrapped_content, all_completed
