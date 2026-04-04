from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, Callable, TYPE_CHECKING

import history

if TYPE_CHECKING:
    import discord

DISCORD_MESSAGE_LIMIT = 2000
DISCORD_EMBED_MAX_FIELDS = 25
DISCORD_EMBED_TOTAL_TEXT_LIMIT = 5800
DISCORD_EMBED_FIELD_NAME_LIMIT = 256
DISCORD_EMBED_FIELD_VALUE_LIMIT = 1024
EXECUTION_PLAN_PROGRESS_UPDATE_INTERVAL_SECONDS = 1
THINKING_DOT_SEQUENCE = (".", "..", "...")
EXECUTION_PLAN_WAITING_EMOJI = "⏳"
EXECUTION_PLAN_IN_PROGRESS_EMOJI = "🔄"
EXECUTION_PLAN_COMPLETED_EMOJI = "✅"
EXECUTION_PLAN_EMBED_COLOR_WAITING = 0x95A5A6
EXECUTION_PLAN_EMBED_COLOR_IN_PROGRESS = 0x3498DB
EXECUTION_PLAN_EMBED_COLOR_COMPLETED = 0x2ECC71
EXECUTION_PLAN_EMBED_COLOR_ERROR = 0xE74C3C
PLANNER_PLAN_MESSAGE_HEADER = "Sub-agent planner plan progress"
EXECUTION_PLAN_MESSAGE_HEADER = "Sub-agent execution plan progress"
PLAN_KIND_TO_MESSAGE_HEADER = {
    "planner": PLANNER_PLAN_MESSAGE_HEADER,
    "execution": EXECUTION_PLAN_MESSAGE_HEADER,
}
PLAN_KIND_TO_EMBED_TITLE = {
    "planner": "Planning...",
    "execution": "Executing...",
}
PLAN_KIND_TO_MANAGER_ROLE = {
    "planner": "PlannerManager",
    "execution": "ExecutionManager",
}
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


def build_execution_plan_ascii_diagram(
    execution_plan: list[dict],
    history_file: str,
    plan_kind: str = "execution",
) -> str:
    lines: list[str] = []
    normalized_plan_kind = (plan_kind or "execution").strip().lower() or "execution"
    title = "Planner Plan" if normalized_plan_kind == "planner" else "Execution Plan"
    lines.append(f"+-- {title} ({history_file})")

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
            force_tool = str(agent.get("force_tool", "")).strip() or "none"
            preview = truncate_instruction_preview(instruction)
            lines.append(f"|   |-- Agent {agent_index}: {task_name}")
            lines.append(f"|   |   instruction: {preview}")
            lines.append(f"|   |   thinking_level: {plan_thinking_level}")
            lines.append(f"|   |   force_tool: {force_tool}")

    return "\n".join(lines)


def dispatch_execution_plan_preview_async(
    execution_plan: list[dict],
    history_file: str,
    execution_plan_notifier: Callable[..., None] | None,
    attempt_number: int,
    reset_previous_preview: bool,
    plan_kind: str = "execution",
) -> None:
    if execution_plan_notifier is None or not execution_plan:
        return

    def _worker() -> None:
        try:
            diagram = build_execution_plan_ascii_diagram(execution_plan, history_file, plan_kind=plan_kind)
            if diagram:
                try:
                    execution_plan_notifier(
                        diagram,
                        execution_plan,
                        attempt_number,
                        reset_previous_preview,
                        plan_kind,
                    )
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
        self._execution_plan_started_at: dict[str, float] = {}

    def get_active_tasks(self) -> list[asyncio.Task[None]]:
        return [
            task
            for task in self._execution_plan_progress_tasks.values()
            if not task.done()
        ]

    def clear_state(self) -> None:
        self._execution_plan_progress_tasks.clear()
        self._execution_plan_progress_messages.clear()
        self._execution_plan_started_at.clear()

    def build_execution_plan_notifier(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        channel: discord.abc.Messageable,
        history_file: str,
    ) -> Callable[..., None]:
        def _notifier(
            diagram_text: str,
            execution_plan: list[dict[str, Any]],
            attempt_number: int = 1,
            reset_previous_preview: bool = False,
            plan_kind: str = "execution",
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
                        plan_kind=plan_kind,
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
        plan_kind: str,
    ) -> None:
        _ = reset_previous_preview
        tracker_key = f"{id(channel)}::{history_file}"
        existing_task = self._execution_plan_progress_tasks.get(tracker_key)
        if existing_task is not None and not existing_task.done():
            existing_task.cancel()
            await asyncio.gather(existing_task, return_exceptions=True)

        self._execution_plan_started_at.setdefault(tracker_key, time.monotonic())

        task = asyncio.create_task(
            self._run_execution_plan_progress_tracker(
                channel=channel,
                history_file=history_file,
                execution_plan=execution_plan,
                attempt_number=attempt_number,
                plan_kind=plan_kind,
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
        plan_kind: str,
    ) -> None:
        import discord

        tracker_key = f"{id(channel)}::{history_file}"
        progress_message: discord.Message | None = self._execution_plan_progress_messages.get(tracker_key)
        last_sent_state_signature: str | None = None
        embed_enabled = True
        thinking_step = 0
        started_at = self._execution_plan_started_at.get(tracker_key)
        if started_at is None:
            started_at = time.monotonic()
            self._execution_plan_started_at[tracker_key] = started_at

        while True:
            elapsed_seconds = time.monotonic() - started_at
            (
                message_content,
                all_completed,
                progress_embed,
                state_signature,
            ) = await self._build_execution_plan_progress_message(
                history_file=history_file,
                execution_plan=execution_plan,
                attempt_number=attempt_number,
                plan_kind=plan_kind,
                elapsed_seconds=elapsed_seconds,
                include_embed=embed_enabled,
            )
            thinking_content = self._build_thinking_indicator(thinking_step)
            use_embed_payload = embed_enabled and progress_embed is not None
            effective_message_content = message_content if use_embed_payload else thinking_content
            effective_state_signature = (
                state_signature
                if use_embed_payload
                else self._build_payload_signature(effective_message_content, None)
            )

            try:
                if progress_message is None:
                    progress_message, _, embed_enabled = await self._send_progress_message(
                        channel=channel,
                        history_file=history_file,
                        message_content=effective_message_content,
                        progress_embed=progress_embed,
                        embed_enabled=embed_enabled,
                    )
                    last_sent_state_signature = effective_state_signature
                    self._execution_plan_progress_messages[tracker_key] = progress_message
                elif effective_state_signature != last_sent_state_signature:
                    progress_message, _, embed_enabled = await self._edit_progress_message(
                        channel=channel,
                        history_file=history_file,
                        progress_message=progress_message,
                        message_content=effective_message_content,
                        progress_embed=progress_embed,
                        embed_enabled=embed_enabled,
                    )
                    last_sent_state_signature = effective_state_signature
                    self._execution_plan_progress_messages[tracker_key] = progress_message
            except Exception as exc:
                logging.exception(
                    "Failed to send/edit execution progress message for history '%s': %s",
                    history_file,
                    exc,
                )
                return

            if all_completed:
                return

            thinking_step += 1
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
        self._execution_plan_started_at.pop(tracker_key, None)

        progress_message = self._execution_plan_progress_messages.pop(tracker_key, None)
        if progress_message is not None:
            try:
                await progress_message.delete()
            except Exception:
                pass

    @staticmethod
    def _truncate_text(value: str, limit: int) -> str:
        if limit <= 0:
            return ""
        if len(value) <= limit:
            return value
        if limit <= 3:
            return "." * limit
        return value[: limit - 3] + "..."

    @staticmethod
    def _build_agent_status_emoji(
        key: tuple[int, int],
        completed_keys: set[tuple[int, int]],
        in_progress_keys: set[tuple[int, int]],
    ) -> str:
        if key in completed_keys:
            return EXECUTION_PLAN_COMPLETED_EMOJI
        if key in in_progress_keys:
            return EXECUTION_PLAN_IN_PROGRESS_EMOJI
        return EXECUTION_PLAN_WAITING_EMOJI

    @staticmethod
    def _build_thinking_indicator(step: int) -> str:
        normalized_step = max(0, int(step))
        return THINKING_DOT_SEQUENCE[normalized_step % len(THINKING_DOT_SEQUENCE)]

    @staticmethod
    def _build_payload_signature(
        message_content: str,
        progress_embed: Any | None,
        include_footer: bool = True,
    ) -> str:
        if progress_embed is None:
            return f"text::{message_content}"

        embed_color = int(getattr(progress_embed, "color", 0) or 0)
        embed_title = str(getattr(progress_embed, "title", "") or "")
        embed_description = str(getattr(progress_embed, "description", "") or "")
        footer = ""
        if include_footer:
            footer = getattr(getattr(progress_embed, "footer", None), "text", "") or ""

        fields_signature: list[str] = []
        for field in getattr(progress_embed, "fields", []):
            fields_signature.append(
                f"{field.name}\n{field.value}\n{int(bool(field.inline))}"
            )

        embed_signature = "\n---\n".join(fields_signature)
        return (
            f"embed::{embed_color}\n{embed_title}\n{embed_description}\n{footer}\n"
            f"{embed_signature}\ncontent::{message_content}"
        )

    @staticmethod
    def _format_elapsed_footer_text(elapsed_seconds: float) -> str:
        total_seconds = max(0, int(elapsed_seconds))

        if total_seconds < 60:
            return f"Elapsed: {total_seconds}s"

        total_minutes, seconds = divmod(total_seconds, 60)
        if total_minutes < 60:
            if seconds == 0:
                return f"Elapsed: {total_minutes}m"
            return f"Elapsed: {total_minutes}m {seconds}s"

        hours, minutes = divmod(total_minutes, 60)
        if minutes == 0 and seconds == 0:
            return f"Elapsed: {hours}h"
        if seconds == 0:
            return f"Elapsed: {hours}h {minutes}m"
        return f"Elapsed: {hours}h {minutes}m {seconds}s"

    @staticmethod
    def _compute_execution_plan_progress_state(
        execution_plan: list[dict[str, Any]],
        history_entries: list[dict[str, Any]],
        normalized_plan_kind: str,
    ) -> dict[str, Any]:
        manager_role = PLAN_KIND_TO_MANAGER_ROLE.get(normalized_plan_kind, "manager")

        latest_manager_index = -1
        for index in range(len(history_entries) - 1, -1, -1):
            role = str(history_entries[index].get("speaker") or "").strip()
            if role.casefold() == manager_role.casefold():
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
        if active_stage_index is not None and active_stage_index < len(execution_plan):
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

        all_completed = bool(stage_completion) and all(stage_completion)
        if not execution_plan:
            all_completed = True

        return {
            "completed_keys": completed_keys,
            "in_progress_keys": in_progress_keys,
            "stage_completion": stage_completion,
            "all_completed": all_completed,
        }

    def _build_execution_plan_progress_embed(
        self,
        *,
        history_file: str,
        execution_plan: list[dict[str, Any]],
        attempt_number: int,
        normalized_plan_kind: str,
        elapsed_seconds: float,
        completed_keys: set[tuple[int, int]],
        in_progress_keys: set[tuple[int, int]],
        stage_completion: list[bool],
        all_completed: bool,
        has_error: bool = False,
    ) -> Any | None:
        import discord

        if has_error:
            embed_color = EXECUTION_PLAN_EMBED_COLOR_ERROR
        elif all_completed:
            embed_color = EXECUTION_PLAN_EMBED_COLOR_COMPLETED
        elif in_progress_keys:
            embed_color = EXECUTION_PLAN_EMBED_COLOR_IN_PROGRESS
        else:
            embed_color = EXECUTION_PLAN_EMBED_COLOR_WAITING

        completed_stage_count = sum(1 for done in stage_completion if done)
        total_stage_count = len(execution_plan)
        embed_title = PLAN_KIND_TO_EMBED_TITLE.get(normalized_plan_kind, "Executing...")

        embed = discord.Embed(
            title=embed_title,
            description=(
                f"Attempt {max(1, int(attempt_number))} - "
                f"Stages complete: {completed_stage_count}/{total_stage_count}"
            ),
            color=embed_color,
        )
        embed.set_footer(text=self._format_elapsed_footer_text(elapsed_seconds))

        used_chars = len(embed.title or "") + len(embed.description or "")
        omitted_stages = 0

        for stage_index, stage in enumerate(execution_plan, start=1):
            if len(embed.fields) >= DISCORD_EMBED_MAX_FIELDS:
                omitted_stages = len(execution_plan) - stage_index + 1
                break

            mode = str(stage.get("mode", "serial"))
            sub_agents = stage.get("sub_agents", []) if isinstance(stage.get("sub_agents"), list) else []
            field_name = self._truncate_text(
                f"Stage {stage_index} [{mode}]",
                DISCORD_EMBED_FIELD_NAME_LIMIT,
            )

            stage_lines: list[str] = []
            for agent_index, agent in enumerate(sub_agents, start=1):
                if not isinstance(agent, dict):
                    continue

                key = (stage_index - 1, agent_index - 1)
                emoji = self._build_agent_status_emoji(key, completed_keys, in_progress_keys)
                task_name = str(agent.get("task_name", "unnamed_task"))
                thinking_level = str(agent.get("thinking_level", "MEDIUM")).strip().upper()
                thinking_emoji = THINKING_LEVEL_TO_EMOJI.get(thinking_level, THINKING_LEVEL_TO_EMOJI["MEDIUM"])
                if key in in_progress_keys:
                    stage_lines.append(f"{emoji} **{task_name}** {thinking_emoji}")
                else:
                    stage_lines.append(f"{emoji} {task_name} {thinking_emoji}")

                if key in in_progress_keys:
                    instruction_preview = truncate_instruction_preview(str(agent.get("instruction", "")))
                    stage_lines.append(f"instruction: {instruction_preview}")

            if not stage_lines:
                stage_lines.append("No sub-agents.")

            available_chars = min(
                DISCORD_EMBED_FIELD_VALUE_LIMIT,
                max(0, DISCORD_EMBED_TOTAL_TEXT_LIMIT - used_chars),
            )
            if available_chars <= 0:
                omitted_stages = len(execution_plan) - stage_index + 1
                break

            field_value = "\n".join(stage_lines)
            field_value = self._truncate_text(field_value, available_chars)
            embed.add_field(name=field_name, value=field_value, inline=False)
            used_chars += len(field_name) + len(field_value)

        if omitted_stages > 0 and len(embed.fields) < DISCORD_EMBED_MAX_FIELDS:
            omitted_value = self._truncate_text(
                f"{omitted_stages} additional stage(s) omitted due to embed size limits.",
                DISCORD_EMBED_FIELD_VALUE_LIMIT,
            )
            embed.add_field(name="Additional stages", value=omitted_value, inline=False)

        return embed

    async def _send_progress_message(
        self,
        *,
        channel: discord.abc.Messageable,
        history_file: str,
        message_content: str,
        progress_embed: Any | None,
        embed_enabled: bool,
    ) -> tuple[discord.Message, str, bool]:
        import discord

        if embed_enabled and progress_embed is not None:
            try:
                message = await channel.send(embed=progress_embed)
                return message, self._build_payload_signature("", progress_embed), True
            except Exception as embed_exc:
                logging.warning(
                    "Failed sending execution progress embed for history '%s'; falling back to text: %s",
                    history_file,
                    embed_exc,
                )

        fallback_content = message_content or self._build_thinking_indicator(0)
        message = await channel.send(fallback_content)
        return message, self._build_payload_signature(fallback_content, None), False

    async def _edit_progress_message(
        self,
        *,
        channel: discord.abc.Messageable,
        history_file: str,
        progress_message: discord.Message,
        message_content: str,
        progress_embed: Any | None,
        embed_enabled: bool,
    ) -> tuple[discord.Message, str, bool]:
        import discord

        if embed_enabled and progress_embed is not None:
            try:
                await progress_message.edit(content=None, embed=progress_embed)
                return progress_message, self._build_payload_signature("", progress_embed), True
            except discord.NotFound:
                return await self._send_progress_message(
                    channel=channel,
                    history_file=history_file,
                    message_content=message_content,
                    progress_embed=progress_embed,
                    embed_enabled=embed_enabled,
                )
            except Exception as embed_exc:
                logging.warning(
                    "Failed editing execution progress embed for history '%s'; falling back to text: %s",
                    history_file,
                    embed_exc,
                )

        try:
            fallback_content = message_content or self._build_thinking_indicator(0)
            await progress_message.edit(content=fallback_content, embed=None)
            return progress_message, self._build_payload_signature(fallback_content, None), False
        except discord.NotFound:
            return await self._send_progress_message(
                channel=channel,
                history_file=history_file,
                message_content=message_content,
                progress_embed=None,
                embed_enabled=False,
            )

    async def _build_execution_plan_progress_message(
        self,
        history_file: str,
        execution_plan: list[dict[str, Any]],
        attempt_number: int,
        plan_kind: str,
        elapsed_seconds: float,
        include_embed: bool = True,
    ) -> tuple[str, bool, Any | None, str]:
        history_entries = await asyncio.to_thread(history.parse_history_file, history_file)

        normalized_plan_kind = (plan_kind or "execution").strip().lower() or "execution"
        progress_state = self._compute_execution_plan_progress_state(
            execution_plan=execution_plan,
            history_entries=history_entries,
            normalized_plan_kind=normalized_plan_kind,
        )
        completed_keys: set[tuple[int, int]] = progress_state["completed_keys"]
        in_progress_keys: set[tuple[int, int]] = progress_state["in_progress_keys"]
        stage_completion: list[bool] = progress_state["stage_completion"]
        all_completed: bool = progress_state["all_completed"]
        message_content = ""

        progress_embed = None
        if include_embed:
            try:
                progress_embed = self._build_execution_plan_progress_embed(
                    history_file=history_file,
                    execution_plan=execution_plan,
                    attempt_number=attempt_number,
                    normalized_plan_kind=normalized_plan_kind,
                    elapsed_seconds=elapsed_seconds,
                    completed_keys=completed_keys,
                    in_progress_keys=in_progress_keys,
                    stage_completion=stage_completion,
                    all_completed=all_completed,
                )
            except Exception as embed_exc:
                logging.warning(
                    "Failed building execution progress embed for history '%s'; using text fallback: %s",
                    history_file,
                    embed_exc,
                )

        state_signature = self._build_payload_signature(
            message_content,
            progress_embed,
            include_footer=False,
        )

        return message_content, all_completed, progress_embed, state_signature
