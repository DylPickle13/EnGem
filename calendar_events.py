from __future__ import annotations

import json
import sys
import threading
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Callable

from tools.access_google_workspace import _run_gws_command


def _sync_check_active_events(
	calendar_id: str = "primary",
	*,
	lookback_hours: int = 1,
	lookahead_hours: int = 1,
	timeout_seconds: int = 60,
	retries: int = 2,
) -> list[dict[str, Any]]:
	"""Return a list of active event objects (synchronous).

	This helper performs a single gws CLI query and does not print.
	"""

	def _to_rfc3339_utc(value: datetime) -> str:
		normalized = value.astimezone(timezone.utc).replace(microsecond=0)
		return normalized.isoformat().replace("+00:00", "Z")

	def _parse_event_time(value: str) -> datetime | None:
		if not value:
			return None
		candidate = value.strip()
		if candidate.endswith("Z"):
			candidate = candidate[:-1] + "+00:00"
		try:
			parsed = datetime.fromisoformat(candidate)
		except Exception:
			return None
		if parsed.tzinfo is None:
			parsed = parsed.replace(tzinfo=timezone.utc)
		return parsed.astimezone(timezone.utc)

	def _event_time(event: dict[str, Any], key: str) -> datetime | None:
		boundary = event.get(key)
		if not isinstance(boundary, dict):
			return None
		date_time = boundary.get("dateTime")
		if isinstance(date_time, str):
			return _parse_event_time(date_time)
		date_value = boundary.get("date")
		if isinstance(date_value, str):
			try:
				parsed = datetime.fromisoformat(date_value)
				return parsed.replace(tzinfo=timezone.utc)
			except Exception:
				return None
		return None

	now_utc = datetime.now(timezone.utc)
	params = {
		"calendarId": calendar_id,
		"timeMin": _to_rfc3339_utc(now_utc - timedelta(hours=lookback_hours)),
		"timeMax": _to_rfc3339_utc(now_utc + timedelta(hours=lookahead_hours)),
		"singleEvents": True,
		"orderBy": "startTime",
	}

	# Attempt the gws call with retries and exponential backoff.
	attempt = 0
	max_attempts = max(1, int(retries) + 1)
	result = None
	last_details = ""
	while attempt < max_attempts:
		result = _run_gws_command(
			["calendar", "events", "list", "--params", json.dumps(params), "--format", "json"],
			runtime_data={},
			timeout=timeout_seconds,
		)
		if result.get("ok"):
			break

		details = (result.get("stderr") or result.get("stdout") or "").strip()
		last_details = details
		logging.warning(
			"gws command failed (attempt %d/%d): %s; command: %s",
			attempt + 1,
			max_attempts,
			details or "No output",
			result.get("command"),
		)

		attempt += 1
		if attempt < max_attempts:
			sleep_seconds = min(2 ** attempt, 8)
			time.sleep(sleep_seconds)

	if not result or not result.get("ok"):
		details = last_details or ""
		raise RuntimeError(f"gws command failed: {details}")

	stdout = str(result.get("stdout", "")).strip()
	if not stdout:
		return []

	try:
		parsed = json.loads(stdout)
	except json.JSONDecodeError:
		raise RuntimeError("gws returned non-JSON output")

	items = parsed.get("items") if isinstance(parsed, dict) else []
	active: list[dict[str, Any]] = []
	for ev in items or []:
		start = _event_time(ev, "start")
		end = _event_time(ev, "end")
		if start is None and end is None:
			continue
		if start is None:
			is_active = now_utc < end
		elif end is None:
			is_active = start <= now_utc
		else:
			is_active = start <= now_utc < end
		if is_active:
			active.append(ev)

	return active


def check_active_events(
	calendar_id: str = "primary",
	poll_interval_seconds: float = 5.0,
	*,
	lookback_hours: int = 1,
	lookahead_hours: int = 1,
	timeout_seconds: int = 60,
	gws_retries: int = 2,
	callback: Callable[[list[dict[str, Any]]], None] | None = None,
	event_processor: Callable[[dict[str, Any]], bool] | None = None,
	stop_event: threading.Event | None = None,
	daemon: bool = True,
) -> tuple[threading.Thread, threading.Event]:
	"""Start a background thread that checks for active events every interval.

	The thread invokes `callback(results)` when a check completes (if provided).
	Returns (thread, stop_event).
	"""
	stop_signal = stop_event or threading.Event()

	def _worker() -> None:
		while not stop_signal.is_set():
			try:
				active_events = _sync_check_active_events(
					calendar_id, lookback_hours=lookback_hours, lookahead_hours=lookahead_hours, timeout_seconds=timeout_seconds, retries=gws_retries
				)
				if callback:
					try:
						callback(active_events)
					except Exception as cb_exc:
						logging.exception("Callback error while processing active events: %s", cb_exc)

				# Process and delete each active event
				for ev in active_events:
					try:
						processed = False
						if event_processor is not None:
							processed = bool(event_processor(ev))
						else:
							# TODO: process the event here (e.g., save data, notify services, etc.)
							logging.info("No event processor configured for event: %s", ev.get("id"))

						if not processed:
							continue

						event_id = ev.get("id")
						if not event_id:
							logging.warning("Skipping event without id: %s", ev)
							continue
						params = {"calendarId": calendar_id, "eventId": event_id}
						delete_result = _run_gws_command(
							["calendar", "events", "delete", "--params", json.dumps(params)], runtime_data={}, timeout=timeout_seconds
						)
						if not delete_result.get("ok"):
							logging.warning(
								"Failed to delete event %s: %s",
								event_id,
								(delete_result.get("stderr") or delete_result.get("stdout") or "").strip(),
							)
					except Exception:
						logging.exception("Error processing/deleting event: %s", ev)
			except Exception as exc:
				logging.exception("Error checking calendar: %s", exc)
			stop_signal.wait(poll_interval_seconds)

	thread = threading.Thread(target=_worker, name=f"check-active-events:{calendar_id}", daemon=daemon)
	thread.start()
	return thread, stop_signal


if __name__ == "__main__":
	cal = sys.argv[1] if len(sys.argv) > 1 else "primary"
	thread, stop_event = check_active_events(cal, poll_interval_seconds=5.0, daemon=False)
	try:
		while thread.is_alive():
			thread.join(1)
	except KeyboardInterrupt:
		pass
	finally:
		stop_event.set()
		thread.join(5)