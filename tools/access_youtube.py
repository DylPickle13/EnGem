from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import MINIMAL_MODEL as MINIMAL_MODEL, MEDIUM_MODEL as MEDIUM_MODEL

try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
except Exception as e:  # pragma: no cover - helpful import-time error
    raise ImportError(
        "Missing Google API dependencies. Install with:\n"
        "pip install google-api-python-client google-auth-oauthlib google-auth-httplib2"
    ) from e

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]
SMOKE_TEST_QUERIES: list[tuple[str, str]] = [
    (
        "search via api",
        "Call search.list with params part=snippet, q='Python automation', type=video, maxResults=3.",
    ),
    (
        "video details via api",
        "Call videos.list with params part=snippet,contentDetails,statistics and id=dQw4w9WgXcQ.",
    ),
    (
        "my channels via api",
        "Call channels.list with params part=snippet,contentDetails,statistics, mine=true, maxResults=10.",
    ),
]


def _get_credentials(client_secrets_file: Optional[str] = None, credentials_file: str = "youtube_token.json", scopes=SCOPES) -> Credentials:
    creds: Optional[Credentials] = None
    if credentials_file and os.path.exists(credentials_file):
        try:
            creds = Credentials.from_authorized_user_file(credentials_file, scopes)
        except Exception:
            creds = None

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        if credentials_file:
            with open(credentials_file, "w") as f:
                f.write(creds.to_json())
        return creds

    if not client_secrets_file:
        raise ValueError("No valid credentials found and no client_secrets_file provided to run OAuth flow.")

    flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, scopes)
    creds = flow.run_local_server(port=0)
    if credentials_file:
        with open(credentials_file, "w") as f:
            f.write(creds.to_json())
    return creds


def _build_service(client_secrets_file: Optional[str] = None, credentials_file: Optional[str] = None):
    """Return an authenticated YouTube API service object using OAuth credentials.

    The function will try to load existing credentials from `credentials_file`.
    If none are present or valid, it will run the OAuth flow using `client_secrets_file`.
    """
    creds = _get_credentials(client_secrets_file, credentials_file or "youtube_token.json")
    return build("youtube", "v3", credentials=creds)


def _coerce_max_results(value: Any, default: int = 5, minimum: int = 1, maximum: int = 50) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(minimum, min(parsed, maximum))


# --- Plain-text query helpers ------------------------------------


def _json_dumps(value: Any) -> str:
    def _json_default(obj: Any):
        if isinstance(obj, bytes):
            return {
                "type": "bytes_base64",
                "value": base64.b64encode(obj).decode("ascii"),
            }
        return str(obj)

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=_json_default)


def _parse_json_response(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def _extract_first_json_object(text: str) -> dict | None:
    stripped = (text or "").strip()
    parsed = _parse_json_response(stripped)
    if isinstance(parsed, dict):
        return parsed

    start = stripped.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = stripped[start : index + 1]
                parsed = _parse_json_response(candidate)
                if isinstance(parsed, dict):
                    return parsed
                return None

    return None


def _format_output(output: str, *, pretty: bool) -> str:
    if not pretty:
        return output
    parsed = _parse_json_response(output)
    if parsed is None:
        return output
    return json.dumps(parsed, indent=2, ensure_ascii=False)


def _error_payload(message: str, **extra: Any) -> str:
    payload = {"ok": False, "error": message}
    payload.update(extra)
    return _json_dumps(payload)


def _plan_query_payload(query: str, *, feedback: list[dict[str, Any]] | None = None) -> tuple[dict | None, str]:
    import llm

    system_instructions = (
        "You are a YouTube API planner. Given a user request, return a single JSON object describing the action to take. "
        "Allowed actions: 'api' only. "
        "Return only one JSON object using these fields: action, "
        "resource, method, params, body, media_file, media_mime, page_all, max_pages. "
        "The JSON object MUST be the first and only JSON object in your response. Examples:\n"
        "{\"action\":\"api\",\"resource\":\"search\",\"method\":\"list\",\"params\":{\"part\":\"snippet\",\"q\":\"funny cats\",\"type\":\"video\",\"maxResults\":5}}\n"
        "{\"action\":\"api\",\"resource\":\"channels\",\"method\":\"list\",\"params\":{\"part\":\"snippet,contentDetails,statistics\",\"mine\":true,\"maxResults\":10}}\n"
        "{\"action\":\"api\",\"resource\":\"playlists\",\"method\":\"list\",\"params\":{\"part\":\"snippet,contentDetails\",\"mine\":true,\"maxResults\":10}}\n"
        "Map every request to the correct YouTube Data API resource and method."
    )

    prompt_parts = [f"User request:\n{query}"]
    if feedback:
        prompt_parts.append(
            "Previous planner feedback JSON. Repair your next payload based on these validation/execution issues:\n"
            f"{_json_dumps(feedback)}"
        )

    planner_output = llm._run_model_api(
        text="\n\n".join(prompt_parts),
        system_instructions=system_instructions,
        model=MINIMAL_MODEL,
        tool_use_allowed=False,
        force_tool="",
        temperature=0.2,
        thinking_level="low",
    )
    return _extract_first_json_object(planner_output), planner_output


def _validate_planned_payload(payload: dict) -> str | None:
    action = str(payload.get("action", "")).strip().lower()
    if action != "api":
        return "Unsupported or missing action. Use: api."

    resource = payload.get("resource")
    method = payload.get("method")
    if not str(resource or "").strip():
        return "api action requires `resource` (for example: videos, playlists, channels)."
    if not str(method or "").strip():
        return "api action requires `method` (for example: list, insert, update, delete)."
    params = payload.get("params")
    if params is not None and not isinstance(params, dict):
        return "api action `params` must be an object when provided."
    body = payload.get("body")
    if body is not None and not isinstance(body, dict):
        return "api action `body` must be an object when provided."
    media_file = payload.get("media_file")
    if media_file is not None and not str(media_file).strip():
        return "api action `media_file` must be a non-empty string when provided."
    return None


def _payload_to_cli_query(payload: dict[str, Any]) -> str | None:
    action = str(payload.get("action", "")).strip().lower()
    if action == "api":
        resource = str(payload.get("resource", "")).strip()
        method = str(payload.get("method", "")).strip()
        if not resource or not method:
            return None
        parts = [f"youtube.{resource}.{method}"]
        params = payload.get("params")
        if isinstance(params, dict) and params:
            parts.extend(["--params", _json_dumps(params)])
        body = payload.get("body")
        if isinstance(body, dict) and body:
            parts.extend(["--json", _json_dumps(body)])
        media_file = payload.get("media_file")
        if str(media_file or "").strip():
            parts.extend(["--media-file", str(media_file)])
        return shlex.join(parts)

    return None


def _extract_cli_queries_from_response(raw_response: dict[str, Any]) -> list[str]:
    query = _payload_to_cli_query(raw_response.get("planned_payload", {})) if isinstance(raw_response.get("planned_payload"), dict) else None
    return [query] if query else []


def _execute_generic_api_action(service, payload: dict) -> dict[str, Any]:
    resource = str(payload.get("resource", "")).strip()
    method = str(payload.get("method", "")).strip()
    if not resource or not method:
        raise ValueError("api action requires non-empty `resource` and `method`.")

    if not resource.replace("_", "").isalnum() or not method.replace("_", "").isalnum():
        raise ValueError("api action resource/method must be alphanumeric identifiers.")

    params = payload.get("params")
    if params is None:
        request_kwargs: dict[str, Any] = {}
    elif isinstance(params, dict):
        request_kwargs = dict(params)
    else:
        raise ValueError("api action `params` must be an object.")

    body = payload.get("body")
    if body is not None:
        if not isinstance(body, dict):
            raise ValueError("api action `body` must be an object.")
        request_kwargs["body"] = body

    media_file = payload.get("media_file")
    if str(media_file or "").strip():
        media_path = Path(str(media_file)).expanduser()
        if not media_path.exists():
            raise FileNotFoundError(f"media file not found: {media_path}")
        media_mime = str(payload.get("media_mime") or "").strip() or None
        request_kwargs["media_body"] = MediaFileUpload(str(media_path.resolve()), mimetype=media_mime, resumable=True)

    resource_factory = getattr(service, resource, None)
    if not callable(resource_factory):
        raise ValueError(f"Unknown YouTube resource: {resource}")
    resource_handle = resource_factory()

    method_callable = getattr(resource_handle, method, None)
    if not callable(method_callable):
        raise ValueError(f"Unsupported method '{method}' for resource '{resource}'")

    page_all = bool(payload.get("page_all", False))
    max_pages = _coerce_max_results(payload.get("max_pages", 3), default=3, minimum=1, maximum=20)

    if page_all:
        responses: list[Any] = []
        all_items: list[Any] = []
        current_kwargs = dict(request_kwargs)
        for _ in range(max_pages):
            response = method_callable(**current_kwargs).execute()
            responses.append(response)
            if isinstance(response, dict):
                items = response.get("items")
                if isinstance(items, list):
                    all_items.extend(items)
                next_token = response.get("nextPageToken")
                if not next_token:
                    break
                current_kwargs["pageToken"] = next_token
                continue
            break
        return {
            "resource": resource,
            "method": method,
            "page_all": True,
            "pages": len(responses),
            "items": all_items,
            "responses": responses,
        }

    request = method_callable(**request_kwargs)
    # HttpRequest always exposes next_chunk(), but it is only valid for resumable uploads/downloads.
    resumable = getattr(request, "resumable", None)
    if resumable is not None and hasattr(request, "next_chunk"):
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                logger.info("Request progress: %s%%", int(status.progress() * 100))
        return {
            "resource": resource,
            "method": method,
            "response": response,
        }

    response = request.execute()
    return {
        "resource": resource,
        "method": method,
        "response": response,
    }


def _execute_structured_payload(payload: dict, *, runtime_data: dict[str, Any]) -> str:
    action = str(payload.get("action", "")).strip().lower()
    if action == "api":
        client_secrets = runtime_data.get("client_secrets") or payload.get("client_secrets")
        creds_file = runtime_data.get("credentials_file") or payload.get("credentials_file") or "youtube_token.json"
        try:
            service = _build_service(client_secrets_file=client_secrets, credentials_file=creds_file)
        except Exception as exc:
            return _error_payload("api action requires OAuth client secrets via `--client-secrets` or a valid credentials file.", error=str(exc))
        try:
            result = _execute_generic_api_action(service, payload)
        except Exception as exc:
            return _error_payload("api action failed", error=str(exc))
        return _json_dumps({"ok": True, "result": result})

    return _error_payload(f"Unsupported action: {action}")


def _interpret_query_response(query: str, raw_response: str) -> str:
    import llm

    system_instructions = (
        "You are a YouTube API response interpreter. Given the original user request and a raw JSON tool response, "
        "return only the information needed to satisfy the original request. If the action failed, briefly explain the failure and include the most relevant error detail."
    )
    prompt = f"Original user request:\n{query}\n\nRaw tool response JSON:\n{raw_response}\n\nReturn a concise final answer for the user."
    return llm._run_model_api(
        text=prompt,
        system_instructions=system_instructions,
        model=MINIMAL_MODEL,
        tool_use_allowed=False,
        force_tool="",
        temperature=0.2,
        thinking_level="low",
    )


def _run_query_action(query: str, *, runtime_data: dict[str, Any]) -> str:
    normalized_query = (query or "").strip()
    if not normalized_query:
        return _error_payload("access_youtube requires a plain-text query.", executed=False)

    max_attempts = runtime_data.get("max_planner_attempts", runtime_data.get("planner_attempts", 4))
    try:
        max_attempts = max(1, min(int(max_attempts), 5))
    except Exception:
        max_attempts = 4

    planner_feedback: list[dict[str, Any]] = []
    last_payload: dict[str, Any] | None = None
    last_planner_output = ""

    for attempt in range(1, max_attempts + 1):
        planned_payload, planner_output = _plan_query_payload(normalized_query, feedback=planner_feedback)
        last_planner_output = planner_output

        if not isinstance(planned_payload, dict):
            planner_feedback.append(
                {
                    "attempt": attempt,
                    "error": "Planner failed to produce a valid JSON payload.",
                    "planner_output": planner_output,
                }
            )
            continue

        last_payload = planned_payload
        validation_error = _validate_planned_payload(planned_payload)
        if validation_error:
            planner_feedback.append(
                {
                    "attempt": attempt,
                    "error": validation_error,
                    "planned_payload": planned_payload,
                }
            )
            continue

        execution_text = _execute_structured_payload(planned_payload, runtime_data=runtime_data)
        parsed_result = _parse_json_response(execution_text)
        execution_result: Any = parsed_result if parsed_result is not None else execution_text
        ok = bool(not (isinstance(execution_result, dict) and execution_result.get("ok") is False))

        if not ok and attempt < max_attempts:
            planner_feedback.append(
                {
                    "attempt": attempt,
                    "error": "Execution failed. Repair the payload using this feedback.",
                    "planned_payload": planned_payload,
                    "execution_result": execution_result,
                }
            )
            continue

        return _json_dumps(
            {
                "ok": ok,
                "executed": True,
                "query": normalized_query,
                "planner_attempts": attempt,
                "planned_payload": planned_payload,
                "planned_cli_query": _payload_to_cli_query(planned_payload),
                "result": execution_result,
            }
        )

    return _json_dumps(
        {
            "ok": False,
            "executed": False,
            "query": normalized_query,
            "planner_attempts": max_attempts,
            "planned_payload": last_payload,
            "planned_cli_query": _payload_to_cli_query(last_payload) if isinstance(last_payload, dict) else None,
            "planner_output": last_planner_output,
            "feedback": planner_feedback,
        }
    )


def access_youtube(query: str = "") -> str:
    """
    Plain-text YouTube entry point backed by the YouTube Data API v3.

    Examples:
    - Call search.list with params part=snippet, q="python automation", type=video, maxResults=5.
    - Call videos.list with params part=snippet,contentDetails,statistics and id=dQw4w9WgXcQ.
    - Call channels.list with params part=snippet,contentDetails,statistics, mine=true, maxResults=10.
    - Call playlists.list with params part=snippet,contentDetails, mine=true, maxResults=15.
    """
    output = _run_query_action(query, runtime_data={})
    parsed_output = _parse_json_response(output)

    prefix_lines: list[str] = []
    if isinstance(parsed_output, dict):
        attempts = parsed_output.get("planner_attempts")
        if attempts is not None:
            prefix_lines.append(f"Planner attempts: {attempts}")

        cli_queries = _extract_cli_queries_from_response(parsed_output)
        if len(cli_queries) == 1:
            prefix_lines.append(f"CLI query: {cli_queries[0]}")
        elif len(cli_queries) > 1:
            prefix_lines.append("CLI queries:")
            prefix_lines.extend(f"{index}. {cli_query}" for index, cli_query in enumerate(cli_queries, start=1))

    try:
        interpreted = _interpret_query_response(query, output)
    except Exception:
        interpreted = output

    if prefix_lines:
        return "\n".join([*prefix_lines, "", interpreted])
    return interpreted


def _build_runtime_data(args: argparse.Namespace) -> dict[str, Any]:
    runtime_data: dict[str, Any] = {}
    if getattr(args, "client_secrets", None):
        runtime_data["client_secrets"] = args.client_secrets
    if getattr(args, "credentials_file", None):
        runtime_data["credentials_file"] = args.credentials_file
    return runtime_data


def _run_smoke_suite(*, pretty: bool, runtime_data: dict[str, Any]) -> int:
    failures = 0
    for index, (label, query) in enumerate(SMOKE_TEST_QUERIES, start=1):
        output = _run_query_action(query, runtime_data=runtime_data)
        print(f"=== Smoke Test {index}: {label} ===")
        print(query)
        print(_format_output(output, pretty=pretty))
        print()

        parsed_output = _parse_json_response(output)
        if not isinstance(parsed_output, dict) or not parsed_output.get("ok"):
            failures += 1

    if failures:
        print(f"Smoke suite completed with {failures} failure(s).")
        return 1
    print(f"Smoke suite completed successfully with {len(SMOKE_TEST_QUERIES)} test(s).")
    return 0


def _main():
    parser = argparse.ArgumentParser(description="Plain-text YouTube wrapper backed by YouTube Data API v3")
    parser.add_argument(
        "query",
        nargs="?",
        help="Plain-text query. If omitted, query may be read from STDIN.",
    )
    parser.add_argument("-f", "--file", dest="query_file", help="Path to a plain-text query file")
    parser.add_argument("--smoke", action="store_true", help="Run the built-in smoke suite")
    parser.add_argument("--client-secrets", help="Path to OAuth client_secrets.json")
    parser.add_argument("--credentials-file", help="Where to save/load OAuth token", default="youtube_token.json")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output when possible")
    args = parser.parse_args()

    runtime_data = _build_runtime_data(args)

    try:
        if args.smoke:
            raise SystemExit(_run_smoke_suite(pretty=args.pretty, runtime_data=runtime_data))

        if args.query_file:
            query_path = Path(args.query_file).expanduser()
            if not query_path.exists():
                print(f"Error: query file not found: {query_path}", file=sys.stderr)
                raise SystemExit(2)
            query_text = query_path.read_text(encoding="utf-8")
        elif args.query:
            query_text = str(args.query)
        elif not sys.stdin.isatty():
            query_text = sys.stdin.read()
        else:
            # No query provided: run hardcoded smoke tests for quick local validation.
            raise SystemExit(_run_smoke_suite(pretty=args.pretty, runtime_data=runtime_data))

        output = _run_query_action(query_text, runtime_data=runtime_data)
        print(_format_output(output, pretty=args.pretty))
    except SystemExit:
        raise
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    _main()
