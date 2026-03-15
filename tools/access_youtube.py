from __future__ import annotations

import argparse
import json
import logging
import os
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
    ("search", "Find 3 recent videos about Python automation."),
    ("video details", "Get details for video id dQw4w9WgXcQ."),
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


def _search_videos(service, query: str, max_results: int = 5, order: str = "relevance", published_after: Optional[str] = None) -> List[Dict[str, Any]]:
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "order": order,
        "maxResults": max_results,
    }
    if published_after:
        params["publishedAfter"] = published_after
    resp = service.search().list(**params).execute()
    items: List[Dict[str, Any]] = []
    for it in resp.get("items", []):
        vid = it.get("id", {}).get("videoId")
        snip = it.get("snippet", {})
        items.append(
            {
                "videoId": vid,
                "title": snip.get("title"),
                "description": snip.get("description"),
                "channelTitle": snip.get("channelTitle"),
                "publishedAt": snip.get("publishedAt"),
            }
        )
    return items


def _get_video_details(service, video_id: str) -> Dict[str, Any]:
    resp = service.videos().list(part="snippet,contentDetails,statistics", id=video_id).execute()
    items = resp.get("items", [])
    return items[0] if items else {}


def _upload_video(service, file_path: str, title: str, description: str = "", tags: Optional[List[str]] = None, categoryId: str = "22", privacyStatus: str = "private") -> Dict[str, Any]:
    if not hasattr(service, "videos"):
        raise ValueError("Uploads require an authenticated `service` built with OAuth2 credentials (client_secrets_file).")
    body = {
        "snippet": {"title": title, "description": description, "tags": tags or [], "categoryId": categoryId},
        "status": {"privacyStatus": privacyStatus},
    }
    media = MediaFileUpload(file_path, chunksize=256 * 1024, resumable=True)
    request = service.videos().insert(part="snippet,status", body=body, media_body=media)
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            logger.info("Upload progress: %s%%", int(status.progress() * 100))
    return response


# --- Plain-text query helpers ------------------------------------


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


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


def _plan_query_payload(query: str) -> tuple[dict | None, str]:
    import llm

    system_instructions = (
        "You are a YouTube API planner. Given a user request, return a single JSON object describing the action to take. "
        "Allowed actions: 'search' (returns matching videos), 'get' (return details for a video id), 'upload' (upload a local file). "
        "Return only one JSON object using these fields: action, query, max_results, video_id, file, title, description, tags, privacyStatus. "
        "The JSON object MUST be the first and only JSON object in your response. Examples:\n"
        "{\"action\":\"search\",\"query\":\"funny cats\",\"max_results\":5}\n"
        "{\"action\":\"get\",\"video_id\":\"VIDEO_ID\"}\n"
        "{\"action\":\"upload\",\"file\":\"/path/to/file.mp4\",\"title\":\"Title\"}\n"
    )

    planner_output = llm._run_model_api(
        text=f"User request:\n{query}",
        system_instructions=system_instructions,
        model=MINIMAL_MODEL,
        tool_use_allowed=False,
        force_tool=False,
        temperature=0.2,
        thinking_level="low",
    )
    return _extract_first_json_object(planner_output), planner_output


def _execute_structured_payload(payload: dict, *, runtime_data: dict[str, Any]) -> str:
    action = str(payload.get("action", "")).strip().lower()
    if action == "search":
        q = payload.get("query") or payload.get("q")
        if not q:
            return _error_payload("search action requires `query`.")
        client_secrets = runtime_data.get("client_secrets") or payload.get("client_secrets")
        creds_file = runtime_data.get("credentials_file") or payload.get("credentials_file") or "youtube_token.json"
        try:
            service = _build_service(client_secrets_file=client_secrets, credentials_file=creds_file)
        except Exception as exc:
            return _error_payload("search action requires OAuth client secrets via `--client-secrets` or a valid credentials file.", error=str(exc))
        max_results = int(payload.get("max_results", 5))
        items = _search_videos(service, str(q), max_results=max_results)
        return _json_dumps({"ok": True, "items": items})

    if action == "get":
        video_id = payload.get("video_id") or payload.get("id")
        if not video_id:
            return _error_payload("get action requires `video_id`.")
        client_secrets = runtime_data.get("client_secrets") or payload.get("client_secrets")
        creds_file = runtime_data.get("credentials_file") or payload.get("credentials_file") or "youtube_token.json"
        try:
            service = _build_service(client_secrets_file=client_secrets, credentials_file=creds_file)
        except Exception as exc:
            return _error_payload("get action requires OAuth client secrets via `--client-secrets` or a valid credentials file.", error=str(exc))
        details = _get_video_details(service, str(video_id))
        return _json_dumps({"ok": True, "video": details})

    if action == "upload":
        file_path = payload.get("file")
        if not file_path:
            return _error_payload("upload action requires a local `file` path.")
        client_secrets = runtime_data.get("client_secrets") or payload.get("client_secrets")
        if not client_secrets:
            return _error_payload("upload action requires OAuth client secrets via `--client-secrets`.")
        creds_file = runtime_data.get("credentials_file") or payload.get("credentials_file") or "youtube_token.json"
        service = _build_service(client_secrets_file=client_secrets, credentials_file=creds_file)
        title = payload.get("title") or payload.get("name") or Path(str(file_path)).stem
        description = payload.get("description", "")
        tags = payload.get("tags") or []
        privacy = payload.get("privacyStatus", "private")
        try:
            file_path_resolved = Path(str(file_path)).expanduser()
            if not file_path_resolved.exists():
                return _error_payload("upload failed", error=f"file not found: {file_path_resolved}")
            resp = _upload_video(service, str(file_path_resolved.resolve()), title, description=description, tags=tags, privacyStatus=privacy)
        except Exception as exc:
            return _error_payload("upload failed", error=str(exc))
        return _json_dumps({"ok": True, "result": resp})

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
        force_tool=False,
        temperature=0.2,
        thinking_level="low",
    )


def _run_query_action(query: str, *, runtime_data: dict[str, Any]) -> str:
    normalized_query = (query or "").strip()
    if not normalized_query:
        return _error_payload("access_youtube requires a plain-text query.", executed=False)

    planned_payload, planner_output = _plan_query_payload(normalized_query)
    if not isinstance(planned_payload, dict):
        return _error_payload(
            "Planner failed to produce a valid JSON payload.",
            executed=False,
            planner_output=planner_output,
        )

    execution_text = _execute_structured_payload(planned_payload, runtime_data=runtime_data)
    parsed_result = _parse_json_response(execution_text)
    execution_result: Any = parsed_result if parsed_result is not None else execution_text
    ok = bool(not (isinstance(execution_result, dict) and execution_result.get("ok") is False))

    return _json_dumps(
        {
            "ok": ok,
            "executed": True,
            "query": normalized_query,
            "planned_payload": planned_payload,
            "result": execution_result,
        }
    )


def access_youtube(query: str = "") -> str:
    """
    Plain-text YouTube entry point backed by the YouTube Data API v3.

    Examples:
    - Find 3 recent videos about Python automation.
    - Get details for video id dQw4w9WgXcQ.
    - Upload a local file /path/to/video.mp4 with title "My Video".
    """
    output = _run_query_action(query, runtime_data={})
    try:
        return _interpret_query_response(query, output)
    except Exception:
        return output


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
