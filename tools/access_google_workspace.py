from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import copy
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import (
    DEFAULT_INFERENCE_MODE as DEFAULT_INFERENCE_MODE,
    INFERENCE_MODE_FLEX as INFERENCE_MODE_FLEX,
    MINIMAL_MODEL as MINIMAL_MODEL,
    GOOGLE_API_KEY as GOOGLE_API_KEY,
    GOOGLE_API_KEY_PATH as GOOGLE_API_KEY_PATH,
)

PLANNER_INSTRUCTIONS_FILE = _REPO_ROOT / "agent_instructions" / "google_workspace_planner.md"
RUNTIME_SETTING_KEYS = [
    "api_key",
    "api_key_file",
    "token",
    "credentials_file",
    "client_id",
    "client_secret",
    "config_dir",
    "project_id",
    "sanitize_template",
    "sanitize_mode",
    "timeout",
]
SMOKE_TEST_QUERIES: list[tuple[str, str]] = [
    ("services", "List the available Google Workspace services."),
    ("gmail labels", "Show the Gmail labels for the authenticated user."),
    ("docs create help", "How do I create a Google Docs document?"),
    ("drive files", "List 2 non-trashed Google Drive files."),
    ("calendar list", "List my calendars."),
    ("gmail schema", "Show the schema for listing Gmail messages."),
    ("sheets get help", "How do I read a value range from Google Sheets?"),
    ("sheet dry run", "Dry run creating a Google Sheet titled 'EnGem Smoke Test Sheet'."),
    ("doc dry run", "Dry run creating a Google Docs document titled 'EnGem Smoke Test Doc'."),
    ("slides dry run", "Dry run creating a Google Slides presentation titled 'EnGem Smoke Test Deck'."),
    (
        "doc workflow",
        "Create a Google Doc titled 'EnGem One Shot Smoke Test' containing a bullet list of the first 3 Google Drive file names, then verify the document and return the documentId and full text.",
    ),
    (
        "sheet count workflow",
        "Create a Google Sheet named 'Drive File Count Smoke Test', count the first 10 Google Drive files, write that count into cell A1, then verify A1 and return the spreadsheetId and A1 value.",
    ),
]

_FLEX_SUPPORTED_MODELS = {
    "gemini-3.1-flash-lite-preview",
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "gemini-3-pro-image-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-image",
    "gemini-2.5-flash-lite",
}
_MINIMAL_MODEL_SUPPORTS_FLEX = str(MINIMAL_MODEL).strip().lower() in _FLEX_SUPPORTED_MODELS
DEFAULT_API_KEY_FILE = str(GOOGLE_API_KEY_PATH or "").strip() or "google_api_key.txt"


def _repo_root() -> Path:
    return _REPO_ROOT


def _read_api_key_file(path_value: Any) -> str:
    path_text = str(path_value or "").strip()
    if not path_text:
        return ""

    key_path = Path(path_text).expanduser()
    if not key_path.exists():
        return ""

    try:
        key_contents = key_path.read_text(encoding="utf-8")
    except Exception:
        return ""

    for line in key_contents.splitlines():
        candidate = line.strip()
        if candidate:
            return candidate
    return ""


def _resolve_api_key_from_payload(payload: dict[str, Any]) -> str:
    direct_api_key = str(payload.get("api_key") or "").strip()
    if direct_api_key:
        return direct_api_key

    env_api_key = str(os.getenv("GOOGLE_API_KEY") or GOOGLE_API_KEY or "").strip()
    if env_api_key:
        return env_api_key

    api_key_file = str(payload.get("api_key_file") or DEFAULT_API_KEY_FILE).strip() or DEFAULT_API_KEY_FILE
    return _read_api_key_file(api_key_file)


def _merge_api_key_into_params(
    params_value: dict[str, Any] | str | None,
    *,
    api_key: str,
) -> dict[str, Any] | str | None:
    if not api_key:
        return params_value

    if params_value is None:
        return {"key": api_key}

    if isinstance(params_value, dict):
        merged = dict(params_value)
        merged.setdefault("key", api_key)
        return merged

    if isinstance(params_value, str):
        parsed = _parse_json_response(params_value)
        if isinstance(parsed, dict):
            parsed.setdefault("key", api_key)
            return parsed

    return params_value


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _parse_json_response(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
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


def _load_planner_system_instructions() -> str:
    if not PLANNER_INSTRUCTIONS_FILE.exists():
        raise FileNotFoundError(f"Planner instructions file not found: {PLANNER_INSTRUCTIONS_FILE}")
    return PLANNER_INSTRUCTIONS_FILE.read_text(encoding="utf-8")


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
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


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _normalize_parts(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(part) for part in value if part is not None and str(part) != ""]
    if value is None or value == "":
        return []
    return [str(value)]


def _strip_gws_prefix(args: list[str]) -> list[str]:
    if args and args[0] == "gws":
        return args[1:]
    return args


def _repo_relative_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (_repo_root() / path).resolve()


def _build_gws_env(runtime_data: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    payload_to_env = {
        "token": "GOOGLE_WORKSPACE_CLI_TOKEN",
        "credentials_file": "GOOGLE_WORKSPACE_CLI_CREDENTIALS_FILE",
        "client_id": "GOOGLE_WORKSPACE_CLI_CLIENT_ID",
        "client_secret": "GOOGLE_WORKSPACE_CLI_CLIENT_SECRET",
        "config_dir": "GOOGLE_WORKSPACE_CLI_CONFIG_DIR",
        "project_id": "GOOGLE_WORKSPACE_PROJECT_ID",
        "sanitize_template": "GOOGLE_WORKSPACE_CLI_SANITIZE_TEMPLATE",
        "sanitize_mode": "GOOGLE_WORKSPACE_CLI_SANITIZE_MODE",
    }
    for payload_key, env_key in payload_to_env.items():
        value = runtime_data.get(payload_key)
        if value is not None and value != "":
            env[env_key] = str(value)

    api_key = _resolve_api_key_from_payload(runtime_data)
    if api_key:
        env.setdefault("GOOGLE_API_KEY", api_key)

    return env


def _run_gws_command(
    args: list[str],
    *,
    runtime_data: dict[str, Any] | None = None,
    timeout: int = 300,
) -> dict[str, Any]:
    if shutil.which("gws") is None:
        return {
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": (
                "gws CLI not found in PATH. Install it with `npm install -g @googleworkspace/cli` "
                "and authenticate with `gws auth login`."
            ),
            "command": ["gws", *_strip_gws_prefix(args)],
        }

    command_data = runtime_data or {}
    command = ["gws", *_strip_gws_prefix(args)]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=int(command_data.get("timeout", timeout)),
            cwd=str(_repo_root()),
            env=_build_gws_env(command_data),
        )
        return {
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": (result.stdout or "").strip(),
            "stderr": (result.stderr or "").strip(),
            "command": command,
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": f"command timed out after {int(command_data.get('timeout', timeout))}s.",
            "command": command,
        }
    except Exception as exc:
        return {
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": f"Error running gws command: {exc}",
            "command": command,
        }


def _run_gws(
    args: list[str],
    *,
    runtime_data: dict[str, Any] | None = None,
    timeout: int = 300,
    success_message: str | None = None,
) -> str:
    result = _run_gws_command(args, runtime_data=runtime_data, timeout=timeout)
    stdout = str(result.get("stdout", "")).strip()
    stderr = str(result.get("stderr", "")).strip()

    if not result.get("ok"):
        details = "\n".join(part for part in (stdout, stderr) if part).strip() or "No output"
        return f"Error (exit {result.get('returncode')}): {details}"

    if stdout:
        return stdout
    if stderr:
        return stderr
    if success_message:
        return success_message
    return "OK"


def _parse_help_sections(text: str) -> dict[str, Any]:
    import re

    lines = text.splitlines()
    description_lines: list[str] = []
    usage = ""
    commands: list[dict[str, Any]] = []
    options: list[str] = []
    section: str | None = None
    command_pattern = re.compile(r"^\s{2,}(\S+)\s{2,}(.+)$")
    option_pattern = re.compile(r"^\s{2,}(-.+)$")

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            if section is None and description_lines:
                section = "preamble_done"
            continue

        if stripped.startswith("Usage:"):
            usage = stripped
            section = None
            continue
        if stripped == "Commands:":
            section = "commands"
            continue
        if stripped == "Options:":
            section = "options"
            continue

        if section == "commands":
            match = command_pattern.match(line)
            if match:
                name = match.group(1)
                description = match.group(2).strip()
                commands.append(
                    {
                        "name": name,
                        "description": description,
                        "kind": "helper" if name.startswith("+") else "command",
                    }
                )
            elif commands:
                commands[-1]["description"] = f"{commands[-1]['description']} {stripped}".strip()
            continue

        if section == "options":
            option_match = option_pattern.match(line)
            if option_match:
                options.append(option_match.group(1).strip())
            elif options:
                options[-1] = f"{options[-1]} {stripped}".strip()
            continue

        if not usage:
            description_lines.append(stripped)

    return {
        "description": " ".join(description_lines).strip(),
        "usage": usage,
        "commands": commands,
        "options": options,
    }


def _normalize_json_argument(
    value: Any,
    *,
    field_name: str,
    allow_list: bool,
) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    if isinstance(value, dict):
        return _json_dumps(value), None
    if allow_list and isinstance(value, list):
        return _json_dumps(value), None
    if isinstance(value, str) and value.strip():
        return value.strip(), None

    if allow_list:
        return None, f"Error: `{field_name}` must be a JSON object, array, or string."
    return None, f"Error: `{field_name}` must be a JSON object or string."


def _combine_params(
    payload: dict[str, Any],
    *,
    base: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | str | None, str | None]:
    merged = dict(base or {})

    fields = payload.get("fields")
    if fields is not None and "fields" not in merged:
        merged["fields"] = fields

    params = payload.get("params")
    if params is None:
        return merged, None
    if isinstance(params, dict):
        merged.update(params)
        return merged, None
    if isinstance(params, str) and params.strip():
        if merged:
            return None, "Error: `params` cannot be a raw JSON string when shorthand parameters are also used."
        return params.strip(), None
    return None, "Error: `params` must be a JSON object or JSON string."


def _append_params_flag(args: list[str], params_value: dict[str, Any] | str | None) -> None:
    if params_value is None:
        return
    if isinstance(params_value, str):
        args.extend(["--params", params_value])
        return
    if params_value:
        args.extend(["--params", _json_dumps(params_value)])


def _append_json_flag(args: list[str], value: Any, *, field_name: str = "json") -> str | None:
    json_value, error = _normalize_json_argument(value, field_name=field_name, allow_list=True)
    if error:
        return error
    if json_value:
        args.extend(["--json", json_value])
    return None


def _append_common_flags(
    args: list[str],
    payload: dict[str, Any],
    *,
    allow_format: bool = True,
    allow_paging: bool = True,
    allow_dry_run: bool = True,
    allow_api_version: bool = True,
    allow_sanitize: bool = True,
) -> None:
    if allow_format:
        output_format = payload.get("format")
        if output_format:
            args.extend(["--format", str(output_format)])

    if allow_paging:
        if _as_bool(payload.get("page_all"), default=False):
            args.append("--page-all")

        page_limit = payload.get("page_limit")
        if page_limit is not None:
            args.extend(["--page-limit", str(page_limit)])

        page_delay = payload.get("page_delay")
        if page_delay is not None:
            args.extend(["--page-delay", str(page_delay)])

    if allow_dry_run and _as_bool(payload.get("dry_run"), default=False):
        args.append("--dry-run")

    if allow_api_version:
        api_version = payload.get("api_version")
        if api_version:
            args.extend(["--api-version", str(api_version)])

    if allow_sanitize:
        sanitize = payload.get("sanitize")
        if sanitize:
            args.extend(["--sanitize", str(sanitize)])

    raw_flags = str(payload.get("raw_flags", "")).strip()
    if raw_flags:
        args.extend(shlex.split(raw_flags))


def _resource_segments(payload: dict[str, Any]) -> list[str]:
    segments: list[str] = []
    segments.extend(_normalize_parts(payload.get("resources")))
    if not segments:
        segments.extend(_normalize_parts(payload.get("resource")))
    segments.extend(_normalize_parts(payload.get("subresources")))
    segments.extend(_normalize_parts(payload.get("subresource")))
    return segments


def _build_path_segments_from_payload(payload: dict[str, Any]) -> list[str]:
    segments: list[str] = []
    service = payload.get("service")
    if service:
        segments.append(str(service))
    segments.extend(_resource_segments(payload))
    method = payload.get("method")
    if method:
        segments.append(str(method))
    return segments


def _build_call_args(payload: dict[str, Any]) -> tuple[list[str], str | None] | str:
    service = payload.get("service")
    method = payload.get("method")
    if not service or not method:
        return "Error: internal call payloads require `service` and `method`."

    args = [str(service), *_resource_segments(payload), str(method)]

    positional = payload.get("positional")
    if isinstance(positional, list):
        args.extend(str(part) for part in positional if part is not None and str(part) != "")

    params_value, error = _combine_params(payload)
    if error:
        return error
    params_value = _merge_api_key_into_params(params_value, api_key=_resolve_api_key_from_payload(payload))
    _append_params_flag(args, params_value)

    json_error = _append_json_flag(args, payload.get("json", payload.get("body")))
    if json_error:
        return json_error

    upload = payload.get("upload")
    if upload:
        args.extend(["--upload", str(_repo_relative_path(str(upload)))])

    output = payload.get("output", payload.get("out"))
    if output:
        output_path = _repo_relative_path(str(output))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        args.extend(["--output", str(output_path)])

    _append_common_flags(args, payload)
    return args, None


def _build_raw_args(payload: dict[str, Any]) -> tuple[list[str], str | None] | str:
    args_value = payload.get("args")
    if isinstance(args_value, list):
        args = _strip_gws_prefix([str(part) for part in args_value if part is not None and str(part) != ""])
    elif payload.get("command"):
        args = _strip_gws_prefix(shlex.split(str(payload["command"])))
    else:
        return _build_call_args(payload)

    positional = payload.get("positional")
    if isinstance(positional, list):
        args.extend(str(part) for part in positional if part is not None and str(part) != "")

    params_value, error = _combine_params(payload)
    if error:
        return error

    if args and args[0] not in {"auth", "schema"} and "--help" not in args:
        params_value = _merge_api_key_into_params(params_value, api_key=_resolve_api_key_from_payload(payload))

    _append_params_flag(args, params_value)

    json_error = _append_json_flag(args, payload.get("json", payload.get("body")))
    if json_error:
        return json_error

    upload = payload.get("upload")
    if upload:
        args.extend(["--upload", str(_repo_relative_path(str(upload)))])

    output = payload.get("output", payload.get("out"))
    if output:
        output_path = _repo_relative_path(str(output))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        args.extend(["--output", str(output_path)])

    _append_common_flags(args, payload)
    return args, None


def _build_help_args(payload: dict[str, Any]) -> tuple[list[str], str | None]:
    topic = payload.get("topic", payload.get("help_for"))
    if isinstance(topic, list):
        args = _strip_gws_prefix([str(part) for part in topic if part is not None and str(part) != ""])
    elif isinstance(topic, str) and topic.strip():
        args = _strip_gws_prefix(shlex.split(topic.strip()))
    else:
        args = _build_path_segments_from_payload(payload)

    if not args or args[-1] != "--help":
        args.append("--help")
    return args, None


def _build_schema_args(payload: dict[str, Any]) -> tuple[list[str], str | None] | str:
    target = payload.get("target", payload.get("schema"))
    if not target:
        segments = _build_path_segments_from_payload(payload)
        if len(segments) >= 2:
            target = ".".join(segments)

    if not target:
        return "Error: schema mode requires `target` like `gmail.users.messages.list`."

    args = ["schema", str(target)]
    if _as_bool(payload.get("resolve_refs"), default=False):
        args.append("--resolve-refs")
    return args, None


def _build_services_args(_: dict[str, Any]) -> tuple[list[str], str | None]:
    return ["--help"], None


def _build_auth_args(payload: dict[str, Any]) -> tuple[list[str], str | None]:
    auth_action = payload.get("auth_action", payload.get("subcommand", payload.get("method")))
    args = ["auth"]
    if auth_action:
        args.append(str(auth_action))

    scopes = payload.get("scopes")
    if isinstance(scopes, list):
        scope_values = [str(scope) for scope in scopes if scope is not None and str(scope) != ""]
        if scope_values:
            args.extend(["--scopes", ",".join(scope_values)])
    elif isinstance(scopes, str) and scopes.strip():
        args.extend(["--scopes", scopes.strip()])

    positional = payload.get("positional")
    if isinstance(positional, list):
        args.extend(str(part) for part in positional if part is not None and str(part) != "")

    raw_flags = str(payload.get("raw_flags", "")).strip()
    if raw_flags:
        args.extend(shlex.split(raw_flags))
    return args, None


def _resolve_action(payload: dict[str, Any]) -> str:
    action = str(payload.get("action", "")).strip().lower()
    aliases = {
        "workflow": "workflow",
        "multi_step": "workflow",
        "call": "call",
        "request": "call",
        "execute": "call",
        "invoke": "call",
        "raw": "raw",
        "discover": "discover",
        "describe": "discover",
        "introspect": "discover",
        "catalog": "discover",
        "validate": "validate",
        "check": "validate",
        "check_command": "validate",
        "schema": "schema",
        "help": "help",
        "services": "services",
        "auth": "auth",
    }
    if action:
        return aliases.get(action, action)

    if isinstance(payload.get("steps"), list) and payload.get("steps"):
        return "workflow"
    if payload.get("schema") or payload.get("target"):
        return "schema"
    if payload.get("command") or payload.get("args"):
        return "raw"
    if _as_bool(payload.get("discover"), default=False):
        return "discover"
    if _as_bool(payload.get("validate"), default=False):
        return "validate"
    if payload.get("service") and payload.get("method"):
        return "call"
    if _as_bool(payload.get("services"), default=False):
        return "services"
    return ""


def _build_internal_command(payload: dict[str, Any]) -> tuple[list[str], str | None] | str:
    action = _resolve_action(payload)

    if action == "call":
        return _build_call_args(payload)
    if action == "raw":
        return _build_raw_args(payload)
    if action == "help":
        return _build_help_args(payload)
    if action == "schema":
        return _build_schema_args(payload)
    if action == "services":
        return _build_services_args(payload)
    if action == "auth":
        return _build_auth_args(payload)

    return "Error: unsupported internal action."


def _coerce_scalar(value: Any, value_type: str | None) -> Any:
    if value is None:
        return None
    if value_type == "boolean":
        if isinstance(value, bool):
            return value
        lowered = str(value).strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        return bool(value)
    if value_type == "integer":
        try:
            return int(str(value))
        except Exception:
            return value
    if value_type == "number":
        try:
            return float(str(value))
        except Exception:
            return value
    return value


def _sample_parameter_value(name: str, schema: dict[str, Any]) -> Any:
    default = schema.get("default")
    if default is not None:
        return default if schema.get("type") not in {"boolean", "integer", "number"} else _coerce_scalar(default, schema.get("type"))

    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and enum_values:
        return enum_values[0]

    value_type = schema.get("type")
    if value_type == "boolean":
        return False
    if value_type == "integer":
        return 0
    if value_type == "number":
        return 0.0
    return f"<{name}>"


def _preferred_body_keys(properties: dict[str, Any], required_keys: list[str]) -> list[str]:
    if required_keys:
        return required_keys

    read_only_like = {
        "id",
        "kind",
        "etag",
        "spreadsheetId",
        "spreadsheetUrl",
        "documentId",
        "revisionId",
        "presentationId",
    }
    filtered: list[str] = []
    for key, value in properties.items():
        if key in read_only_like:
            continue
        description = str(value.get("description", "")).strip().lower() if isinstance(value, dict) else ""
        if description.startswith("output only") or description.startswith("read-only"):
            continue
        filtered.append(key)
    if filtered:
        return filtered[:3]
    return list(properties.keys())[:3]


def _sample_schema_value(schema: dict[str, Any], *, field_name: str = "value", depth: int = 0) -> Any:
    if not isinstance(schema, dict):
        return f"<{field_name}>"

    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and enum_values:
        return enum_values[0]
    if depth >= 2 and schema.get("$ref"):
        return f"<{schema['$ref']}>"

    properties = schema.get("properties")
    schema_type = schema.get("type")
    if isinstance(properties, dict) or schema_type == "object":
        property_map = properties if isinstance(properties, dict) else {}
        keys = _preferred_body_keys(property_map, list(schema.get("required", [])))
        value: dict[str, Any] = {}
        for key in keys[:3]:
            child_schema = property_map.get(key, {})
            value[key] = _sample_schema_value(child_schema, field_name=key, depth=depth + 1)
        if value:
            return value
        if schema.get("$ref"):
            return f"<{schema['$ref']}>"
        return {}
    if schema_type == "array":
        return [_sample_schema_value(schema.get("items", {}), field_name=field_name, depth=depth + 1)]
    if schema_type == "boolean":
        return False
    if schema_type == "integer":
        return 0
    if schema_type == "number":
        return 0.0
    if schema.get("$ref"):
        return f"<{schema['$ref']}>"
    return f"<{field_name}>"


def _summarize_parameters(parameters: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    summary = {"required": [], "optional": [], "path": [], "query": []}
    for name, raw_schema in parameters.items():
        if not isinstance(raw_schema, dict):
            continue
        entry: dict[str, Any] = {
            "name": name,
            "location": str(raw_schema.get("location", "query")),
            "required": bool(raw_schema.get("required")),
            "type": str(raw_schema.get("type", "any")),
            "sample": _sample_parameter_value(name, raw_schema),
        }
        if raw_schema.get("default") is not None:
            entry["default"] = _coerce_scalar(raw_schema.get("default"), raw_schema.get("type"))
        if isinstance(raw_schema.get("enum"), list) and raw_schema.get("enum"):
            entry["enum"] = raw_schema.get("enum")
        description = str(raw_schema.get("description", "")).strip()
        if description:
            entry["description"] = description
        location_key = "path" if entry["location"] == "path" else "query"
        summary[location_key].append(entry)
        summary["required" if entry["required"] else "optional"].append(entry)
    return summary


def _build_action_payload_for_path(path: list[str], *, action: str, treat_last_as_method: bool) -> dict[str, Any]:
    if not path:
        return {"action": action}

    payload: dict[str, Any] = {"action": action, "service": path[0]}
    middle_segments = path[1:-1] if treat_last_as_method and len(path) > 1 else path[1:]
    if len(middle_segments) == 1:
        payload["resource"] = middle_segments[0]
    elif len(middle_segments) > 1:
        payload["resources"] = middle_segments
    if treat_last_as_method and len(path) > 1:
        payload["method"] = path[-1]
    return payload


def _build_navigation_queries(path: list[str], help_info: dict[str, Any]) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    for command in help_info.get("commands", []):
        if not isinstance(command, dict):
            continue
        name = str(command.get("name", "")).strip()
        if not name or name == "help":
            continue
        child_path = [*path, name]
        queries.append(
            {
                "name": name,
                "description": str(command.get("description", "")).strip(),
                "kind": str(command.get("kind", "command")),
                "discover_payload": _build_action_payload_for_path(child_path, action="discover", treat_last_as_method=False),
            }
        )
    return queries


def _build_method_payload_template(path: list[str], schema: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {"action": "call", "service": path[0], "method": path[-1]}
    resources = path[1:-1]
    if len(resources) == 1:
        payload["resource"] = resources[0]
    elif len(resources) > 1:
        payload["resources"] = resources

    parameters = schema.get("parameters")
    if isinstance(parameters, dict):
        required_params = {
            name: _sample_parameter_value(name, raw_schema)
            for name, raw_schema in parameters.items()
            if isinstance(raw_schema, dict) and raw_schema.get("required")
        }
        if required_params:
            payload["params"] = required_params

    request_body = schema.get("requestBody")
    if isinstance(request_body, dict) and isinstance(request_body.get("schema"), dict):
        payload["json"] = _sample_schema_value(request_body["schema"], field_name="json")

    return payload


def _build_query_guide(
    *,
    path: list[str],
    help_info: dict[str, Any],
    schema: dict[str, Any] | None,
    schema_target: str | None,
) -> dict[str, Any]:
    guide: dict[str, Any] = {
        "field_rules": {
            "service": "First gws path segment.",
            "resource": "Use for a single intermediate path segment.",
            "resources": "Use for multiple intermediate path segments in order.",
            "method": "Final API method once you have navigated to a method-level path.",
            "params": "All API path and query parameters go here.",
            "json": "Request body goes here when the method has a requestBody schema.",
            "upload": "Local file path for media uploads when the gws method supports upload.",
            "output": "Local file path where downloaded output should be written.",
            "dry_run": "Set to true to validate a mutating request without sending it.",
        }
    }

    if not path:
        guide["discover_payload"] = {"action": "discover"}
        guide["next_queries"] = _build_navigation_queries(path, help_info)
        return guide

    if schema is None:
        guide["discover_payload"] = _build_action_payload_for_path(path, action="discover", treat_last_as_method=False)
        guide["current_path"] = {"service": path[0], "resources": path[1:], "method": None}
        guide["next_queries"] = _build_navigation_queries(path, help_info)
        guide["how_to_continue"] = (
            "Pick one of the `next_queries` entries and run discover again until you reach a method. "
            "Once you are at a method-level path, use the returned `call_payload_template`."
        )
        return guide

    payload_template = _build_method_payload_template(path, schema)
    parameter_summary = _summarize_parameters(schema.get("parameters", {})) if isinstance(schema.get("parameters"), dict) else {
        "required": [],
        "optional": [],
        "path": [],
        "query": [],
    }
    guide["current_path"] = {"service": path[0], "resources": path[1:-1], "method": path[-1]}
    guide["discover_payload"] = _build_action_payload_for_path(path, action="discover", treat_last_as_method=True)
    guide["operation"] = {
        "http_method": schema.get("httpMethod"),
        "api_path": schema.get("path"),
        "has_request_body": bool(isinstance(schema.get("requestBody"), dict) and schema["requestBody"].get("schema")),
        "scopes": schema.get("scopes", []),
    }
    guide["call_payload_template"] = payload_template
    guide["validate_payload"] = _build_action_payload_for_path(path, action="validate", treat_last_as_method=True)
    guide["schema_payload"] = {"action": "schema", "target": schema_target, "resolve_refs": True} if schema_target else None
    guide["params"] = parameter_summary

    optional_params_template = {
        entry["name"]: entry["sample"]
        for entry in parameter_summary.get("optional", [])
        if isinstance(entry, dict) and entry.get("name")
    }
    if optional_params_template:
        guide["optional_params_template"] = optional_params_template

    request_body = schema.get("requestBody")
    if isinstance(request_body, dict) and isinstance(request_body.get("schema"), dict):
        guide["json_body"] = {
            "schema_ref": request_body.get("schemaRef"),
            "template": payload_template.get("json"),
            "note": "If the template still contains <TypeName> placeholders, rerun schema with resolve_refs=true to expand nested object types.",
        }

    return guide


def _build_discover_args(payload: dict[str, Any]) -> list[str]:
    topic = payload.get("topic", payload.get("path", payload.get("command")))
    if isinstance(topic, list):
        args = _strip_gws_prefix([str(part) for part in topic if part is not None and str(part) != ""])
    elif isinstance(topic, str) and topic.strip():
        args = _strip_gws_prefix(shlex.split(topic.strip()))
    else:
        args = _build_path_segments_from_payload(payload)

    if not args or args[-1] != "--help":
        args.append("--help")
    return args


def _build_schema_target(payload: dict[str, Any]) -> str | None:
    target = payload.get("target", payload.get("schema"))
    if target:
        return str(target)
    segments = _build_path_segments_from_payload(payload)
    if len(segments) >= 2:
        return ".".join(segments)
    return None


def _run_discover_action(payload: dict[str, Any], *, runtime_data: dict[str, Any]) -> str:
    args = _build_discover_args(payload)
    result = _run_gws_command(args, runtime_data=runtime_data)
    path = _strip_gws_prefix(args[:-1])
    response: dict[str, Any] = {
        "ok": bool(result.get("ok")),
        "path": path,
        "command": result.get("command"),
        "returncode": result.get("returncode"),
    }

    stdout = str(result.get("stdout", ""))
    stderr = str(result.get("stderr", ""))
    text = stdout or stderr
    response["help"] = _parse_help_sections(text) if text else {}
    if stderr:
        response["stderr"] = stderr

    schema_target: str | None = None
    schema_payload: dict[str, Any] | None = None
    if result.get("ok") and payload.get("method"):
        schema_target = _build_schema_target(payload)
        if schema_target:
            schema_result = _run_gws_command(["schema", schema_target], runtime_data=runtime_data)
            response["schema_target"] = schema_target
            if schema_result.get("ok"):
                parsed_schema = _parse_json_response(str(schema_result.get("stdout", "")))
                if isinstance(parsed_schema, dict):
                    response["schema"] = parsed_schema
                    schema_payload = parsed_schema
            else:
                response["schema_error"] = str(schema_result.get("stderr") or schema_result.get("stdout") or "")

    response["query_guide"] = _build_query_guide(
        path=path,
        help_info=response.get("help", {}),
        schema=schema_payload,
        schema_target=schema_target,
    )
    return _json_dumps(response)


def _extract_suggestions(text: str) -> list[str]:
    import re

    suggestions: list[str] = []
    seen: set[str] = set()
    marker = "some similar subcommands exist:"
    for match in re.finditer(r"'([^']+)'", text):
        start = max(0, match.start() - 120)
        context = text[start:match.start()].lower()
        if marker not in context:
            continue
        candidate = match.group(1).strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            suggestions.append(candidate)
    return suggestions


def _run_validate_action(payload: dict[str, Any], *, runtime_data: dict[str, Any]) -> str:
    args = _build_discover_args(payload)
    result = _run_gws_command(args, runtime_data=runtime_data)
    stdout = str(result.get("stdout", ""))
    stderr = str(result.get("stderr", ""))
    text = stdout or stderr
    response: dict[str, Any] = {
        "valid": bool(result.get("ok")),
        "path": _strip_gws_prefix(args[:-1]),
        "command": result.get("command"),
        "returncode": result.get("returncode"),
        "suggestions": _extract_suggestions(text),
    }
    if result.get("ok"):
        response["help"] = _parse_help_sections(text)
    else:
        response["error"] = text
    return _json_dumps(response)


def _contains_unresolved_placeholders(value: Any) -> bool:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("<") and stripped.endswith(">"):
            return True
        if "{{" in value or "}}" in value:
            return True
        if "{" in value and "}" in value and not stripped.startswith("{"):
            return True
        return False
    if isinstance(value, list):
        return any(_contains_unresolved_placeholders(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_unresolved_placeholders(item) for item in value.values())
    return False


def _collect_missing_required_params(payload: dict[str, Any], discover_payload: dict[str, Any]) -> list[str]:
    guide = discover_payload.get("query_guide")
    if not isinstance(guide, dict):
        return []
    params_info = guide.get("params")
    if not isinstance(params_info, dict):
        return []
    required_entries = params_info.get("required")
    if not isinstance(required_entries, list):
        return []

    payload_params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
    missing: list[str] = []
    for entry in required_entries:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip()
        if not name:
            continue
        if name not in payload_params or _contains_unresolved_placeholders(payload_params.get(name)):
            missing.append(name)
    return missing


def _tokenize_reference_path(path: str) -> list[str]:
    import re

    tokens: list[str] = []
    for match in re.finditer(r"([^.\[\]]+)|\[(\*|\d+)\]", path):
        token = match.group(1) if match.group(1) is not None else match.group(2)
        if token is not None and token != "":
            tokens.append(token)
    return tokens


def _extract_reference_value(context: dict[str, Any], path: str) -> Any:
    tokens = _tokenize_reference_path(path)
    if not tokens:
        raise ValueError(f"Invalid workflow reference path: {path}")

    values: list[Any] = [context]
    wildcard_used = False
    for token in tokens:
        next_values: list[Any] = []
        if token == "*":
            wildcard_used = True
            for value in values:
                if isinstance(value, list):
                    next_values.extend(value)
            values = next_values
            continue

        for value in values:
            if isinstance(value, dict) and token in value:
                next_values.append(value[token])
                continue
            if isinstance(value, list) and token.isdigit():
                index = int(token)
                if 0 <= index < len(value):
                    next_values.append(value[index])
        values = next_values

    if not values:
        raise ValueError(f"Workflow reference not found: {path}")
    if wildcard_used or len(values) > 1:
        return values
    return values[0]


def _extract_google_doc_text(document: Any) -> str:
    pieces: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            text_run = value.get("textRun")
            if isinstance(text_run, dict):
                content = text_run.get("content")
                if isinstance(content, str):
                    pieces.append(content)
            for key, child in value.items():
                if key != "textRun":
                    visit(child)
            return
        if isinstance(value, list):
            for item in value:
                visit(item)

    visit(document)
    return "".join(pieces).strip()


def _apply_workflow_transform(name: str, source: Any, spec: dict[str, Any]) -> Any:
    transform = name.strip().lower()

    if transform == "first":
        if isinstance(source, list):
            return source[0] if source else None
        return source
    if transform == "json":
        return _json_dumps(source)
    if transform == "doc_text":
        return _extract_google_doc_text(source)
    if transform == "count":
        if isinstance(source, list):
            return len(source)
        if source is None:
            return 0
        return 1

    if isinstance(source, list):
        items = ["" if item is None else str(item) for item in source]
    elif source is None:
        items = []
    else:
        items = [str(source)]

    prefix = str(spec.get("prefix", ""))
    suffix = str(spec.get("suffix", ""))
    if transform == "join_lines":
        return prefix + "\n".join(items) + suffix
    if transform == "bulleted_lines":
        item_prefix = str(spec.get("item_prefix", "- "))
        return prefix + "\n".join(f"{item_prefix}{item}" for item in items) + suffix
    if transform == "numbered_lines":
        return prefix + "\n".join(f"{index}. {item}" for index, item in enumerate(items, start=1)) + suffix

    raise ValueError(f"Unsupported workflow transform: {name}")


def _resolve_interpolated_string(text: str, context: dict[str, Any]) -> Any:
    import re

    pattern = re.compile(r"\$\{([^}]+)\}")
    matches = list(pattern.finditer(text))
    if not matches:
        return text
    if len(matches) == 1 and matches[0].span() == (0, len(text)):
        return _extract_reference_value(context, matches[0].group(1).strip())

    result = text
    for match in reversed(matches):
        replacement = _extract_reference_value(context, match.group(1).strip())
        result = result[: match.start()] + str(replacement) + result[match.end() :]
    return result


def _resolve_workflow_value(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        if set(value.keys()) == {"$ref"}:
            return _extract_reference_value(context, str(value["$ref"]).strip())
        if "$transform" in value:
            source = _resolve_workflow_value(value.get("source"), context)
            resolved_spec = {
                key: _resolve_workflow_value(spec_value, context)
                for key, spec_value in value.items()
                if key not in {"$transform", "source"}
            }
            return _apply_workflow_transform(str(value["$transform"]), source, resolved_spec)
        return {key: _resolve_workflow_value(item, context) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_workflow_value(item, context) for item in value]
    if isinstance(value, str):
        return _resolve_interpolated_string(value, context)
    return value


def _is_likely_datetime_string(s: str) -> bool:
    if not isinstance(s, str):
        return False
    s = s.strip()
    if not s:
        return False
    # Only treat strings that include a time-of-day or explicit T as datetimes
    return "T" in s or ":" in s


def _parse_iso_like_datetime(s: str) -> datetime | None:
    candidate = s.strip()
    if not candidate:
        return None
    # Handle trailing Z
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    # If there's a space between date and time, make it ISO-like
    if " " in candidate and "T" not in candidate and ":" in candidate:
        candidate = candidate.replace(" ", "T", 1)
    try:
        return datetime.fromisoformat(candidate)
    except Exception:
        return None


def _convert_datetimes_to_utc_in_value(value: Any, *, user_tz: str = "America/Toronto") -> Any:
    if isinstance(value, str):
        s = value.strip()
        if not _is_likely_datetime_string(s):
            return value
        dt = _parse_iso_like_datetime(s)
        if dt is None:
            return value
        if dt.tzinfo is None:
            try:
                local_tz = ZoneInfo(user_tz)
            except Exception:
                local_tz = ZoneInfo("UTC")
            dt_local = dt.replace(tzinfo=local_tz)
        else:
            dt_local = dt
        dt_utc = dt_local.astimezone(timezone.utc)
        return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    if isinstance(value, dict):
        return {k: _convert_datetimes_to_utc_in_value(v, user_tz=user_tz) for k, v in value.items()}
    if isinstance(value, list):
        return [
            _convert_datetimes_to_utc_in_value(item, user_tz=user_tz) for item in value
        ]
    return value


def _convert_payload_times_to_utc(payload: dict[str, Any], *, user_tz: str = "America/Toronto") -> dict[str, Any]:
    new = copy.deepcopy(payload)
    # Convert common places where timestamps appear (`params` and `json`),
    # then walk other fields conservatively (only strings/dicts/lists).
    if isinstance(new.get("params"), dict):
        new["params"] = _convert_datetimes_to_utc_in_value(new["params"], user_tz=user_tz)
    if "json" in new and new.get("json") is not None:
        new["json"] = _convert_datetimes_to_utc_in_value(new["json"], user_tz=user_tz)

    for key, val in list(new.items()):
        if key in {"params", "json"}:
            continue
        if isinstance(val, (dict, list, str)):
            new[key] = _convert_datetimes_to_utc_in_value(val, user_tz=user_tz)

    return new


def _inherit_runtime_settings(payload: dict[str, Any], runtime_data: dict[str, Any]) -> dict[str, Any]:
    merged = dict(payload)
    for key in RUNTIME_SETTING_KEYS:
        if key not in merged and key in runtime_data:
            merged[key] = runtime_data[key]
    return merged


def _extract_step_payload(step: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in step.items() if key not in {"id", "description"}}


def _method_is_mutating(method: Any) -> bool:
    normalized = str(method or "").strip().lower()
    return normalized in {
        "create",
        "update",
        "batchupdate",
        "append",
        "insert",
        "delete",
        "move",
        "copy",
        "send",
        "patch",
        "modify",
    }


def _workflow_requires_verification(workflow: dict[str, Any]) -> bool:
    steps = workflow.get("steps")
    if not isinstance(steps, list) or not steps:
        return False

    mutating = False
    step_map: dict[str, dict[str, Any]] = {}
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            continue
        step_id = str(step.get("id", f"step_{index}")).strip()
        if step_id:
            step_map[step_id] = step
        action = _resolve_action(step)
        if action == "call" and (_method_is_mutating(step.get("method")) or step.get("json") is not None or step.get("upload") is not None):
            mutating = True

    if not mutating:
        return False

    final_step_id = str(workflow.get("final_step", "")).strip()
    final_step = step_map.get(final_step_id) if final_step_id else None
    if final_step is None:
        for step in reversed(steps):
            if isinstance(step, dict):
                final_step = step
                break
    if final_step is None:
        return True

    final_action = _resolve_action(final_step)
    if final_action in {"discover", "help", "schema", "validate"}:
        return False
    if final_action == "call" and not _method_is_mutating(final_step.get("method")):
        return False
    return True


def _execute_structured_payload(payload: dict[str, Any], *, runtime_data: dict[str, Any]) -> str:
    action = _resolve_action(payload)
    if action == "discover":
        return _run_discover_action(payload, runtime_data=runtime_data)
    if action == "validate":
        return _run_validate_action(payload, runtime_data=runtime_data)
    if action == "workflow":
        return _run_workflow_action(payload, runtime_data=runtime_data)

    # Convert any ISO-like local datetimes (assumed America/Toronto) to UTC
    try:
        exec_payload = _convert_payload_times_to_utc(payload)
    except Exception:
        exec_payload = payload

    built_command = _build_internal_command(exec_payload)
    if isinstance(built_command, str):
        return built_command
    args, success_message = built_command
    return _run_gws(args, runtime_data=runtime_data, success_message=success_message)


def _run_workflow_action(workflow: dict[str, Any], *, runtime_data: dict[str, Any]) -> str:
    steps = workflow.get("steps")
    if not isinstance(steps, list) or not steps:
        return _error_payload("workflow action requires a non-empty `steps` array.", executed=False)

    context: dict[str, Any] = {"steps": {}}
    step_summaries: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    last_step_id: str | None = None

    for index, raw_step in enumerate(steps, start=1):
        if not isinstance(raw_step, dict):
            return _error_payload("Each workflow step must be a JSON object.", executed=False, failed_step=index)

        step_id = str(raw_step.get("id", f"step_{index}")).strip()
        if not step_id:
            return _error_payload("Workflow steps require a non-empty `id`.", executed=False, failed_step=index)
        if step_id in seen_ids:
            return _error_payload(f"Duplicate workflow step id: {step_id}", executed=False, failed_step=step_id)
        seen_ids.add(step_id)
        last_step_id = step_id

        try:
            resolved_payload = _resolve_workflow_value(_extract_step_payload(raw_step), context)
        except Exception as exc:
            return _error_payload(
                f"Failed to resolve workflow step references: {exc}",
                executed=False,
                failed_step=step_id,
            )

        if not isinstance(resolved_payload, dict):
            return _error_payload(
                "Resolved workflow step payload must be a JSON object.",
                executed=False,
                failed_step=step_id,
            )

        resolved_payload = _inherit_runtime_settings(resolved_payload, runtime_data)
        step_action = _resolve_action(resolved_payload)
        if step_action in {"", "workflow"}:
            return _error_payload(
                "Workflow steps must resolve to call, discover, schema, help, validate, raw, services, or auth actions.",
                executed=False,
                failed_step=step_id,
                payload=resolved_payload,
            )

        validation: dict[str, Any] | None = None
        if step_action == "call":
            validation_text = _run_validate_action(resolved_payload, runtime_data=runtime_data)
            parsed_validation = _parse_json_response(validation_text)
            validation = parsed_validation if isinstance(parsed_validation, dict) else {"raw": validation_text}
            if not validation.get("valid"):
                return _error_payload(
                    "Workflow step failed path validation.",
                    executed=False,
                    failed_step=step_id,
                    payload=resolved_payload,
                    validation=validation,
                )

        if _contains_unresolved_placeholders(resolved_payload.get("params")) or _contains_unresolved_placeholders(resolved_payload.get("json")):
            return _error_payload(
                "Workflow step still contains unresolved placeholders after binding.",
                executed=False,
                failed_step=step_id,
                payload=resolved_payload,
            )

        execution_text = _execute_structured_payload(resolved_payload, runtime_data=runtime_data)
        parsed_execution = _parse_json_response(execution_text)
        execution_result: Any = parsed_execution if parsed_execution is not None else execution_text

        step_summary: dict[str, Any] = {"id": step_id, "payload": resolved_payload, "result": execution_result}
        description = str(raw_step.get("description", "")).strip()
        if description:
            step_summary["description"] = description
        if validation is not None:
            step_summary["validation"] = validation

        step_summaries.append(step_summary)
        context["steps"][step_id] = step_summary

        if isinstance(execution_result, str) and execution_result.startswith("Error"):
            return _error_payload(
                execution_result,
                executed=False,
                failed_step=step_id,
                steps=step_summaries,
            )

    try:
        if "result" in workflow:
            final_result = _resolve_workflow_value(workflow["result"], context)
            final_step_id = str(workflow.get("final_step", last_step_id or "")).strip() or None
        else:
            final_step_id = str(workflow.get("final_step", last_step_id or "")).strip()
            if not final_step_id or final_step_id not in context["steps"]:
                return _error_payload(
                    "Workflow final_step is missing or does not match an executed step.",
                    executed=False,
                    steps=step_summaries,
                )
            final_result = context["steps"][final_step_id]["result"]
    except Exception as exc:
        return _error_payload(
            f"Failed to resolve workflow result: {exc}",
            executed=False,
            steps=step_summaries,
        )

    return _json_dumps(
        {
            "ok": True,
            "executed": True,
            "steps": step_summaries,
            "final_step": final_step_id,
            "result": final_result,
        }
    )


def _build_query_planner_prompt(
    query: str,
    *,
    root_help: dict[str, Any],
    feedback: list[dict[str, Any]],
) -> str:
    prompt_parts = [
        f"User request:\n{query}",
        "Available top-level gws help JSON:",
        _json_dumps(root_help),
    ]
    if feedback:
        prompt_parts.extend(
            [
                "Previous planner feedback JSON. Fix the next payload using this live validation/execution feedback:",
                _json_dumps(feedback),
            ]
        )
    return "\n\n".join(prompt_parts)


def _query_looks_multi_step(query: str, root_help: dict[str, Any]) -> bool:
    lowered = query.strip().lower()
    if not lowered:
        return False

    service_names: set[str] = set()
    for command in root_help.get("commands", []):
        if not isinstance(command, dict):
            continue
        name = str(command.get("name", "")).strip().lower()
        if name and name != "help" and name in lowered:
            service_names.add(name)

    if len(service_names) >= 2:
        return True

    bridge_markers = [
        " then ",
        " and then ",
        "populate",
        "append",
        "using ",
        " from ",
        " into ",
        "based on",
        "with a list of",
        "write that",
        "write the total",
    ]
    action_verbs = ["create", "update", "send", "copy", "move", "write", "draft"]
    source_verbs = ["list", "find", "search", "get", "read", "retrieve", "count"]
    has_bridge = any(marker in lowered for marker in bridge_markers)
    has_action = any(f"{verb} " in lowered for verb in action_verbs)
    has_source = any(f"{verb} " in lowered for verb in source_verbs)
    return has_bridge and has_action and has_source


def _query_requests_explicit_result(query: str) -> bool:
    lowered = query.strip().lower()
    if not lowered:
        return False
    markers = [
        " return ",
        " print ",
        " show ",
        " full text",
        " documentid",
        " document id",
        " spreadsheetid",
        " spreadsheet id",
        " a1 value",
        " value in a1",
        " text",
        " count",
    ]
    padded = f" {lowered} "
    return any(marker in padded for marker in markers)


def _plan_query_payload(
    query: str,
    *,
    root_help: dict[str, Any],
    feedback: list[dict[str, Any]],
    model_name: str,
    thinking_level: str,
) -> tuple[dict[str, Any] | None, str]:
    import llm

    planner_output = llm._run_model_api(
        text=_build_query_planner_prompt(query, root_help=root_help, feedback=feedback),
        system_instructions=_load_planner_system_instructions(),
        model=model_name,
        tool_use_allowed=False,
        force_tool="",
        temperature=0.2,
        thinking_level=thinking_level,
        inference_mode=(
            INFERENCE_MODE_FLEX
            if str(model_name).strip().lower() in _FLEX_SUPPORTED_MODELS
            else DEFAULT_INFERENCE_MODE
        ),
    )
    return _extract_first_json_object(planner_output), planner_output


def _run_query_action(query: str, *, runtime_data: dict[str, Any]) -> str:
    normalized_query = (query or "").strip()
    if not normalized_query:
        return _error_payload("access_google_workspace requires a plain-text query.", executed=False)

    root_help_result = _run_gws_command(["--help"], runtime_data=runtime_data)
    root_help_text = str(root_help_result.get("stdout") or root_help_result.get("stderr") or "")
    root_help = _parse_help_sections(root_help_text) if root_help_text else {}

    max_attempts = runtime_data.get("max_planner_attempts", runtime_data.get("planner_attempts", 4))
    try:
        max_attempts = max(1, min(int(max_attempts), 5))
    except Exception:
        max_attempts = 4

    multi_step_request = _query_looks_multi_step(normalized_query, root_help)
    explicit_result_requested = _query_requests_explicit_result(normalized_query)
    planner_feedback: list[dict[str, Any]] = []
    last_payload: dict[str, Any] | None = None
    last_planner_output = ""

    for attempt in range(1, max_attempts + 1):
        planned_payload, planner_output = _plan_query_payload(
            normalized_query,
            root_help=root_help,
            feedback=planner_feedback,
            model_name=MINIMAL_MODEL,
            thinking_level="low"
        )
        last_planner_output = planner_output

        if not isinstance(planned_payload, dict):
            planner_feedback.append(
                {
                    "attempt": attempt,
                    "error": "Planner did not return a strict JSON object payload.",
                    "planner_output": planner_output,
                }
            )
            continue

        last_payload = planned_payload
        action = _resolve_action(planned_payload)
        if action == "" or action not in {"workflow", "call", "discover", "schema", "help", "validate", "raw", "services", "auth"}:
            planner_feedback.append(
                {
                    "attempt": attempt,
                    "error": "Planner returned an unsupported action. Return a structured payload using workflow, call, discover, schema, help, validate, raw, services, or auth.",
                    "planned_payload": planned_payload,
                }
            )
            continue

        if multi_step_request and action != "workflow":
            planner_feedback.append(
                {
                    "attempt": attempt,
                    "error": "This request needs multiple dependent steps. Return action=workflow with step ids, refs, and verification.",
                    "planned_payload": planned_payload,
                }
            )
            continue

        if action == "workflow":
            if explicit_result_requested and "result" not in planned_payload:
                planner_feedback.append(
                    {
                        "attempt": attempt,
                        "error": "The user asked for a specific returned output. Add a top-level workflow `result` object built from refs/transforms.",
                        "planned_payload": planned_payload,
                    }
                )
                continue

            if _workflow_requires_verification(planned_payload):
                planner_feedback.append(
                    {
                        "attempt": attempt,
                        "error": "Mutating workflows must include a final verification/read step after the write step. Set final_step to that verification step.",
                        "planned_payload": planned_payload,
                    }
                )
                continue

            workflow_text = _run_workflow_action(planned_payload, runtime_data=runtime_data)
            parsed_workflow = _parse_json_response(workflow_text)
            workflow_result: Any = parsed_workflow if parsed_workflow is not None else workflow_text
            if isinstance(workflow_result, dict) and not workflow_result.get("ok") and attempt < max_attempts:
                planner_feedback.append(
                    {
                        "attempt": attempt,
                        "error": "Workflow execution failed. Repair the workflow using this live feedback.",
                        "planned_payload": planned_payload,
                        "workflow_result": workflow_result,
                    }
                )
                continue

            return _json_dumps(
                {
                    "ok": bool(isinstance(workflow_result, dict) and workflow_result.get("ok", False)),
                    "executed": bool(isinstance(workflow_result, dict) and workflow_result.get("executed", False)),
                    "query": normalized_query,
                    "planner_attempts": attempt,
                    "planned_payload": planned_payload,
                    "result": workflow_result,
                }
            )

        validation: dict[str, Any] | None = None
        discover_info: dict[str, Any] | None = None
        if action == "call":
            validation_text = _run_validate_action(planned_payload, runtime_data=runtime_data)
            parsed_validation = _parse_json_response(validation_text)
            validation = parsed_validation if isinstance(parsed_validation, dict) else {"raw": validation_text}
            if not validation.get("valid"):
                planner_feedback.append(
                    {
                        "attempt": attempt,
                        "error": "Planned call path failed validation.",
                        "planned_payload": planned_payload,
                        "validation": validation,
                    }
                )
                continue

            discover_text = _run_discover_action(planned_payload, runtime_data=runtime_data)
            parsed_discover = _parse_json_response(discover_text)
            if isinstance(parsed_discover, dict):
                discover_info = parsed_discover

            missing_required = _collect_missing_required_params(planned_payload, discover_info or {})
            has_placeholders = _contains_unresolved_placeholders(planned_payload.get("params")) or _contains_unresolved_placeholders(planned_payload.get("json"))
            if missing_required or has_placeholders:
                planner_feedback.append(
                    {
                        "attempt": attempt,
                        "error": "Planned call is missing required params or still contains placeholders.",
                        "planned_payload": planned_payload,
                        "validation": validation,
                        "query_guide": (discover_info or {}).get("query_guide"),
                        "missing_required_params": missing_required,
                    }
                )
                continue

        execution_text = _execute_structured_payload(_inherit_runtime_settings(planned_payload, runtime_data), runtime_data=runtime_data)
        parsed_result = _parse_json_response(execution_text)
        execution_result: Any = parsed_result if parsed_result is not None else execution_text

        if isinstance(execution_result, str) and execution_result.startswith("Error") and attempt < max_attempts:
            planner_feedback.append(
                {
                    "attempt": attempt,
                    "error": "Execution failed. Repair the payload using this live feedback.",
                    "planned_payload": planned_payload,
                    "validation": validation,
                    "query_guide": (discover_info or {}).get("query_guide"),
                    "execution_error": execution_result,
                }
            )
            continue

        return _json_dumps(
            {
                "ok": not (isinstance(execution_result, str) and execution_result.startswith("Error")),
                "executed": True,
                "query": normalized_query,
                "planner_attempts": attempt,
                "planned_payload": planned_payload,
                "validation": validation,
                "query_guide": (discover_info or {}).get("query_guide") or (execution_result.get("query_guide") if isinstance(execution_result, dict) else None),
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
            "planner_output": last_planner_output,
            "feedback": planner_feedback,
        }
    )


def _interpret_query_response(query: str, raw_response: str) -> str:
    import llm

    system_instructions = (
        "You are a Google Workspace API response interpreter. "
        "Given the original user request and a raw JSON tool response, "
        "return only the information needed to satisfy the original request. "
        "Do not include planner metadata, payload internals, schema/help blocks, or debugging details. "
        "If the action failed, briefly explain the failure and include the most relevant error detail."
    )
    prompt = (
        f"Original user request:\n{query}\n\n"
        f"Raw tool response JSON:\n{raw_response}\n\n"
        "Return a concise final answer for the user."
    )

    return llm._run_model_api(
        text=prompt,
        system_instructions=system_instructions,
        model=MINIMAL_MODEL,
        tool_use_allowed=False,
        force_tool="",
        temperature=0.2,
        thinking_level="low",
        inference_mode=(
            INFERENCE_MODE_FLEX if _MINIMAL_MODEL_SUPPORTS_FLEX else DEFAULT_INFERENCE_MODE
        ),
    )


def _payload_to_cli_query(payload: dict[str, Any]) -> str | None:
    try:
        prepared_payload = _inherit_runtime_settings(payload, {})
        try:
            prepared_payload = _convert_payload_times_to_utc(prepared_payload)
        except Exception:
            pass

        built = _build_internal_command(prepared_payload)
        if isinstance(built, str):
            return None

        args, _ = built
        stripped_args = _strip_gws_prefix(args)
        return f"gws {shlex.join(stripped_args)}" if stripped_args else "gws"
    except Exception:
        return None


def _extract_cli_queries_from_response(raw_response: dict[str, Any]) -> list[str]:
    queries: list[str] = []
    seen: set[str] = set()

    def add_query(value: str | None) -> None:
        if not value:
            return
        if value in seen:
            return
        seen.add(value)
        queries.append(value)

    planned_payload = raw_response.get("planned_payload")
    if isinstance(planned_payload, dict):
        if _resolve_action(planned_payload) == "workflow":
            steps = planned_payload.get("steps")
            if isinstance(steps, list):
                for step in steps:
                    if not isinstance(step, dict):
                        continue
                    add_query(_payload_to_cli_query(_extract_step_payload(step)))
        else:
            add_query(_payload_to_cli_query(planned_payload))

    if queries:
        return queries

    result = raw_response.get("result")
    if isinstance(result, dict):
        steps = result.get("steps")
        if isinstance(steps, list):
            for step in steps:
                if not isinstance(step, dict):
                    continue
                payload = step.get("payload")
                if isinstance(payload, dict):
                    add_query(_payload_to_cli_query(payload))

    return queries


def _run_smoke_suite(*, pretty: bool, timeout: int | None = None) -> int:
    failures = 0
    for index, (label, query) in enumerate(SMOKE_TEST_QUERIES, start=1):
        runtime_data: dict[str, Any] = {}
        if timeout is not None:
            runtime_data["timeout"] = int(timeout)

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


def access_google_workspace(query: str = "") -> str:
    """
    Plain-text Google Workspace entry point backed by the gws CLI.

    Examples:
    - Show the Gmail labels for the authenticated user.
    - How do I create a Google Docs document?
    - Create a Google Doc populated with a list of all Google Drive file names.
    - Create a Google Sheet named Drive File Count and write the total number of files into cell A1.
    - Read the content of Google Doc with ID 'DOCUMENT_ID' and print the full text.
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


def _build_runtime_data(args: Any) -> dict[str, Any]:
    runtime_data: dict[str, Any] = {}
    if getattr(args, "timeout", None) is not None:
        runtime_data["timeout"] = int(args.timeout)
    if getattr(args, "api_key", None):
        runtime_data["api_key"] = str(args.api_key)
    if getattr(args, "api_key_file", None):
        runtime_data["api_key_file"] = str(args.api_key_file)
    return runtime_data


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Plain-text Google Workspace wrapper backed by the gws CLI")
    parser.add_argument(
        "query",
        nargs="?",
        help="Plain-text query. If omitted, query may be read from STDIN.",
    )
    parser.add_argument("-f", "--file", dest="query_file", help="Path to a plain-text query file")
    parser.add_argument("--smoke", action="store_true", help="Run the built-in smoke suite")
    parser.add_argument("-t", "--timeout", type=int, help="Override timeout in seconds")
    parser.add_argument("--api-key", help="Google API key (same key used by access_youtube)")
    parser.add_argument(
        "--api-key-file",
        default=DEFAULT_API_KEY_FILE,
        help="Path to text file containing Google API key",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output when possible")
    args = parser.parse_args()

    runtime_data = _build_runtime_data(args)

    try:
        if args.smoke:
            raise SystemExit(_run_smoke_suite(pretty=args.pretty, timeout=args.timeout))

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
            raise SystemExit(_run_smoke_suite(pretty=args.pretty, timeout=args.timeout))

        output = _run_query_action(query_text, runtime_data=runtime_data)
        print(_format_output(output, pretty=args.pretty))
    except SystemExit:
        raise
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
