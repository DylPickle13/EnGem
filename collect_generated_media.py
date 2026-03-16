import json
from importlib import import_module
from pathlib import Path

import history


def _find_repo_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "requirements.txt").exists() or (candidate / "generated_files").exists():
            return candidate
    return current


_REPO_ROOT = _find_repo_root()
_OUTPUTS_DIR = _REPO_ROOT / "generated_files"
_LEGACY_OUTPUT_DIRS = (
    _REPO_ROOT / "generated_images",
    _REPO_ROOT / "generated_videos",
    _REPO_ROOT / "generated_documents",
)
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"}
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
_DOCUMENT_EXTENSIONS = {
    ".pdf", ".txt", ".md", ".csv", ".json", ".xml", ".yaml", ".yml",
    ".html", ".htm", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".tex",
}

MEDIA_SELECTOR_FILE = Path(__file__).parent / "agent_instructions" / "media_selector.md"


# Directories and allowed/restricted outputs (match llm.py expectations)
GENERATED_IMAGES_DIR = (_REPO_ROOT / "generated_images").resolve()
GENERATED_VIDEOS_DIR = (_REPO_ROOT / "generated_videos").resolve()
GENERATED_FILES_DIR = (_REPO_ROOT / "generated_files").resolve()
GENERATED_DOCUMENTS_DIR = (_REPO_ROOT / "generated_documents").resolve()
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


def parse_selected_media_paths(selector_response: str, max_results: int = 10) -> list[str]:
    """Parse a model selector response and return validated, normalized media file paths.

    The function extracts a JSON payload containing `media_paths`, validates each
    path is an existing file under allowed output directories, applies restricted
    directory checks, and returns up to `max_results` unique normalized paths.
    """
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
        if len(normalized) >= max_results:
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
    if not any(_is_under_directory(path, directory) for directory in ALLOWED_OUTPUT_DIRECTORIES):
        return None
    if any(_is_under_directory(path, directory) for directory in RESTRICTED_OUTPUT_DIRECTORIES):
        if path.suffix.lower() not in SUPPORTED_OUTPUT_FILE_EXTENSIONS:
            return None
    return str(path)


def _is_under_directory(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except Exception:
        return False


def get_generated_media(max_items: str = "80") -> str:
    """Return a JSON list of recent generated output files.

    The output is a JSON object with a single key "media" that contains
    images, videos, and generated document/file outputs, sorted by newest first.
    """
    try:
        limit = int(str(max_items).strip())
    except Exception:
        limit = 80

    limit = max(1, min(limit, 300))
    media = _collect_media_catalog(limit=limit)
    return json.dumps({"media": media}, ensure_ascii=False)


def select_media_paths(
    history_file: str,
    user_message: str,
    temperature: float,
    history_cache: object | None = None,
) -> list[str]:
    try:
        llm = import_module("llm")
        run_model_api = llm._run_model_api
    except Exception as exc:
        print(f"Error importing media selection model helper: {exc}")
        return []

    catalog_json = get_generated_media("120")
    selector_input = (
        "Latest user request:\n"
        f"{user_message}\n\n"
        "Use the conversation history and this generated media catalog JSON:\n"
        f"{catalog_json}"
    )

    if history_cache is not None:
        selector_response = run_model_api(
            text=selector_input,
            system_instructions=MEDIA_SELECTOR_FILE.read_text(encoding="utf-8"),
            model=llm.MINIMAL_MODEL,
            tool_use_allowed=False,
            force_tool="",
            temperature=temperature,
            thinking_level="low",
            history_cache=history_cache,
            current_history_text=history_cache.history_text,
        )
    else:
        selector_response = run_model_api(
            text=selector_input,
            system_instructions=MEDIA_SELECTOR_FILE.read_text(encoding="utf-8"),
            model=llm.MINIMAL_MODEL,
            tool_use_allowed=False,
            force_tool="",
            temperature=temperature,
            thinking_level="low",
            current_history_text=history.get_conversation_history(history_file=history_file),
        )

    return parse_selected_media_paths(selector_response)


def _collect_media_catalog(limit: int = 80) -> list[dict]:
    entries: list[dict] = []

    for file_path in _iter_files():
        entries.append(_build_entry(file_path, _infer_media_type(file_path)))

    entries.sort(key=lambda item: item.get("modified_ts", 0.0), reverse=True)
    return entries[:limit]


def _iter_files():
    seen_paths: set[Path] = set()

    for folder in (_OUTPUTS_DIR, *_LEGACY_OUTPUT_DIRS):
        if not folder.exists() or not folder.is_dir():
            continue

        for path in folder.iterdir():
            if not path.is_file():
                continue

            resolved_path = path.resolve()
            if resolved_path in seen_paths:
                continue

            seen_paths.add(resolved_path)
            yield path


def _infer_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _IMAGE_EXTENSIONS:
        return "image"
    if suffix in _VIDEO_EXTENSIONS:
        return "video"
    if suffix in _DOCUMENT_EXTENSIONS:
        return "document"
    return "file"


def _build_entry(path: Path, media_type: str) -> dict:
    try:
        stat = path.stat()
        modified_ts = float(stat.st_mtime)
        size_bytes = int(stat.st_size)
    except Exception:
        modified_ts = 0.0
        size_bytes = 0

    return {
        "type": media_type,
        "name": path.name,
        "path": str(path),
        "modified_ts": modified_ts,
        "size_bytes": size_bytes,
    }