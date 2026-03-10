import json
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent
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