import json
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent
_IMAGES_DIR = _REPO_ROOT / "generated_images"
_VIDEOS_DIR = _REPO_ROOT / "generated_videos"
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"}
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}


def get_generated_media(max_items: str = "80") -> str:
    """Return a JSON list of recent generated media files.

    The output is a JSON object with a single key "media" that contains
    both images and videos, sorted by newest first.
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

    for file_path in _iter_files(_IMAGES_DIR, _IMAGE_EXTENSIONS):
        entries.append(_build_entry(file_path, "image"))

    for file_path in _iter_files(_VIDEOS_DIR, _VIDEO_EXTENSIONS):
        entries.append(_build_entry(file_path, "video"))

    entries.sort(key=lambda item: item.get("modified_ts", 0.0), reverse=True)
    return entries[:limit]


def _iter_files(folder: Path, valid_suffixes: set[str]):
    if not folder.exists() or not folder.is_dir():
        return

    for path in folder.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in valid_suffixes:
            continue
        yield path


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