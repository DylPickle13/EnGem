import ast
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Maximum number of characters to keep from tool stdout
MAX_OUTPUT_CHARS = 100_000
REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATED_IMAGES_DIR = REPO_ROOT / "generated_images"
GENERATED_VIDEOS_DIR = REPO_ROOT / "generated_videos"
GENERATED_FILES_DIR = REPO_ROOT / "generated_files"
GENERATED_DOCUMENTS_DIR = REPO_ROOT / "generated_documents"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
DOCUMENT_EXTENSIONS = {
    ".pdf", ".txt", ".md", ".csv", ".json", ".xml", ".yaml", ".yml",
    ".html", ".htm", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip",
}
TRACKED_OUTPUT_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS | DOCUMENT_EXTENSIONS


def _snapshot_repo_root_files() -> set[Path]:
    return {
        path.resolve()
        for path in REPO_ROOT.iterdir()
        if path.is_file()
    }


def _ensure_generated_output_directories() -> None:
    for directory in (
        GENERATED_IMAGES_DIR,
        GENERATED_VIDEOS_DIR,
        GENERATED_FILES_DIR,
        GENERATED_DOCUMENTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def _get_output_directory(path: Path) -> Path | None:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return GENERATED_IMAGES_DIR
    if suffix in VIDEO_EXTENSIONS:
        return GENERATED_VIDEOS_DIR
    if suffix in DOCUMENT_EXTENSIONS:
        return GENERATED_DOCUMENTS_DIR
    return None


def _relocate_generated_outputs(previous_snapshot: set[Path]) -> list[Path]:
    _ensure_generated_output_directories()
    relocated_paths: list[Path] = []

    for path in REPO_ROOT.iterdir():
        if not path.is_file() or path.suffix.lower() not in TRACKED_OUTPUT_EXTENSIONS:
            continue

        resolved_path = path.resolve()
        if resolved_path in previous_snapshot:
            continue

        destination_dir = _get_output_directory(path)
        if destination_dir is None:
            continue

        destination_path = destination_dir / path.name
        if destination_path.exists():
            destination_path.unlink()
        shutil.move(str(path), str(destination_path))
        relocated_paths.append(destination_path.resolve())

    return relocated_paths


def run_python(code: str) -> str:
    """
    Run Python code safely. 
    Use print statements to ensure code execution is visible in the output.
    """
    os.chdir(REPO_ROOT)
    try:
        ast.parse(code, mode="exec")
    except SyntaxError as e:
        output_text = f"SyntaxError: {e}"
        return f"Code run:\n```python{code}```\n\nOutput:\n{output_text}"

    preexisting_root_files = _snapshot_repo_root_files()

    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
        f.write(code.encode())
        filename = f.name

    relocated_outputs: list[Path] = []
    try:
        result = subprocess.run(
            [sys.executable, filename],
            capture_output=True,
            text=True,
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""

        def _truncate_stdout(s: str) -> str:
            if len(s) <= MAX_OUTPUT_CHARS:
                return s
            return s[:MAX_OUTPUT_CHARS] + f"\n\n...[truncated stdout; original length={len(s)} chars]"

        truncated_stdout = _truncate_stdout(stdout)

        if stderr and stderr.strip():
            # include stderr and the (possibly truncated) stdout
            if truncated_stdout and truncated_stdout.strip():
                output_text = f"STDERR:\n{stderr}\n\nSTDOUT:\n{truncated_stdout}"
            else:
                output_text = f"STDERR:\n{stderr}"
        else:
            output_text = truncated_stdout
    except subprocess.TimeoutExpired as e:
        output_text = f"TimeoutExpired: {e}"
    finally:
        try:
            relocated_outputs = _relocate_generated_outputs(preexisting_root_files)
        except Exception as e:
            if output_text.strip():
                output_text += f"\n\nArtifact relocation error: {e}"
            else:
                output_text = f"Artifact relocation error: {e}"
        try:
            os.unlink(filename)
        except OSError:
            pass

    if relocated_outputs:
        artifact_lines = "\n".join(str(path) for path in relocated_outputs)
        if output_text.strip():
            output_text += f"\n\nGenerated artifacts:\n{artifact_lines}"
        else:
            output_text = f"Generated artifacts:\n{artifact_lines}"

    return f"Code run:\n```python{code}```\n\nOutput:\n{output_text}"