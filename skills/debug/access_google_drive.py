import json
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
	return Path(__file__).resolve().parent.parent


def _parse_payload(payload: str) -> dict[str, Any]:
	text = (payload or "").strip()
	if not text:
		return {}
	try:
		parsed = json.loads(text)
	except json.JSONDecodeError:
		return {"value": text}
	if isinstance(parsed, dict):
		return parsed
	return {"value": parsed}


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


def _run_gog(args: list[str], *, account: str | None = None, client: str | None = None, use_json: bool = True, timeout: int = 300) -> str:
	if shutil.which("gog") is None:
		return "Error: gog CLI not found in PATH. Install it and verify `gog --version` works."

	command = ["gog"]
	if client:
		command.extend(["--client", client])
	if account:
		command.extend(["--account", account])
	if use_json:
		command.append("--json")
	command.extend(args)

	try:
		result = subprocess.run(
			command,
			capture_output=True,
			text=True,
			timeout=timeout,
			cwd=str(_repo_root()),
		)
	except subprocess.TimeoutExpired:
		return f"Error: command timed out after {timeout}s."
	except Exception as exc:
		return f"Error running gog command: {exc}"

	stdout = (result.stdout or "").strip()
	stderr = (result.stderr or "").strip()

	if result.returncode != 0:
		details = stderr or stdout or "No output"
		return f"Error (exit {result.returncode}): {details}"

	if stdout:
		return stdout
	if stderr:
		return stderr
	return "OK"


def list_drive_files(payload: str = "") -> str:
	"""
	List files in Google Drive.

	Payload examples:
	- ""
	- "{\"max\": 20}"
	- "{\"parent\": \"<folderId>\", \"max\": 50, \"account\": \"you@gmail.com\"}"
	- "{\"no_all_drives\": true, \"raw_flags\": \"--plain\"}"
	"""
	data = _parse_payload(payload)
	max_items = str(data.get("max", 20))
	account = data.get("account")
	client = data.get("client")

	args = ["drive", "ls", "--max", max_items]

	parent = data.get("parent")
	if parent:
		args.extend(["--parent", str(parent)])

	if _as_bool(data.get("no_all_drives"), default=False):
		args.append("--no-all-drives")

	raw_flags = str(data.get("raw_flags", "")).strip()
	if raw_flags:
		args.extend(shlex.split(raw_flags))

	return _run_gog(args, account=account, client=client, use_json=True)


def search_drive_files(payload: str) -> str:
	"""
	Search Google Drive files by query.

	Payload examples:
	- "invoice"
	- "{\"query\": \"invoice\", \"max\": 20}"
	- "{\"query\": \"mimeType = 'application/pdf'\", \"raw_query\": true}"
	"""
	data = _parse_payload(payload)
	query = data.get("query", data.get("value"))
	if not query:
		return "Error: missing search query."

	max_items = str(data.get("max", 20))
	account = data.get("account")
	client = data.get("client")

	args = ["drive", "search", str(query), "--max", max_items]

	if _as_bool(data.get("raw_query"), default=False):
		args.append("--raw-query")
	if _as_bool(data.get("no_all_drives"), default=False):
		args.append("--no-all-drives")

	raw_flags = str(data.get("raw_flags", "")).strip()
	if raw_flags:
		args.extend(shlex.split(raw_flags))

	return _run_gog(args, account=account, client=client, use_json=True)


def get_drive_file(payload: str) -> str:
	"""
	Get metadata for a Drive file by file ID.

	Payload examples:
	- "<fileId>"
	- "{\"file_id\": \"<fileId>\"}"
	"""
	data = _parse_payload(payload)
	file_id = data.get("file_id", data.get("id", data.get("value")))
	if not file_id:
		return "Error: missing file ID."

	account = data.get("account")
	client = data.get("client")
	args = ["drive", "get", str(file_id)]
	return _run_gog(args, account=account, client=client, use_json=True)


def upload_drive_file(payload: str) -> str:
	"""
	Upload a file to Google Drive.

	Payload examples:
	- "./local/path/file.txt"
	- "{\"path\": \"./report.pdf\", \"parent\": \"<folderId>\"}"
	- "{\"path\": \"./doc.docx\", \"convert\": true, \"name\": \"Doc Copy\"}"
	- "{\"path\": \"./file.txt\", \"replace\": \"<fileId>\"}"
	"""
	data = _parse_payload(payload)
	path = data.get("path", data.get("file_path", data.get("value")))
	if not path:
		return "Error: missing local file path to upload."

	account = data.get("account")
	client = data.get("client")

	args = ["drive", "upload", str(path)]

	parent = data.get("parent")
	if parent:
		args.extend(["--parent", str(parent)])

	replace = data.get("replace")
	if replace:
		args.extend(["--replace", str(replace)])

	name = data.get("name")
	if name:
		args.extend(["--name", str(name)])

	if _as_bool(data.get("convert"), default=False):
		args.append("--convert")

	convert_to = data.get("convert_to")
	if convert_to:
		args.extend(["--convert-to", str(convert_to)])

	raw_flags = str(data.get("raw_flags", "")).strip()
	if raw_flags:
		args.extend(shlex.split(raw_flags))

	return _run_gog(args, account=account, client=client, use_json=True)


def download_drive_file(payload: str) -> str:
	"""
	Download a file from Google Drive by file ID.

	Payload examples:
	- "<fileId>"
	- "{\"file_id\": \"<fileId>\", \"out\": \"./download.bin\"}"
	- "{\"file_id\": \"<fileId>\", \"format\": \"pdf\", \"out\": \"./doc.pdf\"}"
	"""
	data = _parse_payload(payload)
	file_id = data.get("file_id", data.get("id", data.get("value")))
	if not file_id:
		return "Error: missing file ID to download."

	account = data.get("account")
	client = data.get("client")

	args = ["drive", "download", str(file_id)]

	file_format = data.get("format")
	if file_format:
		args.extend(["--format", str(file_format)])

	out = data.get("out", data.get("output"))
	if out:
		args.extend(["--out", str(out)])

	raw_flags = str(data.get("raw_flags", "")).strip()
	if raw_flags:
		args.extend(shlex.split(raw_flags))

	return _run_gog(args, account=account, client=client, use_json=True)


def create_drive_folder(payload: str) -> str:
	"""
	Create a folder in Google Drive.

	Payload examples:
	- "My Folder"
	- "{\"name\": \"My Folder\", \"parent\": \"<folderId>\"}"
	"""
	data = _parse_payload(payload)
	folder_name = data.get("name", data.get("value"))
	if not folder_name:
		return "Error: missing folder name."

	account = data.get("account")
	client = data.get("client")

	args = ["drive", "mkdir", str(folder_name)]
	parent = data.get("parent")
	if parent:
		args.extend(["--parent", str(parent)])

	return _run_gog(args, account=account, client=client, use_json=True)


def move_drive_file(payload: str) -> str:
	"""
	Move a file into another folder.

	Payload example:
	- "{\"file_id\": \"<fileId>\", \"parent\": \"<destinationFolderId>\"}"
	"""
	data = _parse_payload(payload)
	file_id = data.get("file_id", data.get("id"))
	parent = data.get("parent")
	if not file_id or not parent:
		return "Error: both file_id and parent are required."

	account = data.get("account")
	client = data.get("client")

	args = ["drive", "move", str(file_id), "--parent", str(parent)]
	return _run_gog(args, account=account, client=client, use_json=True)


def delete_drive_file(payload: str) -> str:
	"""
	Delete a file by ID (trash by default; permanent when requested).

	Payload examples:
	- "<fileId>"
	- "{\"file_id\": \"<fileId>\", \"permanent\": true}"
	"""
	data = _parse_payload(payload)
	file_id = data.get("file_id", data.get("id", data.get("value")))
	if not file_id:
		return "Error: missing file ID to delete."

	account = data.get("account")
	client = data.get("client")

	args = ["drive", "delete", str(file_id)]
	if _as_bool(data.get("permanent"), default=False):
		args.append("--permanent")

	return _run_gog(args, account=account, client=client, use_json=True)
