from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Ensure repository root is on sys.path so top-level modules (like config)
# can be imported when this module is run directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from config import MEDIUM_MODEL, get_paid_gemini_api_key
except Exception:
    MEDIUM_MODEL = ""

    def get_paid_gemini_api_key() -> str:
        return str(os.getenv("PAID_GEMINI_API_KEY", "")).strip()

COPILOT_COMMAND = "copilot"
COPILOT_TIMEOUT_SECONDS = 300
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
COPILOT_PROVIDER_BASE_URL = str(os.getenv("COPILOT_PROVIDER_BASE_URL", GEMINI_OPENAI_BASE_URL)).strip() or GEMINI_OPENAI_BASE_URL
COPILOT_PROVIDER_TYPE = str(os.getenv("COPILOT_PROVIDER_TYPE", "openai")).strip() or "openai"
GEMINI_COMPAT_DEFAULT_MODEL = "gemini-3-flash-preview"
_CONFIG_DEFAULT_MODEL = str(MEDIUM_MODEL or "").strip()
COPILOT_DEFAULT_MODEL = (
    str(os.getenv("COPILOT_DEFAULT_MODEL", _CONFIG_DEFAULT_MODEL)).strip()
    or GEMINI_COMPAT_DEFAULT_MODEL
)
COPILOT_MODEL = str(os.getenv("COPILOT_MODEL", COPILOT_DEFAULT_MODEL)).strip() or COPILOT_DEFAULT_MODEL
COPILOT_DISABLE_BUILTIN_MCPS = str(os.getenv("COPILOT_DISABLE_BUILTIN_MCPS", "true")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
COPILOT_DISABLE_TOOL_CALLS = str(os.getenv("COPILOT_DISABLE_TOOL_CALLS", "false")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
COPILOT_AUTOPILOT = str(os.getenv("COPILOT_AUTOPILOT", "true")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
COPILOT_STREAM_MODE = str(os.getenv("COPILOT_STREAM_MODE", "off")).strip().lower()
if COPILOT_STREAM_MODE not in {"on", "off"}:
    COPILOT_STREAM_MODE = "off"
COPILOT_EXCLUDED_TOOLS = str(
    os.getenv(
        "COPILOT_EXCLUDED_TOOLS",
        "bash,write_bash,read_bash,stop_bash,list_bash,view,create,edit,web_fetch,"
        "report_intent,fetch_copilot_cli_documentation,skill,sql,read_agent,list_agents,"
        "grep,glob,task",
    )
).strip()
COPILOT_INSTALL_PROMPT_MARKERS = (
    "cannot find github copilot cli",
    "install github copilot cli",
)
COPILOT_AUTH_ERROR_MARKERS = (
    "no authentication information found",
    "authenticate with github",
    "copilot can be authenticated with github",
)
MAIN_TEST_PROMPT = "read each tool in the tools folder, and summarize their uses. "


def _looks_like_install_prompt(*texts: str) -> bool:
    combined_text = "\n".join(str(text or "") for text in texts).lower()
    return any(marker in combined_text for marker in COPILOT_INSTALL_PROMPT_MARKERS)


def _looks_like_auth_error(*texts: str) -> bool:
    combined_text = "\n".join(str(text or "") for text in texts).lower()
    return any(marker in combined_text for marker in COPILOT_AUTH_ERROR_MARKERS)


def _run_copilot_command(prompt: str, model_name: str) -> tuple[str, str]:
    copilot_path = shutil.which(COPILOT_COMMAND)
    if copilot_path is None:
        raise RuntimeError(
            "GitHub Copilot CLI was not found in PATH. Install it and ensure the `copilot` command is available."
        )

    cleaned_prompt = str(prompt or "").strip()
    provider_api_key = str(os.getenv("COPILOT_PROVIDER_API_KEY", "")).strip() or str(get_paid_gemini_api_key() or "").strip()
    if not provider_api_key:
        raise RuntimeError(
            "Gemini API key is missing. Set COPILOT_PROVIDER_API_KEY or PAID_GEMINI_API_KEY, "
            "or configure get_paid_gemini_api_key() in config.py."
        )

    copilot_env = os.environ.copy()
    copilot_env["COPILOT_PROVIDER_BASE_URL"] = COPILOT_PROVIDER_BASE_URL
    copilot_env["COPILOT_PROVIDER_TYPE"] = COPILOT_PROVIDER_TYPE
    copilot_env["COPILOT_PROVIDER_API_KEY"] = provider_api_key
    copilot_env["COPILOT_MODEL"] = model_name

    command = [
        copilot_path,
        "--prompt",
        cleaned_prompt,
        "--model",
        model_name,
        "--allow-all-tools",
        "--stream",
        COPILOT_STREAM_MODE,
    ]
    if COPILOT_AUTOPILOT:
        command.append("--autopilot")
    if COPILOT_DISABLE_BUILTIN_MCPS:
        command.append("--disable-builtin-mcps")
    if COPILOT_DISABLE_TOOL_CALLS and COPILOT_EXCLUDED_TOOLS:
        # Optional fallback if a provider/tooling combo regresses.
        command.append(f"--excluded-tools={COPILOT_EXCLUDED_TOOLS}")

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=COPILOT_TIMEOUT_SECONDS,
            cwd=str(_REPO_ROOT),
            env=copilot_env,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Copilot command timed out after {COPILOT_TIMEOUT_SECONDS}s: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Error running Copilot command: {exc}") from exc

    stdout = result.stdout or ""
    stderr = result.stderr or ""

    if _looks_like_install_prompt(stdout, stderr):
        raise RuntimeError(
            "GitHub Copilot CLI is not installed. The resolved `copilot` command is an installer wrapper "
            "that prompted for installation. Install Copilot CLI interactively first, then rerun this tool."
        )

    if _looks_like_auth_error(stdout, stderr):
        raise RuntimeError(
            "GitHub Copilot CLI is installed but not authenticated. Run `copilot` and use `/login`, "
            "set COPILOT_GITHUB_TOKEN, GH_TOKEN, or GITHUB_TOKEN, or run `gh auth login`, then rerun this tool."
        )

    if result.returncode != 0:
        details = "\n".join(part for part in (stdout, stderr) if part) or "No output"
        raise RuntimeError(f"Copilot command failed (exit {result.returncode}): {details}")

    return stdout, stderr


def run_copilot(prompt: str) -> str:
    """
    Run GitHub Copilot CLI for a prompt and return its captured output.
    """
    cleaned_prompt = str(prompt or "").strip()
    if not cleaned_prompt:
        return "Prompt is empty. Provide a prompt for GitHub Copilot CLI."

    model_name = COPILOT_MODEL

    try:
        stdout, stderr = _run_copilot_command(cleaned_prompt, model_name)
    except Exception as exc:
        return f"Error running Copilot: {exc}"

    if not stdout and not stderr:
        return "Copilot returned no output."

    if stdout and stderr:
        separator = "" if stdout.endswith("\n") else "\n"
        return f"{stdout}{separator}{stderr}"

    return stdout or stderr


def main() -> None:
    """
    Simple local entry point for trying run_copilot with a hardcoded test prompt.
    Update MAIN_TEST_PROMPT to quickly try different prompt wording.
    """
    print(run_copilot(MAIN_TEST_PROMPT))


if __name__ == "__main__":
    main()
