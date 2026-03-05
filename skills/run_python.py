import ast
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Maximum number of characters to keep from tool stdout
MAX_OUTPUT_CHARS = 100_000


def run_python(code: str) -> str:
    """
    Run Python code safely. 
    Use print statements to ensure code execution is visible in the output.
    """
    os.chdir(Path(__file__).resolve().parent.parent)
    try:
        ast.parse(code, mode="exec")
    except SyntaxError as e:
        output_text = f"SyntaxError: {e}"
        return f"Code run:\n```python{code}```\n\nOutput:\n{output_text}"

    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
        f.write(code.encode())
        filename = f.name

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
            os.unlink(filename)
        except OSError:
            pass

    return f"Code run:\n```python{code}```\n\nOutput:\n{output_text}"