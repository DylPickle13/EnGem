import ast
import os
import subprocess
import sys
import tempfile


def run_python(code: str) -> str:
    """
    Run Python code safely and return a single string containing:
    1) the code that was executed, and
    2) either stderr (if present) or stdout.

    code: the Python code to execute
    """
    try:
        ast.parse(code, mode="exec")
    except SyntaxError as e:
        output_text = f"SyntaxError: {e}"
        return f"Code run:\n{code}\n\nOutput:\n{output_text}"

    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
        f.write(code.encode())
        filename = f.name

    try:
        result = subprocess.run(
            [sys.executable, filename],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output_text = result.stderr if result.stderr else result.stdout
    except subprocess.TimeoutExpired as e:
        output_text = f"TimeoutExpired: {e}"
    finally:
        try:
            os.unlink(filename)
        except OSError:
            pass

    return f"Code run:\n{code}\n\nOutput:\n{output_text}"