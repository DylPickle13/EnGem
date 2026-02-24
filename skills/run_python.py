import ast
import os
import subprocess
import sys
import tempfile
from pathlib import Path


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
        return f"Code run:\n{code}\n\nOutput:\n{output_text}"

    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
        f.write(code.encode())
        filename = f.name

    try:
        result = subprocess.run(
            [sys.executable, filename],
            capture_output=True,
            text=True,
            timeout=300,
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